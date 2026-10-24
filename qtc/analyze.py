#!/usr/bin/env python3
"""Statistics, tables and figures for the Quantize-then-Correct study (CPU; numpy/scipy/pandas/matplotlib).

Inputs (from a results checkout or the pod volume):  <runs>/meta/*.json, <runs>/qtc/{metrics,gate,recon}/*.json,
                                                      optional <runs>/qtc/bench/*.json
Outputs: <out>/tables/*.csv + *.tex (booktabs), <out>/figures/*.png (300 dpi) + *.pdf, <out>/summary.json

Analyses
  A. Noise floor: FP16 seed-7 vs seed-42 of the same clip. For every gate quantity the 95th percentile over clips
     defines the "within noise" bound; the gate's false-positive rate is the failure rate of these pairs.
  B. Absolute kinematic levels per arm (variant x drive) and post-processing (none | kapt_smoothnet).
  C. Paired tests over clip-seed pairs (Wilcoxon signed-rank, Holm across the metrics of each family,
     Cliff's delta, bootstrap 95 % CI of the median paired difference, 10 000 resamples):
       H1 damage   : quant/raw/none            vs fp16/raw/none
       H2 pre      : quant/pre/none            vs quant/raw/none
       H3 post     : quant/raw/kapt_smoothnet  vs quant/raw/none
       H4 combined : quant/pre/kapt_smoothnet  vs fp16/raw/none
  D. Gate pass rates (Wilson 95 % CI) and within-noise rates per arm and post.
  E. Kinematic-pixel disagreement: Kendall tau between arm rankings by LPIPS and by kinematic deltas;
     Spearman rho over pairs between LPIPS and |relative acceleration-variance delta|.
  F. Cost: DiT weight memory, peak VRAM and seconds per video per arm (from meta; bench files if present).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

KIN = ["acceleration_variance", "mean_limb_length_std", "var_accel_norm", "violation_fraction"]
KIN_LABEL = {"acceleration_variance": "Accel. var.", "mean_limb_length_std": "Limb-length s.d.",
             "var_accel_norm": "Norm. jitter", "violation_fraction": "Violation frac."}
DELTAS = ["rel_delta_acceleration_variance", "rel_delta_mean_limb_length_std", "rel_delta_var_accel_norm",
          "signed_delta_violation_fraction", "mean_NPE", "p95_NPE"]
DELTA_LABEL = {"rel_delta_acceleration_variance": r"$|\Delta|$ accel. var. (rel.)",
               "rel_delta_mean_limb_length_std": r"$|\Delta|$ limb s.d. (rel.)",
               "rel_delta_var_accel_norm": r"$|\Delta|$ jitter (rel.)",
               "signed_delta_violation_fraction": r"$\Delta$ violation frac.",
               "mean_NPE": "mean NPE", "p95_NPE": "NPE$_{95}$"}
VAR_LABEL = {"fp16": "FP16", "vidit": "W8A8", "w4a8": "W4A8", "w4a8mp": "W4A8-MP"}
RNG = np.random.default_rng(20260922)


# ----------------------------------------------------------------------------- loading
def parse_uid(uid: str) -> dict:
    p = uid.split("-")
    return {"variant": p[0], "clip": p[3], "seed": int(p[4][1:]), "drive": "pre" if "-dpre" in uid else "raw"}


def obsolete(uid: str) -> bool:
    """Pilot-only pre-correction fields (v1 '-dpre', v2 '-dpre2') are excluded; the study uses v3 ('-dpre3')."""
    return uid.endswith("-dpre") or uid.endswith("-dpre2")


def load(runs: Path):
    q = runs / "qtc"
    rows = []
    for f in sorted((q / "metrics").glob("*.json")):
        r = json.loads(f.read_text())
        rows.append({**parse_uid(r["uid"]), "uid": r["uid"], "post": r["post"], **{k: r.get(k) for k in KIN}})
    absdf = pd.DataFrame(rows)
    rows = []
    for f in sorted((q / "gate").glob("*.json")):
        r = json.loads(f.read_text())
        d = {**parse_uid(r["cand"]), "cand": r["cand"], "ref": r["ref"], "tag": r["tag"], "post": r["post"],
             "passed": bool(r["passed"])}
        d.update({k: r["deltas"].get(k) for k in DELTAS})
        d.update({f"check_{k}": v for k, v in r["checks"].items() if not k.startswith("finite_")})
        rows.append(d)
    gate = pd.DataFrame(rows)
    rows = []
    for f in sorted((q / "recon").glob("*.json")):
        r = json.loads(f.read_text())
        rows.append({"cand": r["cand"], "ref": r["ref"], **{k: r[k] for k in ("psnr_db", "ssim", "lpips", "tlpips", "tmse")}})
    recon = pd.DataFrame(rows)
    rows = []
    for f in sorted((runs / "meta").glob("*.json")):
        m = json.loads(f.read_text())
        if "uid" not in m:
            continue
        rows.append({"uid": m["uid"], **parse_uid(m["uid"]), "gen_seconds": m.get("gen_seconds"),
                     "peak_vram_gb": m.get("peak_vram_gb"), "int8_layers": m.get("int8_layers"),
                     "int4_layers": m.get("int4_layers"), "dit_params": m.get("dit_params"),
                     "int8_param_fraction": m.get("int8_param_fraction"),
                     "w4a8_packed_weight_bytes": m.get("w4a8_packed_weight_bytes"),
                     "w4a8_dense_weight_bytes": m.get("w4a8_dense_weight_bytes")})
    meta = pd.DataFrame(rows)
    if not meta.empty and not absdf.empty:
        meta = meta[meta["uid"].isin(set(absdf["uid"]))]          # only videos of this study
    absdf = absdf[~absdf["uid"].map(obsolete)] if not absdf.empty else absdf
    gate = gate[~gate["cand"].map(obsolete)] if not gate.empty else gate
    recon = recon[~recon["cand"].map(obsolete)] if not recon.empty else recon
    meta = meta[~meta["uid"].map(obsolete)] if not meta.empty else meta
    bench = [json.loads(f.read_text()) for f in sorted((q / "bench").glob("*.json"))] if (q / "bench").is_dir() else []
    return absdf, gate, recon, meta, bench


# ----------------------------------------------------------------------------- statistics
def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return float((gt - lt) / (len(x) * len(y)))


def boot_ci_median(d: np.ndarray, n: int = 10_000, alpha: float = 0.05):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return float("nan"), float("nan")
    idx = RNG.integers(0, len(d), size=(n, len(d)))
    meds = np.median(d[idx], axis=1)
    return float(np.quantile(meds, alpha / 2)), float(np.quantile(meds, 1 - alpha / 2))


def holm(pvals):
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan")
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def paired(df_a: pd.DataFrame, df_b: pd.DataFrame, metrics, label_a: str, label_b: str, family: str):
    """Paired comparison a vs b on (clip, seed). Returns rows with Holm-adjusted p within the family."""
    m = df_a.merge(df_b, on=["clip", "seed"], suffixes=("_a", "_b"))
    rows = []
    for k in metrics:
        a, b = m[f"{k}_a"].to_numpy(float), m[f"{k}_b"].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        d = a - b
        if len(d) >= 5 and np.any(d != 0):
            p = float(stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
        else:
            p = float("nan")
        lo, hi = boot_ci_median(d)
        rows.append({"family": family, "a": label_a, "b": label_b, "metric": k, "n": int(len(d)),
                     "median_a": float(np.median(a)) if len(a) else np.nan,
                     "median_b": float(np.median(b)) if len(b) else np.nan,
                     "median_diff": float(np.median(d)) if len(d) else np.nan, "ci_lo": lo, "ci_hi": hi,
                     "frac_a_lower": float(np.mean(d < 0)) if len(d) else np.nan,
                     "cliffs_delta": cliffs_delta(a, b) if len(d) else np.nan, "p": p})
    adj = holm([r["p"] for r in rows])
    for r, q in zip(rows, adj):
        r["p_holm"] = float(q)
    return rows


def sel(df, **kw):
    for k, v in kw.items():
        df = df[df[k] == v]
    return df


# ----------------------------------------------------------------------------- tables
def to_latex(df: pd.DataFrame, path: Path, caption: str, label: str, fmt: dict | None = None):
    fmt = fmt or {}
    cols = list(df.columns)
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{@{}" + "l" * 1 + "r" * (len(cols) - 1) + r"@{}}", r"\toprule",
             " & ".join(str(c).replace("_", r"\_") for c in cols) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append("--" if not np.isfinite(v) else (fmt.get(c, "{:.3g}").format(v)))
            else:
                cells.append(str(v).replace("_", r"\_"))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- figures
def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight", "pdf.fonttype": 42})
    return plt


PALETTE = {"FP16": "#4d4d4d", "W8A8": "#2c6e9b", "W4A8": "#b5553c", "W4A8-MP": "#7a6a2f"}


def savefig(fig, out: Path, name: str):
    (out / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figures" / f"{name}.png", dpi=300)
    fig.savefig(out / "figures" / f"{name}.pdf")


def fig_deltas(gate: pd.DataFrame, floor: dict, out: Path):
    plt = style()
    arms = [(v, d) for v in ("vidit", "w4a8", "w4a8mp") for d in ("raw", "pre") if len(sel(gate, variant=v, drive=d))]
    if not arms:
        return
    qs = ["rel_delta_acceleration_variance", "rel_delta_var_accel_norm", "rel_delta_mean_limb_length_std", "mean_NPE"]
    fig, axs = plt.subplots(1, len(qs), figsize=(7.16, 2.1))
    for ax, k in zip(axs, qs):
        pos = 0
        ticks, labels = [], []
        for post, hatch in (("none", None), ("kapt_smoothnet", "////")):
            for v, d in arms:
                s = sel(gate, variant=v, drive=d, post=post)[k].dropna().to_numpy()
                if not len(s):
                    continue
                col = PALETTE[VAR_LABEL[v]]
                b = ax.boxplot([s], positions=[pos], widths=0.6, patch_artist=True, showfliers=False)
                for p in b["boxes"]:
                    p.set_facecolor(col if d == "raw" else "white")
                    p.set_edgecolor(col)
                    if hatch:
                        p.set_hatch(hatch)
                for m in b["medians"]:
                    m.set_color("black")
                ax.plot(np.full(len(s), pos) + RNG.uniform(-0.15, 0.15, len(s)), s, ".", ms=1.5, color="black",
                        alpha=0.5)
                ticks.append(pos)
                labels.append(f"{VAR_LABEL[v]}{'+pre' if d == 'pre' else ''}{'+post' if post != 'none' else ''}")
                pos += 1
            pos += 0.5
        if k in floor:
            ax.axhline(floor[k], color="#999999", lw=0.8, ls="--")
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=60, ha="right")
        ax.set_title(DELTA_LABEL[k])
    fig.tight_layout()
    savefig(fig, out, "deltas_by_arm")
    plt.close(fig)


def fig_frontier(cost: pd.DataFrame, out: Path):
    plt = style()
    if cost.empty:
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for _, r in cost.iterrows():
        col = PALETTE.get(r["arm_label"].split("+")[0], "#000000")
        mk = "o" if r["post"] == "none" else "s"
        fc = col if r["drive"] == "raw" else "white"
        ax.errorbar(r["weight_gb"], r["median_abs_rel_accel"], yerr=[[r["median_abs_rel_accel"] - r["ci_lo"]],
                    [r["ci_hi"] - r["median_abs_rel_accel"]]], fmt=mk, mfc=fc, mec=col, ecolor=col, ms=4, lw=0.8)
        lab = r["arm_label"].replace("FP16", "BF16")
        if lab.startswith("W4A8"):
            lab += " (simulated; computed footprint)"
        ax.annotate(lab, (r["weight_gb"], r["median_abs_rel_accel"]), fontsize=5.5,
                    xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("DiT weight memory (GB); W4A8: packed footprint, not measured")
    ax.set_ylabel(r"median $|\Delta|$ accel. var. vs BF16 (rel.)")
    fig.tight_layout()
    savefig(fig, out, "cost_plausibility_frontier")
    plt.close(fig)


def fig_disagreement(merged: pd.DataFrame, out: Path):
    plt = style()
    if merged.empty:
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for v in merged["variant"].unique():
        s = merged[merged["variant"] == v]
        ax.scatter(s["lpips"], s["rel_delta_acceleration_variance"], s=6, color=PALETTE.get(VAR_LABEL.get(v, v)),
                   label=VAR_LABEL.get(v, v), alpha=0.8)
    ax.set_xlabel("LPIPS vs FP16")
    ax.set_ylabel(r"$|\Delta|$ accel. var. (rel.)")
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig(fig, out, "kinematic_vs_pixel")
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/workspace/runs")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    runs, out = Path(a.runs), Path(a.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    absdf, gate, recon, meta, bench = load(runs)
    summary: dict = {"n_abs": len(absdf), "n_gate": len(gate), "n_recon": len(recon), "n_meta": len(meta)}
    if absdf.empty or gate.empty:
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary))
        return

    # A. noise floor (raw skeletons and post-processed skeletons separately)
    floor, floor_rows = {}, []
    for post in ("none", "kapt_smoothnet"):
        nf = sel(gate, tag="fp16_seed", post=post)
        fr = {"post": post, "n_pairs": len(nf), "gate_fail_rate": float(1 - nf["passed"].mean()) if len(nf) else np.nan}
        for k in DELTAS:
            v = nf[k].dropna().to_numpy()
            fr[f"p95_{k}"] = float(np.percentile(np.abs(v) if k.startswith("signed") else v, 95)) if len(v) else np.nan
            if post == "none" and len(v):
                floor[k] = fr[f"p95_{k}"]
        floor_rows.append(fr)
    nfdf = pd.DataFrame(floor_rows)
    nfdf.to_csv(out / "tables" / "noise_floor.csv", index=False)
    summary["noise_floor"] = floor_rows

    # B. absolute levels
    lv = (absdf.groupby(["variant", "drive", "post"])[KIN].median().reset_index())
    lv.to_csv(out / "tables" / "absolute_levels_median.csv", index=False)
    to_latex(lv.assign(variant=lv["variant"].map(VAR_LABEL)), out / "tables" / "absolute_levels.tex",
             "Median kinematic metrics per arm and post-processing (lower is better).", "tab:qtc_levels")

    # C. paired hypothesis tests
    rows = []
    for v in sorted(set(absdf["variant"]) - {"fp16"}):
        base = sel(absdf, variant="fp16", drive="raw", post="none")
        rows += paired(sel(absdf, variant=v, drive="raw", post="none"), base, KIN, f"{v}/raw", "fp16/raw", f"H1_damage_{v}")
        rows += paired(sel(absdf, variant=v, drive="pre", post="none"), sel(absdf, variant=v, drive="raw", post="none"),
                       KIN, f"{v}/pre", f"{v}/raw", f"H2_pre_{v}")
        rows += paired(sel(absdf, variant=v, drive="raw", post="kapt_smoothnet"),
                       sel(absdf, variant=v, drive="raw", post="none"), KIN, f"{v}/raw+post", f"{v}/raw", f"H3_post_{v}")
        rows += paired(sel(absdf, variant=v, drive="pre", post="kapt_smoothnet"), base, KIN, f"{v}/pre+post",
                       "fp16/raw", f"H4_combined_{v}")
    tests = pd.DataFrame(rows)
    tests.to_csv(out / "tables" / "paired_tests.csv", index=False)
    if not tests.empty:
        t = tests[["family", "metric", "n", "median_diff", "ci_lo", "ci_hi", "cliffs_delta", "p_holm"]].copy()
        t["metric"] = t["metric"].map(KIN_LABEL)
        to_latex(t, out / "tables" / "paired_tests.tex",
                 "Paired Wilcoxon signed-rank tests over clip--seed pairs (Holm-adjusted within each family); "
                 "median paired difference with bootstrap 95\\% CI and Cliff's $\\delta$.", "tab:qtc_tests")

    # D. pass and within-noise rates
    rows = []
    for (v, d, post), s in gate[gate["tag"] != "fp16_seed"].groupby(["variant", "drive", "post"]):
        k = int(s["passed"].sum())
        lo, hi = wilson(k, len(s))
        r = {"variant": VAR_LABEL.get(v, v), "drive": d, "post": post, "n": len(s), "passed": k,
             "pass_rate": k / len(s), "wilson_lo": lo, "wilson_hi": hi}
        within = np.ones(len(s), bool)
        for q in ("rel_delta_acceleration_variance", "rel_delta_var_accel_norm", "rel_delta_mean_limb_length_std",
                  "mean_NPE"):
            if q in floor:
                within &= (s[q].to_numpy(float) <= floor[q])
        r["within_noise_rate"] = float(within.mean())
        rows.append(r)
    rates = pd.DataFrame(rows)
    rates.to_csv(out / "tables" / "pass_rates.csv", index=False)
    to_latex(rates, out / "tables" / "pass_rates.tex",
             "Gate pass rates (Wilson 95\\% CI) and within-noise rates per arm; post-processed candidates are "
             "compared with equally post-processed FP16 references.", "tab:qtc_pass")

    # E. kinematic-pixel disagreement (raw skeletons, post = none)
    g0 = sel(gate, post="none")
    g0 = g0[g0["tag"] != "fp16_seed"]
    merged = g0.merge(recon, on=["cand", "ref"], how="inner") if not recon.empty else pd.DataFrame()
    if not merged.empty:
        arm = merged.groupby(["variant", "drive"]).agg(lpips=("lpips", "mean"), psnr=("psnr_db", "mean"),
                                                        accel=("rel_delta_acceleration_variance", "median"),
                                                        jitter=("rel_delta_var_accel_norm", "median"),
                                                        npe=("mean_NPE", "median"), pass_rate=("passed", "mean")).reset_index()
        arm.to_csv(out / "tables" / "pixel_vs_kinematic_arms.csv", index=False)
        dis = {}
        if len(arm) >= 3:
            for k in ("accel", "jitter", "npe"):
                tau = stats.kendalltau(arm["lpips"], arm[k])
                dis[f"kendall_tau_lpips_{k}"] = [float(tau.statistic), float(tau.pvalue)]
        for k in ("rel_delta_acceleration_variance", "rel_delta_var_accel_norm", "mean_NPE"):
            rho = stats.spearmanr(merged["lpips"], merged[k], nan_policy="omit")
            dis[f"spearman_lpips_{k}"] = [float(rho.statistic), float(rho.pvalue), int(len(merged))]
        summary["disagreement"] = dis
        fig_disagreement(merged, out)

    # F. cost and frontier
    cost_rows = []
    if not meta.empty:
        cm = meta.groupby(["variant", "drive"]).agg(sec_per_video=("gen_seconds", "median"),
                                                    peak_vram_gb=("peak_vram_gb", "median"),
                                                    int8_param_fraction=("int8_param_fraction", "median"),
                                                    w4_packed=("w4a8_packed_weight_bytes", "median"),
                                                    w4_dense=("w4a8_dense_weight_bytes", "median"),
                                                    dit_params=("dit_params", "median")).reset_index()
        cm.to_csv(out / "tables" / "cost_from_meta.csv", index=False)
        fp_params = float(meta["dit_params"].dropna().max()) if meta["dit_params"].notna().any() else 5.77e9
        for (v, d, post), s in gate[gate["tag"] != "fp16_seed"].groupby(["variant", "drive", "post"]):
            mm = sel(meta, variant=v)
            frac = float(mm["int8_param_fraction"].median()) if len(mm) else 0.0
            if v == "vidit":
                gb = fp_params * (2 - frac) / 1e9            # INT8 layers 1 B/param, the rest bf16
            elif v in ("w4a8", "w4a8mp"):
                dense = float(mm["w4a8_dense_weight_bytes"].median()) if len(mm) else 0.0
                packed = float(mm["w4a8_packed_weight_bytes"].median()) if len(mm) else 0.0
                gb = (fp_params * 2 - dense + packed) / 1e9 if v == "w4a8" else float("nan")
            else:
                gb = fp_params * 2 / 1e9
            x = s["rel_delta_acceleration_variance"].dropna().to_numpy()
            lo, hi = boot_ci_median(x)
            label = VAR_LABEL.get(v, v) + ("+pre" if d == "pre" else "") + ("+post" if post != "none" else "")
            cost_rows.append({"variant": v, "drive": d, "post": post, "arm_label": label, "weight_gb": gb,
                              "median_abs_rel_accel": float(np.median(x)) if len(x) else np.nan, "ci_lo": lo, "ci_hi": hi})
        fp = sel(gate, tag="fp16_seed", post="none")["rel_delta_acceleration_variance"].dropna().to_numpy()
        if len(fp):
            lo, hi = boot_ci_median(fp)
            cost_rows.append({"variant": "fp16", "drive": "raw", "post": "none", "arm_label": "FP16 (seed-to-seed)",
                              "weight_gb": fp_params * 2 / 1e9, "median_abs_rel_accel": float(np.median(fp)),
                              "ci_lo": lo, "ci_hi": hi})
    cost = pd.DataFrame(cost_rows)
    cost.to_csv(out / "tables" / "frontier.csv", index=False)
    fig_frontier(cost.dropna(subset=["weight_gb"]) if not cost.empty else cost, out)
    fig_deltas(gate[gate["tag"] != "fp16_seed"], floor, out)
    if bench:
        pd.DataFrame(bench).to_csv(out / "tables" / "bench.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k != "noise_floor"}, default=float))


if __name__ == "__main__":
    main()
