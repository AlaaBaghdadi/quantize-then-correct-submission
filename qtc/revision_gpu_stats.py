#!/usr/bin/env python3
"""IJIES R1 revision: analysis of the revision GPU runs (pod results branch).

    python -m qtc.revision_gpu_stats --new <checkout of results/<pod>> --sub <checkout of results/wo0ny5hy0vo2vt>

Inputs (JSON records written by qtc/score.py on the pods):
    --new  runs/qtc/{metrics,gate,recon}, runs/meta, runs/meta/layersets   (refs, heldout, seeds sets)
    --sub  the same for the submitted run (per-clip calibration arms and the submitted BF16 references)
Outputs:
    revision/ijies-r1/stats/revision_gpu.json   numbers and the rows of the manuscript tables "heldout" and "seeds"
    revision/ijies-r1/stats/gpu_*.csv           long tables
    a section of revision/ijies-r1/RESULTS.md between <!-- GPU:BEGIN --> and <!-- GPU:END -->

Analyses
    A  determinism: re-rendered BF16 references (seeds 42, 7) against the submitted ones (video hash, metrics)
    B  extended noise floor: four seed pairs per clip (seeds 1, 2, 3, 7 against 42); fixed-threshold failures;
       LOCO thresholds from the 44 seed pairs of the other clips (maximum and 95th percentile); clustered
       comparison of the quantization deltas with the per-clip mean seed delta; equivalence margins
    C  held-out (other-subject) calibration: quantization deltas against seed deltas, absolute statistics
       against BF16, pose and pixel error, LOCO pass counts, paired comparison with per-clip calibration,
       overlap of the protected layer sets
All clustered procedures follow qtc/revision_stats.py (clip-level Wilcoxon, clip resampling, mixed model).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from qtc import revision_stats as RS

ROOT = Path(__file__).resolve().parents[1]
CHECKS = {"accel": "rel_delta_acceleration_variance", "limb": "rel_delta_mean_limb_length_std",
          "jitter": "rel_delta_var_accel_norm", "viol": "abs_delta_violation_fraction", "npe": "mean_NPE",
          "npe95": "p95_NPE", "under": "under_motion_violations"}
OKMAP = {"accel": "acceleration_variance", "limb": "mean_limb_length_std", "jitter": "var_accel_norm",
         "viol": "violation_fraction", "npe": "mean_NPE", "npe95": "p95_NPE", "under": "under_motion_violations"}
CK = list(CHECKS)
KIN = RS.KIN
SEED_REF = 42
SEEDS_FLOOR = [7, 1, 2, 3]
B = 10_000


def parse(uid: str) -> dict:
    p = uid.split("-")
    return {"variant": p[0], "calib": p[2], "clip": p[3], "seed": int(p[4][1:]),
            "drive": "pre" if uid.endswith("-dpre3") else "raw"}


def ok_uid(uid):
    return not (uid.endswith("-dpre") or uid.endswith("-dpre2"))


def load(runs: Path):
    q = runs / "qtc"
    g, a, x = [], [], []
    for f in sorted((q / "gate").glob("*.json")):
        r = json.loads(f.read_text())
        if not ok_uid(r["cand"]):
            continue
        d = {**parse(r["cand"]), "cand": r["cand"], "ref": r["ref"], "tag": r["tag"], "post": r["post"],
             "passed": bool(r["passed"]), "ref_seed": parse(r["ref"])["seed"]}
        d.update({k: float(r["deltas"][v]) for k, v in CHECKS.items()})
        d.update({f"ok_{k}": bool(r["checks"][OKMAP[k]]) for k in CK})
        g.append(d)
    for f in sorted((q / "metrics").glob("*.json")):
        r = json.loads(f.read_text())
        if not ok_uid(r["uid"]):
            continue
        a.append({**parse(r["uid"]), "uid": r["uid"], "post": r["post"], "skeleton_sha": Path(r["skeleton"]).stem,
                  **{k: float(r[k]) for k in KIN}})
    for f in sorted((q / "recon").glob("*.json")):
        r = json.loads(f.read_text())
        if not ok_uid(r["cand"]):
            continue
        x.append({**parse(r["cand"]), "cand": r["cand"], "ref": r["ref"], "tag": r["tag"],
                  **{k: r[k] for k in ("psnr_db", "ssim", "lpips")}})
    ls = {}
    for f in sorted((runs / "meta" / "layersets").glob("*.json")):
        ls[f.stem] = json.loads(f.read_text())
    meta = {}
    for f in sorted((runs / "meta").glob("*.json")):
        m = json.loads(f.read_text())
        if "uid" in m:
            meta[m["uid"]] = m
    xcols = ["variant", "calib", "clip", "seed", "drive", "cand", "ref", "tag", "psnr_db", "ssim", "lpips"]
    return pd.DataFrame(g), pd.DataFrame(a), pd.DataFrame(x, columns=xcols) if not x else pd.DataFrame(x), ls, meta


def mat(df, clips, col, seeds):
    p = df.pivot_table(index="clip", columns="seed", values=col, aggfunc="first")
    return p.reindex(index=clips, columns=seeds).to_numpy(float)


def wil(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    return float(stats.wilcoxon(d).pvalue) if len(d) >= 5 and np.any(d != 0) else float("nan")


def boot_clip(fn, n, key, Bn=B):
    rng = RS.rng_for("gpu", *key)
    out = np.array([fn(rng.integers(0, n, n)) for _ in range(Bn)])
    return RS.q(out, 0.025, 0.975), RS.q(out, 0.05, 0.95)


def r(s):
    return f"«{s}»"


def pf(p):
    return RS.pfmt(p)


# ============================================================================ A determinism
def determinism(a_new, a_sub, clips):
    rows = []
    for c in clips:
        for s in (42, 7):
            for post in ("none", "kapt_smoothnet"):
                n = a_new[(a_new.variant == "fp16") & (a_new["clip"] == c) & (a_new.seed == s) & (a_new.post == post)]
                o = a_sub[(a_sub.variant == "fp16") & (a_sub["clip"] == c) & (a_sub.seed == s) & (a_sub.post == post)]
                if n.empty or o.empty:
                    continue
                n, o = n.iloc[0], o.iloc[0]
                rel = {k: abs(n[k] - o[k]) / max(abs(o[k]), 1e-12) for k in KIN}
                rows.append({"clip": c, "seed": s, "post": post, "sub_group": o.get("group", ""), "same_video_hash": bool(n.skeleton_sha == o.skeleton_sha)
                             if post == "none" else None, **{f"reldiff_{k}": v for k, v in rel.items()}})
    d = pd.DataFrame(rows)
    if d.empty:
        return {"n": 0}, d
    nn = d[d.post == "none"]
    return {"n_refs": int(len(nn)), "n_identical_video": int(nn.same_video_hash.sum()),
            "non_identical": [f"{r_.clip}-s{r_.seed}" for r_ in nn.itertuples() if not r_.same_video_hash],
            "max_reldiff": {k: float(d[f"reldiff_{k}"].max()) for k in KIN},
            "median_reldiff": {k: float(d[f"reldiff_{k}"].median()) for k in KIN}}, d


# ============================================================================ B extended noise floor
def seed_matrix(g_new, clips, post, k):
    s = g_new[(g_new.variant == "fp16") & (g_new.post == post) & (g_new.ref_seed == SEED_REF)]
    return mat(s, clips, k, SEEDS_FLOOR)                      # clips x 4 (seeds 7, 1, 2, 3 vs 42)


def loco_eval(S4, X, cidx, rule):
    """S4: clips x pairs x checks seed deltas; X: videos x checks; thresholds from the other clips."""
    n = S4.shape[0]
    out = []
    for i in range(len(X)):
        oth = np.delete(S4, cidx[i], axis=0).reshape(-1, S4.shape[2])
        oth = oth[np.all(np.isfinite(oth), axis=1)]
        tau = oth.max(axis=0) if rule == "max" else np.quantile(oth, 0.95, axis=0)
        out.append(bool(np.all(X[i] <= tau + 1e-12)))
    return np.array(out)


def floor_analysis(g_new, g_sub, clips):
    res = {}
    for post in ("none", "kapt_smoothnet"):
        S4 = np.stack([seed_matrix(g_new, clips, post, k) for k in CK], axis=2)   # clips x 4 x checks
        flat = S4.reshape(-1, len(CK))
        flat = flat[np.all(np.isfinite(flat), axis=1)]
        e = {"n_pairs": int(len(flat)),
             "median": {k: float(np.median(flat[:, j])) for j, k in enumerate(CK)},
             "iqr": {k: [float(np.quantile(flat[:, j], .25)), float(np.quantile(flat[:, j], .75))] for j, k in enumerate(CK)}}
        s = g_new[(g_new.variant == "fp16") & (g_new.post == post) & (g_new.ref_seed == SEED_REF)]
        e["fixed_fail"] = int((~s.passed).sum())
        e["fixed_n"] = int(len(s))
        e["check_fail_rate"] = {k: float(1 - s[f"ok_{k}"].mean()) for k in CK}
        # seed 7 only (the submitted floor re-estimated on the re-rendered references)
        S1 = S4[:, :1, :]
        # LOCO with 4 seed pairs per clip: seed pairs themselves, per-clip-calibrated arms (submitted), held-out arms
        evals = {}
        Xs, cs = S4.reshape(-1, len(CK)), np.repeat(np.arange(len(clips)), 4)
        evals["bf16_seed"] = (Xs, cs)
        for src, gg, calib in (("own", g_sub, "own"), ("heldout", g_new, "loso")):
            for v in ("vidit", "w4a8"):
                for drv in ("raw", "pre"):
                    a = gg[(gg.variant == v) & (gg.calib == calib) & (gg.drive == drv) & (gg.post == post)]
                    if a.empty:
                        continue
                    a = a.set_index(["clip", "seed"]).reindex([(c, sd) for c in clips for sd in (42, 7)])
                    evals[f"{v}_{drv}_{src}"] = (a[CK].to_numpy(float), np.repeat(np.arange(len(clips)), 2))
        e["loco"] = {}
        for name, (X, cidx) in evals.items():
            ok = np.all(np.isfinite(X), axis=1)
            for rule in ("max", "q95"):
                pas = loco_eval(S4, X[ok], cidx[ok], rule)
                e["loco"].setdefault(name, {})[rule] = {"pass": int(pas.sum()), "n": int(len(pas)),
                                                        "rate": float(pas.mean()), "wilson": RS.wilson(int(pas.sum()), len(pas))}
            p1 = loco_eval(S1, X[ok], cidx[ok], "max")       # one seed pair per clip (submitted rule)
            e["loco"][name]["max_1pair"] = {"pass": int(p1.sum()), "n": int(len(p1)), "rate": float(p1.mean())}
        # stability with four pairs: clip bootstrap of the calibration, max rule, quantized raw arms and seed pairs
        rng = RS.rng_for("gpu_floor_boot", post)
        st = {}
        for name in [k for k in evals if k in ("bf16_seed", "vidit_raw_own", "w4a8_raw_own", "vidit_raw_heldout",
                                                "w4a8_raw_heldout")]:
            X, cidx = evals[name]
            okm = np.all(np.isfinite(X), axis=1)
            X, cidx = X[okm], cidx[okm]
            rates, base = [], loco_eval(S4, X, cidx, "max")
            flips = np.zeros(len(X))
            for _ in range(1000):
                idx = rng.integers(0, len(clips), len(clips))
                Sb = S4[idx]
                pas = []
                for i in range(len(X)):
                    keep = idx != cidx[i]
                    oth = Sb[keep].reshape(-1, len(CK))
                    oth = oth[np.all(np.isfinite(oth), axis=1)]
                    pas.append(bool(np.all(X[i] <= oth.max(axis=0) + 1e-12)) if len(oth) else False)
                pas = np.array(pas)
                rates.append(pas.mean())
                flips += pas != base
            st[name] = {"calboot_ci95": RS.q(rates, 0.025, 0.975), "flip_rate_mean": float(np.mean(flips / 1000))}
        e["stability"] = st
        res[post] = e
    return res


def vs_floor(g_arm, g_new, clips, k, key):
    """Quantization delta (2 seeds/clip) against the per-clip mean of the four seed deltas (clip-level)."""
    S = seed_matrix(g_new, clips, "none", k)                  # clips x 4
    Q = mat(g_arm, clips, k, [42, 7])                         # clips x 2
    qc, sc = np.nanmean(Q, axis=1), np.nanmean(S, axis=1)
    diff = float(np.nanmedian(Q) - np.nanmedian(S))
    n = len(clips)

    def stat(idx):
        return np.nanmedian(Q[idx]) - np.nanmedian(S[idx])
    c95, c90 = boot_clip(stat, n, key)
    margin = 0.5 * float(np.nanmedian(S))
    return {"median_quant": float(np.nanmedian(Q)), "median_seed": float(np.nanmedian(S)), "diff_median": diff,
            "p_clip": wil(qc - sc), "ci95": c95, "ci90": c90, "margin": margin,
            "equivalent": bool(margin > 0 and c90[0] > -margin and c90[1] < margin),
            "noninferior": bool(margin > 0 and c90[1] < margin),
            "margin_needed_over_median_seed": float(max(abs(c90[0]), abs(c90[1])) / np.nanmedian(S)) if np.nanmedian(S) > 0 else float("nan")}


# ============================================================================ C held-out calibration
def heldout_analysis(g_new, g_sub, a_new, x_new, x_sub, ls_new, ls_sub, clips):
    out = {}
    for v in ("vidit", "w4a8"):
        for drv in ("raw", "pre"):
            key = f"{v}_{drv}"
            gh = g_new[(g_new.variant == v) & (g_new.calib == "loso") & (g_new.drive == drv) & (g_new.post == "none")]
            go = g_sub[(g_sub.variant == v) & (g_sub.calib == "own") & (g_sub.drive == drv) & (g_sub.post == "none")]
            if gh.empty:
                continue
            e = {"n": int(len(gh)), "vs_floor": {k: vs_floor(gh, g_new, clips, k, ("hv", key, k)) for k in ("accel", "jitter", "limb", "npe", "npe95")}}
            # paired: held-out minus per-clip calibration, same clip and seed (clip-level Wilcoxon)
            e["vs_own"] = {}
            for k in ("accel", "jitter", "limb", "npe", "npe95"):
                H, O = mat(gh, clips, k, [42, 7]), mat(go, clips, k, [42, 7])
                d = np.nanmean(H - O, axis=1)
                e["vs_own"][k] = {"median_heldout": float(np.nanmedian(H)), "median_own": float(np.nanmedian(O)),
                                  "median_diff": float(np.nanmedian(H - O)), "p_clip": wil(d)}
            xh = x_new[(x_new.variant == v) & (x_new.calib == "loso") & (x_new.drive == drv)]
            xo = x_sub[(x_sub.variant == v) & (x_sub.calib == "own") & (x_sub.drive == drv)]
            for m in ("psnr_db", "lpips"):
                H, O = mat(xh, clips, m, [42, 7]), mat(xo, clips, m, [42, 7])
                e["vs_own"][m] = {"median_heldout": float(np.nanmedian(H)), "median_own": float(np.nanmedian(O)),
                                  "median_diff": float(np.nanmedian(H - O)), "p_clip": wil(np.nanmean(H - O, axis=1))}
            # absolute statistics against BF16 (raw drive only): clip-level test and equivalence on log ratio
            if drv == "raw":
                e["abs"] = {}
                for post in ("none", "kapt_smoothnet"):
                    A = a_new[(a_new.variant == v) & (a_new.calib == "loso") & (a_new.drive == "raw") & (a_new.post == post)]
                    F = a_new[(a_new.variant == "fp16") & (a_new.post == post)]
                    for kk in KIN:
                        QA, QB = mat(A, clips, kk, [42, 7]), mat(F, clips, kk, [42, 7])
                        logm = kk != "violation_fraction" and np.all(QA[np.isfinite(QA)] > 0) and np.all(QB[np.isfinite(QB)] > 0)
                        D = np.log(QA / QB) if logm else QA - QB
                        sd = np.abs(np.log(QB[:, 1] / QB[:, 0])) if logm else np.abs(QB[:, 1] - QB[:, 0])
                        margin = 0.5 * float(np.nanmedian(sd))
                        dc = np.nanmean(D, axis=1)
                        c95, c90 = boot_clip(lambda idx: np.nanmean(dc[idx]), len(clips), ("ha", key, post, kk))
                        e["abs"][f"{post}|{kk}"] = {"mean_contrast": float(np.nanmean(dc)), "p_clip": wil(dc),
                                                    "ci90": c90, "ci95": c95, "margin": margin,
                                                    "equivalent": bool(c90[0] > -margin and c90[1] < margin)}
                        if logm:
                            e["abs"][f"{post}|{kk}"]["ratio"] = float(np.exp(np.nanmean(dc)))
            # Holm within families: held-out vs per-clip (all quantities of the arm); absolute statistics per post
            ks = list(e["vs_own"])
            for k_, ph in zip(ks, RS.holm([e["vs_own"][k_]["p_clip"] for k_ in ks])):
                e["vs_own"][k_]["p_clip_holm"] = float(ph)
            if "abs" in e:
                for post in ("none", "kapt_smoothnet"):
                    ks = [k_ for k_ in e["abs"] if k_.startswith(post + "|")]
                    for k_, ph in zip(ks, RS.holm([e["abs"][k_]["p_clip"] for k_ in ks])):
                        e["abs"][k_]["p_clip_holm"] = float(ph)
            out[key] = e
    # protected-layer overlap (K): held-out against per-clip
    ov = []
    for c in clips:
        for v in ("vidit", "w4a8"):
            h = ls_new.get(f"{v}-full-loso-{c}")
            o = ls_sub.get(f"{v}-full-own-{c}")
            if h and o:
                kh, ko = set(h["K"]), set(o["K"])
                ov.append({"clip": c, "variant": v, "k_heldout": len(kh), "k_own": len(ko),
                           "jaccard": len(kh & ko) / max(len(kh | ko), 1)})
    out["k_overlap"] = {"n": len(ov), "jaccard_median": float(np.median([o["jaccard"] for o in ov])) if ov else float("nan"),
                        "jaccard_min": float(np.min([o["jaccard"] for o in ov])) if ov else float("nan")}
    return out


# ============================================================================ tables and report
def tables(F, H):
    nm = {"accel": "Δrel(σ²acc)", "jitter": "Δrel(σ²jit)", "limb": "Δrel(σlimb)", "npe": "NPE", "npe95": "NPE95"}
    e = F["none"]
    seeds_head = [[r("Check"), r("Median [IQR], 48 seed pairs"), r("Fixed τ fail"),
                   r("LOCO max, 1 pair/clip"), r("LOCO max, 4 pairs/clip"), r("LOCO q95, 4 pairs/clip")]]
    seeds_rows = []
    for k in ("accel", "limb", "jitter", "viol", "npe", "npe95"):
        seeds_rows.append([r(nm.get(k, "Δfviol")), r(f"{RS.f3(e['median'][k])} [{RS.f3(e['iqr'][k][0])}, {RS.f3(e['iqr'][k][1])}]"),
                           r(f"{100 * e['check_fail_rate'][k]:.0f}%"), "", "", ""])
    for name, lab in (("bf16_seed", "BF16 seed pairs"), ("vidit_raw_own", "W8A8, per-clip calib."),
                      ("w4a8_raw_own", "W4A8, per-clip calib."), ("vidit_raw_heldout", "W8A8, held-out calib."),
                      ("w4a8_raw_heldout", "W4A8, held-out calib.")):
        L = e["loco"].get(name)
        if not L:
            continue
        seeds_rows.append([r(f"Pass: {lab}"), "", "",
                           r(f"{L['max_1pair']['pass']}/{L['max_1pair']['n']}"),
                           r(f"{L['max']['pass']}/{L['max']['n']} ({100 * L['max']['rate']:.0f}%)"),
                           r(f"{L['q95']['pass']}/{L['q95']['n']} ({100 * L['q95']['rate']:.0f}%)")])
    ho_head = [[r("Quantity"), r("Arm"), r("Per-clip calib."), r("Held-out calib."), r("pHolm, held-out vs per-clip"),
                r("pclip, held-out vs seed pairs")]]
    ho_rows = []
    for k in ("accel", "jitter", "limb", "npe", "npe95", "psnr_db", "lpips"):
        for i, v in enumerate(("vidit_raw", "w4a8_raw")):
            if v not in H:
                continue
            o = H[v]["vs_own"][k]
            pv = H[v]["vs_floor"][k]["p_clip"] if k in H[v]["vs_floor"] else float("nan")
            ho_rows.append([r({"psnr_db": "PSNR (dB)", "lpips": "LPIPS"}.get(k, nm.get(k, k))) if i == 0 else "",
                            r("W8A8" if v.startswith("vidit") else "W4A8"),
                            r(f"{o['median_own']:.1f}" if k == "psnr_db" else RS.f3(o["median_own"])),
                            r(f"{o['median_heldout']:.1f}" if k == "psnr_db" else RS.f3(o["median_heldout"])), r(pf(o.get("p_clip_holm", o["p_clip"]))), r(pf(pv))])
    return {"head": seeds_head, "rows": seeds_rows}, {"head": ho_head, "rows": ho_rows}


def report(G, md_path: Path):
    D, F, H = G["determinism"], G["floor"], G["heldout"]
    L = ["<!-- GPU:BEGIN -->", "\n## Revision GPU runs (R1.2 / R2.2 held-out calibration; R1.5 / R2.4 extra seeds)\n",
         "Generated by `qtc/revision_gpu_stats.py`; do not edit by hand.\n",
         RS.table([[k, f"`{v}`"] for k, v in G["provenance"].items()], ["Item", "Value"]),
         "\n### A. Determinism of the re-rendered BF16 references\n",
         f"{D.get('n_identical_video', 0)} of {D.get('n_refs', 0)} re-rendered reference videos are bit-identical to the "
         f"submitted ones (same content hash); not identical: {', '.join(D.get('non_identical', [])) or 'none'}. Maximum relative difference of the absolute statistics: "
         + ", ".join(f"{k} {v:.2e}" for k, v in D.get("max_reldiff", {}).items()) + ".\n",
         "\n### B. Extended noise floor (seeds 7, 1, 2, 3 against 42)\n"]
    for post in ("none", "kapt_smoothnet"):
        e = F[post]
        L.append(f"\nPost `{post}`: {e['n_pairs']} seed pairs; fixed-threshold gate failures {e['fixed_fail']}/{e['fixed_n']}.\n\n")
        L.append(RS.table([[k, RS.f3(e["median"][k]), f"[{RS.f3(e['iqr'][k][0])}, {RS.f3(e['iqr'][k][1])}]",
                            f"{100 * e['check_fail_rate'][k]:.0f}%"] for k in CK], ["Check", "Median", "IQR", "Fixed τ fail"]))
        L.append("\n" + RS.table([[n, f"{x['max_1pair']['pass']}/{x['max_1pair']['n']}",
                                   f"{x['max']['pass']}/{x['max']['n']} ({x['max']['rate']:.3f}) {RS.ci_s(x['max']['wilson'], 2)}",
                                   f"{x['q95']['pass']}/{x['q95']['n']} ({x['q95']['rate']:.3f})"] for n, x in e["loco"].items()],
                                 ["Arm", "LOCO max, 1 pair/clip", "LOCO max, 4 pairs/clip (Wilson)", "LOCO q95, 4 pairs/clip"]))
        L.append("\n" + RS.table([[n, RS.ci_s(x["calboot_ci95"], 2), f"{x['flip_rate_mean']:.3f}"] for n, x in e["stability"].items()],
                                 ["Arm", "Calibration bootstrap 95% (4 pairs/clip)", "Mean verdict flip rate"]))
    L.append("\n### C. Held-out (other-subject) calibration\n")
    for key, e in H.items():
        if key == "k_overlap":
            continue
        L.append(f"\n**{key}** (n = {e['n']})\n\n")
        L.append(RS.table([[k, RS.f3(x["median_quant"]), RS.f3(x["median_seed"]), RS.pfmt(x["p_clip"]), RS.ci_s(x["ci90"]),
                            RS.f3(x["margin"]), "yes" if x["equivalent"] else "no", "yes" if x["noninferior"] else "no",
                            f"{x['margin_needed_over_median_seed']:.2f}"] for k, x in e["vs_floor"].items()],
                          ["Delta", "Median held-out", "Median seed (4/clip)", "p clip", "90% CI diff.", "Margin", "Equiv.",
                           "Non-inf.", "Margin needed"]))
        L.append("\n" + RS.table([[k, RS.f3(x["median_own"]), RS.f3(x["median_heldout"]), RS.f3(x["median_diff"]), RS.pfmt(x["p_clip"]),
                                   RS.pfmt(x.get("p_clip_holm"))] for k, x in e["vs_own"].items()],
                                  ["Quantity", "Per-clip calib.", "Held-out calib.", "Median diff.", "p clip", "p clip Holm"]))
        if "abs" in e:
            L.append("\n" + RS.table([[k, RS.f3(x["mean_contrast"]), RS.pfmt(x["p_clip"]), RS.pfmt(x.get("p_clip_holm")), RS.ci_s(x["ci90"]),
                                       RS.f3(x["margin"]), "yes" if x["equivalent"] else "no"] for k, x in e["abs"].items()],
                                     ["Post | statistic", "Mean contrast vs BF16", "p clip", "p clip Holm", "90% CI", "Margin", "Equiv."]))
    ko = H.get("k_overlap", {})
    L.append(f"\nProtected-layer sets K, held-out vs per-clip: Jaccard median {ko.get('jaccard_median', float('nan')):.3f}, "
             f"minimum {ko.get('jaccard_min', float('nan')):.3f} over {ko.get('n', 0)} clip-variant pairs.\n")
    L.append("<!-- GPU:END -->\n")
    block = "\n".join(L)
    old = md_path.read_text() if md_path.is_file() else ""
    if "<!-- GPU:BEGIN -->" in old:
        old = re.sub(r"<!-- GPU:BEGIN -->.*<!-- GPU:END -->\n?", "", old, flags=re.S)
    md_path.write_text(old.rstrip() + "\n\n" + block)


def sha(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--sub", required=True)
    ap.add_argument("--out", default=str(ROOT / "revision" / "ijies-r1" / "stats"))
    ap.add_argument("--md", default=str(ROOT / "revision" / "ijies-r1" / "RESULTS.md"))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    global B
    if a.quick:
        B = 300
    new, sub, out = Path(a.new) / "runs", Path(a.sub) / "runs", Path(a.out)
    g_new, a_new, x_new, ls_new, meta_new = load(new)
    g_sub, a_sub, x_sub, ls_sub, _ = load(sub)
    clips = RS.QU_clips() if hasattr(RS, "QU_clips") else sorted(g_sub["clip"].unique())
    G = {"determinism": determinism(a_new, a_sub, clips)[0], "floor": floor_analysis(g_new, g_sub, clips),
         "heldout": heldout_analysis(g_new, g_sub, a_new, x_new, x_sub, ls_new, ls_sub, clips)}
    G["counts"] = {"new_videos": len(meta_new), "new_gate_records": int(len(g_new)), "new_metric_sets": int(len(a_new))}
    ts, th = tables(G["floor"], G["heldout"])
    G["table_seeds"], G["table_heldout"] = ts, th
    G["provenance"] = {"script": "qtc/revision_gpu_stats.py", "script_sha256_16": sha(Path(__file__)),
                       "git_commit_at_run": RS.git("rev-parse", "HEAD"), "new_runs": str(Path(a.new).name),
                       "new_branch_commit": RS.git("-C", str(Path(a.new)), "rev-parse", "HEAD"),
                       "sub_branch_commit": RS.git("-C", str(Path(a.sub)), "rev-parse", "HEAD"), "bootstrap": B}
    out.mkdir(parents=True, exist_ok=True)
    (out / "revision_gpu.json").write_text(json.dumps(G, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    g_new.to_csv(out / "gpu_gate_long.csv", index=False)
    a_new.to_csv(out / "gpu_absolute_long.csv", index=False)
    x_new.to_csv(out / "gpu_pixel_long.csv", index=False)
    if not a.quick:
        report(G, Path(a.md))
    print(json.dumps({"counts": G["counts"], "determinism": G["determinism"],
                      "loco_none": {k: v["max"] for k, v in G["floor"]["none"]["loco"].items()}}, indent=1))


if __name__ == "__main__":
    main()
