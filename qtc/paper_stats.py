#!/usr/bin/env python3
"""Numbers reported in the Article 3 manuscript, computed from a results checkout.

    python -m qtc.paper_stats --runs <results>/runs --out paper/data

Writes numbers.json (every scalar quoted in the text) and CSV tables consumed by the figure script.
Inputs are the same JSON files read by qtc.analyze (metrics, gate, recon, meta) plus the post-corrected
skeletons (runs/qtc/post) for the violation diagnosis and the driving-correction fields.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qtc import analyze as A  # noqa: E402

CHECKS = {  # gate check -> (delta key in the gate JSON, fixed Article 2 threshold)
    "accel": ("rel_delta_acceleration_variance", 0.10),
    "limb": ("rel_delta_mean_limb_length_std", 0.10),
    "jitter": ("rel_delta_var_accel_norm", 0.15),
    "viol": ("abs_delta_violation_fraction", 0.02),
    "npe": ("mean_NPE", 0.15),
    "npe95": ("p95_NPE", 0.30),
    "under": ("under_motion_violations", 0.0),
}
ARMS = [("vidit", "raw"), ("vidit", "pre"), ("w4a8", "raw"), ("w4a8", "pre")]
POSTS = ["none", "kapt_smoothnet"]


def load_gate(runs: Path) -> pd.DataFrame:
    rows = []
    for f in sorted((runs / "qtc" / "gate").glob("*.json")):
        g = json.loads(f.read_text())
        if A.obsolete(g["cand"]):
            continue
        u = A.parse_uid(g["cand"])
        d = {**u, "tag": g["tag"], "post": g["post"], "passed": bool(g["passed"]), "cand": g["cand"]}
        d.update({k: float(g["deltas"][v[0]]) for k, v in CHECKS.items()})
        d.update({f"ok_{k}": bool(g["checks"][{"accel": "acceleration_variance", "limb": "mean_limb_length_std",
                                                "jitter": "var_accel_norm", "viol": "violation_fraction",
                                                "npe": "mean_NPE", "npe95": "p95_NPE",
                                                "under": "under_motion_violations"}[k]]) for k in CHECKS})
        rows.append(d)
    return pd.DataFrame(rows)


def med_iqr(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {"median": float(np.median(x)), "q1": float(np.quantile(x, 0.25)), "q3": float(np.quantile(x, 0.75)),
            "min": float(np.min(x)), "max": float(np.max(x)), "n": int(len(x))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs, out = Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    absdf, _, recon, meta, _ = A.load(runs)
    g = load_gate(runs)
    N: dict = {"counts": {"videos": int(len(meta)), "metric_sets": int(len(absdf)), "gate_pairs": int(len(g)),
                          "pixel_pairs": int(len(recon))}}

    # ------------------------------------------------------------------ noise floor (FP16 s7 vs s42)
    nf = g[g.tag == "fp16_seed"]
    N["noise_floor"] = {}
    for post in POSTS:
        s = nf[nf.post == post]
        N["noise_floor"][post] = {
            "n": int(len(s)), "fail_rate": float(1 - s.passed.mean()), "n_fail": int((~s.passed).sum()),
            "check_fail_rate": {k: float(1 - s[f"ok_{k}"].mean()) for k in CHECKS},
            "delta": {k: med_iqr(s[k]) for k in CHECKS}}

    # ------------------------------------------------------------------ quantized arms (same-seed vs FP16)
    N["arms"] = {}
    loco_rows = []
    for (v, drv) in ARMS:
        for post in POSTS:
            s = g[(g.variant == v) & (g.drive == drv) & (g.post == post)]
            ref = nf[nf.post == post]
            key = f"{v}_{drv}_{post}"
            ent = {"n": int(len(s)), "pass_fixed": int(s.passed.sum()), "pass_rate_fixed": float(s.passed.mean()),
                   "wilson_fixed": A.wilson(int(s.passed.sum()), len(s)),
                   "check_fail_rate": {k: float(1 - s[f"ok_{k}"].mean()) for k in CHECKS},
                   "delta": {k: med_iqr(s[k]) for k in CHECKS}, "vs_seed_noise": {}}
            # quantization perturbation vs reseeding perturbation (unpaired, two-sided Mann-Whitney)
            for k in CHECKS:
                x, y = s[k].to_numpy(float), ref[k].to_numpy(float)
                if np.all(np.concatenate([x, y]) == x[0]):
                    p = float("nan")
                else:
                    p = float(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue)
                ent["vs_seed_noise"][k] = {"p": p, "cliffs_delta": A.cliffs_delta(x, y),
                                           "median_ratio": float(np.median(x) / np.median(y)) if np.median(y) > 0 else float("nan")}
            # noise-floor-calibrated gate: thresholds = max over FP16 seed pairs of the OTHER clips (leave-one-clip-out)
            ok_all, ok_noviol = [], []
            for _, r in s.iterrows():
                oth = ref[ref["clip"] != r["clip"]]
                tau = {k: float(oth[k].max()) for k in CHECKS}
                oks = {k: r[k] <= tau[k] + 1e-12 for k in CHECKS}
                ok_all.append(all(oks.values()))
                ok_noviol.append(all(v_ for k, v_ in oks.items() if k != "viol"))
                loco_rows.append({"arm": key, "clip": r["clip"], "seed": r["seed"], **{f"within_{k}": oks[k] for k in CHECKS}})
            ent["pass_loco"] = int(np.sum(ok_all))
            ent["pass_rate_loco"] = float(np.mean(ok_all))
            ent["wilson_loco"] = A.wilson(int(np.sum(ok_all)), len(ok_all))
            ent["pass_rate_loco_noviol"] = float(np.mean(ok_noviol))
            N["arms"][key] = ent
    # the calibration applied to the FP16 seed pairs themselves (leave-one-clip-out): specificity check
    N["noise_floor_loco_self"] = {}
    for post in POSTS:
        ref = nf[nf.post == post]
        ok = []
        for _, r in ref.iterrows():
            oth = ref[ref["clip"] != r["clip"]]
            ok.append(all(r[k] <= oth[k].max() + 1e-12 for k in CHECKS))
        N["noise_floor_loco_self"][post] = {"pass_rate": float(np.mean(ok)), "n_pass": int(np.sum(ok)), "n": len(ok)}
    pd.DataFrame(loco_rows).to_csv(out / "loco_gate.csv", index=False)
    g.to_csv(out / "gate_long.csv", index=False)

    # ------------------------------------------------------------------ absolute levels (median, IQR)
    N["absolute"] = {}
    for (v, drv) in [("fp16", "raw")] + ARMS:
        for post in POSTS:
            s = absdf[(absdf.variant == v) & (absdf.drive == drv) & (absdf.post == post)]
            N["absolute"][f"{v}_{drv}_{post}"] = {k: med_iqr(s[k]) for k in A.KIN}
    absdf.to_csv(out / "absolute_long.csv", index=False)

    # ------------------------------------------------------------------ paired tests (from qtc.analyze)
    pt = pd.read_csv(runs / "qtc" / "analysis" / "tables" / "paired_tests.csv") if (
        runs / "qtc" / "analysis" / "tables" / "paired_tests.csv").is_file() else None
    if pt is not None:
        N["paired"] = {f"{r.family}|{r.metric}": {c: (None if pd.isna(r[c]) else float(r[c])) for c in
                       ["n", "median_a", "median_b", "median_diff", "ci_lo", "ci_hi", "cliffs_delta", "p", "p_holm"]}
                       for _, r in pt.iterrows()}
    # matched post-processed H1 (quantized+post vs FP16+post), not in qtc.analyze
    N["paired_post_matched"] = {}
    for v in ("vidit", "w4a8"):
        qa = absdf[(absdf.variant == v) & (absdf.drive == "raw") & (absdf.post == "kapt_smoothnet")]
        fb = absdf[(absdf.variant == "fp16") & (absdf.post == "kapt_smoothnet")]
        rows = A.paired(qa, fb, A.KIN, f"{v}/raw+post", "fp16/raw+post", f"H1post_{v}")
        for r in rows:
            N["paired_post_matched"][f"{v}|{r['metric']}"] = {k: r[k] for k in
                                                               ["n", "median_a", "median_b", "median_diff", "cliffs_delta", "p", "p_holm"]}

    # ------------------------------------------------------------------ pixel fidelity
    rc = recon.copy()
    rc["tag"] = rc["cand"].map(lambda u: "fp16_seed" if u.startswith("fp16") else
                               f"{A.parse_uid(u)['variant']}_{A.parse_uid(u)['drive']}")
    N["pixel"] = {t: {m: med_iqr(s[m]) for m in ["psnr_db", "ssim", "lpips", "tlpips", "tmse"]}
                  for t, s in rc.groupby("tag")}
    rc.to_csv(out / "pixel_long.csv", index=False)
    # quantized vs seed-noise pixel distance (unpaired)
    N["pixel_vs_seed"] = {}
    fs = rc[rc.tag == "fp16_seed"]
    for t in ["vidit_raw", "w4a8_raw", "vidit_pre", "w4a8_pre"]:
        s = rc[rc.tag == t]
        N["pixel_vs_seed"][t] = {m: float(stats.mannwhitneyu(s[m], fs[m], alternative="two-sided").pvalue)
                                 for m in ["psnr_db", "lpips"]}

    # ------------------------------------------------------------------ kinematic-pixel association (raw drive, no post)
    k = g[(g.post == "none") & (g.tag != "fp16_seed")].merge(rc[["cand", "lpips", "psnr_db"]], on="cand")
    N["assoc"] = {}
    for m in ["accel", "jitter", "npe", "npe95"]:
        r, p = stats.spearmanr(k["lpips"], k[m])
        N["assoc"][f"lpips_{m}"] = {"rho": float(r), "p": float(p), "n": int(len(k))}
    k.to_csv(out / "assoc_long.csv", index=False)

    # ------------------------------------------------------------------ cost
    mt = meta.copy()
    N["cost"] = {}
    for (v, drv), s in mt.groupby(["variant", "drive"]):
        N["cost"][f"{v}_{drv}"] = {"sec_mean": float(s.gen_seconds.mean()), "sec_sd": float(s.gen_seconds.std()),
                                   "vram_mean": float(s.peak_vram_gb.mean()), "vram_max": float(s.peak_vram_gb.max()),
                                   "n": int(len(s))}
    rawmeta = {json.loads(f.read_text())["uid"]: json.loads(f.read_text()) for f in (runs / "meta").glob("*.json")
               if f.name.endswith("r0.json")}
    q = [m for m in rawmeta.values() if m.get("variant") in ("vidit", "w4a8") and not A.obsolete(m["uid"])]
    dit = q[0]["dit_params"]
    frac = [m["int8_param_fraction"] for m in q]
    i8 = np.mean([m["int8_params"] for m in q])
    pk = np.mean([m["w4a8_packed_weight_bytes"] for m in q if m.get("w4a8_packed_weight_bytes")])
    N["census"] = {"dit_params": int(dit), "quant_layers": int(q[0].get("w8a8_layers") or q[0].get("w4a8_layers")),
                   "quant_param_fraction_mean": float(np.mean(frac)), "quant_param_fraction_min": float(np.min(frac)),
                   "quant_param_fraction_max": float(np.max(frac)),
                   "weight_gb_fp16": dit * 2 / 1e9, "weight_gb_w8a8": (i8 * 1 + (dit - i8) * 2) / 1e9,
                   "weight_gb_w4a8": (pk + (dit - i8) * 2) / 1e9}

    # ------------------------------------------------------------------ violation diagnosis on post-corrected skeletons
    sys.path.insert(0, str(ROOT / "phase1"))
    from video_metrics import compute_biomechanical_plausibility
    cnt = {}
    vf = []
    for f in sorted((runs / "qtc" / "metrics").glob("*__kapt_smoothnet.json")):
        m = json.loads(f.read_text())
        if A.obsolete(m["uid"]):
            continue
        p = runs / "qtc" / "post" / Path(m["skeleton"]).name
        d = json.loads(p.read_text())
        r = compute_biomechanical_plausibility(d["frames"], image_size=d.get("image_size"))
        vf.append(r["violation_fraction"])
        for kk, vv in r.items():
            if kk.startswith("per_angle_violation_count_"):
                cnt[kk[26:]] = cnt.get(kk[26:], 0) + vv
    tot = sum(cnt.values())
    N["post_violation"] = {"n_videos": len(vf), "median_fraction": float(np.median(vf)), "counts": cnt,
                           "yaw_share": float((cnt.get("left_shoulder_yaw", 0) + cnt.get("right_shoulder_yaw", 0)) / tot)
                           if tot else float("nan")}

    # ------------------------------------------------------------------ driving-correction field (v3)
    N["field"] = {}
    for f in sorted((runs / "qtc" / "driving_correction").glob("*_correction_v3.json")):
        d = json.loads(f.read_text())
        st = d["stats"]
        W, H = st.get("image_size", [1080, 1920])
        N["field"][d["motion"]] = {"mean_abs": st["mean_abs_delta"], "p95_abs": st["p95_abs_delta"],
                                   "mean_abs_px_width": st["mean_abs_delta"] * W, "joints_set": st["joints_set"],
                                   "capped": st["capped"], "guard_zeroed": st["guard_zeroed"], "frames": st["frames"]}

    (out / "numbers.json").write_text(json.dumps(N, indent=1, default=float))
    print(json.dumps({"counts": N["counts"], "nf": {p: N["noise_floor"][p]["fail_rate"] for p in POSTS},
                      "loco": {k_: (v_["pass_rate_fixed"], v_["pass_rate_loco"], v_["pass_rate_loco_noviol"])
                               for k_, v_ in N["arms"].items()},
                      "loco_self": N["noise_floor_loco_self"], "census": N["census"],
                      "post_violation": N["post_violation"]}, indent=1))


if __name__ == "__main__":
    main()
