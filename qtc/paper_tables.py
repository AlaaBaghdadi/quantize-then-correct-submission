#!/usr/bin/env python3
"""LaTeX tables of the Article 3 manuscript, generated from paper/data/numbers.json (no hand-typed numbers).

    python -m qtc.paper_tables --data paper/data --out paper/tables
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CHK = [("accel", r"$\drel(\sigma^2_\mathrm{acc})$", "0.10"), ("limb", r"$\drel(\bar\sigma_\mathrm{limb})$", "0.10"),
       ("jitter", r"$\drel(\sigma^2_\mathrm{jit})$", "0.15"), ("viol", r"$\Delta f_\mathrm{viol}$", "0.02"),
       ("npe", "NPE", "0.15"), ("npe95", r"NPE$_{95}$", "0.30"), ("under", "under-motion", "0")]
KIN = [("acceleration_variance", r"$\sigma^2_\mathrm{acc}$"), ("mean_limb_length_std", r"$\bar\sigma_\mathrm{limb}$"),
       ("var_accel_norm", r"$\sigma^2_\mathrm{jit}$"), ("violation_fraction", r"$f_\mathrm{viol}$")]


def f(x, d=3):
    if x is None or x != x:
        return "--"
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}"
    return f"{x:.{d}f}"


def p_fmt(p):
    if p is None or p != p:
        return "--"
    if p < 0.001:
        return r"$<$0.001"
    return f"{p:.3f}"


def mi(d, k=None, dd=3):
    return f"{f(d['median'], dd)} [{f(d['q1'], dd)}, {f(d['q3'], dd)}]"


def tab_noise(N, out):
    nf = N["noise_floor"]["none"]
    a8 = N["arms"]["vidit_raw_none"]
    a4 = N["arms"]["w4a8_raw_none"]
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Gate quantities for the \fp{} seed pairs ($n=12$) and the quantization pairs ($n=24$ per bit-width), unprocessed skeletons. Entries are median [interquartile range] and the percentage of pairs failing the fixed threshold $\tau$. $p$: two-sided Mann--Whitney test of the quantization-pair against the seed-pair distribution.}",
         r"\label{tab:noise}", r"\small", r"\begin{tabular}{@{}llrlrlrrr@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{\fp{} seed pairs} & \multicolumn{2}{c}{\wea{} vs \fp} & \multicolumn{2}{c}{\wfa{} vs \fp} & \multicolumn{2}{c}{$p$ vs seed} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(l){9-10}",
         r"Check & $\tau$ & median [IQR] & fail & median [IQR] & fail & median [IQR] & fail & \wea & \wfa \\", r"\midrule"]
    L[5] = r"\begin{tabular}{@{}llrrrrrrrr@{}}"
    for k, name, tau in CHK:
        row = [name, tau]
        for e in (nf, a8, a4):
            row += [mi(e["delta"][k]), f"{100 * e['check_fail_rate'][k]:.0f}\\%"]
        row += [p_fmt(a8["vs_seed_noise"][k]["p"]), p_fmt(a4["vs_seed_noise"][k]["p"])]
        L.append(" & ".join(row) + r" \\")
    L.append(r"\midrule")
    L.append(r"All seven checks & & \multicolumn{2}{r}{" + f"{nf['n_fail']}/{nf['n']} fail" + r"} & \multicolumn{2}{r}{"
             + f"{a8['n'] - a8['pass_fixed']}/{a8['n']} fail" + r"} & \multicolumn{2}{r}{" + f"{a4['n'] - a4['pass_fixed']}/{a4['n']} fail" + r"} & & \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (out / "tab_noise.tex").write_text("\n".join(L) + "\n")


def tab_abs(N, out):
    A = N["absolute"]
    P = N["paired"]
    PM = N["paired_post_matched"]
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Absolute kinematic statistics (median over 24 clip--seed units) and paired comparison of each quantized arm with \fp{} (two-sided Wilcoxon signed-rank test, Holm-adjusted within each family of four metrics; $\delta$: Cliff's delta). Post-corrected statistics are compared with the post-corrected \fp{} statistics.}",
         r"\label{tab:abs}", r"\small", r"\begin{tabular}{@{}llrrrrrrr@{}}", r"\toprule",
         r"Skeletons & Metric & \fp & \wea & $\delta$ & $p_\mathrm{Holm}$ & \wfa & $\delta$ & $p_\mathrm{Holm}$ \\", r"\midrule"]
    for post, lab in (("none", "unprocessed"), ("kapt_smoothnet", "post-corrected")):
        for i, (k, name) in enumerate(KIN):
            row = [lab if i == 0 else "", name, f(A[f"fp16_raw_{post}"][k]["median"])]
            for v in ("vidit", "w4a8"):
                if post == "none":
                    t = P[f"H1_damage_{v}|{k}"]
                else:
                    t = PM[f"{v}|{k}"]
                row += [f(A[f"{v}_raw_{post}"][k]["median"]), f"{t['cliffs_delta']:+.2f}", p_fmt(t["p_holm"])]
            L.append(" & ".join(row) + r" \\")
        if post == "none":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (out / "tab_abs.tex").write_text("\n".join(L) + "\n")


def tab_corr(N, out):
    """Pre-correction (vs raw drive) and post-correction (vs unprocessed) effects, unprocessed/processed skeletons."""
    P = N["paired"]
    A = N["absolute"]
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Effect of the correction strategies on absolute kinematic statistics (median over 24 clip--seed units; Wilcoxon signed-rank test, Holm-adjusted within each family of four metrics). Pre-correction: pre-corrected drive against raw drive, unprocessed skeletons. Post-correction: post-corrected against unprocessed skeletons of the same videos (raw drive).}",
         r"\label{tab:corr}", r"\small", r"\begin{tabular}{@{}llrrrrrrrr@{}}", r"\toprule",
         r" & & \multicolumn{4}{c}{\wea} & \multicolumn{4}{c}{\wfa} \\ \cmidrule(lr){3-6}\cmidrule(l){7-10}",
         r"Strategy & Metric & before & after & $\delta$ & $p_\mathrm{Holm}$ & before & after & $\delta$ & $p_\mathrm{Holm}$ \\", r"\midrule"]
    for fam, lab in (("H2_pre", "pre-correction"), ("H3_post", "post-correction")):
        for i, (k, name) in enumerate(KIN):
            row = [lab if i == 0 else "", name]
            for v in ("vidit", "w4a8"):
                t = P[f"{fam}_{v}|{k}"]
                row += [f(t["median_b"]), f(t["median_a"]), f"{t['cliffs_delta']:+.2f}", p_fmt(t["p_holm"])]
            L.append(" & ".join(row) + r" \\")
        if fam == "H2_pre":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (out / "tab_corr.tex").write_text("\n".join(L) + "\n")


def tab_gate(N, out):
    arms = [("vidit_raw", r"\wea"), ("vidit_pre", r"\wea{}+pre"), ("w4a8_raw", r"\wfa"), ("w4a8_pre", r"\wfa{}+pre")]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Gate pass counts with the fixed thresholds of \cite{albaghdadi2026gate} and with leave-one-clip-out (LOCO) thresholds calibrated on the \fp{} seed pairs, with Wilson 95\% intervals of the pass rate. For the \fp{} seed pairs the LOCO row gives the expected pass rate of output indistinguishable from re-seeding.}",
         r"\label{tab:gate}", r"\small", r"\begin{tabular}{@{}lllcc@{}}", r"\toprule",
         r"Skeletons & Arm & $n$ & fixed $\tau$ & LOCO $\tau$ \\", r"\midrule"]

    def cell(k, n, lo, hi):
        return f"{k}/{n} ({100 * lo:.0f}--{100 * hi:.0f}\\%)"
    from math import sqrt

    def wil(k, n, z=1.96):
        ph = k / n
        den = 1 + z * z / n
        c = (ph + z * z / (2 * n)) / den
        h = z * sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
        return c - h, c + h
    for post, lab in (("none", "unprocessed"), ("kapt_smoothnet", "post-corrected")):
        nf = N["noise_floor"][post]
        ls = N["noise_floor_loco_self"][post]
        kfix = nf["n"] - nf["n_fail"]
        L.append(" & ".join([lab, r"\fp{} seed", str(nf["n"]), cell(kfix, nf["n"], *wil(kfix, nf["n"])),
                             cell(ls["n_pass"], ls["n"], *wil(ls["n_pass"], ls["n"]))]) + r" \\")
        for key, name in arms:
            e = N["arms"][f"{key}_{post}"]
            L.append(" & ".join(["", name, str(e["n"]), cell(e["pass_fixed"], e["n"], *e["wilson_fixed"]),
                                 cell(e["pass_loco"], e["n"], *e["wilson_loco"])]) + r" \\")
        if post == "none":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "tab_gate.tex").write_text("\n".join(L) + "\n")


def tab_pixel_cost(N, out):
    px = N["pixel"]
    co = N["cost"]
    ce = N["census"]
    rows = [("fp16_seed", r"\fp{} (seed 7 vs 42)", "fp16_raw", ce["weight_gb_fp16"]),
            ("vidit_raw", r"\wea", "vidit_raw", ce["weight_gb_w8a8"]), ("vidit_pre", r"\wea{}+pre", "vidit_pre", None),
            ("w4a8_raw", r"\wfa$^\dagger$", "w4a8_raw", ce["weight_gb_w4a8"]), ("w4a8_pre", r"\wfa{}+pre$^\dagger$", "w4a8_pre", None)]
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Pixel fidelity against the \fp{} reference (median over pairs; the \fp{} row compares the two seeds) and cost per video on one A100-SXM4 80\,GB GPU (25 steps, $768\times1360$, 37 frames; mean $\pm$ s.d. over 24 videos). Weight memory is that of the denoiser. $^\dagger$Simulated W4A8: numerics of a W4A8 kernel, latency and peak memory of a bfloat16 execution; weight memory is the packed 4-bit footprint.}",
         r"\label{tab:pixelcost}", r"\small", r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
         r"Arm & PSNR (dB) & SSIM & LPIPS & tLPIPS & tMSE & time (s) & peak mem. (GB) & weights (GB) \\", r"\midrule"]
    for pk, name, ck, w in rows:
        p = px[pk]
        c = co[ck]
        L.append(" & ".join([name, f"{p['psnr_db']['median']:.1f}", f"{p['ssim']['median']:.3f}", f"{p['lpips']['median']:.3f}",
                             f"{p['tlpips']['median']:.4f}", f"{p['tmse']['median']:.4f}",
                             f"{c['sec_mean']:.0f} $\\pm$ {c['sec_sd']:.0f}", f"{c['vram_mean']:.1f}",
                             f"{w:.2f}" if w else "--"]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (out / "tab_pixelcost.tex").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    N = json.loads((Path(a.data) / "numbers.json").read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tab_noise(N, out)
    tab_abs(N, out)
    tab_corr(N, out)
    tab_gate(N, out)
    tab_pixel_cost(N, out)
    print(sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
