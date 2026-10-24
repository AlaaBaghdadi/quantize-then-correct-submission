#!/usr/bin/env python3
"""Figures of the IJIES R1 revision (journal style, IJIES format rules).

    python -m qtc.revision_figures [--data paper/data] [--stats revision/ijies-r1/stats] [--out paper/ijies/figs_r1]

Differences from qtc/paper_figures.py (submitted version): the figures are drawn at the printed width
(16.8 cm, both columns) with all lettering at 10 pt, the unquantized reference is labelled BF16, no
p-values of tests that ignore clip clustering appear inside a figure, and one figure is added for the
stability of the leave-one-clip-out gate (from qtc/revision_stats.py). Output: 300 dpi PNG and vector PDF.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

W_IN = 16.8 / 2.54           # printed width (both columns)
FS = 10                      # IJIES: letters >= 10 pt at printed size
C = {"fp16": "#6b6b6b", "vidit": "#2b6cb0", "w4a8": "#c8612f", "bf16_seed": "#6b6b6b"}
TH = {"accel": 0.10, "limb": 0.10, "jitter": 0.15, "npe": 0.15, "npe95": 0.30}


def style():
    import glob
    from matplotlib import font_manager
    for f in glob.glob("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-*.otf") + \
            glob.glob("/usr/share/texlive/texmf-dist/fonts/opentype/public/tex-gyre/texgyretermes-*.otf"):
        font_manager.fontManager.addfont(f)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["TeX Gyre Termes", "Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
        "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 300, "pdf.fonttype": 42})


def save(fig, out: Path, name: str):
    # fixed canvas (no bbox="tight"): the PNG/PDF width equals the printed width, so 10 pt stays 10 pt
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)


def strip(ax, groups, colors, labels, log=True, thr=None):
    rng = np.random.default_rng(0)
    for i, (vals, c) in enumerate(zip(groups, colors)):
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        if log:
            v = np.maximum(v, 1e-4)
        bp = ax.boxplot([v], positions=[i], widths=0.55, showfliers=False, patch_artist=True,
                        medianprops={"color": "black", "lw": 0.9}, whiskerprops={"lw": 0.6, "color": c},
                        capprops={"lw": 0.6, "color": c}, boxprops={"lw": 0.6, "edgecolor": c})
        bp["boxes"][0].set_facecolor(matplotlib.colors.to_rgba(c, 0.18))
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=9, color=c, lw=0, alpha=0.85, zorder=3)
    if thr is not None:
        ax.axhline(thr, color="black", lw=0.6, ls=(0, (3, 2)), zorder=1)
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels)
    ax.grid(axis="y", lw=0.3, color="#d9d9d9", zorder=0)
    ax.set_axisbelow(True)


def wilson(k, n, z=1.96):
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def fig_noise_floor(g, px, out):
    s = g[g.post == "none"]
    sets = [("fp16", s[s.tag == "fp16_seed"]), ("vidit", s[s.tag == "vidit_raw"]), ("w4a8", s[s.tag == "w4a8_raw"])]
    labels = ["BF16\nseeds", "W8A8", "W4A8"]
    panels = [("accel", r"$\Delta_{\rm rel}(\sigma^2_{\rm acc})$"), ("jitter", r"$\Delta_{\rm rel}(\sigma^2_{\rm jit})$"),
              ("limb", r"$\Delta_{\rm rel}(\sigma_{\rm limb})$"), ("npe", "mean NPE"), ("npe95", r"NPE$_{95}$")]
    fig, axs = plt.subplots(2, 3, figsize=(W_IN, 4.6))
    axs = axs.ravel()
    for ax, (k, t) in zip(axs, panels):
        strip(ax, [d[k] for _, d in sets], [C[n] for n, _ in sets], labels, thr=TH.get(k))
        ax.set_title(t)
    pxs = [px[px.tag == "fp16_seed"], px[px.tag == "vidit_raw"], px[px.tag == "w4a8_raw"]]
    strip(axs[5], [d["lpips"] for d in pxs], [C["fp16"], C["vidit"], C["w4a8"]], labels, log=True)
    axs[5].set_title("LPIPS")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.1, wspace=0.38, hspace=0.62)
    save(fig, out, "fig_noise_floor")


def fig_assoc(k, out):
    fig, axs = plt.subplots(1, 2, figsize=(W_IN, 3.0))
    for ax, m, t in zip(axs, ["npe", "accel"], ["mean NPE", r"$\Delta_{\rm rel}(\sigma^2_{\rm acc})$"]):
        for tag, mk in [("vidit_raw", "o"), ("vidit_pre", "^"), ("w4a8_raw", "o"), ("w4a8_pre", "^")]:
            s = k[k.tag == tag]
            v = tag.split("_")[0]
            ax.scatter(s["lpips"], s[m], s=16, marker=mk, facecolor=C[v] if tag.endswith("raw") else "white",
                       edgecolor=C[v], lw=0.7, label=("W8A8" if v == "vidit" else "W4A8") + ("" if tag.endswith("raw") else "+pre"))
        ax.set_xlabel("LPIPS against the BF16 reference")
        ax.set_ylabel(t)
        if m == "accel":
            ax.set_yscale("log")
        ax.grid(lw=0.3, color="#d9d9d9")
        ax.set_axisbelow(True)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=4)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.97, bottom=0.3, wspace=0.3)
    save(fig, out, "fig_kin_vs_pixel")


def fig_gate(N, out):
    arms = [("vidit_raw", "W8A8"), ("vidit_pre", "W8A8+pre"), ("w4a8_raw", "W4A8"), ("w4a8_pre", "W4A8+pre")]
    fig, axs = plt.subplots(1, 2, figsize=(W_IN, 3.1), sharey=True)
    for ax, post, title in zip(axs, ["none", "kapt_smoothnet"], ["Unprocessed skeletons", "Post-corrected skeletons"]):
        nf, ls = N["noise_floor"][post], N["noise_floor_loco_self"][post]
        rows = [("BF16 seed pairs", 1 - nf["fail_rate"], ls["pass_rate"], "fp16", nf["n"])]
        for key, lab in arms:
            e = N["arms"][f"{key}_{post}"]
            rows.append((lab, e["pass_rate_fixed"], e["pass_rate_loco"], key.split("_")[0], e["n"]))
        for i, (lab, pf, pl, v, n) in enumerate(rows):
            yy = len(rows) - 1 - i
            for p, fill in ((pf, "white"), (pl, C[v])):
                lo, hi = wilson(round(p * n), n)
                ax.plot([lo, hi], [yy, yy], color=C[v], lw=0.8, alpha=0.6 if fill == "white" else 1)
                ax.scatter([p], [yy], s=30, facecolor=fill, edgecolor=C[v], lw=0.9, zorder=3)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows][::-1])
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("pass rate (Wilson 95% interval)")
        ax.set_title(title)
        ax.grid(axis="x", lw=0.3, color="#d9d9d9")
        ax.set_axisbelow(True)
    h1 = plt.Line2D([], [], marker="o", ls="", mfc="white", mec="black", label="fixed thresholds")
    h2 = plt.Line2D([], [], marker="o", ls="", mfc="black", mec="black", label="LOCO thresholds")
    fig.legend(handles=[h1, h2], loc="lower center", ncol=2, frameon=False)
    fig.subplots_adjust(left=0.17, right=0.99, top=0.91, bottom=0.3, wspace=0.08)
    save(fig, out, "fig_gate_calibration")


def fig_gate_stability(R, out):
    """Left: pass rate against the number of calibration clips; right: held-out 6/6 split pass rates."""
    G = R["main"]["gate"]["none"]["arms"]
    arms = [("bf16_seed", "BF16 seed pairs", "fp16"), ("vidit_raw", "W8A8", "vidit"), ("w4a8_raw", "W4A8", "w4a8")]
    fig, axs = plt.subplots(1, 2, figsize=(W_IN, 3.0), gridspec_kw={"width_ratios": [1.3, 1]})
    for j, (a, lab, c) in enumerate(arms):
        ms = sorted(int(m) for m in G[a]["calsize"])
        mean = [G[a]["calsize"][str(m)]["mean"] for m in ms]
        lo = [G[a]["calsize"][str(m)]["ci95"][0] for m in ms]
        hi = [G[a]["calsize"][str(m)]["ci95"][1] for m in ms]
        x = np.array(ms) + (j - 1) * 0.12
        axs[0].errorbar(x, mean, yerr=[np.maximum(np.array(mean) - lo, 0), np.maximum(np.array(hi) - mean, 0)], color=C[c], marker="o", ms=4,
                        lw=0.9, capsize=2, label=lab)
        sp = G[a]["split66"]
        axs[1].errorbar([j], [sp["mean"]], yerr=[[max(sp["mean"] - sp["ci95"][0], 0)], [max(sp["ci95"][1] - sp["mean"], 0)]],
                        color=C[c], marker="o", ms=5, capsize=3, lw=0.9)
    axs[0].set_xlabel("calibration clips per threshold")
    axs[0].set_ylabel("LOCO pass rate")
    axs[0].set_ylim(0, 1.02)
    axs[0].set_xticks(ms)
    axs[1].set_xticks(range(3))
    axs[1].set_xticklabels(["BF16\nseeds", "W8A8", "W4A8"])
    axs[1].set_ylabel("pass rate, 6/6 splits")
    axs[1].set_ylim(0, 1.02)
    axs[1].set_xlim(-0.5, 2.5)
    for ax in axs:
        ax.grid(lw=0.3, color="#d9d9d9")
        ax.set_axisbelow(True)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=3)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.97, bottom=0.31, wspace=0.32)
    save(fig, out, "fig_gate_stability")


def fig_correction(a, px, out):
    arms = [("fp16", "raw", "BF16"), ("vidit", "raw", "W8A8"), ("vidit", "pre", "W8A8\n+pre"),
            ("w4a8", "raw", "W4A8"), ("w4a8", "pre", "W4A8\n+pre")]
    fig, axs = plt.subplots(2, 2, figsize=(W_IN, 5.0))
    axs = axs.ravel()
    for ax, k, t, log in zip(axs[:3], ["acceleration_variance", "var_accel_norm", "violation_fraction"],
                             [r"$\sigma^2_{\rm acc}$", r"$\sigma^2_{\rm jit}$", r"$f_{\rm viol}$"], [True, True, False]):
        rng = np.random.default_rng(1)
        for i, (v, d, _) in enumerate(arms):
            for j, post in enumerate(["none", "kapt_smoothnet"]):
                vals = a[(a.variant == v) & (a.drive == d) & (a.post == post)][k].to_numpy(float)
                x = i + (j - 0.5) * 0.36
                c = C[v]
                bp = ax.boxplot([vals], positions=[x], widths=0.3, showfliers=False, patch_artist=True,
                                medianprops={"color": "black", "lw": 0.8}, whiskerprops={"lw": 0.5, "color": c},
                                capprops={"lw": 0.5, "color": c}, boxprops={"lw": 0.5, "edgecolor": c})
                bp["boxes"][0].set_facecolor(matplotlib.colors.to_rgba(c, 0.18 if post == "none" else 0.55))
                ax.scatter(x + rng.uniform(-0.08, 0.08, len(vals)), vals, s=5, color=c, lw=0, alpha=0.8, zorder=3)
        if log:
            ax.set_yscale("log")
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([t_ for *_, t_ in arms])
        ax.set_title(t)
        ax.grid(axis="y", lw=0.3, color="#d9d9d9")
        ax.set_axisbelow(True)
    pa = [("fp16_seed", "BF16\nseeds", "fp16"), ("vidit_raw", "W8A8", "vidit"), ("vidit_pre", "W8A8\n+pre", "vidit"),
          ("w4a8_raw", "W4A8", "w4a8"), ("w4a8_pre", "W4A8\n+pre", "w4a8")]
    strip(axs[3], [px[px.tag == t]["psnr_db"] for t, *_ in pa], [C[c] for *_, c in pa], [l for _, l, _ in pa], log=False)
    axs[3].set_title("PSNR against the BF16 reference (dB)")
    h1 = matplotlib.patches.Patch(facecolor=matplotlib.colors.to_rgba("#555", 0.18), edgecolor="#555", lw=0.5, label="unprocessed skeleton")
    h2 = matplotlib.patches.Patch(facecolor=matplotlib.colors.to_rgba("#555", 0.55), edgecolor="#555", lw=0.5, label="post-corrected skeleton")
    fig.legend(handles=[h1, h2], loc="lower center", ncol=2, frameon=False)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.17, wspace=0.25, hspace=0.75)
    save(fig, out, "fig_correction")


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--data", default=str(root / "paper" / "data"))
    ap.add_argument("--stats", default=str(root / "revision" / "ijies-r1" / "stats"))
    ap.add_argument("--out", default=str(root / "paper" / "ijies" / "figs_r1"))
    a = ap.parse_args()
    d, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    g = pd.read_csv(d / "gate_long.csv")
    px = pd.read_csv(d / "pixel_long.csv")
    ab = pd.read_csv(d / "absolute_long.csv")
    k = pd.read_csv(d / "assoc_long.csv")
    N = json.loads((d / "numbers.json").read_text())
    R = json.loads((Path(a.stats) / "revision_stats.json").read_text())
    fig_noise_floor(g, px, out)
    fig_assoc(k, out)
    fig_gate(N, out)
    fig_gate_stability(R, out)
    fig_correction(ab, px, out)
    print("figures ->", out)


if __name__ == "__main__":
    main()
