#!/usr/bin/env python3
"""Journal figures for the Article 3 manuscript (serif, print-safe, 300 dpi PNG + vector PDF).

    python -m qtc.paper_figures --data paper/data --out paper/figures
Reads the CSV/JSON written by qtc.paper_stats.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

C = {"fp16": "#6b6b6b", "vidit": "#2b6cb0", "w4a8": "#c8612f"}
LBL = {"fp16": "BF16 (s7 vs s42)", "vidit": "W8A8", "w4a8": "W4A8"}
FULL_W = 7.16   # double-column text width (in)
COL_W = 3.5     # single column (in)
TH = {"accel": 0.10, "limb": 0.10, "jitter": 0.15, "npe": 0.15, "npe95": 0.30}


def style():
    import glob
    from matplotlib import font_manager
    for f in glob.glob("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-*.otf"):
        font_manager.fontManager.addfont(f)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["TeX Gyre Termes", "Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})


def save(fig, out: Path, name: str):
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
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=6, color=c, lw=0, alpha=0.85, zorder=3)
    if thr is not None:
        ax.axhline(thr, color="black", lw=0.6, ls=(0, (3, 2)), zorder=1)
    if log:
        ax.set_yscale("log")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels)
    ax.grid(axis="y", lw=0.3, color="#d9d9d9", zorder=0)
    ax.set_axisbelow(True)


def fig_noise_floor(g: pd.DataFrame, px: pd.DataFrame, out: Path):
    """Quantized same-seed deltas against the FP16 seed-to-seed floor (unprocessed skeletons)."""
    s = g[g.post == "none"]
    sets = [("fp16", s[s.tag == "fp16_seed"]), ("vidit", s[s.tag == "vidit_raw"]), ("w4a8", s[s.tag == "w4a8_raw"])]
    panels = [("accel", r"$|\Delta_{\rm rel}|$ acceleration variance"), ("jitter", r"$|\Delta_{\rm rel}|$ normalised jitter"),
              ("limb", r"$|\Delta_{\rm rel}|$ limb-length s.d."), ("npe", "mean NPE")]
    fig, axs = plt.subplots(1, 5, figsize=(FULL_W, 2.0))
    labels = ["BF16\nseed", "W8A8", "W4A8"]
    for ax, (k, t) in zip(axs, panels):
        strip(ax, [d[k] for _, d in sets], [C[n] for n, _ in sets], labels, thr=TH.get(k))
        ax.set_title(t)
    pxs = [px[px.tag == "fp16_seed"], px[px.tag == "vidit_raw"], px[px.tag == "w4a8_raw"]]
    strip(axs[4], [d["lpips"] for d in pxs], [C["fp16"], C["vidit"], C["w4a8"]], labels, log=True)
    axs[4].set_title("LPIPS (pixel)")
    fig.tight_layout(w_pad=0.6)
    save(fig, out, "fig_noise_floor")


def fig_gate(N: dict, out: Path):
    """Gate pass rates: fixed Article 2 thresholds vs leave-one-clip-out noise-floor thresholds."""
    arms = [("vidit_raw", "W8A8"), ("vidit_pre", "W8A8+pre"), ("w4a8_raw", "W4A8"), ("w4a8_pre", "W4A8+pre")]
    fig, axs = plt.subplots(1, 2, figsize=(FULL_W, 2.1), sharey=True)
    for ax, post, title in zip(axs, ["none", "kapt_smoothnet"], ["Unprocessed skeletons", "KAPT$\\rightarrow$SmoothNet on both sides"]):
        y = np.arange(len(arms) + 1)
        nf = N["noise_floor"][post]
        loco_self = N["noise_floor_loco_self"][post]
        rows = [("BF16 seed pairs", 1 - nf["fail_rate"], loco_self["pass_rate"], "fp16", nf["n"], loco_self["n_pass"])]
        for key, lab in arms:
            e = N["arms"][f"{key}_{post}"]
            rows.append((lab, e["pass_rate_fixed"], e["pass_rate_loco"], key.split("_")[0], e["n"], e["pass_loco"]))
        for i, (lab, pf, pl, v, n, kl) in enumerate(rows):
            yy = len(rows) - 1 - i
            for p, mk, fill in ((pf, "o", "white"), (pl, "o", C[v])):
                k = round(p * n)
                lo, hi = wilson(k, n)
                ax.plot([lo, hi], [yy, yy], color=C[v], lw=0.8, alpha=0.6 if fill == "white" else 1)
                ax.scatter([p], [yy], s=22, marker=mk, facecolor=fill, edgecolor=C[v], lw=0.9, zorder=3)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows][::-1])
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("gate pass rate (95% Wilson CI)")
        ax.set_title(title)
        ax.grid(axis="x", lw=0.3, color="#d9d9d9")
        ax.set_axisbelow(True)
    h1 = plt.Line2D([], [], marker="o", ls="", mfc="white", mec="black", label="fixed thresholds (Article 2)")
    h2 = plt.Line2D([], [], marker="o", ls="", mfc="black", mec="black", label="noise-floor thresholds (LOCO)")
    fig.legend(handles=[h1, h2], loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.55, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, out, "fig_gate_calibration")


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def fig_correction(a: pd.DataFrame, px: pd.DataFrame, out: Path):
    """Absolute levels per arm, with and without post-correction; PSNR of the pre-corrected arms."""
    arms = [("fp16", "raw", "BF16"), ("vidit", "raw", "W8A8"), ("vidit", "pre", "W8A8\n+pre"),
            ("w4a8", "raw", "W4A8"), ("w4a8", "pre", "W4A8\n+pre")]
    fig, axs = plt.subplots(1, 3, figsize=(FULL_W, 2.15), gridspec_kw={"width_ratios": [1.25, 1.25, 1]})
    for ax, k, t in zip(axs[:2], ["acceleration_variance", "var_accel_norm"],
                        ["acceleration variance (px$^2$/frame$^4$)", "normalised jitter"]):
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
                ax.scatter(x + rng.uniform(-0.08, 0.08, len(vals)), vals, s=4, color=c, lw=0, alpha=0.8, zorder=3)
        ax.set_yscale("log")
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([t_ for *_, t_ in arms])
        ax.set_title(t)
        ax.grid(axis="y", lw=0.3, color="#d9d9d9")
        ax.set_axisbelow(True)
    h1 = matplotlib.patches.Patch(facecolor=matplotlib.colors.to_rgba("#555", 0.18), edgecolor="#555", lw=0.5, label="unprocessed")
    h2 = matplotlib.patches.Patch(facecolor=matplotlib.colors.to_rgba("#555", 0.55), edgecolor="#555", lw=0.5, label="post-corrected")
    fig.legend(handles=[h1, h2], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.36, 1.06))
    pa = [("fp16_seed", "BF16\nseed", "fp16"), ("vidit_raw", "W8A8", "vidit"), ("vidit_pre", "W8A8\n+pre", "vidit"),
          ("w4a8_raw", "W4A8", "w4a8"), ("w4a8_pre", "W4A8\n+pre", "w4a8")]
    strip(axs[2], [px[px.tag == t]["psnr_db"] for t, *_ in pa], [C[c] for *_, c in pa], [l for _, l, _ in pa], log=False)
    axs[2].set_title("PSNR vs BF16 reference (dB)")
    fig.tight_layout(w_pad=0.8)
    save(fig, out, "fig_correction")


def fig_assoc(k: pd.DataFrame, N: dict, out: Path):
    fig, axs = plt.subplots(1, 2, figsize=(COL_W * 2 * 0.95, 2.1))
    for ax, m, t in zip(axs, ["npe", "accel"], ["mean NPE", r"$|\Delta_{\rm rel}|$ acceleration variance"]):
        for tag, mk in [("vidit_raw", "o"), ("vidit_pre", "^"), ("w4a8_raw", "o"), ("w4a8_pre", "^")]:
            s = k[k.tag == tag]
            v = tag.split("_")[0]
            ax.scatter(s["lpips"], s[m], s=10, marker=mk, facecolor=C[v] if tag.endswith("raw") else "white",
                       edgecolor=C[v], lw=0.7, label=("W8A8" if v == "vidit" else "W4A8") + ("" if tag.endswith("raw") else "+pre"))
        r = N["assoc"][f"lpips_{m}"]
        ax.set_title(rf"Spearman $\rho$ = {r['rho']:.2f} (p = {r['p']:.3f}, n = {r['n']})")
        ax.set_xlabel("LPIPS vs BF16 same seed")
        ax.set_ylabel(t)
        if m == "accel":
            ax.set_yscale("log")
        ax.grid(lw=0.3, color="#d9d9d9")
        ax.set_axisbelow(True)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, out, "fig_kin_vs_pixel")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    g = pd.read_csv(d / "gate_long.csv")
    px = pd.read_csv(d / "pixel_long.csv")
    ab = pd.read_csv(d / "absolute_long.csv")
    k = pd.read_csv(d / "assoc_long.csv")
    N = json.loads((d / "numbers.json").read_text())
    fig_noise_floor(g, px, out)
    fig_gate(N, out)
    fig_correction(ab, px, out)
    fig_assoc(k, N, out)
    print(sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
