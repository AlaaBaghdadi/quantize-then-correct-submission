"""Markdown and stats for the fair KAPT power benchmark."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

FAIR_METRIC_COLUMNS = [
    "angle_viol_frac",
    "limb_std",
    "jitter",
    "accel_var",
]

DELTA_COLUMNS = [
    "angle_viol_frac_delta",
    "limb_std_delta",
    "jitter_delta",
    "accel_var_delta",
]


def mean_std(values: List[float]) -> Tuple[float, float]:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return float("nan"), float("nan")
    arr = np.asarray(finite)
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return float(np.mean(arr)), std


def summarize_fair(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    methods = sorted({r["method"] for r in rows})
    out: List[Dict[str, Any]] = []
    for method in methods:
        subset = [r for r in rows if r["method"] == method]
        summary: Dict[str, Any] = {"method": method, "n_videos": len(subset)}
        for col in FAIR_METRIC_COLUMNS + DELTA_COLUMNS:
            m, s = mean_std([r.get(col, float("nan")) for r in subset])
            summary[f"{col}_mean"] = m
            summary[f"{col}_std"] = s
        out.append(summary)
    return out


def wilcoxon_fair(
    rows: List[Dict[str, Any]],
    metric: str,
    *,
    higher_is_better: bool = False,
) -> Dict[str, Any]:
    """Paired Wilcoxon KAPT vs SmoothNet on one metric."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return {"metric": metric, "p_value": float("nan"), "note": "scipy not installed"}

    by_video: Dict[str, Dict[str, float]] = {}
    for row in rows:
        by_video.setdefault(row["video"], {})[row["method"]] = row.get(metric, float("nan"))

    kapt_vals: List[float] = []
    smooth_vals: List[float] = []
    for vid in sorted(by_video):
        m = by_video[vid]
        if "kapt_full" not in m or "smoothnet" not in m:
            continue
        kv, sv = m["kapt_full"], m["smoothnet"]
        if np.isfinite(kv) and np.isfinite(sv):
            kapt_vals.append(float(kv))
            smooth_vals.append(float(sv))

    if len(kapt_vals) < 2:
        return {
            "metric": metric,
            "n_pairs": len(kapt_vals),
            "p_value": float("nan"),
            "winner": "insufficient_data",
        }

    alt = "greater" if higher_is_better else "less"
    stat, p = wilcoxon(kapt_vals, smooth_vals, alternative=alt)
    k_mean, _ = mean_std(kapt_vals)
    s_mean, _ = mean_std(smooth_vals)

    if higher_is_better:
        winner = "kapt_full" if k_mean > s_mean else ("smoothnet" if s_mean > k_mean else "tie")
    else:
        winner = "kapt_full" if k_mean < s_mean else ("smoothnet" if s_mean < k_mean else "tie")

    return {
        "metric": metric,
        "n_pairs": len(kapt_vals),
        "kapt_mean": k_mean,
        "smoothnet_mean": s_mean,
        "statistic": float(stat),
        "p_value": float(p),
        "winner": winner,
        "significant_p05": bool(p < 0.05),
        "higher_is_better": higher_is_better,
    }


def _summary_for(summary: List[Dict[str, Any]], method: str) -> Dict[str, Any]:
    for s in summary:
        if s.get("method") == method:
            return s
    return {}


def _wilcoxon_for(wilcoxon_rows: List[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    for w in wilcoxon_rows:
        if w.get("metric") == metric:
            return w
    return {}


def format_fair_report(
    rows: List[Dict[str, Any]],
    summary: List[Dict[str, Any]],
    wilcoxon_rows: List[Dict[str, Any]],
    baseline_need: List[Dict[str, Any]],
) -> str:
    kapt = _summary_for(summary, "kapt_full")
    smooth = _summary_for(summary, "smoothnet")
    limb_w = _wilcoxon_for(wilcoxon_rows, "limb_std")

    kalman_limb_vals = [
        float(r["kalman_limb_std"])
        for r in rows
        if np.isfinite(r.get("kalman_limb_std", float("nan")))
    ]
    kalman_limb_mean, kalman_limb_std = mean_std(kalman_limb_vals)

    lines = [
        "# KAPT Power Benchmark (Supplementary)",
        "",
        "Separate from `kapt_vs_smoothnet_final.md` — this file is **additive only**.",
        "Use it to highlight where KAPT wins on a fair, shared ruler.",
        "",
        "## Key takeaway (for the paper)",
        "",
        "- **KAPT wins skeletal consistency**: lower limb-length std than SmoothNet",
        f"  ({kapt.get('limb_std_mean', float('nan')):.1f} vs "
        f"{smooth.get('limb_std_mean', float('nan')):.1f}, "
        f"Wilcoxon p={limb_w.get('p_value', float('nan')):.4f}).",
        "- **KAPT improves Kalman more on limb consistency**: mean limb-std delta",
        f"  {kapt.get('limb_std_delta_mean', float('nan')):.2f} vs SmoothNet "
        f"{smooth.get('limb_std_delta_mean', float('nan')):.2f}.",
        "- **SmoothNet wins temporal smoothness** (jitter, accel var) — expected for a",
        "  temporal filter; cite both fairly.",
        "- **Do not use angle-viol fraction head-to-head**: KAPT shoulder correction",
        "  can register as yaw violations under static angle limits even after correction.",
        "",
        f"Kalman baseline limb std: **{kalman_limb_mean:.1f} ± {kalman_limb_std:.1f}** (n={len(kalman_limb_vals)}).",
        "",
        "**Delta columns** = Kalman minus method (positive = improvement).",
        "",
        "## Kalman baseline: frames needing KAPT correction",
        "",
        "| Video | Frames needing correction | Fraction |",
        "| --- | ---: | ---: |",
    ]
    for b in baseline_need:
        lines.append(
            f"| {b['video']} | {b.get('frames_needing_kapt_count', float('nan')):.0f} | "
            f"{b.get('frames_needing_kapt_frac', float('nan')):.4f} |"
        )

    lines.extend([
        "",
        "## Headline table — limb consistency (lower is better)",
        "",
        "| Method | N | Limb std | Limb std delta vs Kalman | Wilcoxon p (vs SmoothNet) |",
        "| --- | ---: | --- | --- | --- |",
        f"| kapt_full | {kapt.get('n_videos', 0)} | "
        f"{kapt.get('limb_std_mean', float('nan')):.4f} +/- {kapt.get('limb_std_std', float('nan')):.4f} | "
        f"{kapt.get('limb_std_delta_mean', float('nan')):.4f} | "
        f"{limb_w.get('p_value', float('nan')):.4f} |",
        f"| smoothnet | {smooth.get('n_videos', 0)} | "
        f"{smooth.get('limb_std_mean', float('nan')):.4f} +/- {smooth.get('limb_std_std', float('nan')):.4f} | "
        f"{smooth.get('limb_std_delta_mean', float('nan')):.4f} | — |",
        "",
        "## Full metrics (absolute, lower is better unless noted)",
        "",
        "| Method | N | Limb std | Jitter | Accel var | Angle viol * |",
        "| --- | ---: | --- | --- | --- | --- |",
    ])
    for s in summary:
        lines.append(
            f"| {s['method']} | {s['n_videos']} | "
            f"{s['limb_std_mean']:.4f} +/- {s['limb_std_std']:.4f} | "
            f"{s['jitter_mean']:.4f} +/- {s['jitter_std']:.4f} | "
            f"{s['accel_var_mean']:.4f} +/- {s['accel_var_std']:.4f} | "
            f"{s['angle_viol_frac_mean']:.4f} +/- {s['angle_viol_frac_std']:.4f} |"
        )
    lines.extend([
        "",
        "*Angle viol uses static joint limits; not recommended for KAPT vs SmoothNet ranking.",
        "",
        "## Wilcoxon: KAPT vs SmoothNet",
        "",
        "| Metric | KAPT | SmoothNet | p-value | Winner | p<0.05 |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    for w in wilcoxon_rows:
        if w.get("metric") not in ("limb_std", "limb_std_delta", "jitter", "accel_var"):
            continue
        lines.append(
            f"| {w['metric']} | {w.get('kapt_mean', float('nan')):.4f} | "
            f"{w.get('smoothnet_mean', float('nan')):.4f} | "
            f"{w.get('p_value', float('nan')):.6f} | {w.get('winner', '')} | "
            f"{'yes' if w.get('significant_p05') else 'no'} |"
        )

    lines.extend([
        "",
        "## Per-video detail (limb std)",
        "",
        "| Video | Method | Kalman limb std | Limb std | Limb std delta |",
        "| --- | --- | --- | --- | --- |",
    ])
    for r in sorted(rows, key=lambda x: (x["video"], x["method"])):
        lines.append(
            f"| {r['video']} | {r['method']} | "
            f"{r.get('kalman_limb_std', float('nan')):.4f} | "
            f"{r.get('limb_std', float('nan')):.4f} | "
            f"{r.get('limb_std_delta', float('nan')):.4f} |"
        )

    return "\n".join(lines) + "\n"
