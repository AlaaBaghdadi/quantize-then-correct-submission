"""Stats and reporting for KAPT + SmoothNet hybrid benchmark."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from experiments.benchmark_comparison.fair_stats import mean_std

HYBRID_METHODS = ["smoothnet", "kapt_full", "kapt_then_smoothnet"]

METHOD_LABELS = {
    "smoothnet": "SmoothNet (Kalman only)",
    "kapt_full": "KAPT (full)",
    "kapt_then_smoothnet": "KAPT -> SmoothNet",
}


def summarize_hybrid(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from experiments.benchmark_comparison.fair_stats import DELTA_COLUMNS, FAIR_METRIC_COLUMNS

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


def wilcoxon_pair(
    rows: List[Dict[str, Any]],
    method_a: str,
    method_b: str,
    metric: str,
    *,
    higher_is_better: bool = False,
) -> Dict[str, Any]:
    """Paired Wilcoxon between two methods on one metric."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return {
            "comparison": f"{method_a} vs {method_b}",
            "metric": metric,
            "p_value": float("nan"),
            "note": "scipy not installed",
        }

    by_video: Dict[str, Dict[str, float]] = {}
    for row in rows:
        by_video.setdefault(row["video"], {})[row["method"]] = row.get(metric, float("nan"))

    a_vals: List[float] = []
    b_vals: List[float] = []
    for vid in sorted(by_video):
        m = by_video[vid]
        if method_a not in m or method_b not in m:
            continue
        av, bv = m[method_a], m[method_b]
        if np.isfinite(av) and np.isfinite(bv):
            a_vals.append(float(av))
            b_vals.append(float(bv))

    if len(a_vals) < 2:
        return {
            "comparison": f"{method_a} vs {method_b}",
            "metric": metric,
            "n_pairs": len(a_vals),
            "p_value": float("nan"),
            "winner": "insufficient_data",
        }

    alt = "greater" if higher_is_better else "less"
    stat, p = wilcoxon(a_vals, b_vals, alternative=alt)
    a_mean, _ = mean_std(a_vals)
    b_mean, _ = mean_std(b_vals)

    if higher_is_better:
        winner = method_a if a_mean > b_mean else (method_b if b_mean > a_mean else "tie")
    else:
        winner = method_a if a_mean < b_mean else (method_b if b_mean < a_mean else "tie")

    row: Dict[str, Any] = {
        "comparison": f"{method_a} vs {method_b}",
        "metric": metric,
        "n_pairs": len(a_vals),
        f"{method_a}_mean": a_mean,
        f"{method_b}_mean": b_mean,
        "statistic": float(stat),
        "p_value": float(p),
        "winner": winner,
        "significant_p05": bool(p < 0.05),
        "higher_is_better": higher_is_better,
    }
    return row


def format_hybrid_report(
    rows: List[Dict[str, Any]],
    summary: List[Dict[str, Any]],
    wilcoxon_rows: List[Dict[str, Any]],
) -> str:
    hybrid = next((s for s in summary if s["method"] == "kapt_then_smoothnet"), {})
    kapt = next((s for s in summary if s["method"] == "kapt_full"), {})
    smooth = next((s for s in summary if s["method"] == "smoothnet"), {})

    lines = [
        "# KAPT + SmoothNet Hybrid Benchmark",
        "",
        "Pipeline: **Kalman -> KAPT -> SmoothNet** (`kapt_then_smoothnet`).",
        "Compared against Kalman-only SmoothNet and KAPT alone.",
        "All metrics evaluated vs the same Kalman baseline. Lower is better unless noted.",
        "",
        "## Summary (mean ± std)",
        "",
        "| Method | Limb std | Jitter | Accel var | Angle viol * |",
        "| --- | --- | --- | --- | --- |",
    ]
    for s in summary:
        label = METHOD_LABELS.get(s["method"], s["method"])
        lines.append(
            f"| {label} | "
            f"{s['limb_std_mean']:.4f} ± {s['limb_std_std']:.4f} | "
            f"{s['jitter_mean']:.4f} ± {s['jitter_std']:.4f} | "
            f"{s['accel_var_mean']:.4f} ± {s['accel_var_std']:.4f} | "
            f"{s['angle_viol_frac_mean']:.4f} ± {s['angle_viol_frac_std']:.4f} |"
        )

    lines.extend([
        "",
        "*Angle viol uses static joint limits; interpret cautiously for KAPT variants.",
        "",
        "## Improvement vs Kalman (delta = Kalman - method; higher is better)",
        "",
        "| Method | Delta limb std | Delta jitter | Delta accel var |",
        "| --- | --- | --- | --- |",
    ])
    for s in summary:
        label = METHOD_LABELS.get(s["method"], s["method"])
        lines.append(
            f"| {label} | "
            f"{s['limb_std_delta_mean']:.4f} ± {s['limb_std_delta_std']:.4f} | "
            f"{s['jitter_delta_mean']:.4f} ± {s['jitter_delta_std']:.4f} | "
            f"{s['accel_var_delta_mean']:.4f} ± {s['accel_var_delta_std']:.4f} |"
        )

    lines.extend([
        "",
        "## Key comparisons (Wilcoxon, paired)",
        "",
        "| Comparison | Metric | A mean | B mean | p-value | Winner | p<0.05 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])
    for w in wilcoxon_rows:
        comp = w.get("comparison", "")
        parts = comp.split(" vs ")
        if len(parts) != 2:
            continue
        a, b = parts
        a_mean = w.get(f"{a}_mean", float("nan"))
        b_mean = w.get(f"{b}_mean", float("nan"))
        lines.append(
            f"| {comp} | {w['metric']} | {a_mean:.4f} | {b_mean:.4f} | "
            f"{w.get('p_value', float('nan')):.6f} | {w.get('winner', '')} | "
            f"{'yes' if w.get('significant_p05') else 'no'} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
    ])

    h_jitter = hybrid.get("jitter_mean", float("nan"))
    k_jitter = kapt.get("jitter_mean", float("nan"))
    h_limb = hybrid.get("limb_std_mean", float("nan"))
    k_limb = kapt.get("limb_std_mean", float("nan"))
    s_limb = smooth.get("limb_std_mean", float("nan"))

    if np.isfinite(h_jitter) and np.isfinite(k_jitter) and h_jitter < k_jitter:
        lines.append(
            f"- Hybrid jitter ({h_jitter:.2f}) beats KAPT alone ({k_jitter:.2f})."
        )
    if np.isfinite(h_limb) and np.isfinite(k_limb) and np.isfinite(s_limb):
        if h_limb <= min(k_limb, s_limb):
            lines.append(
                f"- Hybrid limb std ({h_limb:.2f}) is best among all three "
                f"(KAPT {k_limb:.2f}, SmoothNet {s_limb:.2f})."
            )
        elif h_limb < s_limb:
            lines.append(
                f"- Hybrid ({h_limb:.2f}) retains KAPT's limb advantage over "
                f"SmoothNet alone ({s_limb:.2f})."
            )
        else:
            lines.append(
                f"- SmoothNet post-processing may relax KAPT limb gains "
                f"(hybrid {h_limb:.2f} vs KAPT {k_limb:.2f})."
            )

    lines.extend([
        "",
        "## Per-video detail",
        "",
        "| Video | Method | Limb std | Jitter | Accel var | Delta limb std |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    order = {m: i for i, m in enumerate(HYBRID_METHODS)}
    for r in sorted(rows, key=lambda x: (x["video"], order.get(x["method"], 99))):
        label = METHOD_LABELS.get(r["method"], r["method"])
        lines.append(
            f"| {r['video']} | {label} | "
            f"{r.get('limb_std', float('nan')):.4f} | "
            f"{r.get('jitter', float('nan')):.4f} | "
            f"{r.get('accel_var', float('nan')):.4f} | "
            f"{r.get('limb_std_delta', float('nan')):.4f} |"
        )

    return "\n".join(lines) + "\n"
