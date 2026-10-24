"""Aggregate metrics and paired Wilcoxon tests."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from experiments.benchmark_comparison.config import METRIC_COLUMNS


def mean_std(values: List[float]) -> Tuple[float, float]:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return float("nan"), float("nan")
    arr = np.asarray(finite)
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return float(np.mean(arr)), std


def summarize_by_method(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    methods = sorted({r["method"] for r in rows})
    out: List[Dict[str, Any]] = []
    for method in methods:
        subset = [r for r in rows if r["method"] == method]
        summary: Dict[str, Any] = {"method": method, "n_videos": len(subset)}
        for col in METRIC_COLUMNS:
            vals = [r.get(col, float("nan")) for r in subset]
            m, s = mean_std(vals)
            summary[f"{col}_mean"] = m
            summary[f"{col}_std"] = s
        out.append(summary)
    return out


def wilcoxon_kapt_vs_smoothnet(
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Paired Wilcoxon: KAPT vs SmoothNet per metric (lower is better)."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return [{"metric": "all", "p_value": float("nan"), "note": "scipy not installed"}]

    by_video: Dict[str, Dict[str, Dict[str, float]]] = {}
    for row in rows:
        vid = row["video"]
        method = row["method"]
        by_video.setdefault(vid, {})[method] = row

    results: List[Dict[str, Any]] = []
    for metric in METRIC_COLUMNS:
        kapt_vals: List[float] = []
        smooth_vals: List[float] = []
        for vid, methods in sorted(by_video.items()):
            if "kapt_full" not in methods or "smoothnet" not in methods:
                continue
            kv = methods["kapt_full"].get(metric, float("nan"))
            sv = methods["smoothnet"].get(metric, float("nan"))
            if np.isfinite(kv) and np.isfinite(sv):
                kapt_vals.append(float(kv))
                smooth_vals.append(float(sv))

        if len(kapt_vals) < 2:
            results.append({
                "comparison": "kapt_full vs smoothnet",
                "metric": metric,
                "n_pairs": len(kapt_vals),
                "p_value": float("nan"),
                "winner": "insufficient_data",
            })
            continue

        stat, p = wilcoxon(kapt_vals, smooth_vals, alternative="less")
        k_mean, _ = mean_std(kapt_vals)
        s_mean, _ = mean_std(smooth_vals)
        winner = "kapt_full" if k_mean < s_mean else ("smoothnet" if s_mean < k_mean else "tie")
        results.append({
            "comparison": "kapt_full vs smoothnet",
            "metric": metric,
            "n_pairs": len(kapt_vals),
            "kapt_mean": k_mean,
            "smoothnet_mean": s_mean,
            "statistic": float(stat),
            "p_value": float(p),
            "winner": winner,
            "significant_p05": bool(p < 0.05),
        })
    return results


def format_summary_markdown(
    per_video_rows: List[Dict[str, Any]],
    method_summary: List[Dict[str, Any]],
    wilcoxon_rows: List[Dict[str, Any]],
) -> str:
    lines = [
        "# KAPT vs SmoothNet — Benchmark Results",
        "",
        "Baseline for all methods: Kalman-smoothed keypoints (`*_smoothed_keypoints.json`).",
        "Lower is better for all metrics.",
        "",
        "## Per-method mean ± std",
        "",
        "| Method | N | Accel var | Viol frac | Limb std | Jitter |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for s in method_summary:
        lines.append(
            f"| {s['method']} | {s['n_videos']} | "
            f"{s['accel_var_mean']:.4f} ± {s['accel_var_std']:.4f} | "
            f"{s['viol_frac_mean']:.4f} ± {s['viol_frac_std']:.4f} | "
            f"{s['limb_std_mean']:.4f} ± {s['limb_std_std']:.4f} | "
            f"{s['jitter_mean']:.4f} ± {s['jitter_std']:.4f} |"
        )

    lines.extend([
        "",
        "## Paired Wilcoxon (KAPT vs SmoothNet, alternative=less)",
        "",
        "| Metric | KAPT mean | SmoothNet mean | p-value | Winner | p<0.05 |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    for w in wilcoxon_rows:
        if w.get("metric") == "all":
            lines.append(f"| (error) | — | — | — | — | {w.get('note', '')} |")
            continue
        lines.append(
            f"| {w['metric']} | {w.get('kapt_mean', float('nan')):.4f} | "
            f"{w.get('smoothnet_mean', float('nan')):.4f} | "
            f"{w.get('p_value', float('nan')):.6f} | {w.get('winner', '')} | "
            f"{'yes' if w.get('significant_p05') else 'no'} |"
        )

    lines.extend([
        "",
        "## Per-video detail",
        "",
        "| Video | Method | Accel var | Viol frac | Limb std | Jitter |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    for r in sorted(per_video_rows, key=lambda x: (x["video"], x["method"])):
        lines.append(
            f"| {r['video']} | {r['method']} | "
            f"{r.get('accel_var', float('nan')):.4f} | "
            f"{r.get('viol_frac', float('nan')):.4f} | "
            f"{r.get('limb_std', float('nan')):.4f} | "
            f"{r.get('jitter', float('nan')):.4f} |"
        )

    return "\n".join(lines) + "\n"
