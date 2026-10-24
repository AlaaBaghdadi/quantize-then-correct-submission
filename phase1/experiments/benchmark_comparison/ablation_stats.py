"""Summary statistics and Markdown report for KAPT ablation study."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from experiments.benchmark_comparison.ablation_config import ABLATION_VARIANTS, REFERENCE_METHOD
from experiments.benchmark_comparison.fair_stats import DELTA_COLUMNS, FAIR_METRIC_COLUMNS, mean_std

ABLATION_METRIC_COLUMNS = list(FAIR_METRIC_COLUMNS)
ABLATION_DELTA_COLUMNS = list(DELTA_COLUMNS)

FUSED_BASELINE_COLUMNS = [
    "fused_angle_viol_frac",
    "fused_limb_std",
    "fused_jitter",
    "fused_accel_var",
]


def summarize_ablation(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mean ± std per ablation method."""
    method_order = [v.method_id for v in ABLATION_VARIANTS]
    methods = [m for m in method_order if any(r["method"] == m for r in rows)]
    extra = sorted({r["method"] for r in rows} - set(methods))
    methods.extend(extra)

    out: List[Dict[str, Any]] = []
    for method in methods:
        subset = [r for r in rows if r["method"] == method]
        summary: Dict[str, Any] = {"method": method, "n_videos": len(subset)}
        display = next((v.display_name for v in ABLATION_VARIANTS if v.method_id == method), method)
        summary["display_name"] = display
        for col in ABLATION_METRIC_COLUMNS + ABLATION_DELTA_COLUMNS:
            m, s = mean_std([r.get(col, float("nan")) for r in subset])
            summary[f"{col}_mean"] = m
            summary[f"{col}_std"] = s
        out.append(summary)
    return out


def wilcoxon_ablation_vs_full(
    rows: List[Dict[str, Any]],
    metric: str,
    *,
    ablation_method: str,
    higher_is_better: bool = False,
) -> Dict[str, Any]:
    """Paired Wilcoxon: kapt_full vs one ablation variant."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return {
            "metric": metric,
            "ablation_method": ablation_method,
            "p_value": float("nan"),
            "note": "scipy not installed",
        }

    by_video: Dict[str, Dict[str, float]] = {}
    for row in rows:
        by_video.setdefault(row["video"], {})[row["method"]] = row.get(metric, float("nan"))

    full_vals: List[float] = []
    abl_vals: List[float] = []
    for vid in sorted(by_video):
        m = by_video[vid]
        if REFERENCE_METHOD not in m or ablation_method not in m:
            continue
        fv, av = m[REFERENCE_METHOD], m[ablation_method]
        if np.isfinite(fv) and np.isfinite(av):
            full_vals.append(float(fv))
            abl_vals.append(float(av))

    if len(full_vals) < 2:
        return {
            "metric": metric,
            "ablation_method": ablation_method,
            "n_pairs": len(full_vals),
            "p_value": float("nan"),
            "winner": "insufficient_data",
        }

    f_mean, _ = mean_std(full_vals)
    a_mean, _ = mean_std(abl_vals)
    diffs = np.asarray(full_vals, dtype=float) - np.asarray(abl_vals, dtype=float)

    if np.allclose(diffs, 0.0, rtol=0.0, atol=0.0, equal_nan=False):
        return {
            "metric": metric,
            "ablation_method": ablation_method,
            "n_pairs": len(full_vals),
            "kapt_full_mean": f_mean,
            "ablation_mean": a_mean,
            "statistic": 0.0,
            "p_value": 1.0,
            "winner": "tie",
            "significant_p05": False,
            "higher_is_better": higher_is_better,
            "note": "all_pairs_equal",
        }

    alt = "greater" if higher_is_better else "less"
    try:
        stat, p = wilcoxon(full_vals, abl_vals, alternative=alt, zero_method="zsplit")
    except ValueError as exc:
        return {
            "metric": metric,
            "ablation_method": ablation_method,
            "n_pairs": len(full_vals),
            "kapt_full_mean": f_mean,
            "ablation_mean": a_mean,
            "statistic": float("nan"),
            "p_value": float("nan"),
            "winner": "tie",
            "significant_p05": False,
            "higher_is_better": higher_is_better,
            "note": str(exc),
        }

    if higher_is_better:
        winner = REFERENCE_METHOD if f_mean > a_mean else (ablation_method if a_mean > f_mean else "tie")
    else:
        winner = REFERENCE_METHOD if f_mean < a_mean else (ablation_method if a_mean < f_mean else "tie")

    return {
        "metric": metric,
        "ablation_method": ablation_method,
        "n_pairs": len(full_vals),
        "kapt_full_mean": f_mean,
        "ablation_mean": a_mean,
        "statistic": float(stat),
        "p_value": float(p),
        "winner": winner,
        "significant_p05": bool(p < 0.05),
        "higher_is_better": higher_is_better,
    }


def build_wilcoxon_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Wilcoxon kapt_full vs each non-reference ablation, primary metrics."""
    ablation_methods = [
        v.method_id for v in ABLATION_VARIANTS if v.method_id != REFERENCE_METHOD
    ]
    metrics = [
        ("limb_std", False),
        ("jitter", False),
        ("accel_var", False),
        ("angle_viol_frac", False),
        ("limb_std_delta", True),
        ("jitter_delta", True),
        ("accel_var_delta", True),
    ]
    out: List[Dict[str, Any]] = []
    for abl in ablation_methods:
        if not any(r["method"] == abl for r in rows):
            continue
        for metric, higher in metrics:
            out.append(wilcoxon_ablation_vs_full(rows, metric, ablation_method=abl, higher_is_better=higher))
    return out


def _summary_for(summary: List[Dict[str, Any]], method: str) -> Dict[str, Any]:
    for s in summary:
        if s.get("method") == method:
            return s
    return {}


def _best_method(summary: List[Dict[str, Any]], metric_key: str) -> str:
    """Return method id with lowest mean for metric_key."""
    best_m = ""
    best_v = float("inf")
    for s in summary:
        v = s.get(f"{metric_key}_mean", float("nan"))
        if np.isfinite(v) and v < best_v:
            best_v = v
            best_m = str(s.get("method", ""))
    return best_m


def format_ablation_report(
    rows: List[Dict[str, Any]],
    summary: List[Dict[str, Any]],
    wilcoxon_rows: List[Dict[str, Any]],
    manifest: Dict[str, Any],
) -> str:
    """Paper-ready Markdown for ablation study."""
    full = _summary_for(summary, REFERENCE_METHOD)
    no_corr = _summary_for(summary, "ablation_no_correction")
    no_kalman = _summary_for(summary, "ablation_no_kalman")
    no_cone = _summary_for(summary, "ablation_no_soft_cone")
    no_len = _summary_for(summary, "ablation_no_length_stab")

    lines = [
        "# KAPT Ablation Study",
        "",
        "Component ablation on the reviewer benchmark clips (6 videos).",
        "Baseline for delta columns: **Kalman-smoothed** keypoints (no biomechanical correction).",
        "Lower is better for all absolute metrics unless noted.",
        "",
        f"Generated: {manifest.get('timestamp', 'unknown')}",
        f"Videos: {', '.join(manifest.get('videos', []))}",
        "",
        "## Key findings",
        "",
    ]

    if full and no_corr:
        lines.append(
            f"- **Biomechanical correction** (A2 vs A0): violation fraction "
            f"{no_corr.get('angle_viol_frac_mean', float('nan')):.4f} -> "
            f"{full.get('angle_viol_frac_mean', float('nan')):.4f}; "
            f"limb std {no_corr.get('limb_std_mean', float('nan')):.1f} -> "
            f"{full.get('limb_std_mean', float('nan')):.1f}."
        )
    if full and no_kalman:
        lines.append(
            f"- **Kalman smoothing** (A1 vs A0): jitter "
            f"{no_kalman.get('jitter_mean', float('nan')):.2f} vs full "
            f"{full.get('jitter_mean', float('nan')):.2f}; "
            f"accel var {no_kalman.get('accel_var_mean', float('nan')):.1f} vs "
            f"{full.get('accel_var_mean', float('nan')):.1f}."
        )
    if full and no_cone:
        lines.append(
            f"- **Soft cone** (A3 vs A0): hard clamp jitter "
            f"{no_cone.get('jitter_mean', float('nan')):.2f} vs soft cone "
            f"{full.get('jitter_mean', float('nan')):.2f}."
        )
    if full and no_len:
        lines.append(
            f"- **Length stabilizer** (A4 vs A0): limb std "
            f"{no_len.get('limb_std_mean', float('nan')):.2f} vs full "
            f"{full.get('limb_std_mean', float('nan')):.2f}."
        )

    if not any(r["method"] == "ablation_no_kalman" for r in rows):
        lines.extend([
            "",
            "> **Note:** `w/o Kalman` (A1) was not evaluated. Re-run with "
            "`--refresh_fused` after placing source videos under `media/` "
            "so the Kalman pipeline can export `*_fused_keypoints.json`.",
        ])

    lines.extend([
        "",
        "## Summary table (mean +/- std across videos)",
        "",
        "| Variant | N | Limb std ↓ | Jitter ↓ | Accel var ↓ | Angle viol ↓ |",
        "| --- | ---: | --- | --- | --- | --- |",
    ])

    for s in summary:
        name = s.get("display_name", s.get("method", ""))
        lines.append(
            f"| {name} | {s.get('n_videos', 0)} | "
            f"{s.get('limb_std_mean', float('nan')):.4f} +/- {s.get('limb_std_std', float('nan')):.4f} | "
            f"{s.get('jitter_mean', float('nan')):.4f} +/- {s.get('jitter_std', float('nan')):.4f} | "
            f"{s.get('accel_var_mean', float('nan')):.4f} +/- {s.get('accel_var_std', float('nan')):.4f} | "
            f"{s.get('angle_viol_frac_mean', float('nan')):.4f} +/- {s.get('angle_viol_frac_std', float('nan')):.4f} |"
        )

    lines.extend([
        "",
        "## Improvement vs Kalman (delta = Kalman - method; positive = better)",
        "",
        "| Variant | Limb std d | Jitter d | Accel var d | Angle viol d |",
        "| --- | --- | --- | --- | --- |",
    ])
    for s in summary:
        name = s.get("display_name", s.get("method", ""))
        lines.append(
            f"| {name} | "
            f"{s.get('limb_std_delta_mean', float('nan')):.4f} | "
            f"{s.get('jitter_delta_mean', float('nan')):.4f} | "
            f"{s.get('accel_var_delta_mean', float('nan')):.4f} | "
            f"{s.get('angle_viol_frac_delta_mean', float('nan')):.4f} |"
        )

    lines.extend([
        "",
        "## Wilcoxon: KAPT full vs ablations",
        "",
        "| Ablation | Metric | KAPT full | Ablation | p-value | Winner | p<0.05 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])
    for w in wilcoxon_rows:
        if w.get("metric") not in ("limb_std", "jitter", "accel_var", "limb_std_delta", "jitter_delta"):
            continue
        abl = w.get("ablation_method", "")
        display = next((v.display_name for v in ABLATION_VARIANTS if v.method_id == abl), abl)
        lines.append(
            f"| {display} | {w.get('metric', '')} | "
            f"{w.get('kapt_full_mean', float('nan')):.4f} | "
            f"{w.get('ablation_mean', float('nan')):.4f} | "
            f"{w.get('p_value', float('nan')):.6f} | "
            f"{w.get('winner', '')} | "
            f"{'yes' if w.get('significant_p05') else 'no'} |"
        )

    lines.extend([
        "",
        "## Per-video detail",
        "",
        "| Video | Variant | Limb std | Jitter | Accel var | Angle viol | Limb std d |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])
    method_order = {v.method_id: i for i, v in enumerate(ABLATION_VARIANTS)}
    for r in sorted(rows, key=lambda x: (x["video"], method_order.get(x["method"], 99))):
        display = next((v.display_name for v in ABLATION_VARIANTS if v.method_id == r["method"]), r["method"])
        lines.append(
            f"| {r['video']} | {display} | "
            f"{r.get('limb_std', float('nan')):.4f} | "
            f"{r.get('jitter', float('nan')):.4f} | "
            f"{r.get('accel_var', float('nan')):.4f} | "
            f"{r.get('angle_viol_frac', float('nan')):.4f} | "
            f"{r.get('limb_std_delta', float('nan')):.4f} |"
        )

    lines.append("")
    return "\n".join(lines)
