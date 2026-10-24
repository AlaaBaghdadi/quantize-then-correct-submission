"""
Export benchmark CSV data to LaTeX table fragments for report/tables/.

Usage:
  python -m experiments.benchmark_comparison.export_latex_tables
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FAIR_CSV = PROJECT_ROOT / "experiments/reviewer_r5_r6/tables/benchmark_fair.csv"
DEFAULT_WILCOXON_CSV = PROJECT_ROOT / "experiments/reviewer_r5_r6/tables/kapt_power_wilcoxon.csv"
DEFAULT_ABLATION_SUMMARY_CSV = PROJECT_ROOT / "experiments/reviewer_r5_r6/tables/ablation_summary.csv"
DEFAULT_ABLATION_WILCOXON_CSV = PROJECT_ROOT / "experiments/reviewer_r5_r6/tables/ablation_wilcoxon.csv"
DEFAULT_ABLATION_PER_VIDEO_CSV = PROJECT_ROOT / "experiments/reviewer_r5_r6/tables/ablation_per_video.csv"
DEFAULT_SENSITIVITY_SUMMARY_CSV = (
    PROJECT_ROOT / "experiments/reviewer_r5_r6/tables/sensitivity_summary.csv"
)
DEFAULT_HYBRID_SUMMARY_CSV = PROJECT_ROOT / "experiments/reviewer_r5_r6/tables/hybrid_summary.csv"
DEFAULT_HYBRID_WILCOXON_CSV = PROJECT_ROOT / "experiments/reviewer_r5_r6/tables/hybrid_wilcoxon.csv"
DEFAULT_HYBRID_BENCHMARK_CSV = PROJECT_ROOT / "experiments/reviewer_r5_r6/tables/hybrid_benchmark.csv"
DEFAULT_SENSITIVITY_FIGURES_DIR = PROJECT_ROOT / "experiments/reviewer_r5_r6/figures"
OUTPUT_DIR = PROJECT_ROOT / "report/tables"
REPORT_FIGURES_DIR = PROJECT_ROOT / "report/figures"

ABLATION_METHOD_ORDER = [
    ("kapt_full", "KAPT (full)"),
    ("ablation_no_kalman", "w/o Kalman"),
    ("ablation_no_correction", "w/o correction"),
    ("ablation_no_soft_cone", "w/o soft cone"),
    ("ablation_no_length_stab", "w/o length stab."),
]

ABLATION_SIGNIFICANCE_LABELS = {
    "ablation_no_kalman": "w/o Kalman",
    "ablation_no_correction": "w/o correction",
    "ablation_no_soft_cone": "w/o soft cone",
    "ablation_no_length_stab": "w/o length stab.",
}

SENSITIVITY_PARAM_ORDER = [
    ("alpha", "EMA $\\alpha$"),
    ("max_drift", "\\texttt{max\\_drift}"),
    ("process_noise", "$Q$ (process noise)"),
]

SENSITIVITY_FIGURE_NAMES = [
    "sensitivity_all.pdf",
    "sensitivity_alpha.pdf",
    "sensitivity_max_drift.pdf",
    "sensitivity_process_noise.pdf",
]

HYBRID_METHOD_ORDER = [
    ("smoothnet", "SmoothNet (Kalman only)"),
    ("kapt_full", "KAPT (full)"),
    ("kapt_then_smoothnet", "KAPT $\\rightarrow$ SmoothNet"),
]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _f(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(v: float, digits: int = 2) -> str:
    if not np.isfinite(v):
        return "---"
    return f"{v:.{digits}f}"


def _mean_std(values: List[float]) -> Tuple[float, float]:
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return float("nan"), float("nan")
    arr = np.asarray(finite)
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return float(np.mean(arr)), std


def _method_label(method: str) -> str:
    return "KAPT" if method == "kapt_full" else "SmoothNet"


def export_summary(fair_rows: List[Dict[str, str]], wilcoxon_rows: List[Dict[str, str]]) -> str:
    by_method: Dict[str, List[Dict[str, str]]] = {}
    for row in fair_rows:
        by_method.setdefault(row["method"], []).append(row)

    wilcoxon = {r["metric"]: r for r in wilcoxon_rows}
    limb_p = _f(wilcoxon.get("limb_std", {}).get("p_value", float("nan")))

    lines = [
        "% Auto-generated from benchmark_fair.csv — do not edit by hand.",
        "\\begin{table}[H]",
        "  \\centering",
        "  \\caption{Summary comparison: KAPT vs.\\ SmoothNet on Kalman-smoothed keypoint tracks ($n=6$ videos). Lower is better. $\\Delta$ denotes improvement over the Kalman baseline (Kalman $-$ method).}",
        "  \\label{tab:benchmark_summary}",
        "  \\begin{tabular}{@{}lccc@{}}",
        "    \\toprule",
        "    Method & Limb std & $\\Delta$ limb std & Wilcoxon $p$ \\\\",
        "    \\midrule",
    ]

    for method_key, label in [("kapt_full", "KAPT"), ("smoothnet", "SmoothNet")]:
        subset = by_method.get(method_key, [])
        limb_vals = [_f(r["limb_std"]) for r in subset]
        delta_vals = [_f(r["limb_std_delta"]) for r in subset]
        limb_m, limb_s = _mean_std(limb_vals)
        delta_m, _ = _mean_std(delta_vals)
        p_cell = _fmt(limb_p, 3) if method_key == "kapt_full" else "---"
        lines.append(
            f"    {label} & {_fmt(limb_m)} $\\pm$ {_fmt(limb_s)} "
            f"& {_fmt(delta_m)} & {p_cell} \\\\"
        )

    lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)


def export_temporal(fair_rows: List[Dict[str, str]], wilcoxon_rows: List[Dict[str, str]]) -> str:
    by_method: Dict[str, List[Dict[str, str]]] = {}
    for row in fair_rows:
        by_method.setdefault(row["method"], []).append(row)

    lines = [
        "% Auto-generated from benchmark_fair.csv — do not edit by hand.",
        "\\begin{table}[H]",
        "  \\centering",
        "  \\caption{Temporal smoothness metrics (lower is better). SmoothNet, as a dedicated temporal refiner, attains lower jitter and acceleration variance.}",
        "  \\label{tab:benchmark_temporal}",
        "  \\begin{tabular}{@{}lcc@{}}",
        "    \\toprule",
        "    Method & Mean jitter & Accel.\\ variance \\\\",
        "    \\midrule",
    ]

    for method_key, label in [("kapt_full", "KAPT"), ("smoothnet", "SmoothNet")]:
        subset = by_method.get(method_key, [])
        jitter_m, jitter_s = _mean_std([_f(r["jitter"]) for r in subset])
        accel_m, accel_s = _mean_std([_f(r["accel_var"]) for r in subset])
        lines.append(
            f"    {label} & {_fmt(jitter_m)} $\\pm$ {_fmt(jitter_s)} "
            f"& {_fmt(accel_m)} $\\pm$ {_fmt(accel_s)} \\\\"
        )

    lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)


def export_per_video(fair_rows: List[Dict[str, str]]) -> str:
    videos = sorted({r["video"] for r in fair_rows})
    lines = [
        "% Auto-generated from benchmark_fair.csv — do not edit by hand.",
        "\\begin{table}[H]",
        "  \\centering",
        "  \\caption{Per-video limb-length consistency (mean limb length std; lower is better). Bold indicates the better method per video.}",
        "  \\label{tab:benchmark_per_video}",
        "  \\small",
        "  \\begin{tabular}{@{}lcccc@{}}",
        "    \\toprule",
        "    Video & Kalman & KAPT & SmoothNet & $\\Delta$ (KAPT) \\\\",
        "    \\midrule",
    ]

    for video in videos:
        rows = {r["method"]: r for r in fair_rows if r["video"] == video}
        kalman = _f(rows.get("kapt_full", rows.get("smoothnet", {})).get("kalman_limb_std", float("nan")))
        kapt = _f(rows.get("kapt_full", {}).get("limb_std", float("nan")))
        smooth = _f(rows.get("smoothnet", {}).get("limb_std", float("nan")))
        delta = _f(rows.get("kapt_full", {}).get("limb_std_delta", float("nan")))

        if np.isfinite(kapt) and np.isfinite(smooth):
            if kapt < smooth:
                kapt_cell = f"\\textbf{{{_fmt(kapt)}}}"
                smooth_cell = _fmt(smooth)
            elif smooth < kapt:
                kapt_cell = _fmt(kapt)
                smooth_cell = f"\\textbf{{{_fmt(smooth)}}}"
            else:
                kapt_cell = _fmt(kapt)
                smooth_cell = _fmt(smooth)
        else:
            kapt_cell = _fmt(kapt)
            smooth_cell = _fmt(smooth)

        vid = video.replace("_", "\\_")
        lines.append(
            f"    {vid} & {_fmt(kalman)} & {kapt_cell} & {smooth_cell} & {_fmt(delta)} \\\\"
        )

    lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)


def export_context_table() -> str:
    return r"""% Contextual reference — published 3D text-to-motion methods (not directly comparable).
\begin{table}[H]
  \centering
  \caption{Representative physics-aware 3D text-to-motion methods (context only; not evaluated on our 2D video benchmark).}
  \label{tab:benchmark_context}
  \small
  \begin{tabular}{@{}lll@{}}
    \toprule
    Method & Setting & Reported focus \\
    \midrule
    PhysDiff~\cite{yuan2023physdiff} & HumanML3D & Physics-guided 3D motion diffusion \\
    ReinDiffuse~\cite{han2025reindiffuse} & HumanML3D & RL + diffusion for physical plausibility \\
    Morph~\cite{li2025morph} & HumanML3D & Motion-free physics optimization \\
    SmoothNet~\cite{zeng2022smoothnet} & 2D/3D pose tracks & Post-hoc temporal pose refinement \\
    \bottomrule
  \end{tabular}
\end{table}
"""


def _ablation_summary_row(summary_rows: List[Dict[str, str]], method_id: str) -> Dict[str, str]:
    for row in summary_rows:
        if row.get("method") == method_id:
            return row
    return {}


def _bold_cell(value: float, best: float, *, lower_is_better: bool = True) -> str:
    if not np.isfinite(value) or not np.isfinite(best):
        return _fmt(value)
    if abs(value - best) < 1e-9:
        return f"\\textbf{{{_fmt(value)}}}"
    return _fmt(value)


def export_ablation_summary(summary_rows: List[Dict[str, str]]) -> str:
    rows_out: List[Dict[str, float]] = []
    for method_id, _ in ABLATION_METHOD_ORDER:
        row = _ablation_summary_row(summary_rows, method_id)
        rows_out.append({
            "limb_std": _f(row.get("limb_std_mean")),
            "limb_std_std": _f(row.get("limb_std_std")),
            "jitter": _f(row.get("jitter_mean")),
            "jitter_std": _f(row.get("jitter_std")),
            "accel_var": _f(row.get("accel_var_mean")),
            "accel_std": _f(row.get("accel_var_std")),
            "delta_limb": _f(row.get("limb_std_delta_mean")),
        })

    best_limb = min(r["limb_std"] for r in rows_out)
    best_jitter = min(r["jitter"] for r in rows_out)
    best_accel = min(r["accel_var"] for r in rows_out)
    best_delta = max(r["delta_limb"] for r in rows_out)

    lines = [
        "% Auto-generated from ablation_summary.csv — do not edit by hand.",
        "\\begin{table}[H]",
        "  \\centering",
        "  \\caption{Ablation study: mean $\\pm$ std across six videos ($n=6$). Lower is better for limb std, jitter, and acceleration variance. $\\Delta$ limb std is improvement over the Kalman baseline (Kalman $-$ method; higher is better). Bold indicates best per column.}",
        "  \\label{tab:ablation_summary}",
        "  \\small",
        "  \\begin{tabular}{@{}lcccc@{}}",
        "    \\toprule",
        "    Variant & Limb std & Jitter & Accel.\\ var. & $\\Delta$ limb std \\\\",
        "    \\midrule",
    ]

    for (method_id, label), metrics in zip(ABLATION_METHOD_ORDER, rows_out):
        lines.append(
            f"    {label} & "
            f"{_bold_cell(metrics['limb_std'], best_limb)} $\\pm$ {_fmt(metrics['limb_std_std'])} & "
            f"{_bold_cell(metrics['jitter'], best_jitter)} $\\pm$ {_fmt(metrics['jitter_std'])} & "
            f"{_bold_cell(metrics['accel_var'], best_accel)} $\\pm$ {_fmt(metrics['accel_std'])} & "
            f"{_bold_cell(metrics['delta_limb'], best_delta, lower_is_better=False)} \\\\"
        )

    lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)


def export_ablation_significance(wilcoxon_rows: List[Dict[str, str]]) -> str:
    by_ablation: Dict[str, Dict[str, float]] = {}
    for row in wilcoxon_rows:
        abl = row.get("ablation_method", "")
        metric = row.get("metric", "")
        if abl not in ABLATION_SIGNIFICANCE_LABELS:
            continue
        if metric not in ("limb_std", "jitter", "accel_var", "limb_std_delta"):
            continue
        by_ablation.setdefault(abl, {})[metric] = _f(row.get("p_value"))

    lines = [
        "% Auto-generated from ablation_wilcoxon.csv — do not edit by hand.",
        "\\begin{table}[H]",
        "  \\centering",
        "  \\caption{Paired Wilcoxon signed-rank tests: KAPT (full) vs.\\ each ablation variant ($n=6$). $p$-values for primary metrics (lower absolute metric is better; higher $\\Delta$ limb std is better).}",
        "  \\label{tab:ablation_significance}",
        "  \\small",
        "  \\begin{tabular}{@{}lcccc@{}}",
        "    \\toprule",
        "    Ablation & Limb std $p$ & Jitter $p$ & Accel.\\ var.\\ $p$ & $\\Delta$ limb std $p$ \\\\",
        "    \\midrule",
    ]

    for method_id, label in ABLATION_SIGNIFICANCE_LABELS.items():
        p = by_ablation.get(method_id, {})
        lines.append(
            f"    {label} & {_fmt(p.get('limb_std', float('nan')), 3)} "
            f"& {_fmt(p.get('jitter', float('nan')), 3)} "
            f"& {_fmt(p.get('accel_var', float('nan')), 3)} "
            f"& {_fmt(p.get('limb_std_delta', float('nan')), 3)} \\\\"
        )

    lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)


def export_ablation_per_video(per_video_rows: List[Dict[str, str]]) -> str:
    videos = sorted({r["video"] for r in per_video_rows})
    method_ids = [m for m, _ in ABLATION_METHOD_ORDER]

    lines = [
        "% Auto-generated from ablation_per_video.csv — do not edit by hand.",
        "\\begin{table}[H]",
        "  \\centering",
        "  \\caption{Per-video mean limb-length standard deviation (lower is better) for all ablation variants.}",
        "  \\label{tab:ablation_per_video}",
        "  \\scriptsize",
        "  \\begin{tabular}{@{}lccccc@{}}",
        "    \\toprule",
        "    Video & Full & w/o Kalman & w/o corr. & w/o cone & w/o len.\\\\",
        "    \\midrule",
    ]

    for video in videos:
        by_method = {r["method"]: r for r in per_video_rows if r["video"] == video}
        cells = []
        values = []
        for mid in method_ids:
            v = _f(by_method.get(mid, {}).get("limb_std", float("nan")))
            values.append(v)
            cells.append(_fmt(v))
        best = min(values) if values else float("nan")
        cells = [
            f"\\textbf{{{c}}}" if np.isfinite(values[i]) and abs(values[i] - best) < 1e-9 else c
            for i, c in enumerate(cells)
        ]
        vid = video.replace("_", "\\_")
        lines.append(f"    {vid} & {' & '.join(cells)} \\\\")

    lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)


def _format_sensitivity_value(param: str, value: float) -> str:
    if param == "process_noise":
        if abs(value - 1e-3) < 1e-15:
            return "$10^{-3}$"
        if abs(value - 1e-2) < 1e-15:
            return "$10^{-2}$"
        if abs(value - 1e-1) < 1e-15:
            return "$10^{-1}$"
    if param == "alpha":
        return f"{value:g}"
    return _fmt(value, 2)


def export_sensitivity_summary(summary_rows: List[Dict[str, str]]) -> str:
    lines = [
        "% Auto-generated from sensitivity_summary.csv — do not edit by hand.",
        "\\begin{table}[H]",
        "  \\centering",
        "  \\caption{Hyperparameter sensitivity: mean $\\pm$ std across six videos ($n=6$). One parameter varied at a time; others held at defaults. Lower is better. Bold indicates best per parameter group. Default setting marked with $\\checkmark$.}",
        "  \\label{tab:sensitivity_summary}",
        "  \\small",
        "  \\begin{tabular}{@{}llccc@{}}",
        "    \\toprule",
        "    Parameter & Value & Jitter & Limb std & Accel.\\ var. \\\\",
        "    \\midrule",
    ]

    for param, param_label in SENSITIVITY_PARAM_ORDER:
        subset = sorted(
            [r for r in summary_rows if r.get("parameter") == param],
            key=lambda r: _f(r.get("value")),
        )
        if not subset:
            continue

        metrics = []
        for row in subset:
            metrics.append({
                "value": _f(row.get("value")),
                "is_default": str(row.get("is_default", "")).lower() in ("true", "1", "yes"),
                "jitter": _f(row.get("jitter_mean")),
                "jitter_std": _f(row.get("jitter_std")),
                "limb_std": _f(row.get("limb_std_mean")),
                "limb_std_std": _f(row.get("limb_std_std")),
                "accel_var": _f(row.get("accel_var_mean")),
                "accel_std": _f(row.get("accel_var_std")),
            })

        best_jitter = min(m["jitter"] for m in metrics)
        best_limb = min(m["limb_std"] for m in metrics)
        best_accel = min(m["accel_var"] for m in metrics)

        for i, m in enumerate(metrics):
            param_cell = param_label if i == 0 else ""
            default_mark = " $\\checkmark$" if m["is_default"] else ""
            value_cell = _format_sensitivity_value(param, m["value"]) + default_mark
            lines.append(
                f"    {param_cell} & {value_cell} & "
                f"{_bold_cell(m['jitter'], best_jitter)} $\\pm$ {_fmt(m['jitter_std'])} & "
                f"{_bold_cell(m['limb_std'], best_limb)} $\\pm$ {_fmt(m['limb_std_std'])} & "
                f"{_bold_cell(m['accel_var'], best_accel)} $\\pm$ {_fmt(m['accel_std'])} \\\\"
            )
        lines.append("    \\addlinespace")

    if lines[-1] == "    \\addlinespace":
        lines.pop()

    lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)


def _hybrid_summary_row(summary_rows: List[Dict[str, str]], method_id: str) -> Dict[str, str]:
    for row in summary_rows:
        if row.get("method") == method_id:
            return row
    return {}


def export_hybrid_summary(summary_rows: List[Dict[str, str]]) -> str:
    metrics = []
    for method_id, _ in HYBRID_METHOD_ORDER:
        row = _hybrid_summary_row(summary_rows, method_id)
        if not row:
            continue
        metrics.append({
            "limb_std": _f(row.get("limb_std_mean")),
            "limb_std_std": _f(row.get("limb_std_std")),
            "jitter": _f(row.get("jitter_mean")),
            "jitter_std": _f(row.get("jitter_std")),
            "accel_var": _f(row.get("accel_var_mean")),
            "accel_std": _f(row.get("accel_var_std")),
            "limb_delta": _f(row.get("limb_std_delta_mean")),
        })

    best_limb = min(m["limb_std"] for m in metrics)
    best_jitter = min(m["jitter"] for m in metrics)
    best_accel = min(m["accel_var"] for m in metrics)
    best_delta = max(m["limb_delta"] for m in metrics)

    lines = [
        "% Auto-generated from hybrid_summary.csv — do not edit by hand.",
        "\\begin{table}[H]",
        "  \\centering",
        "  \\caption{Hybrid refinement: mean $\\pm$ std across six videos ($n=6$). "
        "Lower is better for limb std, jitter, and acceleration variance. "
        "$\\Delta$ limb std is improvement over the Kalman baseline (Kalman $-$ method; higher is better). "
        "Bold indicates best per column.}",
        "  \\label{tab:hybrid_summary}",
        "  \\begin{tabular}{@{}lcccc@{}}",
        "    \\toprule",
        "    Method & Limb std & Jitter & Accel.\\ var. & $\\Delta$ limb std \\\\",
        "    \\midrule",
    ]

    for (_, label), m in zip(HYBRID_METHOD_ORDER, metrics):
        lines.append(
            f"    {label} & "
            f"{_bold_cell(m['limb_std'], best_limb)} $\\pm$ {_fmt(m['limb_std_std'])} & "
            f"{_bold_cell(m['jitter'], best_jitter)} $\\pm$ {_fmt(m['jitter_std'])} & "
            f"{_bold_cell(m['accel_var'], best_accel)} $\\pm$ {_fmt(m['accel_std'])} & "
            f"{_bold_cell(m['limb_delta'], best_delta, lower_is_better=False)} \\\\"
        )

    lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)


def export_hybrid_significance(wilcoxon_rows: List[Dict[str, str]]) -> str:
    wanted = [
        ("kapt_then_smoothnet vs kapt_full", "limb_std"),
        ("kapt_then_smoothnet vs smoothnet", "limb_std"),
        ("kapt_then_smoothnet vs kapt_full", "jitter"),
        ("kapt_then_smoothnet vs smoothnet", "jitter"),
        ("kapt_then_smoothnet vs kapt_full", "accel_var"),
        ("kapt_then_smoothnet vs smoothnet", "accel_var"),
    ]
    lookup = {(r.get("comparison"), r.get("metric")): r for r in wilcoxon_rows}

    lines = [
        "% Auto-generated from hybrid_wilcoxon.csv — do not edit by hand.",
        "\\begin{table}[H]",
        "  \\centering",
        "  \\caption{Paired Wilcoxon signed-rank tests: KAPT $\\rightarrow$ SmoothNet vs.\\ each baseline ($n=6$). Lower absolute metric is better.}",
        "  \\label{tab:hybrid_significance}",
        "  \\small",
        "  \\begin{tabular}{@{}llccc@{}}",
        "    \\toprule",
        "    Comparison & Metric & Hybrid mean & Baseline mean & $p$-value \\\\",
        "    \\midrule",
    ]

    for comp, metric in wanted:
        row = lookup.get((comp, metric))
        if not row:
            continue
        parts = comp.split(" vs ")
        hybrid_key = parts[0]
        baseline_key = parts[1]
        hybrid_mean = _f(row.get(f"{hybrid_key}_mean"))
        baseline_mean = _f(row.get(f"{baseline_key}_mean"))
        p_val = _f(row.get("p_value"))
        metric_label = {
            "limb_std": "Limb std",
            "jitter": "Jitter",
            "accel_var": "Accel.\\ var.",
        }.get(metric, metric)
        baseline_label = "KAPT" if baseline_key == "kapt_full" else "SmoothNet"
        lines.append(
            f"    vs.\\ {baseline_label} & {metric_label} & "
            f"{_fmt(hybrid_mean)} & {_fmt(baseline_mean)} & {_fmt(p_val, 3)} \\\\"
        )

    lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)


def export_hybrid_per_video(benchmark_rows: List[Dict[str, str]]) -> str:
    videos = sorted({r["video"] for r in benchmark_rows})
    lines = [
        "% Auto-generated from hybrid_benchmark.csv — do not edit by hand.",
        "\\begin{table}[H]",
        "  \\centering",
        "  \\caption{Per-video limb-length consistency (mean limb length std; lower is better). Bold indicates best method per video.}",
        "  \\label{tab:hybrid_per_video}",
        "  \\small",
        "  \\begin{tabular}{@{}lcccc@{}}",
        "    \\toprule",
        "    Video & Kalman & SmoothNet & KAPT & KAPT $\\rightarrow$ SmoothNet \\\\",
        "    \\midrule",
    ]

    for video in videos:
        rows = {r["method"]: r for r in benchmark_rows if r["video"] == video}
        kalman = _f(rows.get("kapt_full", rows.get("smoothnet", {})).get("kalman_limb_std", float("nan")))
        smooth = _f(rows.get("smoothnet", {}).get("limb_std", float("nan")))
        kapt = _f(rows.get("kapt_full", {}).get("limb_std", float("nan")))
        hybrid = _f(rows.get("kapt_then_smoothnet", {}).get("limb_std", float("nan")))

        vals = {"smoothnet": smooth, "kapt_full": kapt, "kapt_then_smoothnet": hybrid}
        finite = {k: v for k, v in vals.items() if np.isfinite(v)}
        best = min(finite.values()) if finite else float("nan")

        def cell(val: float) -> str:
            if np.isfinite(val) and np.isfinite(best) and abs(val - best) < 1e-9:
                return f"\\textbf{{{_fmt(val)}}}"
            return _fmt(val)

        vid = video.replace("_", "\\_")
        lines.append(
            f"    {vid} & {_fmt(kalman)} & {cell(smooth)} & "
            f"{cell(kapt)} & {cell(hybrid)} \\\\"
        )

    lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)


def copy_sensitivity_figures() -> List[Path]:
    """Copy sensitivity PDFs into report/figures/ for LaTeX includegraphics."""
    REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    copied: List[Path] = []
    for name in SENSITIVITY_FIGURE_NAMES:
        src = DEFAULT_SENSITIVITY_FIGURES_DIR / name
        if not src.is_file():
            continue
        dst = REPORT_FIGURES_DIR / name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def main() -> int:
    fair_csv = DEFAULT_FAIR_CSV
    wilcoxon_csv = DEFAULT_WILCOXON_CSV

    if not fair_csv.is_file():
        print(f"Missing {fair_csv}", file=sys.stderr)
        return 1

    fair_rows = _read_csv(fair_csv)
    wilcoxon_rows = _read_csv(wilcoxon_csv) if wilcoxon_csv.is_file() else []

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUTPUT_DIR / "benchmark_summary.tex").write_text(
        export_summary(fair_rows, wilcoxon_rows), encoding="utf-8"
    )
    (OUTPUT_DIR / "benchmark_temporal.tex").write_text(
        export_temporal(fair_rows, wilcoxon_rows), encoding="utf-8"
    )
    (OUTPUT_DIR / "benchmark_per_video.tex").write_text(
        export_per_video(fair_rows), encoding="utf-8"
    )
    (OUTPUT_DIR / "benchmark_context.tex").write_text(
        export_context_table(), encoding="utf-8"
    )

    ablation_summary_csv = DEFAULT_ABLATION_SUMMARY_CSV
    ablation_wilcoxon_csv = DEFAULT_ABLATION_WILCOXON_CSV
    ablation_per_video_csv = DEFAULT_ABLATION_PER_VIDEO_CSV

    if ablation_summary_csv.is_file():
        ablation_summary = _read_csv(ablation_summary_csv)
        (OUTPUT_DIR / "ablation_summary.tex").write_text(
            export_ablation_summary(ablation_summary), encoding="utf-8"
        )
    else:
        print(f"[warn] Missing {ablation_summary_csv}; skipping ablation tables", file=sys.stderr)

    if ablation_wilcoxon_csv.is_file() and ablation_summary_csv.is_file():
        ablation_wilcoxon = _read_csv(ablation_wilcoxon_csv)
        (OUTPUT_DIR / "ablation_significance.tex").write_text(
            export_ablation_significance(ablation_wilcoxon), encoding="utf-8"
        )

    if ablation_per_video_csv.is_file():
        ablation_per_video = _read_csv(ablation_per_video_csv)
        (OUTPUT_DIR / "ablation_per_video.tex").write_text(
            export_ablation_per_video(ablation_per_video), encoding="utf-8"
        )

    sensitivity_summary_csv = DEFAULT_SENSITIVITY_SUMMARY_CSV
    if sensitivity_summary_csv.is_file():
        sensitivity_summary = _read_csv(sensitivity_summary_csv)
        (OUTPUT_DIR / "sensitivity_summary.tex").write_text(
            export_sensitivity_summary(sensitivity_summary), encoding="utf-8"
        )
        copied = copy_sensitivity_figures()
        for path in copied:
            print(f"[done] Copied figure {path.name} -> {path}")
    else:
        print(
            f"[warn] Missing {sensitivity_summary_csv}; skipping sensitivity tables",
            file=sys.stderr,
        )

    hybrid_summary_csv = DEFAULT_HYBRID_SUMMARY_CSV
    hybrid_wilcoxon_csv = DEFAULT_HYBRID_WILCOXON_CSV
    hybrid_benchmark_csv = DEFAULT_HYBRID_BENCHMARK_CSV

    if hybrid_summary_csv.is_file():
        hybrid_summary = _read_csv(hybrid_summary_csv)
        (OUTPUT_DIR / "hybrid_summary.tex").write_text(
            export_hybrid_summary(hybrid_summary), encoding="utf-8"
        )
    else:
        print(f"[warn] Missing {hybrid_summary_csv}; skipping hybrid tables", file=sys.stderr)

    if hybrid_wilcoxon_csv.is_file():
        hybrid_wilcoxon = _read_csv(hybrid_wilcoxon_csv)
        (OUTPUT_DIR / "hybrid_significance.tex").write_text(
            export_hybrid_significance(hybrid_wilcoxon), encoding="utf-8"
        )

    if hybrid_benchmark_csv.is_file():
        hybrid_benchmark = _read_csv(hybrid_benchmark_csv)
        (OUTPUT_DIR / "hybrid_per_video.tex").write_text(
            export_hybrid_per_video(hybrid_benchmark), encoding="utf-8"
        )

    print(f"[done] Wrote LaTeX tables to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
