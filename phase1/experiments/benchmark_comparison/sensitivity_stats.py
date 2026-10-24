"""Aggregate sensitivity CSVs, build report, and plot figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from experiments.benchmark_comparison.sensitivity_config import (
    PRIMARY_METRICS,
    SENSITIVITY_SWEEPS,
    SensitivityParam,
)


def _finite(v: Any) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0


def summarize_sensitivity(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mean +/- std per (parameter, value) across videos."""
    summary: List[Dict[str, Any]] = []
    for sweep in SENSITIVITY_SWEEPS:
        for value in sweep.values:
            subset = [
                r for r in rows
                if r.get("parameter") == sweep.param and _finite(r.get("value")) == value
            ]
            if not subset:
                continue
            entry: Dict[str, Any] = {
                "parameter": sweep.param,
                "display_name": sweep.display_name,
                "value": value,
                "is_default": value == sweep.default,
                "n_videos": len(subset),
            }
            for metric in PRIMARY_METRICS:
                vals = [_finite(r.get(metric)) for r in subset]
                m, s = mean_std(vals)
                entry[f"{metric}_mean"] = m
                entry[f"{metric}_std"] = s
            for metric in PRIMARY_METRICS:
                vals = [_finite(r.get(f"{metric}_delta")) for r in subset]
                m, s = mean_std(vals)
                entry[f"{metric}_delta_mean"] = m
                entry[f"{metric}_delta_std"] = s
            summary.append(entry)
    return summary


def plot_sensitivity_figures(
    summary: List[Dict[str, Any]],
    figures_dir: Path,
) -> List[Path]:
    """One PDF per parameter with primary + secondary metrics."""
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    for sweep in SENSITIVITY_SWEEPS:
        subset = sorted(
            [r for r in summary if r.get("parameter") == sweep.param],
            key=lambda r: float(r["value"]),
        )
        if not subset:
            continue

        x = [float(r["value"]) for r in subset]
        primary = sweep.primary_metric
        secondary = "angle_viol_frac" if primary != "angle_viol_frac" else "limb_std"

        fig, ax1 = plt.subplots(figsize=(6, 4))
        y1 = [float(r.get(f"{primary}_mean", float("nan"))) for r in subset]
        e1 = [float(r.get(f"{primary}_std", 0.0)) for r in subset]
        ax1.errorbar(x, y1, yerr=e1, marker="o", capsize=4, color="#1f77b4", label=primary)
        ax1.set_xlabel(sweep.display_name)
        ax1.set_ylabel(primary.replace("_", " "), color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        if sweep.param == "process_noise":
            ax1.set_xscale("log")

        ax2 = ax1.twinx()
        y2 = [float(r.get(f"{secondary}_mean", float("nan"))) for r in subset]
        e2 = [float(r.get(f"{secondary}_std", 0.0)) for r in subset]
        ax2.errorbar(
            x, y2, yerr=e2, marker="s", capsize=4, color="#ff7f0e",
            linestyle="--", label=secondary,
        )
        ax2.set_ylabel(secondary.replace("_", " "), color="#ff7f0e")
        ax2.tick_params(axis="y", labelcolor="#ff7f0e")

        default_x = sweep.default
        ax1.axvline(default_x, color="gray", linestyle=":", linewidth=1.5, label="default")

        ax1.set_title(f"Sensitivity: {sweep.display_name}")
        fig.tight_layout()
        out_path = figures_dir / f"sensitivity_{sweep.param}.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)

    combined = figures_dir / "sensitivity_all.pdf"
    if len(written) == len(SENSITIVITY_SWEEPS):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, sweep in zip(axes, SENSITIVITY_SWEEPS):
            subset = sorted(
                [r for r in summary if r.get("parameter") == sweep.param],
                key=lambda r: float(r["value"]),
            )
            if not subset:
                continue
            x = [float(r["value"]) for r in subset]
            primary = sweep.primary_metric
            y = [float(r.get(f"{primary}_mean", float("nan"))) for r in subset]
            e = [float(r.get(f"{primary}_std", 0.0)) for r in subset]
            ax.errorbar(x, y, yerr=e, marker="o", capsize=3)
            ax.axvline(sweep.default, color="gray", linestyle=":", linewidth=1.2)
            ax.set_xlabel(sweep.display_name)
            ax.set_ylabel(primary.replace("_", " "))
            ax.set_title(sweep.display_name)
            if sweep.param == "process_noise":
                ax.set_xscale("log")
        fig.tight_layout()
        fig.savefig(combined, bbox_inches="tight")
        plt.close(fig)
        written.append(combined)

    return written


def format_sensitivity_report(
    rows: List[Dict[str, Any]],
    summary: List[Dict[str, Any]],
    manifest: Dict[str, Any],
) -> str:
    lines = [
        "# KAPT Hyperparameter Sensitivity (Experiment C)",
        "",
        "One-at-a-time sweeps on three parameters with largest expected influence:",
        "**EMA alpha**, **length max_drift**, and **Kalman process noise (Q)**.",
        "All other hyperparameters held at defaults.",
        "",
        f"- Videos: {', '.join(manifest.get('videos', []))}",
        f"- Timestamp: {manifest.get('timestamp', '')}",
        "",
        "## Summary (mean +/- std across videos)",
        "",
    ]

    for sweep in SENSITIVITY_SWEEPS:
        lines.append(f"### {sweep.display_name} (`{sweep.param}`)")
        lines.append("")
        lines.append(
            f"| Value | default? | {sweep.primary_metric} | limb_std | jitter | accel_var |"
        )
        lines.append("|-------|----------|-------------------|----------|--------|-----------|")
        subset = sorted(
            [r for r in summary if r.get("parameter") == sweep.param],
            key=lambda r: float(r["value"]),
        )
        for r in subset:
            default_mark = "yes" if r.get("is_default") else ""
            lines.append(
                f"| {r['value']} | {default_mark} | "
                f"{r.get(f'{sweep.primary_metric}_mean', float('nan')):.2f} "
                f"+/- {r.get(f'{sweep.primary_metric}_std', 0.0):.2f} | "
                f"{r.get('limb_std_mean', float('nan')):.2f} | "
                f"{r.get('jitter_mean', float('nan')):.2f} | "
                f"{r.get('accel_var_mean', float('nan')):.1f} |"
            )
        lines.append("")

    lines.extend([
        "## Notes",
        "",
        "- **alpha**: correction-stage EMA; lower = smoother, higher = more responsive.",
        "- **max_drift**: length stabilizer tolerance; smaller = tighter limb lengths.",
        "- **process_noise**: re-smoothed from fused keypoints (no OpenPose re-run).",
        "",
        f"Per-video rows: {len(rows)}",
    ])
    return "\n".join(lines)
