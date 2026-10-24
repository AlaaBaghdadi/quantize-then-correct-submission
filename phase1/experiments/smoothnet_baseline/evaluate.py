"""Evaluate SmoothNet output using existing PoseEvaluator (read-only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from pose_evaluator import PoseEvaluator, _report_to_markdown, frames_to_landmarks
from pose_handler.common_io import read_json


def _extract_metric(report: Dict[str, Any], *path: str) -> float:
    node: Any = report
    for key in path:
        if not isinstance(node, dict):
            return float("nan")
        node = node.get(key)
    try:
        v = float(node)
        return v if np.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def metrics_dict_from_report(report: Dict[str, Any]) -> Dict[str, float]:
    """Pull CSV columns from a PoseEvaluator report."""
    jitter_corr = report.get("motion_jitter", {})
    jitter_val = float("nan")
    if isinstance(jitter_corr, dict):
        c = jitter_corr.get("corrected") or {}
        if isinstance(c, dict):
            jitter_val = _extract_metric({"v": c.get("mean_accel_norm")}, "v")
            if not np.isfinite(jitter_val):
                jitter_val = _extract_metric({"v": c.get("var_accel_norm")}, "v")

    return {
        "accel_var": _extract_metric(report, "temporal_stability", "corrected", "acceleration_variance"),
        "viol_frac": _extract_metric(report, "biomechanical_plausibility", "corrected", "violation_fraction"),
        "limb_std": _extract_metric(report, "limb_length_consistency", "corrected", "mean_limb_length_std"),
        "jitter": jitter_val,
        "accel_var_orig": _extract_metric(report, "temporal_stability", "original", "acceleration_variance"),
        "viol_frac_orig": _extract_metric(report, "biomechanical_plausibility", "original", "violation_fraction"),
        "limb_std_orig": _extract_metric(report, "limb_length_consistency", "original", "mean_limb_length_std"),
    }


def evaluate_pair(
    original_json: Path,
    corrected_json: Path,
    report_md: Optional[Path] = None,
    key_field: str = "keypoints",
) -> Tuple[Dict[str, Any], Dict[str, float], bool]:
    """
    Compare Kalman (original) vs SmoothNet (corrected) keypoint JSONs.

    Returns (full_report, metrics_dict, success_flag).
    """
    data_o = read_json(original_json)
    data_c = read_json(corrected_json)
    frames_o = data_o.get("frames", [])
    frames_c = data_c.get("frames", [])
    image_size = data_o.get("image_size")

    landmarks_orig = frames_to_landmarks(frames_o, key_field=key_field)
    landmarks_corr = frames_to_landmarks(frames_c, key_field=key_field)

    evaluator = PoseEvaluator()
    report, success = evaluator.generate_report(
        landmarks_original=landmarks_orig,
        landmarks_corrected=landmarks_corr,
        frames_original=frames_o,
        frames_corrected=frames_c,
        image_size=image_size,
    )

    metrics = metrics_dict_from_report(report)

    if report_md is not None:
        report_md.parent.mkdir(parents=True, exist_ok=True)
        title = f"SmoothNet baseline: {original_json.stem}"
        md = _report_to_markdown(title, report)
        header = "## Pose Correction Evaluation Results\n\n"
        report_md.write_text(header + md + "\n", encoding="utf-8")

    return report, metrics, bool(success)
