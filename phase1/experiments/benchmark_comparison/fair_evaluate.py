"""
Fair benchmark metrics for KAPT vs SmoothNet.

Uses angle-limit violations (compute_biomechanical_plausibility) instead of
re-running PoseCorrector on corrected tracks. Also reports improvement vs Kalman.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from pose_evaluator import (
    WRIST_ELBOW_JOINT_INDICES,
    _frames_needing_correction,
    _normalize_keypoints_for_corrector,
    frames_to_landmarks,
)
from pose_evaluator import acceleration_stats
from pose_handler.common_io import read_json
from video_metrics import (
    compute_biomechanical_plausibility,
    compute_limb_length_consistency,
    compute_motion_jitter,
)


def _finite(v: Any) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def evaluate_track(
    json_path: Path,
    key_field: str = "keypoints",
) -> Dict[str, float]:
    """Absolute metrics on a single keypoint track (no corrector re-run)."""
    data = read_json(json_path)
    frames = data.get("frames", [])
    image_size = data.get("image_size")
    landmarks = frames_to_landmarks(frames, key_field=key_field)

    bio = compute_biomechanical_plausibility(
        frames, key_field=key_field, image_size=image_size,
    )
    llc = compute_limb_length_consistency(landmarks)
    jitter = compute_motion_jitter(landmarks, joint_indices=WRIST_ELBOW_JOINT_INDICES)
    accel = acceleration_stats(landmarks, joint_indices=WRIST_ELBOW_JOINT_INDICES)

    jitter_val = float("nan")
    if isinstance(jitter, dict):
        jitter_val = _finite(jitter.get("mean_accel_norm"))
        if not np.isfinite(jitter_val):
            jitter_val = _finite(jitter.get("var_accel_norm"))

    return {
        "angle_viol_frac": _finite(bio.get("violation_fraction")),
        "limb_std": _finite(llc.get("mean_limb_length_std") if llc else float("nan")),
        "jitter": jitter_val,
        "accel_var": _finite(accel.get("acceleration_variance")),
    }


def _load_pose_corrector():
    """Import PoseCorrector with module registered (avoids dataclass import bug)."""
    try:
        root = Path(__file__).resolve().parents[2]
        path = root / "pose-corrector" / "pose_corrector.py"
        if not path.is_file():
            return None
        name = "pose_corrector_fair_benchmark"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return getattr(mod, "PoseCorrector", None)
    except Exception:
        return None


def evaluate_kalman_baseline_need(
    kalman_json: Path,
    key_field: str = "keypoints",
    alpha: float = 0.7,
) -> Dict[str, float]:
    """
    Fraction of frames where KAPT shoulder correction would fire on Kalman input.
    Shows problem severity before refinement (Kalman-only baseline).
    """
    need = _frames_needing_correction(
        read_json(kalman_json).get("frames", []),
        key_field=key_field,
        alpha=alpha,
    )
    if need is not None:
        return {
            "frames_needing_kapt_frac": _finite(need.get("violation_fraction")),
            "frames_needing_kapt_count": _finite(need.get("total_violations")),
        }

    PoseCorrector = _load_pose_corrector()
    if PoseCorrector is None:
        return {
            "frames_needing_kapt_frac": float("nan"),
            "frames_needing_kapt_count": float("nan"),
        }

    frames = read_json(kalman_json).get("frames", [])
    frames_needing = 0
    frames_evaluated = 0
    try:
        corrector = PoseCorrector(alpha=alpha)
        for frame in frames:
            raw = frame.get(key_field) or {}
            kps = _normalize_keypoints_for_corrector(raw) if isinstance(raw, dict) else {}
            if not kps:
                continue
            result = corrector.correct_frame(kps)
            if result is None:
                continue
            corrections = getattr(result, "corrections_applied", None)
            if not isinstance(corrections, dict):
                continue
            frames_evaluated += 1
            if corrections.get("left", False) or corrections.get("right", False):
                frames_needing += 1
    except Exception:
        return {
            "frames_needing_kapt_frac": float("nan"),
            "frames_needing_kapt_count": float("nan"),
        }

    if frames_evaluated == 0:
        return {
            "frames_needing_kapt_frac": float("nan"),
            "frames_needing_kapt_count": float("nan"),
        }
    return {
        "frames_needing_kapt_frac": frames_needing / frames_evaluated,
        "frames_needing_kapt_count": float(frames_needing),
    }


def evaluate_method_vs_kalman(
    kalman_json: Path,
    method_json: Path,
    method_name: str,
    video_stem: str,
) -> Dict[str, Any]:
    """Fair comparison row: absolute method metrics + delta vs Kalman baseline."""
    kalman_m = evaluate_track(kalman_json)
    method_m = evaluate_track(method_json)
    need = evaluate_kalman_baseline_need(kalman_json)

    row: Dict[str, Any] = {
        "video": video_stem,
        "method": method_name,
        **method_m,
        "kalman_angle_viol_frac": kalman_m["angle_viol_frac"],
        "kalman_limb_std": kalman_m["limb_std"],
        "kalman_jitter": kalman_m["jitter"],
        "kalman_accel_var": kalman_m["accel_var"],
        **need,
    }

    # Positive delta = improvement (baseline higher than method for error metrics)
    for key in ("angle_viol_frac", "limb_std", "jitter", "accel_var"):
        b = kalman_m[key]
        m = method_m[key]
        if np.isfinite(b) and np.isfinite(m):
            row[f"{key}_delta"] = b - m
        else:
            row[f"{key}_delta"] = float("nan")

    return row
