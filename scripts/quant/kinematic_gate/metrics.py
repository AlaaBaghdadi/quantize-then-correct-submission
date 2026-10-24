"""Build kinematic metric vectors from phase1 keypoints JSON."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from .security import read_json_with_limit, validate_keypoints_payload

# OpenPose COCO-18 joint indices (see phase1/pose_handler/openpose18_mapping.py)
_OPENPOSE18_KEYPOINT_NAMES: Tuple[str, ...] = (
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
)
L_HIP, R_HIP, NECK = 11, 8, 1
FOCUS_JOINTS: Tuple[int, ...] = (3, 4, 6, 7, 9, 10, 12, 13)
CONF_THRESHOLD = 0.5
_SCALE_EPS = 1e-6
_PATH_EPS = 1e-5
_SIGNIFICANT_PATH = 0.1
_UNDER_MOTION_RATIO = 0.5


def ensure_phase1_on_path(phase1_root: Path) -> Path:
    """Add phase1 to sys.path once; return resolved root."""
    root = phase1_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"phase1 root not found: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def load_keypoints_json(json_path: Path) -> Dict[str, Any]:
    """Load and validate smoothed keypoints JSON written by kalman pipeline."""
    path = Path(json_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Keypoints JSON not found: {path}")
    data = read_json_with_limit(path)
    validate_keypoints_payload(data)
    return data


def compute_metric_vector(
    keypoints_json: Path,
    *,
    phase1_root: Path,
) -> Dict[str, Any]:
    """
    Compute the kinematic metric vector for one video's keypoints JSON.

    Returns dict with scalar metrics; internal _frames/_landmarks for in-process use only.
    """
    ensure_phase1_on_path(phase1_root)

    from pose_evaluator import WRIST_ELBOW_JOINT_INDICES, acceleration_stats, frames_to_landmarks
    from video_metrics import (
        compute_biomechanical_plausibility,
        compute_limb_length_consistency,
        compute_motion_jitter,
    )

    data = load_keypoints_json(keypoints_json)
    frames = data.get("frames", [])
    image_size = data.get("image_size")

    landmarks = frames_to_landmarks(frames, key_field="keypoints")
    accel = acceleration_stats(landmarks, joint_indices=WRIST_ELBOW_JOINT_INDICES)
    llc = compute_limb_length_consistency(landmarks)
    bio = compute_biomechanical_plausibility(frames, key_field="keypoints", image_size=image_size)
    jitter = compute_motion_jitter(landmarks, joint_indices=WRIST_ELBOW_JOINT_INDICES)

    metrics = {
        "keypoints_json": str(keypoints_json.resolve()),
        "num_frames": len(frames),
        "acceleration_variance": float(accel.get("acceleration_variance", float("nan"))),
        "mean_acceleration": float(accel.get("mean_acceleration", float("nan"))),
        "mean_limb_length_std": float(llc.get("mean_limb_length_std", float("nan"))),
        "total_violations": float(bio.get("total_violations", float("nan"))),
        "violation_fraction": float(bio.get("violation_fraction", float("nan"))),
        "frames_evaluated": float(bio.get("frames_evaluated", float("nan"))),
        "var_accel_norm": float(jitter.get("var_accel_norm", float("nan"))),
        "mean_accel_norm": float(jitter.get("mean_accel_norm", float("nan"))),
    }

    _assert_finite_core_metrics(metrics)
    return metrics


def _assert_finite_core_metrics(metrics: Dict[str, Any]) -> None:
    """Fail fast if phase1 returned non-finite core metrics (no silent dummy pass)."""
    required = (
        "acceleration_variance",
        "mean_limb_length_std",
        "var_accel_norm",
        "violation_fraction",
    )
    for key in required:
        value = float(metrics.get(key, float("nan")))
        if not math.isfinite(value):
            raise RuntimeError(
                f"Non-finite kinematic metric {key}={value!r} for {metrics.get('keypoints_json')}"
            )
    if int(metrics.get("num_frames", 0)) <= 0:
        raise RuntimeError("num_frames must be positive after keypoints validation")


def _openpose_joint_names() -> Tuple[str, ...]:
    """Return COCO-18 keypoint names in OpenPose index order."""
    return _OPENPOSE18_KEYPOINT_NAMES


def _frames_to_openpose_xy_conf(frames: List[Mapping[str, Any]]) -> np.ndarray:
    """
    Build [T, 18, 3] array with x, y, confidence from named keypoints JSON.

    Missing joints remain NaN in x/y and 0 in confidence.
    """
    names = _openpose_joint_names()
    t = len(frames)
    out = np.zeros((t, len(names), 3), dtype=np.float64)
    for frame_idx, frame in enumerate(frames):
        kps = frame.get("keypoints", {}) if isinstance(frame, dict) else {}
        if not isinstance(kps, dict):
            continue
        for joint_idx, name in enumerate(names):
            kp = kps.get(name)
            if not isinstance(kp, dict):
                continue
            x = kp.get("x")
            y = kp.get("y")
            if x is None or y is None:
                continue
            out[frame_idx, joint_idx, 0] = float(x)
            out[frame_idx, joint_idx, 1] = float(y)
            out[frame_idx, joint_idx, 2] = float(kp.get("confidence", 0.0))
    return out


def _normalized_focus_positions(
    xy_conf: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Root-normalize focus joint positions per frame.

    root = midpoint(L_HIP, R_HIP); scale = ||NECK - root|| + eps.
    Returns norm_pos [T, len(FOCUS_JOINTS), 2] and valid mask [T, len(FOCUS_JOINTS)].
    """
    left_hip = xy_conf[:, L_HIP, :2]
    right_hip = xy_conf[:, R_HIP, :2]
    neck = xy_conf[:, NECK, :2]

    root = 0.5 * (left_hip + right_hip)
    scale = np.linalg.norm(neck - root, axis=1) + _SCALE_EPS

    focus_xy = xy_conf[:, list(FOCUS_JOINTS), :2]
    focus_conf = xy_conf[:, list(FOCUS_JOINTS), 2]

    norm_pos = (focus_xy - root[:, np.newaxis, :]) / scale[:, np.newaxis, np.newaxis]
    valid = focus_conf > CONF_THRESHOLD
    return norm_pos, valid


def _path_lengths(norm_pos: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    Total trajectory path length per focus joint.

    Step displacement ||p[t] - p[t-1]|| is counted only when the joint is valid
    at both t and t-1. Returns L with shape [len(FOCUS_JOINTS)].
    """
    if norm_pos.shape[0] < 2:
        return np.zeros(norm_pos.shape[1], dtype=np.float64)

    step_valid = valid[1:] & valid[:-1]
    displacements = np.linalg.norm(norm_pos[1:] - norm_pos[:-1], axis=2)
    displacements = np.where(step_valid, displacements, 0.0)
    return np.sum(displacements, axis=0)


def compute_joint_movement_summary_between(
    data_fp16: Mapping[str, Any],
    data_quant: Mapping[str, Any],
) -> Dict[str, float]:
    """
    Cross-video spatial tracking metrics between FP16 and quantized keypoints.

    Computes Normalized Position Error (NPE) on root-normalized focus joints and
    counts under-motion events where quant trajectory path is severely damped
    relative to FP16 baseline.
    """
    validate_keypoints_payload(data_fp16)
    validate_keypoints_payload(data_quant)

    frames_fp16 = data_fp16.get("frames", [])
    frames_quant = data_quant.get("frames", [])
    n = min(len(frames_fp16), len(frames_quant))
    if n == 0:
        raise RuntimeError("Cannot compute movement summary: zero overlapping frames")

    xy_fp16 = _frames_to_openpose_xy_conf(frames_fp16[:n])
    xy_quant = _frames_to_openpose_xy_conf(frames_quant[:n])

    norm_fp16, valid_fp16 = _normalized_focus_positions(xy_fp16)
    norm_quant, valid_quant = _normalized_focus_positions(xy_quant)

    valid_both = valid_fp16 & valid_quant
    if not np.any(valid_both):
        mean_npe = float("nan")
        p95_npe = float("nan")
    else:
        npe = np.linalg.norm(norm_quant - norm_fp16, axis=2)
        npe_valid = npe[valid_both]
        mean_npe = float(np.mean(npe_valid))
        p95_npe = float(np.percentile(npe_valid, 95))

    l_fp16 = _path_lengths(norm_fp16, valid_fp16)
    l_quant = _path_lengths(norm_quant, valid_quant)
    r_vel = l_quant / np.maximum(l_fp16, _PATH_EPS)
    under_motion = int(
        np.sum((l_fp16 > _SIGNIFICANT_PATH) & (r_vel < _UNDER_MOTION_RATIO))
    )

    if not math.isfinite(mean_npe):
        raise RuntimeError(
            "Could not compute finite mean_NPE; insufficient valid joint data in keypoints"
        )

    return {
        "mean_NPE": mean_npe,
        "p95_NPE": p95_npe,
        "under_motion_violations": float(under_motion),
    }


def strip_internal_fields(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Remove non-serializable / internal keys for JSON report."""
    skip = {"_frames", "_landmarks"}
    out: Dict[str, float] = {}
    for k, v in metrics.items():
        if k in skip or k.startswith("_"):
            continue
        if isinstance(v, (int, float)):
            out[k] = float(v)
        elif isinstance(v, np.floating):
            out[k] = float(v)
    return out
