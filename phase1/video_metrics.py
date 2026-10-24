"""
Video / pose metrics utilities.

This module focuses on **skeleton-based** metrics for generative / pose-transfer
pipelines, plus simple frame extraction helpers:

- ``extract_frames_from_video``: dump all frames of a video to a folder.
- ``limb_length_consistency_from_frames``: limb-length stability over time.
- ``biomechanical_plausibility_from_frames``: % of frames with impossible joint angles.
- ``motion_jitter_from_frames``: acceleration-based motion smoothness.

FID-based helpers have been **removed from the pipeline** in favor of
pose-centric metrics. The legacy ``compute_fid_between_dirs`` function is kept
only for backwards compatibility and will raise at runtime if called.

Dependencies:
- Video I/O: ``opencv-python``
- Pose JSON loading: ``pose_handler.common_io``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import cv2
import numpy as np

from pose_handler.common_io import read_json
from pose_evaluator import (
    OPENPOSE18_KEYPOINT_NAMES,
    frames_to_landmarks,
)
from pose_handler.calc_angles import keypoints_to_arrays, calc_angles_from_landmarks


def extract_frames_from_video(video_path: Path, out_dir: Path) -> List[Path]:
    """
    Extract every frame from the video as images in out_dir.
    Returns list of saved image paths.
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    saved: List[Path] = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        out_path = out_dir / f"frame_{i:06d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved.append(out_path)
        i += 1
    cap.release()
    return saved


# ---------------------------------------------------------------------------
# Legacy FID stub (kept only to avoid import-time failures)
# ---------------------------------------------------------------------------

def compute_fid_between_dirs(dir_a: Path, dir_b: Path, mode: str = "clean") -> float:  # pragma: no cover
    """
    Legacy stub kept for backwards compatibility.

    FID-based evaluation has been removed from this project in favor of
    pose-centric metrics. This function will raise at runtime if called.
    """
    raise RuntimeError(
        "compute_fid_between_dirs is deprecated and no longer supported. "
        "Use pose_evaluator.compute_fid or your own FID implementation instead."
    )


# ---------------------------------------------------------------------------
# Helpers: frames → landmarks array
# ---------------------------------------------------------------------------

def _frames_to_landmarks(
    frames: Iterable[Mapping[str, Any]],
    *,
    key_field: str = "keypoints",
    num_joints: int = 18,
) -> np.ndarray:
    """
    Convert list of frame dicts (with per-frame keypoints) to [T, J, 3] array.

    Uses the same COCO-18 ordering as the rest of the project. Missing
    keypoints become NaN.
    """
    frame_list = list(frames)
    if not frame_list:
        return np.zeros((0, num_joints, 3), dtype=np.float64)
    return frames_to_landmarks(frame_list, key_field=key_field, num_joints=num_joints)


def _load_landmarks_from_json(json_path: Path, *, key_field: str = "keypoints") -> np.ndarray:
    """
    Load a Kalman / correction JSON (with ``frames``) and convert to landmarks.
    """
    data = read_json(Path(json_path))
    frames = data.get("frames", [])
    return _frames_to_landmarks(frames, key_field=key_field)


# ---------------------------------------------------------------------------
# Limb Length Consistency (LLC)
# ---------------------------------------------------------------------------

# COCO-18 indices for major limbs (see OPENPOSE18_KEYPOINT_NAMES)
_LIMB_DEFS: Dict[str, Tuple[int, int]] = {
    # Arms
    "right_upper_arm": (2, 3),   # right_shoulder -> right_elbow
    "right_lower_arm": (3, 4),   # right_elbow -> right_wrist
    "left_upper_arm": (5, 6),    # left_shoulder -> left_elbow
    "left_lower_arm": (6, 7),    # left_elbow -> left_wrist
    # Legs
    "right_thigh": (8, 9),       # right_hip -> right_knee
    "right_calf": (9, 10),       # right_knee -> right_ankle
    "left_thigh": (11, 12),      # left_hip -> left_knee
    "left_calf": (12, 13),       # left_knee -> left_ankle
}


def compute_limb_length_consistency(
    landmarks: np.ndarray,
    *,
    limb_defs: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Dict[str, float]:
    """
    Limb Length Consistency (LLC).

    For each limb, we compute the Euclidean distance between adjacent joints
    over time and report the **standard deviation** of that length:

        sigma_limb = std_t ||P_j(t) - P_i(t)||

    Returns:
        dict with per-limb std and an overall ``mean_limb_length_std``.
    """
    if limb_defs is None:
        limb_defs = _LIMB_DEFS

    if landmarks.size == 0:
        return {"mean_limb_length_std": float("nan")}

    T, J, D = landmarks.shape
    if T < 2 or J == 0 or D == 0:
        return {"mean_limb_length_std": float("nan")}

    out: Dict[str, float] = {}
    per_limb_stds: List[float] = []

    for limb_name, (i, j) in limb_defs.items():
        if i >= J or j >= J:
            continue
        vec = landmarks[:, i, :] - landmarks[:, j, :]  # [T, 3]
        dists = np.linalg.norm(vec, axis=-1)  # [T]
        mask = np.isfinite(dists)
        valid = dists[mask]
        if valid.size <= 1:
            continue
        std = float(np.std(valid))
        out[f"{limb_name}_std"] = std
        per_limb_stds.append(std)

    if per_limb_stds:
        out["mean_limb_length_std"] = float(np.mean(per_limb_stds))
    else:
        out["mean_limb_length_std"] = float("nan")
    return out


def limb_length_consistency_from_frames(
    frames: Iterable[Mapping[str, Any]],
    *,
    key_field: str = "keypoints",
) -> Dict[str, float]:
    """
    Convenience wrapper: LLC directly from ``frames`` (Kalman / correction JSON).
    """
    landmarks = _frames_to_landmarks(frames, key_field=key_field)
    return compute_limb_length_consistency(landmarks)


def limb_length_consistency_from_json(
    json_path: Path,
    *,
    key_field: str = "keypoints",
) -> Dict[str, float]:
    """
    Convenience wrapper: LLC from a keypoints JSON on disk.
    """
    landmarks = _load_landmarks_from_json(json_path, key_field=key_field)
    return compute_limb_length_consistency(landmarks)


# ---------------------------------------------------------------------------
# Bio-mechanical Plausibility (joint angle limits)
# ---------------------------------------------------------------------------

# Angle constraints based on pose_corrector.py biomechanical limits
# Shoulder Pitch: [0°, 180°] - rest to overhead
# Left Shoulder Yaw: [-45°, +135°] - across chest to out/back
# Right Shoulder Yaw: [-135°, +45°] - out/back to across chest
# Elbow: [0°, 180°] - fully closed to straight
_ANGLE_CONSTRAINTS = {
    "left_shoulder_pitch": (0.0, 180.0),
    "right_shoulder_pitch": (0.0, 180.0),
    "left_shoulder_yaw": (-45.0, 135.0),
    "right_shoulder_yaw": (-135.0, 45.0),
    "left_elbow_angle": (0.0, 180.0),
    "right_elbow_angle": (0.0, 180.0),
}


def compute_biomechanical_plausibility(
    frames: Iterable[Mapping[str, Any]],
    *,
    key_field: str = "keypoints",
    image_size: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Bio-mechanical Plausibility: count violations of joint angle constraints.

    Uses pose_handler.calc_angles to compute shoulder (pitch, yaw, roll) and elbow angles,
    then checks if they violate biomechanical constraints. Counts total violations across
    all frames.

    Args:
        frames: Iterable of frame dicts with keypoints
        key_field: Key in each frame dict containing keypoints (default: "keypoints")
        image_size: Optional dict with "width" and "height" for scaling

    Returns:
        {
            "total_violations": int,           # total number of angle violations across all frames
            "frames_with_any_violation": int,  # number of frames with at least one violation
            "frames_evaluated": int,           # number of frames where angles could be computed
            "violation_fraction": float,       # fraction of evaluated frames with ≥1 violation
            "per_angle_violation_count_*": int,  # one per angle type (e.g. left_shoulder_pitch)
            "per_angle_violation_fraction_*": float,  # fraction of evaluations that violated
        }
    """
    frame_list = list(frames)
    if not frame_list:
        return {
            "total_violations": 0,
            "frames_with_any_violation": 0,
            "frames_evaluated": 0,
            "violation_fraction": float("nan"),
        }

    # Initialize counters
    total_violations = 0
    frames_evaluated = 0
    frames_with_violation = 0
    angle_violation_counts: Dict[str, int] = {name: 0 for name in _ANGLE_CONSTRAINTS}
    angle_eval_counts: Dict[str, int] = {name: 0 for name in _ANGLE_CONSTRAINTS}

    for frame in frame_list:
        kps = frame.get(key_field, {})
        if not kps:
            continue

        # Convert keypoints to arrays and compute angles using calc_angles.py
        arr = keypoints_to_arrays(kps, image_size=image_size)
        if arr is None:
            continue

        angles = calc_angles_from_landmarks(arr)
        if angles is None:
            continue

        frames_evaluated += 1
        frame_has_violation = False

        # Check shoulder angles
        for side in ("left", "right"):
            shoulder = angles.get(f"{side}_shoulder")
            if shoulder is None:
                continue

            # Check pitch
            pitch = shoulder.get("pitch")
            if pitch is not None and np.isfinite(pitch):
                key = f"{side}_shoulder_pitch"
                angle_eval_counts[key] += 1
                min_val, max_val = _ANGLE_CONSTRAINTS[key]
                if not (min_val <= pitch <= max_val):
                    angle_violation_counts[key] += 1
                    total_violations += 1
                    frame_has_violation = True

            # Check yaw
            yaw = shoulder.get("yaw")
            if yaw is not None and np.isfinite(yaw):
                key = f"{side}_shoulder_yaw"
                angle_eval_counts[key] += 1
                min_val, max_val = _ANGLE_CONSTRAINTS[key]
                if not (min_val <= yaw <= max_val):
                    angle_violation_counts[key] += 1
                    total_violations += 1
                    frame_has_violation = True

        # Check elbow angles
        for side in ("left", "right"):
            elbow = angles.get(f"{side}_elbow")
            if elbow is None:
                continue
            angle = elbow.get("angle")
            if angle is not None and np.isfinite(angle):
                key = f"{side}_elbow_angle"
                angle_eval_counts[key] += 1
                min_val, max_val = _ANGLE_CONSTRAINTS[key]
                if not (min_val <= angle <= max_val):
                    angle_violation_counts[key] += 1
                    total_violations += 1
                    frame_has_violation = True

        if frame_has_violation:
            frames_with_violation += 1

    # Build result dict
    result: Dict[str, float] = {
        "total_violations": float(total_violations),
        "frames_with_any_violation": float(frames_with_violation),
        "frames_evaluated": float(frames_evaluated),
    }

    if frames_evaluated > 0:
        result["violation_fraction"] = frames_with_violation / frames_evaluated
    else:
        result["violation_fraction"] = float("nan")

    # Add per-angle counts and fractions
    for angle_name in _ANGLE_CONSTRAINTS:
        eval_n = angle_eval_counts[angle_name]
        viol_n = angle_violation_counts[angle_name]
        result[f"per_angle_violation_count_{angle_name}"] = float(viol_n)
        if eval_n > 0:
            result[f"per_angle_violation_fraction_{angle_name}"] = viol_n / eval_n
        else:
            result[f"per_angle_violation_fraction_{angle_name}"] = float("nan")

    return result


def biomechanical_plausibility_from_frames(
    frames: Iterable[Mapping[str, Any]],
    *,
    key_field: str = "keypoints",
    image_size: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Convenience wrapper: bio-mechanical plausibility directly from ``frames``.
    """
    return compute_biomechanical_plausibility(frames, key_field=key_field, image_size=image_size)


def biomechanical_plausibility_from_json(
    json_path: Path,
    *,
    key_field: str = "keypoints",
) -> Dict[str, float]:
    """
    Convenience wrapper: bio-mechanical plausibility from JSON on disk.
    """
    data = read_json(json_path)
    frames = data.get("frames", [])
    image_size = data.get("image_size")
    return compute_biomechanical_plausibility(frames, key_field=key_field, image_size=image_size)


# ---------------------------------------------------------------------------
# Temporal Smoothness: Motion Jitter (acceleration energy)
# ---------------------------------------------------------------------------

def compute_motion_jitter(
    landmarks: np.ndarray,
    *,
    joint_indices: Optional[Tuple[int, ...]] = None,
) -> Dict[str, float]:
    """
    Motion jitter based on discrete second derivative of positions.

    For positions P_t, we define acceleration:

        A_t = P_{t+1} - 2 * P_t + P_{t-1}

    and report the average and variance of ||A_t|| over all valid joints
    and time steps. Lower values indicate smoother motion.
    """
    if landmarks.size == 0:
        return {"mean_accel_norm": float("nan"), "var_accel_norm": float("nan")}

    T, J, D = landmarks.shape
    if T < 3 or J == 0 or D == 0:
        return {"mean_accel_norm": float("nan"), "var_accel_norm": float("nan")}

    if joint_indices is None:
        joint_indices = tuple(range(J))

    accel_norms: List[float] = []

    for t in range(1, T - 1):
        p_prev = landmarks[t - 1]
        p_curr = landmarks[t]
        p_next = landmarks[t + 1]
        for j in joint_indices:
            if j >= J:
                continue
            a_prev = p_prev[j]
            a_curr = p_curr[j]
            a_next = p_next[j]
            if not (
                np.all(np.isfinite(a_prev))
                and np.all(np.isfinite(a_curr))
                and np.all(np.isfinite(a_next))
            ):
                continue
            acc_vec = a_next - 2.0 * a_curr + a_prev
            accel_norms.append(float(np.linalg.norm(acc_vec)))

    if not accel_norms:
        return {"mean_accel_norm": float("nan"), "var_accel_norm": float("nan")}

    arr = np.asarray(accel_norms, dtype=np.float64)
    return {
        "mean_accel_norm": float(np.mean(arr)),
        "var_accel_norm": float(np.var(arr)),
    }


def motion_jitter_from_frames(
    frames: Iterable[Mapping[str, Any]],
    *,
    key_field: str = "keypoints",
    joint_indices: Optional[Tuple[int, ...]] = None,
) -> Dict[str, float]:
    """
    Convenience wrapper: motion jitter directly from ``frames``.
    """
    landmarks = _frames_to_landmarks(frames, key_field=key_field)
    return compute_motion_jitter(landmarks, joint_indices=joint_indices)


def motion_jitter_from_json(
    json_path: Path,
    *,
    key_field: str = "keypoints",
    joint_indices: Optional[Tuple[int, ...]] = None,
) -> Dict[str, float]:
    """
    Convenience wrapper: motion jitter from a keypoints JSON on disk.
    """
    landmarks = _load_landmarks_from_json(json_path, key_field=key_field)
    return compute_motion_jitter(landmarks, joint_indices=joint_indices)

