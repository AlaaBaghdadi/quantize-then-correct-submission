"""
Format bridge: Kalman smoothed keypoints JSON <-> H36M-17 arrays for SmoothNet.

OpenPose-18 names in JSON; SmoothNet checkpoint expects 17 H36M joints in 2D.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pose_handler.common_io import read_json, write_json

# H36M 17-joint index -> how to obtain (x, y) from OpenPose-18 named keypoints
# "midpoint": tuple of two key names
# "direct": single key name
H36M_JOINT_SOURCES: List[Tuple[str, Any]] = [
    ("midpoint", ("left_hip", "right_hip")),       # 0 Hip
    ("direct", "right_hip"),                      # 1 RHip
    ("direct", "right_knee"),                     # 2 RKnee
    ("direct", "right_ankle"),                    # 3 RAnkle
    ("direct", "left_hip"),                       # 4 LHip
    ("direct", "left_knee"),                      # 5 LKnee
    ("direct", "left_ankle"),                     # 6 LAnkle
    ("midpoint_hip_neck", None),                  # 7 Spine (hip center + neck)
    ("direct", "neck"),                           # 8 Thorax
    ("direct", "neck"),                           # 9 Neck
    ("direct", "nose"),                           # 10 Head
    ("direct", "left_shoulder"),                  # 11 LShoulder
    ("direct", "left_elbow"),                     # 12 LElbow
    ("direct", "left_wrist"),                     # 13 LWrist
    ("direct", "right_shoulder"),                 # 14 RShoulder
    ("direct", "right_elbow"),                    # 15 RElbow
    ("direct", "right_wrist"),                    # 16 RWrist
]

# H36M joint index -> OpenPose-18 key to write back (None = derived, skip direct write)
H36M_TO_OPENPOSE_WRITE: Dict[int, Optional[str]] = {
    0: None,  # hip center — not a single OP joint
    1: "right_hip",
    2: "right_knee",
    3: "right_ankle",
    4: "left_hip",
    5: "left_knee",
    6: "left_ankle",
    7: None,  # spine — derived
    8: "neck",
    9: "neck",
    10: "nose",
    11: "left_shoulder",
    12: "left_elbow",
    13: "left_wrist",
    14: "right_shoulder",
    15: "right_elbow",
    16: "right_wrist",
}


def _get_kp_xy(
    keypoints: Dict[str, Any],
    name: str,
) -> Optional[Tuple[float, float]]:
    kp = keypoints.get(name)
    if not kp or not isinstance(kp, dict):
        return None
    x, y = kp.get("x"), kp.get("y")
    if x is None or y is None:
        return None
    try:
        return float(x), float(y)
    except (TypeError, ValueError):
        return None


def _midpoint_xy(
    keypoints: Dict[str, Any],
    name_a: str,
    name_b: str,
) -> Optional[Tuple[float, float]]:
    a = _get_kp_xy(keypoints, name_a)
    b = _get_kp_xy(keypoints, name_b)
    if a is None or b is None:
        return None
    return (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0


def frame_to_h36m_2d(keypoints: Dict[str, Any]) -> np.ndarray:
    """Return shape (17, 2) with NaN for missing joints."""
    out = np.full((17, 2), np.nan, dtype=np.float64)
    hip_center: Optional[Tuple[float, float]] = None

    for j, spec in enumerate(H36M_JOINT_SOURCES):
        kind = spec[0]
        if kind == "midpoint":
            _, names = spec
            xy = _midpoint_xy(keypoints, names[0], names[1])
            if j == 0 and xy is not None:
                hip_center = xy
        elif kind == "midpoint_hip_neck":
            neck = _get_kp_xy(keypoints, "neck")
            if hip_center is not None and neck is not None:
                xy = ((hip_center[0] + neck[0]) / 2.0, (hip_center[1] + neck[1]) / 2.0)
            else:
                xy = None
        else:
            _, name = spec
            xy = _get_kp_xy(keypoints, name)

        if xy is not None:
            out[j, 0] = xy[0]
            out[j, 1] = xy[1]

    return out


def frames_to_h36m_sequence(frames: List[Dict[str, Any]], key_field: str = "keypoints") -> np.ndarray:
    """Stack frames -> (T, 17, 2) pixel coordinates."""
    seq = []
    for frame in frames:
        kps = frame.get(key_field) or {}
        seq.append(frame_to_h36m_2d(kps if isinstance(kps, dict) else {}))
    return np.stack(seq, axis=0)


def interpolate_nans_along_time(pose: np.ndarray) -> np.ndarray:
    """
    Linear interpolation along time axis for each joint coordinate.
    pose: (T, 17, 2). Joints entirely NaN remain NaN.
    """
    out = pose.copy()
    t_len = out.shape[0]
    for j in range(out.shape[1]):
        for c in range(2):
            col = out[:, j, c]
            valid = np.isfinite(col)
            if valid.sum() == 0:
                continue
            if valid.sum() == t_len:
                continue
            indices = np.arange(t_len)
            col[~valid] = np.interp(indices[~valid], indices[valid], col[valid])
            out[:, j, c] = col
    return out


def normalize_pixels(pose: np.ndarray, width: float, height: float) -> np.ndarray:
    """H36M-style: (coord - size/2) / (size/2) per axis."""
    out = pose.copy()
    half_w = max(float(width) / 2.0, 1.0)
    half_h = max(float(height) / 2.0, 1.0)
    out[..., 0] = (out[..., 0] - half_w) / half_w
    out[..., 1] = (out[..., 1] - half_h) / half_h
    return out


def denormalize_pixels(pose: np.ndarray, width: float, height: float) -> np.ndarray:
    out = pose.copy()
    half_w = max(float(width) / 2.0, 1.0)
    half_h = max(float(height) / 2.0, 1.0)
    out[..., 0] = out[..., 0] * half_w + half_w
    out[..., 1] = out[..., 1] * half_h + half_h
    return out


def apply_smoothed_h36m_to_frames(
    frames: List[Dict[str, Any]],
    smoothed_pixels: np.ndarray,
    key_field: str = "keypoints",
) -> List[Dict[str, Any]]:
    """
    Write smoothed H36M joints back into OpenPose-18 frame dicts.
    Eyes/ears and unmapped joints keep original values.
    """
    if len(frames) != smoothed_pixels.shape[0]:
        raise ValueError(
            f"Frame count mismatch: {len(frames)} frames vs {smoothed_pixels.shape[0]} poses"
        )

    out_frames: List[Dict[str, Any]] = []
    for t, frame in enumerate(frames):
        new_frame = copy.deepcopy(frame)
        kps = new_frame.get(key_field)
        if not isinstance(kps, dict):
            kps = {}
            new_frame[key_field] = kps

        for h36m_idx, op_name in H36M_TO_OPENPOSE_WRITE.items():
            if op_name is None:
                continue
            x, y = smoothed_pixels[t, h36m_idx, 0], smoothed_pixels[t, h36m_idx, 1]
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            existing = kps.get(op_name)
            if isinstance(existing, dict):
                entry = copy.deepcopy(existing)
            else:
                entry = {"confidence": 1.0, "z": 0.0}
            entry["x"] = float(x)
            entry["y"] = float(y)
            kps[op_name] = entry

        new_frame[key_field] = kps
        out_frames.append(new_frame)

    return out_frames


def load_keypoints_json(path: str) -> Dict[str, Any]:
    return read_json(path)


def save_keypoints_json(path: str, data: Dict[str, Any]) -> None:
    write_json(path, data)


def build_output_payload(
    source_data: Dict[str, Any],
    smoothed_frames: List[Dict[str, Any]],
) -> Dict[str, Any]:
    output = copy.deepcopy(source_data)
    output["frames"] = smoothed_frames
    output["format"] = "smoothnet_baseline/refined_keypoints_v1"
    output["refinement"] = {
        "method": "SmoothNet",
        "reference": "Zeng et al., ECCV 2022",
        "joint_layout": "H36M-17 bridged from OpenPose-18",
    }
    return output
