"""
Convert named keypoints (our format) to OpenPose JSON format and verify with overlay plots.

OpenPose COCO-18 order: Nose, Neck, R Shoulder, R Elbow, R Wrist, L Shoulder, L Elbow, L Wrist,
R Hip, R Knee, R Ankle, L Hip, L Knee, L Ankle, R Eye, L Eye, R Ear, L Ear.
Missing keypoints are output as (0, 0, 0); PISE/make_pise_fashion_sample may use a score
threshold (e.g. 0.15) and set -1 for missing in their CSV.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from pose_handler.openpose18_mapping import OPENPOSE18_CONNECTIONS, OPENPOSE18_KEYPOINT_NAMES


# COCO-18 index from keypoint name (lowercase with underscores)
NAME_TO_COCO18_INDEX: Dict[str, int] = {
    name.lower().replace(" ", "_"): i for i, name in enumerate(OPENPOSE18_KEYPOINT_NAMES)
}


def named_keypoints_to_pose_keypoints_2d(
    keypoints: Dict[str, Any],
    missing_conf: float = 0.0,
) -> List[float]:
    """
    Convert named keypoints dict to OpenPose flat list [x0,y0,c0, x1,y1,c1, ...] in COCO-18 order.

    Args:
        keypoints: Dict of name -> {x, y, confidence} or {x, y, z, confidence}.
        missing_conf: Confidence value for missing keypoints (default 0.0).

    Returns:
        Flat list of length 54 (18 * 3).
    """
    out: List[float] = []
    for i in range(18):
        name = OPENPOSE18_KEYPOINT_NAMES[i].lower().replace(" ", "_")
        kp = keypoints.get(name)
        if kp is None or not isinstance(kp, dict):
            out.extend([0.0, 0.0, missing_conf])
            continue
        x = float(kp.get("x", 0.0))
        y = float(kp.get("y", 0.0))
        c = float(kp.get("confidence", missing_conf))
        out.extend([x, y, c])
    return out


def named_keypoints_to_openpose_json(
    keypoints: Dict[str, Any],
    missing_conf: float = 0.0,
) -> Dict[str, Any]:
    """
    Convert named keypoints to full OpenPose-style JSON (single person).
    Uses COCO-18 (54 values). For body25 format use named_keypoints_to_openpose_json_body25.
    """
    flat = named_keypoints_to_pose_keypoints_2d(keypoints, missing_conf=missing_conf)
    return {"people": [{"pose_keypoints_2d": flat}]}


# Body25 index order used by PISE make_pise_fashion_sample (openpose18 index -> body25 index)
OPENPOSE18_TO_BODY25_INDEX = (0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)


def named_keypoints_to_pose_keypoints_2d_body25(
    keypoints: Dict[str, Any],
    missing_conf: float = 0.0,
) -> List[float]:
    """
    Convert named keypoints to OpenPose BODY_25 flat list (25 * 3 = 75 values).
    Required when run.py / make_pise_fashion_sample use --openpose_model body25.
    """
    flat18 = named_keypoints_to_pose_keypoints_2d(keypoints, missing_conf=missing_conf)
    flat25: List[float] = [0.0] * 75
    for i in range(18):
        b25 = OPENPOSE18_TO_BODY25_INDEX[i]
        flat25[b25 * 3] = flat18[i * 3]
        flat25[b25 * 3 + 1] = flat18[i * 3 + 1]
        flat25[b25 * 3 + 2] = flat18[i * 3 + 2]
    return flat25


def named_keypoints_to_openpose_json_body25(
    keypoints: Dict[str, Any],
    missing_conf: float = 0.0,
) -> Dict[str, Any]:
    """OpenPose-style JSON with pose_keypoints_2d in BODY_25 format (75 values)."""
    flat = named_keypoints_to_pose_keypoints_2d_body25(keypoints, missing_conf=missing_conf)
    return {"people": [{"pose_keypoints_2d": flat}]}


def write_openpose_json(
    path: Path,
    keypoints: Dict[str, Any],
    missing_conf: float = 0.0,
    body25: bool = True,
) -> None:
    """Write named keypoints to a file as OpenPose JSON. Use body25=True for run.py / PISE (default)."""
    if body25:
        data = named_keypoints_to_openpose_json_body25(keypoints, missing_conf=missing_conf)
    else:
        data = named_keypoints_to_openpose_json(keypoints, missing_conf=missing_conf)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def draw_skeleton_on_image(
    img_bgr: np.ndarray,
    keypoints: Dict[str, Any],
    min_conf: float = 0.15,
    bone_color: tuple = (0, 255, 0),
    joint_color: tuple = (255, 0, 0),
) -> None:
    """
    Draw COCO-18 skeleton on BGR image in-place.
    keypoints: named keypoints dict.
    """
    # Build list of 18 points (x, y, conf)
    kps18: List[Optional[tuple]] = [None] * 18
    for i in range(18):
        name = OPENPOSE18_KEYPOINT_NAMES[i].lower().replace(" ", "_")
        kp = keypoints.get(name)
        if kp is None or not isinstance(kp, dict):
            continue
        c = float(kp.get("confidence", 0.0))
        if c < min_conf:
            continue
        x = float(kp.get("x", 0.0))
        y = float(kp.get("y", 0.0))
        kps18[i] = (int(round(x)), int(round(y)))

    for a, b in OPENPOSE18_CONNECTIONS:
        if a >= len(kps18) or b >= len(kps18):
            continue
        pa, pb = kps18[a], kps18[b]
        if pa is None or pb is None:
            continue
        cv2.line(img_bgr, pa, pb, bone_color, 2, cv2.LINE_AA)

    for p in kps18:
        if p is None:
            continue
        cv2.circle(img_bgr, p, 4, joint_color, -1, cv2.LINE_AA)


def verify_openpose_overlay(
    frame_bgr: np.ndarray,
    keypoints: Dict[str, Any],
    out_path: Path,
    min_conf: float = 0.15,
) -> None:
    """
    Overlay skeleton on frame and save to out_path for verification.
    """
    img = frame_bgr.copy()
    draw_skeleton_on_image(img, keypoints, min_conf=min_conf)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def verify_openpose_frames(
    video_path: Path,
    frames_data: List[Dict],
    key_field: str,
    out_dir: Path,
    frame_indices: Optional[List[int]] = None,
    min_conf: float = 0.15,
) -> List[Path]:
    """
    For a subset of frames, extract frame image, overlay keypoints, and save verification images.

    Args:
        video_path: Path to video file.
        frames_data: List of frame dicts; each has key_field (e.g. "keypoints") with named keypoints.
        key_field: Key to use for keypoints in each frame (e.g. "keypoints", "keypoints_corrected").
        out_dir: Directory to write verify_openpose_frame_000.png etc.
        frame_indices: Which frame indices to plot. If None, plot first, middle, last.
        min_conf: Minimum confidence for drawing.

    Returns:
        List of saved image paths.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if frame_indices is None:
        if n_frames <= 3:
            frame_indices = list(range(n_frames))
        else:
            frame_indices = [0, n_frames // 2, n_frames - 1]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    cap = cv2.VideoCapture(str(video_path))
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        if idx < len(frames_data):
            kps = frames_data[idx].get(key_field, {})
            if kps:
                out_path = out_dir / f"verify_openpose_frame_{idx:04d}.png"
                verify_openpose_overlay(frame, kps, out_path, min_conf=min_conf)
                saved.append(out_path)
    cap.release()
    return saved
