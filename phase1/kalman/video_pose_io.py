"""
Helpers for video pose pipeline: load OpenPose frame JSONs, extract MediaPipe from video,
scale MediaPipe Z, convert OpenPose to named keypoints, average frame-by-frame.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from pose_handler.common_io import read_json, write_json
from pose_handler.keypoints_schema import (
    FORMAT_MEDIAPIPE_POSE_V1,
    MEDIAPIPE_POSE_INDEX_BY_NAME,
)


# ----- OpenPose frame JSON discovery -----

def list_openpose_frame_jsons(
    output_dir: Path,
    video_stem: str,
) -> List[Tuple[int, Path]]:
    """
    List OpenPose JSON files for a video, sorted by frame index.
    Expects names like {video_stem}_{frame_index:010d}_keypoints.json
    or {video_stem}_{frame_index}_keypoints.json (variable padding).
    Returns list of (frame_index, path).
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return []
    pattern = re.compile(r"^" + re.escape(video_stem) + r"_(\d+)_keypoints\.json$")
    out: List[Tuple[int, Path]] = []
    for p in output_dir.iterdir():
        if not p.is_file() or p.suffix.lower() != ".json":
            continue
        m = pattern.match(p.name)
        if m:
            frame_idx = int(m.group(1))
            out.append((frame_idx, p))
    out.sort(key=lambda x: x[0])
    return out


def _parse_pose_keypoints_2d(flat: List[float]) -> List[Optional[Dict[str, float]]]:
    """OpenPose format: [x0,y0,c0,x1,y1,c1,...] -> list of {x,y,confidence} or None."""
    out: List[Optional[Dict[str, float]]] = []
    n = len(flat) // 3
    for i in range(n):
        x = float(flat[i * 3 + 0])
        y = float(flat[i * 3 + 1])
        c = float(flat[i * 3 + 2])
        if c <= 0.0:
            out.append(None)
        else:
            out.append({"x": x, "y": y, "confidence": c})
    return out


# COCO-18 names (same order as openpose18_mapping)
OPENPOSE18_KEYPOINT_NAMES = [
    "Nose", "Neck", "Right Shoulder", "Right Elbow", "Right Wrist",
    "Left Shoulder", "Left Elbow", "Left Wrist",
    "Right Hip", "Right Knee", "Right Ankle",
    "Left Hip", "Left Knee", "Left Ankle",
    "Right Eye", "Left Eye", "Right Ear", "Left Ear",
]
COCO18_TO_NAME = {
    i: name.lower().replace(" ", "_") for i, name in enumerate(OPENPOSE18_KEYPOINT_NAMES)
}
BODY25_TO_COCO18 = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
    9: 8, 10: 9, 11: 10, 12: 11, 13: 12, 14: 13,
    15: 14, 16: 15, 17: 16, 18: 17,
}


def openpose_json_to_named_keypoints(
    data: Dict,
    person_index: int = 0,
    min_conf: float = 0.0,
) -> Dict[str, Optional[Dict[str, float]]]:
    """
    Convert OpenPose JSON data (people[].pose_keypoints_2d) to named keypoints dict.
    """
    people = data.get("people", [])
    if not people or person_index >= len(people):
        return {}
    person = people[person_index]
    flat = person.get("pose_keypoints_2d", [])
    if not isinstance(flat, list) or len(flat) < 3:
        return {}
    num_kps = len(flat) // 3
    kps = _parse_pose_keypoints_2d(flat)
    is_body25 = num_kps >= 25

    named: Dict[str, Optional[Dict[str, float]]] = {}
    for coco18_idx, name in COCO18_TO_NAME.items():
        if is_body25:
            body25_idx = None
            for b25, c18 in BODY25_TO_COCO18.items():
                if c18 == coco18_idx:
                    body25_idx = b25
                    break
            kp = kps[body25_idx] if body25_idx is not None and body25_idx < len(kps) else None
        else:
            kp = kps[coco18_idx] if coco18_idx < len(kps) else None
        if kp is None:
            named[name] = None
            continue
        if float(kp.get("confidence", 0.0)) < min_conf:
            named[name] = None
            continue
        named[name] = kp
    return named


# ----- MediaPipe from video -----

def _lm_to_point(lm) -> Dict[str, float]:
    return {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)}


def extract_mediapipe_from_video(
    video_path: Path,
    model_complexity: int = 2,
    min_detection_confidence: float = 0.5,
) -> Tuple[List[Dict], float, int, int]:
    """
    Run MediaPipe Pose on every frame of the video (static_image_mode=True).
    Returns (entries, fps, width, height). Each entry has image_name, image_path,
    image_size, detected, landmarks (same structure as extract_mediapipe_from_folder).
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"Invalid video dimensions: {video_path}")

    entries: List[Dict] = []
    frame_idx = 0

    with mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            if frame_idx == 0 and (width != w or height != h):
                width, height = w, h
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            entry = {
                "image_name": f"frame_{frame_idx:010d}.png",
                "image_path": f"{video_path.as_posix()}#frame_{frame_idx}",
                "image_size": {"width": width, "height": height},
                "detected": bool(results.pose_landmarks and results.pose_world_landmarks),
                "landmarks": {},
            }
            if entry["detected"]:
                pose_landmarks = results.pose_landmarks.landmark
                world_landmarks = results.pose_world_landmarks.landmark
                for name, idx in MEDIAPIPE_POSE_INDEX_BY_NAME.items():
                    lm = pose_landmarks[idx]
                    wlm = world_landmarks[idx]
                    entry["landmarks"][name] = {
                        "visibility": float(lm.visibility),
                        "normalized": _lm_to_point(lm),
                        "world_3d": _lm_to_point(wlm),
                        "pixel": {"x": float(lm.x) * w, "y": float(lm.y) * h},
                    }
            entries.append(entry)
            frame_idx += 1

    cap.release()
    return entries, float(fps), width, height


# ----- Scale MediaPipe Z (same logic as scale_mediapipe_z) -----

def _np2(v: Dict[str, float]) -> np.ndarray:
    return np.array([float(v["x"]), float(v["y"])], dtype=np.float32)


def _calc_scale_factor(landmarks: Dict) -> float:
    required = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    if any(k not in landmarks for k in required):
        return 1.0

    def get_norm_xy(name: str) -> Optional[np.ndarray]:
        if name not in landmarks or landmarks[name] is None:
            return None
        return _np2(landmarks[name]["normalized"])

    def get_world_xy(name: str) -> Optional[np.ndarray]:
        if name not in landmarks or landmarks[name] is None:
            return None
        return _np2(landmarks[name]["world_3d"])

    pairs = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
    ]
    scales: List[float] = []
    for a, b in pairs:
        a_n, b_n = get_norm_xy(a), get_norm_xy(b)
        a_w, b_w = get_world_xy(a), get_world_xy(b)
        if a_n is None or b_n is None or a_w is None or b_w is None:
            continue
        dn = float(np.linalg.norm(a_n - b_n))
        dw = float(np.linalg.norm(a_w - b_w))
        if dw > 1e-10 and dn > 1e-10:
            scales.append(dn / dw)
    if not scales:
        return 1.0
    return float(np.mean(scales))


def scale_mediapipe_entries(entries: List[Dict]) -> List[Dict]:
    """Scale z (world meters -> normalized units) for each entry. Returns new list of entries."""
    out: List[Dict] = []
    for entry in entries:
        detected = bool(entry.get("detected", True))
        lm = entry.get("landmarks", {}) if detected else {}
        scale_factor = _calc_scale_factor(lm) if detected else 1.0
        keypoints = {}
        for name, rec in lm.items():
            x = float(rec["normalized"]["x"])
            y = float(rec["normalized"]["y"])
            z_world = float(rec["world_3d"]["z"])
            keypoints[name] = {
                "x": x,
                "y": y,
                "z_world": z_world,
                "z_scaled": float(z_world * scale_factor),
                "visibility": float(rec.get("visibility", 0.0)),
            }
        out.append({
            "image_name": entry.get("image_name", ""),
            "image_path": entry.get("image_path", ""),
            "image_size": entry.get("image_size", {"width": 0, "height": 0}),
            "detected": detected,
            "scale_factor": float(scale_factor),
            "keypoints": keypoints,
        })
    return out


# ----- Average MediaPipe + OpenPose (per frame) -----

def _get_mediapipe_normalized_coords(entry: Dict, keypoint_name: str) -> Optional[Dict[str, float]]:
    if "keypoints" in entry:
        kp = entry.get("keypoints", {}).get(keypoint_name)
        if not kp:
            return None
        return {
            "x": float(kp.get("x", 0.0)),
            "y": float(kp.get("y", 0.0)),
            "confidence": float(kp.get("visibility", 0.0)),
        }
    if "landmarks" in entry:
        lm = entry.get("landmarks", {}).get(keypoint_name)
        if not lm:
            return None
        norm = lm.get("normalized", {})
        return {
            "x": float(norm.get("x", 0.0)),
            "y": float(norm.get("y", 0.0)),
            "confidence": float(lm.get("visibility", 0.0)),
        }
    return None


def _get_openpose_normalized_coords(
    keypoints: Dict,
    keypoint_name: str,
    image_size: Dict[str, float],
) -> Optional[Dict[str, float]]:
    width = float(image_size.get("width", 0))
    height = float(image_size.get("height", 0))
    if width == 0 or height == 0:
        return None
    kp = keypoints.get(keypoint_name)
    if not kp:
        return None
    return {
        "x": float(kp.get("x", 0.0)) / width,
        "y": float(kp.get("y", 0.0)) / height,
        "confidence": float(kp.get("confidence", 0.0)),
    }


def _get_mediapipe_z(entry: Dict, keypoint_name: str) -> Optional[float]:
    """Return scaled z (normalized units) from MediaPipe entry."""
    if "keypoints" in entry:
        kp = entry.get("keypoints", {}).get(keypoint_name)
        if not kp:
            return None
        if "z_scaled" in kp:
            return float(kp.get("z_scaled", 0.0))
        if "z_world" in kp:
            return float(kp.get("z_world", 0.0))
        return None
    if "landmarks" in entry:
        lm = entry.get("landmarks", {}).get(keypoint_name)
        if not lm:
            return None
        return float(lm.get("world_3d", {}).get("z", 0.0))
    return None


def _get_mediapipe_z_world(entry: Dict, keypoint_name: str) -> Optional[float]:
    """Return raw z (world meters) from MediaPipe entry."""
    if "keypoints" in entry:
        kp = entry.get("keypoints", {}).get(keypoint_name)
        if not kp:
            return None
        if "z_world" in kp:
            return float(kp.get("z_world", 0.0))
        return None
    if "landmarks" in entry:
        lm = entry.get("landmarks", {}).get(keypoint_name)
        if not lm:
            return None
        return float(lm.get("world_3d", {}).get("z", 0.0))
    return None


def _average_keypoints(
    mp_coords: Optional[Dict[str, float]],
    op_coords: Optional[Dict[str, float]],
    min_conf: float,
) -> Optional[Dict[str, float]]:
    if mp_coords is None and op_coords is None:
        return None
    if mp_coords is None:
        if op_coords and float(op_coords.get("confidence", 0.0)) >= min_conf:
            return op_coords.copy()
        return None
    if op_coords is None:
        if mp_coords and float(mp_coords.get("confidence", 0.0)) >= min_conf:
            return mp_coords.copy()
        return None
    mp_conf = float(mp_coords.get("confidence", 0.0))
    op_conf = float(op_coords.get("confidence", 0.0))
    if mp_conf < min_conf and op_conf < min_conf:
        return None
    if mp_conf >= min_conf and op_conf >= min_conf:
        avg_x = (float(mp_coords["x"]) + float(op_coords["x"])) / 2.0
        avg_y = (float(mp_coords["y"]) + float(op_coords["y"])) / 2.0
        avg_conf = (mp_conf + op_conf) / 2.0
    elif mp_conf >= min_conf:
        avg_x, avg_y = float(mp_coords["x"]), float(mp_coords["y"])
        avg_conf = mp_conf
    else:
        avg_x, avg_y = float(op_coords["x"]), float(op_coords["y"])
        avg_conf = op_conf
    return {"x": avg_x, "y": avg_y, "confidence": avg_conf}


def average_frame_keypoints(
    mp_scaled_entry: Dict,
    op_named_keypoints: Dict[str, Optional[Dict[str, float]]],
    image_size: Dict[str, int],
    min_conf: float = 0.3,
) -> Dict[str, Optional[Dict[str, float]]]:
    """
    Average MediaPipe (scaled) and OpenPose keypoints for one frame.
    Returns unified-style keypoints: pixel x, y, z, confidence.
    """
    width = float(image_size.get("width", 0))
    height = float(image_size.get("height", 0))

    mp_keypoint_names = set(mp_scaled_entry.get("keypoints", {}).keys())
    op_keypoint_names = set(op_named_keypoints.keys())
    all_names = sorted(mp_keypoint_names | op_keypoint_names)

    averaged_norm: Dict[str, Optional[Dict[str, float]]] = {}
    for kp_name in all_names:
        mp_coords = _get_mediapipe_normalized_coords(mp_scaled_entry, kp_name)
        op_coords = _get_openpose_normalized_coords(op_named_keypoints, kp_name, image_size)
        averaged_norm[kp_name] = _average_keypoints(mp_coords, op_coords, min_conf)

    # Neck: MediaPipe has no neck
    mp_ls = _get_mediapipe_normalized_coords(mp_scaled_entry, "left_shoulder")
    mp_rs = _get_mediapipe_normalized_coords(mp_scaled_entry, "right_shoulder")
    op_neck = _get_openpose_normalized_coords(op_named_keypoints, "neck", image_size)
    if mp_ls is not None and mp_rs is not None:
        mp_mid = {
            "x": (float(mp_ls["x"]) + float(mp_rs["x"])) / 2.0,
            "y": (float(mp_ls["y"]) + float(mp_rs["y"])) / 2.0,
            "confidence": (float(mp_ls.get("confidence", 0.0)) + float(mp_rs.get("confidence", 0.0))) / 2.0,
        }
        averaged_norm["neck"] = _average_keypoints(mp_mid, op_neck, min_conf)
    elif op_neck is not None and float(op_neck.get("confidence", 0.0)) >= min_conf:
        averaged_norm["neck"] = op_neck.copy()

    scale_factor = float(mp_scaled_entry.get("scale_factor", 1.0))
    averaged_pixels: Dict[str, Optional[Dict[str, float]]] = {}
    for name, kp in averaged_norm.items():
        if kp is None or width == 0 or height == 0:
            averaged_pixels[name] = None
            continue
        z_world = _get_mediapipe_z_world(mp_scaled_entry, name)
        if name == "neck" and z_world is None:
            z_l = _get_mediapipe_z_world(mp_scaled_entry, "left_shoulder")
            z_r = _get_mediapipe_z_world(mp_scaled_entry, "right_shoulder")
            z_world = (z_l + z_r) / 2.0 if (z_l is not None and z_r is not None) else (z_l if z_l is not None else z_r)
        z_scaled = _get_mediapipe_z(mp_scaled_entry, name)
        if name == "neck" and z_scaled is None:
            z_l = _get_mediapipe_z(mp_scaled_entry, "left_shoulder")
            z_r = _get_mediapipe_z(mp_scaled_entry, "right_shoulder")
            z_scaled = (z_l + z_r) / 2.0 if (z_l is not None and z_r is not None) else (z_l if z_l is not None else z_r)
        if z_world is None:
            z_world = 0.0
        if z_scaled is None:
            z_scaled = float(z_world * scale_factor)
        averaged_pixels[name] = {
            "x": float(kp["x"]) * width,
            "y": float(kp["y"]) * height,
            "z": float(z_world),
            "z_scaled": float(z_scaled),
            "confidence": float(kp.get("confidence", 0.0)),
        }
    return averaged_pixels


# ----- Save / load multi-frame keypoints and angles -----

def save_smoothed_keypoints(
    path: Path,
    frames: List[Dict],
    video_stem: str,
    fps: float,
    image_size: Dict[str, int],
) -> None:
    """Save list of per-frame keypoints (each frame is {keypoints: {...}})."""
    payload = {
        "format": "kalman/smoothed_keypoints_v1",
        "video_stem": video_stem,
        "fps": fps,
        "image_size": image_size,
        "num_frames": len(frames),
        "frames": frames,
    }
    write_json(path, payload)


def save_angles_json(
    path: Path,
    frame_angles: List[Dict],
    video_stem: str,
    fps: float,
) -> None:
    """Save angles per frame: list of {frame_index, angles: {...}}."""
    payload = {
        "format": "kalman/angles_v1",
        "video_stem": video_stem,
        "fps": fps,
        "frames": frame_angles,
    }
    write_json(path, payload)
