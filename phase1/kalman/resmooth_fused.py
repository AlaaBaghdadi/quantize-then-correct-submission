"""
Re-apply Kalman smoothing to fused (pre-Kalman) keypoints with custom noise params.

Used for hyperparameter sensitivity without re-running OpenPose + MediaPipe.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional

from kalman.keypoint_smoother import KeypointSmoother
from kalman.video_pose_io import save_smoothed_keypoints
from pose_handler.common_io import read_json, write_json


def resmooth_fused_json(
    fused_json: Path,
    out_json: Path,
    *,
    process_noise: float = 1e-2,
    measurement_noise: float = 1e-1,
    visibility_threshold: float = 0.3,
    force: bool = False,
) -> Path:
    """
    Read fused keypoints JSON, Kalman-smooth with given Q/R, write smoothed JSON.

    Output schema matches kalman/run_video_pipeline smoothed exports (includes
    keypoints_averaged copied from fused input per frame).
    """
    fused_json = fused_json.resolve()
    out_json = out_json.resolve()

    if out_json.is_file() and not force:
        return out_json

    if not fused_json.is_file():
        raise FileNotFoundError(f"Fused keypoints not found: {fused_json}")

    data = read_json(fused_json)
    frames_in = data.get("frames", [])
    if not frames_in:
        raise ValueError(f"No frames in fused JSON: {fused_json}")

    keypoint_names: List[str] = sorted(
        {name for frame in frames_in for name in (frame.get("keypoints") or {}).keys()}
    )
    if not keypoint_names:
        raise ValueError(f"No keypoints found in {fused_json}")

    smoother = KeypointSmoother(
        keypoint_names=keypoint_names,
        dt=1.0,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        visibility_threshold=visibility_threshold,
    )

    smoothed_frames: List[Dict] = []
    for frame in frames_in:
        kps = frame.get("keypoints") or {}
        smoothed_xy = smoother.update(kps)
        merged: Dict[str, Optional[Dict]] = {}
        for name in keypoint_names:
            xy = smoothed_xy.get(name)
            orig = kps.get(name)
            if xy is None:
                merged[name] = None
                continue
            z = float(orig.get("z", 0.0)) if orig else 0.0
            z_scaled = float(orig.get("z_scaled", z)) if orig else z
            conf = float(orig.get("confidence", 0.0)) if orig else 0.0
            merged[name] = {
                "x": xy["x"],
                "y": xy["y"],
                "z": z,
                "z_scaled": z_scaled,
                "confidence": conf,
            }
        smoothed_frames.append({"keypoints": merged, "keypoints_averaged": copy.deepcopy(kps)})

    out_json.parent.mkdir(parents=True, exist_ok=True)
    video_stem = data.get("video_stem", fused_json.stem.replace("_fused_keypoints", ""))
    fps = float(data.get("fps") or 30.0)
    image_size = data.get("image_size") or {"width": 512, "height": 512}

    save_smoothed_keypoints(out_json, smoothed_frames, video_stem, fps, image_size)

    payload = read_json(out_json)
    payload["kalman_params"] = {
        "process_noise": process_noise,
        "measurement_noise": measurement_noise,
        "visibility_threshold": visibility_threshold,
        "source_fused": str(fused_json),
    }
    write_json(out_json, payload)
    return out_json
