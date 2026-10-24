from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from pose_handler.common_io import ensure_dir, read_json, write_json
from pose_handler.keypoints_schema import (
    FORMAT_MEDIAPIPE_POSE_SCALED_V1,
    FORMAT_MEDIAPIPE_POSE_V1,
    FORMAT_UNIFIED_KEYPOINTS_V1,
)
from pose_handler.unified_keypoints_io import load_unified_keypoints
from shared_pose.pose_io import calc_scale_factor


def main() -> int:
    ap = argparse.ArgumentParser(description="Scale MediaPipe world z into normalized units and write a clean scaled file.")
    ap.add_argument(
        "--in_json",
        default="pose_handler_out/mediapipe_pose.json",
        help="Input JSON from extract step (must be mediapipe_pose.json with normalized + world_3d data).",
    )
    ap.add_argument("--out_dir", default="pose_handler_out", help="Output directory.")
    args = ap.parse_args()

    in_path = Path(args.in_json)
    data = read_json(in_path)
    fmt = data.get("format")
    
    out_dir = ensure_dir(args.out_dir)
    out_entries = []

    # We require the full MediaPipe pose file with landmarks (normalized + world_3d)
    if fmt != FORMAT_MEDIAPIPE_POSE_V1:
        raise ValueError(
            f"Unexpected format in {in_path}: {fmt}. "
            f"Expected {FORMAT_MEDIAPIPE_POSE_V1} (mediapipe_pose.json with landmarks)."
        )

    # Old format with entries from the extractor
    entries = data.get("entries", [])

    for entry in entries:
        detected = bool(entry.get("detected", True))

        # Work directly from landmarks: normalized (x,y) and world_3d (x,y,z)
        lm = entry.get("landmarks", {}) if detected else {}
        # Use shared scale-factor implementation (same logic as Kalman/video pipeline):
        # compute scale between normalized (x,y) and world_3d (x,y)
        # distances for shoulders/hips.
        scale_factor = calc_scale_factor(lm) if detected else 1.0

        keypoints = {}
        for name, rec in lm.items():
            x = float(rec["normalized"]["x"])
            y = float(rec["normalized"]["y"])
            z_world = float(rec["world_3d"]["z"])  # meters
            keypoints[name] = {
                "x": x,
                "y": y,
                "z_world": z_world,
                "z_scaled": float(z_world * scale_factor),
                "visibility": float(rec.get("visibility", 0.0)),
            }

        out_entries.append(
            {
                "image_name": entry.get("image_name", ""),
                "image_path": entry.get("image_path", ""),
                "image_size": entry.get("image_size", {"width": 0, "height": 0}),
                "detected": detected,
                "scale_factor": float(scale_factor),
                "keypoints": keypoints,
            }
        )

    # Write per-image files with image name and "scale" keyword
    from pose_handler.unified_keypoints_io import save_unified_keypoints
    
    for entry in out_entries:
        image_name = entry.get("image_name", "unknown")
        image_stem = Path(image_name).stem
        
        # Write scaled format file per image (must end with "scale")
        scaled_path = out_dir / f"{image_stem}_mediapipe_pose_scale.json"
        scaled_payload = {
            "format": FORMAT_MEDIAPIPE_POSE_SCALED_V1,
            "entries": [entry]
        }
        write_json(scaled_path, scaled_payload)
        print(f"Wrote: {scaled_path}")
        
        if not entry.get("detected", False):
            continue
        
        # Write unified format file per image (must end with "scale")
        unified_path = out_dir / f"{image_stem}_unified_scale.json"
        
        # Unified output: keep x/y as normalized, and write z as z_scaled
        keypoints: Dict[str, Optional[Dict[str, float]]] = {}
        
        for kp_name, kp_data in entry.get("keypoints", {}).items():
            if not kp_data:
                keypoints[kp_name] = None
                continue
            x_norm = float(kp_data.get("x", 0.0))
            y_norm = float(kp_data.get("y", 0.0))
            z_scaled = float(kp_data.get("z_scaled", 0.0))
            keypoints[kp_name] = {
                "x": x_norm,
                "y": y_norm,
                "z": z_scaled,  # Include z_scaled in unified format
                "confidence": float(kp_data.get("visibility", 0.0)),
            }
        
        save_unified_keypoints(
            unified_path,
            keypoints,
            image_name,
            entry.get("image_path", ""),
            entry.get("image_size", {}),
        )
        print(f"  Also wrote unified: {unified_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

