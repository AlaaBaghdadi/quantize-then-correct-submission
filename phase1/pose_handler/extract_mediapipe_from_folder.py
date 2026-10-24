from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import cv2
import mediapipe as mp

from pose_handler.common_io import as_posix_str, ensure_dir, list_images, write_json
from pose_handler.keypoints_schema import (
    FORMAT_MEDIAPIPE_POSE_V1,
    MEDIAPIPE_POSE_INDEX_BY_NAME,
)


def _lm_to_point(lm) -> Dict[str, float]:
    return {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)}


def _extract_from_image(
    image_path: Path,
    pose: mp.solutions.pose.Pose,
) -> Dict:
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    entry = {
        "image_name": image_path.name,
        "image_path": as_posix_str(image_path),
        "image_size": {"width": int(w), "height": int(h)},
        "detected": bool(results.pose_landmarks and results.pose_world_landmarks),
        "landmarks": {},
    }

    if not entry["detected"]:
        return entry

    pose_landmarks = results.pose_landmarks.landmark
    world_landmarks = results.pose_world_landmarks.landmark

    for name, idx in MEDIAPIPE_POSE_INDEX_BY_NAME.items():
        lm = pose_landmarks[idx]
        wlm = world_landmarks[idx]
        entry["landmarks"][name] = {
            "visibility": float(lm.visibility),
            "normalized": _lm_to_point(lm),
            "world_3d": _lm_to_point(wlm),
            "pixel": {"x": float(lm.x) * float(w), "y": float(lm.y) * float(h)},
        }

    return entry


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract MediaPipe Pose keypoints from all images in a folder.")
    ap.add_argument("--input_dir", required=True, help="Folder containing images.")
    ap.add_argument("--out_dir", default="pose_handler_out", help="Output directory.")
    ap.add_argument("--model_complexity", type=int, default=2, choices=[0, 1, 2], help="MediaPipe pose model complexity.")
    ap.add_argument("--min_detection_confidence", type=float, default=0.5)
    args = ap.parse_args()

    images = list_images(args.input_dir)
    out_dir = ensure_dir(args.out_dir)

    mp_pose = mp.solutions.pose
    entries = []

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=int(args.model_complexity),
        enable_segmentation=False,
        min_detection_confidence=float(args.min_detection_confidence),
    ) as pose:
        for img_path in images:
            entries.append(_extract_from_image(img_path, pose))

    payload = {"format": FORMAT_MEDIAPIPE_POSE_V1, "entries": entries}

    # Name the MediaPipe JSON with the image stem when there's only one image,
    # otherwise keep the original aggregate filename for compatibility.
    if len(images) == 1:
        single_stem = Path(images[0]).stem
        mediapipe_filename = f"{single_stem}_mediapipe_pose.json"
    else:
        mediapipe_filename = "mediapipe_pose.json"

    out_path = out_dir / mediapipe_filename
    write_json(out_path, payload)
    print(f"Wrote: {out_path}")
    
    # Also write unified format for each entry
    from pose_handler.unified_keypoints_io import save_unified_keypoints
    
    for entry in entries:
        if not entry.get("detected", False):
            continue
        image_name = entry.get("image_name", "unknown")
        unified_path = out_dir / f"{Path(image_name).stem}_unified.json"
        
        # Convert to unified format
        keypoints: Dict[str, Optional[Dict[str, float]]] = {}
        for kp_name, lm in entry.get("landmarks", {}).items():
            if not lm:
                keypoints[kp_name] = None
                continue
            # Use normalized (x, y) and world_3d z in the unified format
            normalized = lm.get("normalized", {})
            world_3d = lm.get("world_3d", {})
            keypoints[kp_name] = {
                "x": float(normalized.get("x", 0.0)),
                "y": float(normalized.get("y", 0.0)),
                "z": float(world_3d.get("z", 0.0)),
                "confidence": float(lm.get("visibility", 0.0)),
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

