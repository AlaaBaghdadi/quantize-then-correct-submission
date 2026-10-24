from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

from pose_handler.common_io import ensure_dir, list_images
from pose_handler.openpose18_mapping import OPENPOSE18_CONNECTIONS, OPENPOSE18_KEYPOINT_NAMES
from pose_handler.unified_keypoints_io import load_unified_keypoints


def _draw_openpose18(
    img_bgr,
    keypoints: List[Optional[dict]],
    min_conf: float,
) -> None:
    # connections
    for a, b in OPENPOSE18_CONNECTIONS:
        ka = keypoints[a] if a < len(keypoints) else None
        kb = keypoints[b] if b < len(keypoints) else None
        if ka is None or kb is None:
            continue
        if float(ka.get("confidence", 0.0)) < min_conf or float(kb.get("confidence", 0.0)) < min_conf:
            continue
        p1 = (int(round(float(ka["x"]))), int(round(float(ka["y"]))))
        p2 = (int(round(float(kb["x"]))), int(round(float(kb["y"]))))
        cv2.line(img_bgr, p1, p2, (0, 255, 0), 2, cv2.LINE_AA)

    # keypoints
    for kp in keypoints:
        if kp is None:
            continue
        if float(kp.get("confidence", 0.0)) < min_conf:
            continue
        p = (int(round(float(kp["x"]))), int(round(float(kp["y"]))))
        cv2.circle(img_bgr, p, 4, (255, 0, 0), -1, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser(description="Draw 2D pose overlay (OpenPose18) on all images in a folder.")
    ap.add_argument("--input_dir", required=True, help="Folder containing images.")
    ap.add_argument(
        "--keypoints_json",
        default="pose_handler_out/openpose18.json",
        help="Either pose_handler_out/openpose18.json or pose_handler_out/mediapipe_pose.json",
    )
    ap.add_argument("--out_dir", default="pose_handler_out", help="Output directory (will create vis_2d/ inside).")
    ap.add_argument("--min_conf", type=float, default=0.3, help="Minimum confidence for drawing.")
    args = ap.parse_args()

    images = list_images(args.input_dir)
    
    # Build lookup by image_name - load unified format for each image
    lookup: Dict[str, List[Optional[dict]]] = {}
    
    # Convert unified keypoints to OpenPose18 format for drawing
    def _unified_to_openpose18(named: Dict[str, Optional[dict]]) -> List[Optional[dict]]:
        """Convert unified format keypoints to OpenPose18 index list."""
        kps: List[Optional[dict]] = [None] * 18
        name_to_idx = {
            name.lower().replace(" ", "_"): idx
            for idx, name in enumerate(OPENPOSE18_KEYPOINT_NAMES)
        }
        for kp_name, kp_data in named.items():
            if kp_data is None:
                continue
            idx = name_to_idx.get(kp_name)
            if idx is not None:
                kps[idx] = {
                    "x": float(kp_data.get("x", 0.0)),
                    "y": float(kp_data.get("y", 0.0)),
                    "confidence": float(kp_data.get("confidence", 0.0)),
                }
        return kps
    
    # Try to load as unified format (works for any format)
    for img_path in images:
        try:
            unified_data = load_unified_keypoints(args.keypoints_json, image_name=img_path.name)
            keypoints_named = unified_data.get("keypoints", {})
            lookup[img_path.name] = _unified_to_openpose18(keypoints_named)
        except Exception as e:
            # If loading fails for this image, skip it
            print(f"Warning: Could not load keypoints for {img_path.name}: {e}")
            continue

    out_root = ensure_dir(args.out_dir)
    out_vis = ensure_dir(out_root / "vis_2d")

    min_conf = float(args.min_conf)
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        kps = lookup.get(img_path.name)
        if kps:
            _draw_openpose18(img, kps, min_conf=min_conf)
        out_path = out_vis / f"{img_path.stem}_vis2d.png"
        cv2.imwrite(str(out_path), img)

    print(f"Wrote overlays to: {out_vis}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

