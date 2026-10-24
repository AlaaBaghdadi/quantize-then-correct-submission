from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from pose_handler.common_io import ensure_dir
from pose_handler.openpose18_mapping import OPENPOSE18_CONNECTIONS, OPENPOSE18_KEYPOINT_NAMES
from pose_handler.unified_keypoints_io import load_unified_keypoints


# OpenPose COCO-18 index -> name (must match `openpose_json_to_named.py` output keys)
# Convert "Right Shoulder" -> "right_shoulder", etc.
COCO18_TO_NAME: Dict[int, str] = {
    idx: name.lower().replace(" ", "_") for idx, name in enumerate(OPENPOSE18_KEYPOINT_NAMES)
}


def _named_to_openpose18(named: Dict[str, Optional[dict]], min_conf: float) -> List[Optional[dict]]:
    kps: List[Optional[dict]] = [None] * 18
    for idx, name in COCO18_TO_NAME.items():
        kp = named.get(name)
        if not kp:
            continue
        if float(kp.get("confidence", 0.0)) < min_conf:
            continue
        kps[idx] = {
            "x": float(kp["x"]),
            "y": float(kp["y"]),
            "confidence": float(kp["confidence"]),
        }
    return kps


def _draw_openpose18(img_bgr, keypoints: List[Optional[dict]], min_conf: float, show_labels: bool = True) -> None:
    # Draw connections (bones)
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

    # Draw keypoints with labels
    for idx, kp in enumerate(keypoints):
        if kp is None:
            continue
        if float(kp.get("confidence", 0.0)) < min_conf:
            continue
        p = (int(round(float(kp["x"]))), int(round(float(kp["y"]))))
        cv2.circle(img_bgr, p, 4, (255, 0, 0), -1, cv2.LINE_AA)
        
        # Add label
        if show_labels and idx < len(COCO18_TO_NAME):
            label = COCO18_TO_NAME[idx]
            # Position label slightly above the point
            label_pos = (p[0] + 8, p[1] - 8)
            # Draw background rectangle for better readability
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
            )
            cv2.rectangle(
                img_bgr,
                (label_pos[0] - 2, label_pos[1] - text_height - 2),
                (label_pos[0] + text_width + 2, label_pos[1] + baseline + 2),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                img_bgr,
                label,
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )


def main() -> int:
    ap = argparse.ArgumentParser(description="Overlay openpose_named.json keypoints on a single image.")
    ap.add_argument("--image", required=True, help="Path to the original image.")
    ap.add_argument("--named_json", required=True, help="Path to pose_handler_out/openpose_named.json.")
    ap.add_argument("--out_dir", default="pose_handler_out", help="Output directory.")
    ap.add_argument("--min_conf", type=float, default=0.3, help="Minimum confidence for drawing.")
    ap.add_argument("--no_labels", action="store_true", help="Don't show keypoint labels on the image.")
    args = ap.parse_args()

    img_path = Path(args.image)
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    # Load any keypoint format and convert to unified
    data = load_unified_keypoints(args.named_json, image_name=img_path.name)
    keypoints_named = data.get("keypoints", {})
    kps18 = _named_to_openpose18(keypoints_named, float(args.min_conf))
    _draw_openpose18(img, kps18, float(args.min_conf), show_labels=not args.no_labels)

    out_root = ensure_dir(args.out_dir)
    out_path = out_root / f"{img_path.stem}_openpose_named_vis2d.png"
    cv2.imwrite(str(out_path), img)
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

