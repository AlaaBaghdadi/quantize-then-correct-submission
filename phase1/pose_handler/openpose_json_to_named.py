from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pose_handler.common_io import ensure_dir, read_json, write_json
from pose_handler.openpose18_mapping import OPENPOSE18_KEYPOINT_NAMES


# OpenPose BODY_25 to COCO-18 mapping (BODY_25 has 25 keypoints, we extract 18)
# Based on OpenPose documentation: BODY_25 indices that map to COCO-18
BODY25_TO_COCO18: Dict[int, int] = {
    0: 0,   # Nose
    1: 1,   # Neck
    2: 2,   # Right Shoulder
    3: 3,   # Right Elbow
    4: 4,   # Right Wrist
    5: 5,   # Left Shoulder
    6: 6,   # Left Elbow
    7: 7,   # Left Wrist
    9: 8,   # Right Hip (BODY_25 index 9 -> COCO-18 index 8)
    10: 9,  # Right Knee
    11: 10, # Right Ankle
    12: 11, # Left Hip
    13: 12, # Left Knee
    14: 13, # Left Ankle
    15: 14, # Right Eye
    16: 15, # Left Eye
    17: 16, # Right Ear
    18: 17, # Left Ear
}

# OpenPose COCO-18 index -> MediaPipe-style snake_case name
COCO18_TO_NAME: Dict[int, str] = {
    idx: name.lower().replace(" ", "_") for idx, name in enumerate(OPENPOSE18_KEYPOINT_NAMES)
}


def _parse_pose_keypoints_2d(flat: List[float]) -> List[Optional[Dict[str, float]]]:
    """
    OpenPose format: flat list [x0,y0,c0,x1,y1,c1,...] for 18 keypoints.
    """
    out: List[Optional[Dict[str, float]]] = []
    n = len(flat) // 3
    for i in range(n):
        x = float(flat[i * 3 + 0])
        y = float(flat[i * 3 + 1])
        c = float(flat[i * 3 + 2])
        # OpenPose can output 0,0,0 for missing
        if c <= 0.0:
            out.append(None)
        else:
            out.append({"x": x, "y": y, "confidence": c})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert an OpenPose JSON (people[].pose_keypoints_2d) into a clean, named keypoints JSON."
    )
    ap.add_argument("--in_json", required=True, help="Path to OpenPose JSON (COCO-18 pose_keypoints_2d).")
    ap.add_argument("--out_dir", default="pose_handler_out", help="Output directory.")
    ap.add_argument("--person_index", type=int, default=0, help="Which person to use from `people[]`.")
    ap.add_argument("--min_conf", type=float, default=0.0, help="If >0, low-confidence keypoints become null.")
    ap.add_argument(
        "--out_json",
        default=None,
        help="Optional output filename (default: openpose_named.json inside out_dir).",
    )
    args = ap.parse_args()

    in_path = Path(args.in_json)
    data = read_json(in_path)

    people = data.get("people", [])
    if not people or args.person_index >= len(people):
        raise ValueError("No people found in OpenPose JSON (or person_index out of range).")

    person = people[int(args.person_index)]
    flat = person.get("pose_keypoints_2d", [])
    if not isinstance(flat, list) or len(flat) < 3:
        raise ValueError("pose_keypoints_2d is missing or invalid.")

    num_kps = len(flat) // 3
    kps = _parse_pose_keypoints_2d(flat)
    min_conf = float(args.min_conf)

    # Detect format: BODY_25 (25 keypoints) or COCO-18 (18 keypoints)
    is_body25 = num_kps >= 25
    if is_body25:
        print(f"Detected BODY_25 format ({num_kps} keypoints), converting to COCO-18...")
    else:
        print(f"Detected COCO-18 format ({num_kps} keypoints)")

    named: Dict[str, Optional[Dict[str, float]]] = {}
    for coco18_idx, name in COCO18_TO_NAME.items():
        if is_body25:
            # Map from BODY_25 index to COCO-18
            body25_idx = None
            for b25, c18 in BODY25_TO_COCO18.items():
                if c18 == coco18_idx:
                    body25_idx = b25
                    break
            if body25_idx is None or body25_idx >= len(kps):
                named[name] = None
                continue
            kp = kps[body25_idx]
        else:
            # Direct COCO-18 mapping
            kp = kps[coco18_idx] if coco18_idx < len(kps) else None
        
        if kp is None:
            named[name] = None
            continue
        if float(kp.get("confidence", 0.0)) < min_conf:
            named[name] = None
            continue
        named[name] = kp

    out_dir = ensure_dir(args.out_dir)
    out_path = Path(args.out_json) if args.out_json else out_dir / "openpose_named.json"

    from pose_handler.unified_keypoints_io import save_unified_keypoints
    from pose_handler.keypoints_schema import FORMAT_UNIFIED_KEYPOINTS_V1
    
    # Write in unified format
    save_unified_keypoints(
        out_path,
        named,
        Path(args.in_json).stem,
        str(in_path.as_posix()),
        {},  # image_size not available from OpenPose JSON
    )
    
    # Also add source info
    data = read_json(out_path)
    data["source"] = str(in_path.as_posix())
    write_json(out_path, data)
    
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

