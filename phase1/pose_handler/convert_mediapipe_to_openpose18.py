from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pose_handler.common_io import ensure_dir, read_json, write_json
from pose_handler.keypoints_schema import FORMAT_MEDIAPIPE_POSE_V1, FORMAT_OPENPOSE18_V1
from pose_handler.openpose18_mapping import (
    OPENPOSE18_CONNECTIONS,
    OPENPOSE18_FROM_MEDIAPIPE,
    OPENPOSE18_KEYPOINT_NAMES,
)


def _neck_from_shoulders(ls: Dict, rs: Dict) -> Tuple[float, float, float]:
    # use pixel coords and visibility
    lx, ly = float(ls["pixel"]["x"]), float(ls["pixel"]["y"])
    rx, ry = float(rs["pixel"]["x"]), float(rs["pixel"]["y"])
    x = (lx + rx) / 2.0
    y = (ly + ry) / 2.0
    c = (float(ls.get("visibility", 0.0)) + float(rs.get("visibility", 0.0))) / 2.0
    return x, y, c


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert MediaPipe pose JSON to OpenPose COCO-18 JSON.")
    ap.add_argument("--in_json", default="pose_handler_out/mediapipe_pose.json", help="Input JSON from extract step.")
    ap.add_argument("--out_dir", default="pose_handler_out", help="Output directory.")
    ap.add_argument("--min_conf", type=float, default=0.0, help="If >0, low-confidence keypoints become null.")
    args = ap.parse_args()

    in_path = Path(args.in_json)
    data = read_json(in_path)
    if data.get("format") != FORMAT_MEDIAPIPE_POSE_V1:
        raise ValueError(f"Unexpected format in {in_path}: {data.get('format')}")

    out_dir = ensure_dir(args.out_dir)
    out_entries = []
    min_conf = float(args.min_conf)

    for entry in data.get("entries", []):
        img_size = entry.get("image_size", {"width": 0, "height": 0})
        detected = bool(entry.get("detected", False))
        lm = entry.get("landmarks", {}) if detected else {}

        keypoints: List[Optional[Dict[str, float]]] = [None] * 18

        # Fill direct mappings
        for op_idx, mp_idx in OPENPOSE18_FROM_MEDIAPIPE.items():
            if mp_idx is None:
                continue
            mp_name = None
            # landmarks dict keys are human names, not indices; map via known names
            # We can reverse by checking known keys that match these indices.
            # Since our extract step includes the exact names we need, use those.
            # indices used here correspond to specific names:
            idx_to_name = {
                0: "nose",
                2: "left_eye",
                5: "right_eye",
                7: "left_ear",
                8: "right_ear",
                11: "left_shoulder",
                12: "right_shoulder",
                13: "left_elbow",
                14: "right_elbow",
                15: "left_wrist",
                16: "right_wrist",
                23: "left_hip",
                24: "right_hip",
                25: "left_knee",
                26: "right_knee",
                27: "left_ankle",
                28: "right_ankle",
            }
            mp_name = idx_to_name.get(mp_idx)
            if not mp_name or mp_name not in lm:
                continue
            rec = lm[mp_name]
            conf = float(rec.get("visibility", 0.0))
            if conf < min_conf:
                continue
            keypoints[op_idx] = {
                "x": float(rec["pixel"]["x"]),
                "y": float(rec["pixel"]["y"]),
                "confidence": conf,
            }

        # Neck (index 1) from shoulders
        if "left_shoulder" in lm and "right_shoulder" in lm:
            x, y, c = _neck_from_shoulders(lm["left_shoulder"], lm["right_shoulder"])
            if c >= min_conf:
                keypoints[1] = {"x": float(x), "y": float(y), "confidence": float(c)}

        else:
            # Unified format
            img_size = entry.get("image_size", {"width": 0, "height": 0})
            keypoints_dict = entry.get("keypoints", {})
            keypoints: List[Optional[Dict[str, float]]] = [None] * 18

            for mp_idx, op_idx in MEDIAPIPE_TO_OPENPOSE18.items():
                mp_name = None
                for name, idx in keypoints_schema.MEDIAPIPE_POSE_INDEX_BY_NAME.items():
                    if idx == mp_idx:
                        mp_name = name
                        break
                if not mp_name or mp_name not in keypoints_dict or keypoints_dict[mp_name] is None:
                    continue
                rec = keypoints_dict[mp_name]
                vis = float(rec.get("confidence", 0.0))
                if vis < min_conf:
                    continue
                keypoints[op_idx] = {
                    "x": float(rec.get("x", 0.0)),
                    "y": float(rec.get("y", 0.0)),
                    "confidence": vis,
                }

            # Neck (index 1) from shoulders
            if "left_shoulder" in keypoints_dict and "right_shoulder" in keypoints_dict:
                ls = keypoints_dict["left_shoulder"]
                rs = keypoints_dict["right_shoulder"]
                if ls is not None and rs is not None:
                    x, y, c = _neck_from_shoulders(ls, rs)
                    if c >= min_conf:
                        keypoints[1] = {"x": float(x), "y": float(y), "confidence": float(c)}

        out_entries.append(
            {
                "image_name": entry.get("image_name", ""),
                "image_path": entry.get("image_path", ""),
                "image_size": img_size,
                "keypoints": keypoints,
            }
        )

    out = {
        "format": FORMAT_OPENPOSE18_V1,
        "keypoint_names": OPENPOSE18_KEYPOINT_NAMES,
        "connections": [[int(a), int(b)] for a, b in OPENPOSE18_CONNECTIONS],
        "entries": out_entries,
    }

    out_path = out_dir / "openpose18.json"
    write_json(out_path, out)
    print(f"Wrote: {out_path}")
    
    # Also write unified format for each entry
    from pose_handler.unified_keypoints_io import save_unified_keypoints
    
    for entry in out_entries:
        image_name = entry.get("image_name", "unknown")
        unified_path = out_dir / f"{Path(image_name).stem}_unified.json"
        
        # Convert OpenPose18 list format to unified named format
        keypoints_list = entry.get("keypoints", [])
        keypoints: Dict[str, Optional[Dict[str, float]]] = {}
        for idx, name in enumerate(OPENPOSE18_KEYPOINT_NAMES):
            kp_name = name.lower().replace(" ", "_")
            if idx < len(keypoints_list) and keypoints_list[idx] is not None:
                kp = keypoints_list[idx]
                keypoints[kp_name] = {
                    "x": float(kp.get("x", 0.0)),
                    "y": float(kp.get("y", 0.0)),
                    "confidence": float(kp.get("confidence", 0.0)),
                }
            else:
                keypoints[kp_name] = None
        
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

