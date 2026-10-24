from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

from pose_handler.common_io import ensure_dir, read_json, write_json


def _get_mediapipe_normalized_coords(entry: Dict, keypoint_name: str) -> Optional[Dict[str, float]]:
    """Extract **normalized** (0-1) coordinates from MediaPipe entry."""
    # Scaled format - already normalized coordinates (0-1)
    if "keypoints" in entry:
        kp = entry.get("keypoints", {}).get(keypoint_name)
        if not kp:
            return None
        return {
            "x": float(kp.get("x", 0.0)),
            "y": float(kp.get("y", 0.0)),
            "confidence": float(kp.get("visibility", 0.0)),
        }
    # Regular format - use the 'normalized' field written by extract_mediapipe_from_folder
    elif "landmarks" in entry:
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
    """Extract **normalized** (0-1) coordinates from OpenPose named keypoints."""
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
    """
    Get z-axis value from MediaPipe entry.
    Preference:
    - If scaled format: use z_scaled (normalized units), fallback to z_world.
    - If regular format: use world_3d.z (meters).
    """
    if "keypoints" in entry:
        kp = entry.get("keypoints", {}).get(keypoint_name)
        if not kp:
            return None
        if "z_scaled" in kp:
            return float(kp.get("z_scaled", 0.0))
        if "z_world" in kp:
            return float(kp.get("z_world", 0.0))
        return None
    elif "landmarks" in entry:
        lm = entry.get("landmarks", {}).get(keypoint_name)
        if not lm:
            return None
        world = lm.get("world_3d", {})
        return float(world.get("z", 0.0))
    return None

def _average_keypoints(
    mp_coords: Optional[Dict[str, float]],
    op_coords: Optional[Dict[str, float]],
    min_conf: float = 0.0,
) -> Optional[Dict[str, float]]:
    """Calculate average of two keypoints. Returns None if both are missing or low confidence."""
    if mp_coords is None and op_coords is None:
        return None
    
    # If only one exists and it's above threshold, use it
    if mp_coords is None:
        if op_coords and float(op_coords.get("confidence", 0.0)) >= min_conf:
            return op_coords.copy()
        return None
    
    if op_coords is None:
        if mp_coords and float(mp_coords.get("confidence", 0.0)) >= min_conf:
            return mp_coords.copy()
        return None
    
    # Both exist - average them
    mp_conf = float(mp_coords.get("confidence", 0.0))
    op_conf = float(op_coords.get("confidence", 0.0))
    
    # If both are below threshold, skip
    if mp_conf < min_conf and op_conf < min_conf:
        return None
    
    # Weighted average based on confidence (or simple average if both above threshold)
    if mp_conf >= min_conf and op_conf >= min_conf:
        # Both valid - simple average
        avg_x = (float(mp_coords["x"]) + float(op_coords["x"])) / 2.0
        avg_y = (float(mp_coords["y"]) + float(op_coords["y"])) / 2.0
        avg_conf = (mp_conf + op_conf) / 2.0
    elif mp_conf >= min_conf:
        # Only MediaPipe valid
        avg_x = float(mp_coords["x"])
        avg_y = float(mp_coords["y"])
        avg_conf = mp_conf
    else:
        # Only OpenPose valid
        avg_x = float(op_coords["x"])
        avg_y = float(op_coords["y"])
        avg_conf = op_conf
    
    return {
        "x": avg_x,
        "y": avg_y,
        "confidence": avg_conf,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Average MediaPipe and OpenPose keypoints to create a combined result."
    )
    ap.add_argument(
        "--mediapipe_json",
        required=True,
        help="Path to mediapipe_pose.json or mediapipe_pose_scaled.json",
    )
    ap.add_argument(
        "--openpose_json",
        required=True,
        help="Path to openpose_named.json",
    )
    ap.add_argument("--out_dir", default="pose_handler_out", help="Output directory.")
    ap.add_argument(
        "--out_json",
        default=None,
        help="Optional output filename (default: averaged_keypoints.json inside out_dir).",
    )
    ap.add_argument(
        "--image_name",
        default=None,
        help="Image name to process (if MediaPipe JSON has multiple entries). If None, uses first entry.",
    )
    ap.add_argument(
        "--min_conf",
        type=float,
        default=0.3,
        help="Minimum confidence/visibility threshold for including keypoints.",
    )
    args = ap.parse_args()

    # Read MediaPipe JSON
    mp_data = read_json(args.mediapipe_json)
    mp_format = mp_data.get("format", "")
    
    if mp_format == "pose_handler/mediapipe_pose_v1" or mp_format == "pose_handler/mediapipe_pose_scaled_v1":
        entries = mp_data.get("entries", [])
        if not entries:
            raise ValueError("MediaPipe JSON has no entries.")
        
        # Find the entry for the specified image
        if args.image_name:
            entry = None
            for e in entries:
                if e.get("image_name") == args.image_name:
                    entry = e
                    break
            if not entry:
                raise ValueError(f"Image '{args.image_name}' not found in MediaPipe JSON.")
        else:
            entry = entries[0]
            print(f"Using first entry: {entry.get('image_name', 'unknown')}")
    else:
        raise ValueError(f"Unsupported MediaPipe format: {mp_format}")

    # Read OpenPose JSON (named or already-unified named format)
    op_data = read_json(args.openpose_json)
    op_fmt = op_data.get("format")
    if op_fmt not in ("pose_handler/openpose_named_v1", "pose_handler/unified_keypoints_v1"):
        raise ValueError(f"Unsupported OpenPose format: {op_fmt}")
    
    op_keypoints = op_data.get("keypoints", {})

    # Get all unique keypoint names from both sources
    mp_keypoint_names = set()
    if "keypoints" in entry:
        mp_keypoint_names = set(entry.get("keypoints", {}).keys())
    elif "landmarks" in entry:
        mp_keypoint_names = set(entry.get("landmarks", {}).keys())
    
    op_keypoint_names = set(op_keypoints.keys())
    all_keypoint_names = sorted(mp_keypoint_names | op_keypoint_names)

    # Average each keypoint in **normalized space** (0-1)
    averaged_norm: Dict[str, Optional[Dict[str, float]]] = {}
    image_size = entry.get("image_size", {})
    for kp_name in all_keypoint_names:
        mp_coords = _get_mediapipe_normalized_coords(entry, kp_name)
        op_coords = _get_openpose_normalized_coords(op_keypoints, kp_name, image_size)
        averaged_norm[kp_name] = _average_keypoints(mp_coords, op_coords, float(args.min_conf))

    # Neck: MediaPipe has no neck; average MediaPipe shoulder midpoint with OpenPose neck
    mp_ls = _get_mediapipe_normalized_coords(entry, "left_shoulder")
    mp_rs = _get_mediapipe_normalized_coords(entry, "right_shoulder")
    op_neck = _get_openpose_normalized_coords(op_keypoints, "neck", image_size)
    if mp_ls is not None and mp_rs is not None:
        mp_mid = {
            "x": (float(mp_ls["x"]) + float(mp_rs["x"])) / 2.0,
            "y": (float(mp_ls["y"]) + float(mp_rs["y"])) / 2.0,
            "confidence": (float(mp_ls.get("confidence", 0.0)) + float(mp_rs.get("confidence", 0.0))) / 2.0,
        }
        averaged_norm["neck"] = _average_keypoints(mp_mid, op_neck, float(args.min_conf))
    elif op_neck is not None and float(op_neck.get("confidence", 0.0)) >= float(args.min_conf):
        averaged_norm["neck"] = op_neck.copy()

    # Get image info
    image_name = entry.get("image_name", "unknown")
    image_path = entry.get("image_path", "")
    image_size = entry.get("image_size", {})

    # Convert back to **pixel** coordinates for unified output and attach z from MediaPipe
    width = float(image_size.get("width", 0))
    height = float(image_size.get("height", 0))
    averaged_pixels: Dict[str, Optional[Dict[str, float]]] = {}
    for name, kp in averaged_norm.items():
        if kp is None or width == 0 or height == 0:
            averaged_pixels[name] = None
            continue
        z_val = _get_mediapipe_z(entry, name)
        if name == "neck" and z_val is None:
            z_l = _get_mediapipe_z(entry, "left_shoulder")
            z_r = _get_mediapipe_z(entry, "right_shoulder")
            z_val = (z_l + z_r) / 2.0 if (z_l is not None and z_r is not None) else (z_l if z_l is not None else z_r)
        averaged_pixels[name] = {
            "x": float(kp["x"]) * width,
            "y": float(kp["y"]) * height,
            "z": z_val if z_val is not None else 0.0,
            "confidence": float(kp.get("confidence", 0.0)),
        }

    # Write output in unified format (pixel coordinates)
    from pose_handler.unified_keypoints_io import save_unified_keypoints
    
    out_dir = ensure_dir(args.out_dir)
    out_path = Path(args.out_json) if args.out_json else out_dir / "averaged_keypoints.json"
    
    save_unified_keypoints(
        out_path,
        averaged_pixels,
        image_name,
        image_path,
        image_size,
    )
    
    # Also add source info
    data = read_json(out_path)
    data["source_mediapipe"] = str(Path(args.mediapipe_json).as_posix())
    data["source_openpose"] = str(Path(args.openpose_json).as_posix())
    write_json(out_path, data)
    
    print(f"Wrote averaged keypoints to: {out_path}")
    
    # Print summary
    valid_count = sum(1 for kp in averaged_pixels.values() if kp is not None)
    print(f"Valid keypoints: {valid_count}/{len(all_keypoint_names)}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
