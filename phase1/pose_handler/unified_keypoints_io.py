from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from pose_handler.common_io import read_json, write_json
from pose_handler.keypoints_schema import (
    FORMAT_MEDIAPIPE_POSE_V1,
    FORMAT_MEDIAPIPE_POSE_SCALED_V1,
    FORMAT_OPENPOSE18_V1,
    FORMAT_UNIFIED_KEYPOINTS_V1,
    ImageSize,
)


def _convert_mediapipe_to_unified(entry: Dict, image_name: Optional[str] = None) -> Dict:
    """Convert MediaPipe entry to unified format."""
    image_name = image_name or entry.get("image_name", "unknown")
    image_path = entry.get("image_path", "")
    image_size = entry.get("image_size", {})
    
    keypoints: Dict[str, Optional[Dict[str, float]]] = {}
    
    # Check if it's scaled format or regular format
    if "keypoints" in entry:
        # Scaled format - normalized coordinates, convert to pixels
        width = float(image_size.get("width", 0))
        height = float(image_size.get("height", 0))
        for kp_name, kp_data in entry.get("keypoints", {}).items():
            if not kp_data:
                keypoints[kp_name] = None
                continue
            x_norm = float(kp_data.get("x", 0.0))
            y_norm = float(kp_data.get("y", 0.0))
            kp_dict = {
                "x": x_norm * width,
                "y": y_norm * height,
                "confidence": float(kp_data.get("visibility", 0.0)),
            }
            # Preserve z_scaled if available (for 3D visualization)
            if "z_scaled" in kp_data:
                kp_dict["z_scaled"] = float(kp_data.get("z_scaled", 0.0))
            keypoints[kp_name] = kp_dict
    elif "landmarks" in entry:
        # Regular format - use pixel coordinates
        for kp_name, lm in entry.get("landmarks", {}).items():
            if not lm:
                keypoints[kp_name] = None
                continue
            pixel = lm.get("pixel", {})
            keypoints[kp_name] = {
                "x": float(pixel.get("x", 0.0)),
                "y": float(pixel.get("y", 0.0)),
                "confidence": float(lm.get("visibility", 0.0)),
            }
    
    return {
        "format": FORMAT_UNIFIED_KEYPOINTS_V1,
        "image_name": image_name,
        "image_path": image_path,
        "image_size": image_size,
        "keypoints": keypoints,
    }


def _convert_openpose18_to_unified(entry: Dict, image_name: Optional[str] = None) -> Dict:
    """Convert OpenPose18 entry to unified format."""
    from pose_handler.openpose18_mapping import OPENPOSE18_KEYPOINT_NAMES
    
    image_name = image_name or entry.get("image_name", "unknown")
    image_path = entry.get("image_path", "")
    image_size = entry.get("image_size", {})
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
    
    return {
        "format": FORMAT_UNIFIED_KEYPOINTS_V1,
        "image_name": image_name,
        "image_path": image_path,
        "image_size": image_size,
        "keypoints": keypoints,
    }


def load_unified_keypoints(json_path: Path | str, image_name: Optional[str] = None) -> Dict:
    """
    Load any keypoint JSON file and convert it to unified format.
    Supports: mediapipe_pose_v1, mediapipe_pose_scaled_v1, openpose18_v1, openpose_named_v1, averaged_keypoints_v1, unified_keypoints_v1
    """
    data = read_json(json_path)
    fmt = data.get("format", "")
    
    if fmt == FORMAT_UNIFIED_KEYPOINTS_V1:
        # Already unified
        return data
    
    if fmt == "pose_handler/openpose_named_v1" or fmt == "pose_handler/averaged_keypoints_v1":
        # Already in unified-like format, just ensure format field is correct
        result = data.copy()
        result["format"] = FORMAT_UNIFIED_KEYPOINTS_V1
        return result
    
    if fmt == FORMAT_MEDIAPIPE_POSE_V1 or fmt == FORMAT_MEDIAPIPE_POSE_SCALED_V1:
        entries = data.get("entries", [])
        if not entries:
            raise ValueError(f"No entries found in {json_path}")
        
        if image_name:
            entry = None
            for e in entries:
                if e.get("image_name") == image_name:
                    entry = e
                    break
            if not entry:
                raise ValueError(f"Image '{image_name}' not found in {json_path}")
        else:
            entry = entries[0]
        
        # For scaled format, we need to convert normalized to pixel coordinates
        if fmt == FORMAT_MEDIAPIPE_POSE_SCALED_V1:
            image_size = entry.get("image_size", {})
            width = float(image_size.get("width", 0))
            height = float(image_size.get("height", 0))
            
            keypoints: Dict[str, Optional[Dict[str, float]]] = {}
            for kp_name, kp_data in entry.get("keypoints", {}).items():
                if not kp_data:
                    keypoints[kp_name] = None
                    continue
                x_norm = float(kp_data.get("x", 0.0))
                y_norm = float(kp_data.get("y", 0.0))
                keypoints[kp_name] = {
                    "x": x_norm * width,
                    "y": y_norm * height,
                    "confidence": float(kp_data.get("visibility", 0.0)),
                    "z_scaled": float(kp_data.get("z_scaled", 0.0)),  # Keep z_scaled for 3D visualization
                }
            
            return {
                "format": FORMAT_UNIFIED_KEYPOINTS_V1,
                "image_name": entry.get("image_name", "unknown"),
                "image_path": entry.get("image_path", ""),
                "image_size": image_size,
                "keypoints": keypoints,
            }
        else:
            return _convert_mediapipe_to_unified(entry)
    
    if fmt == FORMAT_OPENPOSE18_V1:
        entries = data.get("entries", [])
        if not entries:
            raise ValueError(f"No entries found in {json_path}")
        
        if image_name:
            entry = None
            for e in entries:
                if e.get("image_name") == image_name:
                    entry = e
                    break
            if not entry:
                raise ValueError(f"Image '{image_name}' not found in {json_path}")
        else:
            entry = entries[0]
        
        return _convert_openpose18_to_unified(entry)
    
    raise ValueError(f"Unsupported format: {fmt}")


def save_unified_keypoints(
    json_path: Path | str,
    keypoints: Dict[str, Optional[Dict[str, float]]],
    image_name: str,
    image_path: str = "",
    image_size: Optional[ImageSize] = None,
) -> None:
    """Save keypoints in unified format."""
    payload = {
        "format": FORMAT_UNIFIED_KEYPOINTS_V1,
        "image_name": image_name,
        "image_path": image_path,
        "image_size": image_size or {},
        "keypoints": keypoints,
    }
    write_json(json_path, payload)
