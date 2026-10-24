"""
Comparison pipeline: load 3 keypoint sources (pipeline_input_unified, pipeline_input_unified_scale,
averaged_keypoints), optionally scale x,y for pixel-based keypoints using image_size,
compute angles from each via calc_angles, and save all results in one JSON for comparison.

Keypoint types:
  1. pipeline_input_unified: x,y typically normalized (0-1), z raw.
  2. pipeline_input_unified_scale: x,y normalized (0-1), z scaled.
  3. averaged_keypoints (e.g. t-pose): x,y in pixel coords, z scaled; image_size in file
     is used to normalize x,y (divide by width/height) for angle calculation.

All three use format pose_handler/unified_keypoints_v1.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from pose_handler.calc_angles import (
    _angles_from_arrays,
    keypoints_to_arrays,
)
from pose_handler.common_io import ensure_dir, read_json, write_json
from pose_handler.keypoints_schema import FORMAT_UNIFIED_KEYPOINTS_V1


def _is_normalized_xy(keypoints: Dict[str, Any]) -> bool:
    """Return True if x,y appear to be in [0,1] (normalized)."""
    if not keypoints:
        return True
    max_x = max(
        float(kp.get("x", 0.0))
        for kp in keypoints.values()
        if kp is not None
    )
    max_y = max(
        float(kp.get("y", 0.0))
        for kp in keypoints.values()
        if kp is not None
    )
    return max_x <= 1.5 and max_y <= 1.5


def _effective_image_size(
    data: Dict[str, Any],
    keypoints: Dict[str, Any],
) -> Optional[Dict[str, float]]:
    """
    Return image_size to use for keypoints_to_arrays.
    - If x,y are already normalized (0-1), use width=1, height=1 so scale is 1.
    - Else use file's image_size so pixel x,y get normalized by width/height.
    """
    if _is_normalized_xy(keypoints):
        return {"width": 1.0, "height": 1.0}
    im = data.get("image_size")
    if im and im.get("width") and im.get("height"):
        return {"width": float(im["width"]), "height": float(im["height"])}
    return None


def _compute_angles_for_file(json_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load unified keypoints JSON, compute angles, return dict with source, path,
    image_size_used, angles (or error message).
    """
    data = read_json(json_path)
    if data.get("format") != FORMAT_UNIFIED_KEYPOINTS_V1:
        return {
            "source": json_path.name,
            "path": str(json_path.resolve()),
            "error": f"Unsupported format: {data.get('format')}",
            "angles": None,
        }
    keypoints = data.get("keypoints", {})
    image_size = _effective_image_size(data, keypoints)
    arr = keypoints_to_arrays(keypoints, image_size=image_size)
    angles = _angles_from_arrays(arr)
    return {
        "source": json_path.name,
        "path": str(json_path.resolve()),
        "image_size_in_file": data.get("image_size"),
        "image_size_used_for_xy": image_size,
        "angles": angles,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare angles from 3 keypoint sources (unified, unified_scale, averaged_keypoints) "
        "and save one JSON for comparison.",
    )
    ap.add_argument(
        "--unified",
        default=None,
        help="Path to pipeline_input_unified.json (hands-up).",
    )
    ap.add_argument(
        "--unified_scale",
        default=None,
        help="Path to pipeline_input_unified_scale.json (hands-up).",
    )
    ap.add_argument(
        "--averaged",
        default=None,
        help="Path to averaged_keypoints.json (e.g. t-pose; x,y in pixels, scaled using file image_size).",
    )
    ap.add_argument(
        "--out_dir",
        default="pose_handler_out",
        help="Output base directory; comparison JSON is written here.",
    )
    ap.add_argument(
        "--out_file",
        default="angles_comparison.json",
        help="Output filename for the comparison JSON (default: angles_comparison.json).",
    )
    args = ap.parse_args()

    # Default paths relative to out_dir if not given
    out_base = Path(args.out_dir).resolve()
    ensure_dir(out_base)
    hands_up = out_base / "hands-up"
    t_pose = out_base / "t-pose"

    paths: List[Path] = []
    if args.unified:
        paths.append(Path(args.unified).resolve())
    elif (hands_up / "pipeline_input_unified.json").is_file():
        paths.append(hands_up / "pipeline_input_unified.json")
    else:
        paths.append(Path("pose_handler_out/hands-up/pipeline_input_unified.json").resolve())

    if args.unified_scale:
        paths.append(Path(args.unified_scale).resolve())
    elif (hands_up / "pipeline_input_unified_scale.json").is_file():
        paths.append(hands_up / "pipeline_input_unified_scale.json")
    else:
        paths.append(Path("pose_handler_out/hands-up/pipeline_input_unified_scale.json").resolve())

    if args.averaged:
        paths.append(Path(args.averaged).resolve())
    elif (t_pose / "averaged_keypoints.json").is_file():
        paths.append(t_pose / "averaged_keypoints.json")
    else:
        paths.append(Path("pose_handler_out/t-pose/averaged_keypoints.json").resolve())

    results: List[Dict[str, Any]] = []
    for p in paths:
        if not p.is_file():
            results.append({
                "source": p.name,
                "path": str(p),
                "error": "File not found",
                "angles": None,
            })
            continue
        results.append(_compute_angles_for_file(p))

    out_path = out_base / args.out_file
    payload = {
        "format": "pose_handler/angles_comparison_v1",
        "description": "Angles from pipeline_input_unified, pipeline_input_unified_scale, and averaged_keypoints.",
        "entries": results,
    }
    write_json(out_path, payload)
    print(f"Wrote comparison to: {out_path}")
    for r in results:
        src = r.get("source", "?")
        ang = r.get("angles")
        err = r.get("error")
        if err:
            print(f"  {src}: error — {err}")
        else:
            print(f"  {src}: angles computed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
