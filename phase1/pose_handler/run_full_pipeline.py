"""
Full pipeline: original image + OpenPose JSON → reformat OpenPose, (optionally) scale image,
run MediaPipe, average OpenPose + MediaPipe keypoints, extract angles.

Steps:
  1. Reformat OpenPose JSON to named keypoints (openpose_named.json).
  2. Optionally scale the image to --scale_width x --scale_height.
  3. Run MediaPipe on the (possibly scaled) image.
  4. Scale MediaPipe Z (world meters → normalized units).
  5. Scale OpenPose keypoints to match MediaPipe image size (if image was scaled).
  6. Average MediaPipe and OpenPose keypoints → averaged_keypoints.json.
  7. Extract shoulder + elbow angles → averaged_angles.json.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

import cv2
import mediapipe as mp

from pose_handler.common_io import as_posix_str, ensure_dir, read_json, write_json
from pose_handler.keypoints_schema import FORMAT_MEDIAPIPE_POSE_V1
from pose_handler.unified_keypoints_io import save_unified_keypoints


def _run_cmd(cmd: list, step_name: str) -> None:
    """Run a subprocess from project root; raise with stderr on non-zero exit."""
    project_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"{step_name} failed (exit {result.returncode}): {err}")


def _reformat_openpose(in_json: Path, out_dir: Path, out_filename: str = "openpose_named.json") -> Path:
    """Run openpose_json_to_named and return path to the written JSON."""
    out_path = out_dir / out_filename
    cmd = [
        sys.executable,
        "-m",
        "pose_handler.openpose_json_to_named",
        "--in_json",
        str(in_json.resolve()),
        "--out_dir",
        str(out_dir.resolve()),
        "--out_json",
        str(out_path.resolve()),
    ]
    _run_cmd(cmd, "openpose_json_to_named")
    return out_path


def _extract_mediapipe_from_image(image_path: Path, out_dir: Path) -> Dict:
    """
    Run MediaPipe pose extraction on a single image in-process (avoids subprocess exit codes).
    Returns the single entry dict; also writes mediapipe_pose.json and unified JSON to out_dir.
    """
    from pose_handler.extract_mediapipe_from_folder import _extract_from_image

    stem = image_path.stem
    out_dir = out_dir.resolve()
    ensure_dir(out_dir)

    with mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    ) as pose:
        entry = _extract_from_image(image_path, pose)

    payload = {"format": FORMAT_MEDIAPIPE_POSE_V1, "entries": [entry]}
    mediapipe_path = out_dir / f"{stem}_mediapipe_pose.json"
    write_json(mediapipe_path, payload)

    if entry.get("detected", False):
        keypoints = {}
        for kp_name, lm in entry.get("landmarks", {}).items():
            if not lm:
                keypoints[kp_name] = None
                continue
            keypoints[kp_name] = {
                "x": float(lm["normalized"]["x"]),
                "y": float(lm["normalized"]["y"]),
                "z": float(lm["world_3d"]["z"]),
                "confidence": float(lm.get("visibility", 0.0)),
            }
        unified_path = out_dir / f"{stem}_unified.json"
        save_unified_keypoints(
            unified_path,
            keypoints,
            entry.get("image_name", "unknown"),
            as_posix_str(image_path),
            entry.get("image_size", {}),
        )

    return entry


def _scale_keypoints_to_size(
    keypoints: Dict,
    from_width: int,
    from_height: int,
    to_width: int,
    to_height: int,
) -> Dict:
    """Scale keypoint pixel coordinates from one image size to another."""
    if from_width <= 0 or from_height <= 0:
        return keypoints
    sx = to_width / from_width
    sy = to_height / from_height
    out: Dict = {}
    for name, kp in keypoints.items():
        if kp is None:
            out[name] = None
            continue
        out[name] = {
            "x": float(kp.get("x", 0.0)) * sx,
            "y": float(kp.get("y", 0.0)) * sy,
            "confidence": float(kp.get("confidence", 0.0)),
        }
        if "z" in kp:
            out[name]["z"] = float(kp["z"])
        if "z_scaled" in kp:
            out[name]["z_scaled"] = float(kp["z_scaled"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Full pipeline: image + OpenPose JSON → reformat, (optional) scale, MediaPipe, average, angles.",
    )
    ap.add_argument("--image", required=True, help="Path to the original image.")
    ap.add_argument("--openpose_json", required=True, help="Path to OpenPose JSON (people[].pose_keypoints_2d).")
    ap.add_argument(
        "--out_dir",
        default="pose_handler_out",
        help="Output base directory. A subfolder named after the image (stem) is created here; all outputs are saved inside it.",
    )
    ap.add_argument(
        "--scale_width",
        type=int,
        default=None,
        help="If set with --scale_height, scale the image to this width before MediaPipe.",
    )
    ap.add_argument(
        "--scale_height",
        type=int,
        default=None,
        help="If set with --scale_width, scale the image to this height before MediaPipe.",
    )
    ap.add_argument(
        "--person_index",
        type=int,
        default=0,
        help="Which person to use from OpenPose people[] (default: 0).",
    )
    ap.add_argument(
        "--min_conf",
        type=float,
        default=0.3,
        help="Minimum confidence for averaging keypoints (default: 0.3).",
    )
    ap.add_argument(
        "--keep_temp",
        action="store_true",
        help="Keep temporary directory used for MediaPipe input.",
    )
    args = ap.parse_args()

    image_path = Path(args.image).resolve()
    openpose_path = Path(args.openpose_json).resolve()
    out_base = Path(args.out_dir).resolve()
    # Create subfolder named after the image (stem) inside out_dir; save everything there
    run_dir = ensure_dir(out_base / image_path.stem)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not openpose_path.is_file():
        raise FileNotFoundError(f"OpenPose JSON not found: {openpose_path}")

    # Optional scaling
    scale_width = args.scale_width
    scale_height = args.scale_height
    if (scale_width is not None) != (scale_height is not None):
        raise ValueError("Provide both --scale_width and --scale_height, or neither.")
    do_scale = scale_width is not None and scale_height is not None

    # Load image and get original size
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    orig_h, orig_w = img.shape[:2]

    # 1) Reformat OpenPose JSON to named keypoints
    print("Step 1: Reformat OpenPose JSON to named keypoints...")
    openpose_named_path = _reformat_openpose(openpose_path, run_dir, "openpose_named.json")
    openpose_data = read_json(openpose_named_path)
    openpose_keypoints = openpose_data.get("keypoints", {})

    # 2) Prepare image for MediaPipe (optionally scaled)
    if do_scale:
        img_for_mp = cv2.resize(img, (scale_width, scale_height), interpolation=cv2.INTER_LINEAR)
        mp_w, mp_h = scale_width, scale_height
        print(f"Step 2: Scaled image to {mp_w}x{mp_h}.")
    else:
        img_for_mp = img
        mp_w, mp_h = orig_w, orig_h
        print("Step 2: Using original image size (no scaling).")

    # Write image to a temp dir for MediaPipe input
    temp_dir = Path(tempfile.mkdtemp(prefix="pose_pipeline_", dir=run_dir)).resolve()
    try:
        pipeline_image_name = "pipeline_input.png"
        pipeline_image_path = temp_dir / pipeline_image_name
        cv2.imwrite(str(pipeline_image_path), img_for_mp)
        if not pipeline_image_path.is_file():
            raise RuntimeError(f"Failed to write image to {pipeline_image_path}")

        # 3) Run MediaPipe on the (possibly scaled) image (in-process to avoid subprocess exit codes)
        print("Step 3: Extracting MediaPipe pose...")
        _extract_mediapipe_from_image(pipeline_image_path, run_dir)
        stem = pipeline_image_path.stem  # pipeline_input
        mediapipe_json = run_dir / f"{stem}_mediapipe_pose.json"
        if not mediapipe_json.is_file():
            raise RuntimeError(f"MediaPipe output not found: {mediapipe_json}")

        # 4) Scale MediaPipe Z
        print("Step 4: Scaling MediaPipe Z...")
        _run_cmd(
            [
                sys.executable,
                "-m",
                "pose_handler.scale_mediapipe_z",
                "--in_json",
                str(mediapipe_json),
                "--out_dir",
                str(run_dir),
            ],
            "scale_mediapipe_z",
        )
        mediapipe_scaled_json = run_dir / f"{stem}_mediapipe_pose_scale.json"
        if not mediapipe_scaled_json.is_file():
            raise RuntimeError(f"Scaled MediaPipe output not found: {mediapipe_scaled_json}")

        # 5) Scale OpenPose keypoints to MediaPipe image size (if we scaled the image)
        openpose_for_average_path = run_dir / "openpose_named_for_average.json"
        if do_scale and (orig_w != mp_w or orig_h != mp_h):
            print("Step 5: Scaling OpenPose keypoints to MediaPipe image size...")
            scaled_kps = _scale_keypoints_to_size(
                openpose_keypoints,
                orig_w,
                orig_h,
                mp_w,
                mp_h,
            )
            save_unified_keypoints(
                openpose_for_average_path,
                scaled_kps,
                pipeline_image_name,
                str(pipeline_image_path),
                {"width": mp_w, "height": mp_h},
            )
        else:
            # Same size: copy openpose_named and add image_size so averager can normalize
            openpose_data["image_size"] = {"width": mp_w, "height": mp_h}
            openpose_data["image_name"] = pipeline_image_name
            write_json(openpose_for_average_path, openpose_data)

        # 6) Average MediaPipe and OpenPose keypoints
        print("Step 6: Averaging MediaPipe and OpenPose keypoints...")
        _run_cmd(
            [
                sys.executable,
                "-m",
                "pose_handler.average_mediapipe_openpose",
                "--mediapipe_json",
                str(mediapipe_scaled_json),
                "--openpose_json",
                str(openpose_for_average_path),
                "--image_name",
                pipeline_image_name,
                "--out_dir",
                str(run_dir),
                "--min_conf",
                str(args.min_conf),
            ],
            "average_mediapipe_openpose",
        )
    finally:
        if not args.keep_temp and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    # 7) Extract angles from averaged keypoints
    print("Step 7: Extracting angles...")
    averaged_path = run_dir / "averaged_keypoints.json"
    angles_path = run_dir / "averaged_angles.json"
    _run_cmd(
        [
            sys.executable,
            "-m",
            "pose_handler.calc_angles",
            "--in_json",
            str(averaged_path),
            "--out_path",
            str(angles_path),
        ],
        "calc_angles",
    )

    print("Done.")
    print(f"  Output folder:     {run_dir}")
    print(f"  Averaged keypoints: {averaged_path}")
    print(f"  Angles:            {angles_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
