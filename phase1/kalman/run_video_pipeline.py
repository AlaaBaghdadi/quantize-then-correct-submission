"""
Full video pose pipeline: copy video to OpenPose, run OpenPose + MediaPipe,
scale and average frame-by-frame, Kalman smooth, compute angles, render video, cleanup.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import os
import cv2
import numpy as np

from pose_handler.common_io import ensure_dir, read_json
from pose_handler.openpose18_mapping import OPENPOSE18_CONNECTIONS, OPENPOSE18_KEYPOINT_NAMES

from shared_pose.angles.calc_angles import keypoints_to_arrays, _angles_from_arrays
from shared_pose.pose_io import (
    average_frame_keypoints,
    openpose_json_to_named_keypoints,
    scale_mediapipe_entries,
)

from kalman.config import get_default_openpose_root, get_default_out_dir, get_project_root
from kalman.keypoint_smoother import KeypointSmoother
from kalman.video_pose_io import (
    extract_mediapipe_from_video,
    list_openpose_frame_jsons,
    save_angles_json,
    save_smoothed_keypoints,
)


# COCO-18 index -> name for drawing
COCO18_TO_NAME = {
    i: name.lower().replace(" ", "_") for i, name in enumerate(OPENPOSE18_KEYPOINT_NAMES)
}


def _named_to_openpose18_list(
    named: Dict[str, Optional[Dict]],
    min_conf: float,
) -> List[Optional[Dict]]:
    """Convert named keypoints to list of 18 in COCO-18 order for drawing."""
    kps: List[Optional[Dict]] = [None] * 18
    for idx, name in COCO18_TO_NAME.items():
        kp = named.get(name)
        if not kp or float(kp.get("confidence", 0.0)) < min_conf:
            continue
        kps[idx] = {"x": float(kp["x"]), "y": float(kp["y"]), "confidence": float(kp.get("confidence", 0.0))}
    return kps


def _draw_skeleton_and_angles(
    img_bgr,
    keypoints: Dict[str, Optional[Dict]],
    angles: Optional[Dict[str, float]],
    min_conf: float = 0.3,
    white_background: bool = False,
) -> None:
    """Draw OpenPose18 skeleton and angle text overlay."""
    kps18 = _named_to_openpose18_list(keypoints, min_conf)

    # Bones
    for a, b in OPENPOSE18_CONNECTIONS:
        ka = kps18[a] if a < len(kps18) else None
        kb = kps18[b] if b < len(kps18) else None
        if ka is None or kb is None:
            continue
        if float(ka.get("confidence", 0.0)) < min_conf or float(kb.get("confidence", 0.0)) < min_conf:
            continue
        p1 = (int(round(float(ka["x"]))), int(round(float(ka["y"]))))
        p2 = (int(round(float(kb["x"]))), int(round(float(kb["y"]))))
        cv2.line(img_bgr, p1, p2, (0, 255, 0), 2, cv2.LINE_AA)

    # Joints
    for kp in kps18:
        if kp is None or float(kp.get("confidence", 0.0)) < min_conf:
            continue
        p = (int(round(float(kp["x"]))), int(round(float(kp["y"]))))
        cv2.circle(img_bgr, p, 4, (255, 0, 0), -1, cv2.LINE_AA)

    # Angles text panel (top-left) - Clinical Anatomical Angles
    if angles:
        text_color = (0, 0, 0) if white_background else (255, 255, 255)
        outline_color = (255, 255, 255) if white_background else (0, 0, 0)
        lines = [
            f"L shoulder: flex={angles.get('left_shoulder_flexion', 0):.0f} abd={angles.get('left_shoulder_abduction', 0):.0f} rot={angles.get('left_shoulder_rotation', 0):.0f}",
            f"R shoulder: flex={angles.get('right_shoulder_flexion', 0):.0f} abd={angles.get('right_shoulder_abduction', 0):.0f} rot={angles.get('right_shoulder_rotation', 0):.0f}",
            f"L elbow: {angles.get('left_elbow_flexion', 0):.0f}  R elbow: {angles.get('right_elbow_flexion', 0):.0f}",
        ]
        y_offset = 24
        for line in lines:
            cv2.putText(
                img_bgr, line, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, outline_color, 2, cv2.LINE_AA,
            )
            cv2.putText(
                img_bgr, line, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA,
            )
            y_offset += 22


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Video pose pipeline: OpenPose + MediaPipe, average, Kalman smooth, angles, output video.",
    )
    ap.add_argument("--video", required=True, help="Path to input video.")
    ap.add_argument("--out_dir", default=None, help="Output directory (default: kalman_out).")
    ap.add_argument("--openpose_root", default=None, help="OpenPose root (default: openpose-1.7.0.../openpose).")
    ap.add_argument("--keep_temp", action="store_true", help="Keep temporary files (video copy, OpenPose output, etc.).")
    ap.add_argument("--min_conf", type=float, default=0.3, help="Min confidence for averaging and drawing.")
    ap.add_argument("--person_index", type=int, default=0, help="OpenPose person index.")
    ap.add_argument(
        "--process_noise", type=float, default=1e-2,
        help="Kalman process noise Q (default 1e-2).",
    )
    ap.add_argument(
        "--measurement_noise", type=float, default=1e-1,
        help="Kalman measurement noise R (default 1e-1).",
    )
    args = ap.parse_args()

    project_root = get_project_root()
    video_path = Path(args.video).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else get_default_out_dir()
    openpose_root = Path(args.openpose_root).resolve() if args.openpose_root else get_default_openpose_root()

    ensure_dir(out_dir)
    video_stem = video_path.stem

    # ----- Step 1: Copy video to OpenPose media_videos -----
    media_videos = openpose_root / "media_videos"
    media_videos.mkdir(parents=True, exist_ok=True)
    video_copy = media_videos / video_path.name
    print(f"Step 1: Copying video to {video_copy}...")
    shutil.copy2(video_path, video_copy)

    # ----- Step 2: Run OpenPose on video -----
    # [REV-R1] Linux port: OPENPOSE_BIN points at build/examples/openpose/openpose.bin.
    # Headless flags (--display 0 --render_pose 0) do not change the keypoints.
    _env_bin = os.environ.get("OPENPOSE_BIN")
    openpose_exe = Path(_env_bin) if _env_bin else openpose_root / "bin" / "OpenPoseDemo.exe"
    if not openpose_exe.is_file():
        raise FileNotFoundError(f"OpenPose executable not found: {openpose_exe}")
    output_dir = openpose_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(openpose_exe),
        "--video", f"media_videos/{video_path.name}",
        "--write_json", "output/",
    ]
    if _env_bin:
        cmd += ["--display", "0", "--render_pose", "0"]
    print("Step 2: Running OpenPose...")
    result = subprocess.run(cmd, cwd=str(openpose_root), capture_output=True, text=True)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"OpenPose failed (exit {result.returncode}): {err}")

    # ----- Step 3: MediaPipe on video -----
    print("Step 3: Extracting MediaPipe poses from video...")
    mp_entries, fps, width, height = extract_mediapipe_from_video(video_path)
    image_size = {"width": width, "height": height}
    n_frames = len(mp_entries)
    if n_frames == 0:
        raise RuntimeError("No frames extracted from video.")

    # ----- Step 4: Scale MediaPipe Z -----
    print("Step 4: Scaling MediaPipe Z...")
    mp_scaled_entries = scale_mediapipe_entries(mp_entries)

    # ----- Step 5 & 6: Load OpenPose per frame, average with MediaPipe -----
    op_frame_list = list_openpose_frame_jsons(output_dir, video_stem)
    if len(op_frame_list) != n_frames:
        print(f"Warning: OpenPose produced {len(op_frame_list)} JSONs, MediaPipe {n_frames} frames. Aligning by frame index.")
    averaged_frames: List[Dict[str, Optional[Dict]]] = []
    op_by_index = {idx: path for idx, path in op_frame_list}
    for i in range(n_frames):
        mp_entry = mp_scaled_entries[i] if i < len(mp_scaled_entries) else None
        op_path = op_by_index.get(i)
        op_named = {}
        if op_path and op_path.is_file():
            op_data = read_json(op_path)
            op_named = openpose_json_to_named_keypoints(op_data, person_index=args.person_index, min_conf=args.min_conf)
        if mp_entry is None:
            averaged_frames.append(op_named)
            continue
        avg_kps = average_frame_keypoints(mp_entry, op_named, image_size, min_conf=args.min_conf)
        averaged_frames.append(avg_kps)

    # ----- Step 6b: Save fused (pre-Kalman) keypoints -----
    fused_frame_list = [{"keypoints": kps} for kps in averaged_frames]
    fused_path = out_dir / f"{video_stem}_fused_keypoints.json"
    save_smoothed_keypoints(fused_path, fused_frame_list, video_stem, fps, image_size)
    print(f"Saved fused keypoints: {fused_path}")

    # ----- Step 7: Kalman smooth -----
    keypoint_names = sorted(set())
    for kps in averaged_frames:
        keypoint_names = sorted(set(keypoint_names) | set(kps.keys()))
    if not keypoint_names:
        keypoint_names = [name.lower().replace(" ", "_") for name in OPENPOSE18_KEYPOINT_NAMES] + ["neck"]
    smoother = KeypointSmoother(
        keypoint_names=keypoint_names,
        dt=1.0,
        process_noise=args.process_noise,
        measurement_noise=args.measurement_noise,
        visibility_threshold=args.min_conf,
    )
    smoothed_frames: List[Dict] = []
    for kps in averaged_frames:
        smoothed_xy = smoother.update(kps)
        # Merge smoothed (x,y) with z and confidence from averaged
        merged: Dict[str, Optional[Dict]] = {}
        for name in keypoint_names:
            xy = smoothed_xy.get(name)
            orig = kps.get(name)
            if xy is None:
                merged[name] = None
                continue
            z = float(orig.get("z", 0.0)) if orig else 0.0
            z_scaled = float(orig.get("z_scaled", z)) if orig else z
            conf = float(orig.get("confidence", 0.0)) if orig else 0.0
            merged[name] = {
                "x": xy["x"],
                "y": xy["y"],
                "z": z,
                "z_scaled": z_scaled,
                "confidence": conf,
            }
        # Keep both smoothed and non-smoothed (averaged) keypoints per frame for comparison
        smoothed_frames.append({"keypoints": merged, "keypoints_averaged": kps})

    # ----- Step 8: Save smoothed keypoints, compute angles per frame -----
    smoothed_path = out_dir / f"{video_stem}_smoothed_keypoints.json"
    save_smoothed_keypoints(smoothed_path, smoothed_frames, video_stem, fps, image_size)
    print(f"Saved smoothed keypoints: {smoothed_path}")

    frame_angles_list: List[Dict] = []
    for i, frame_data in enumerate(smoothed_frames):
        kps = frame_data.get("keypoints", {})
        arr = keypoints_to_arrays(kps, image_size=image_size)
        angles = _angles_from_arrays(arr)
        frame_angles_list.append({"frame_index": i, "angles": angles})

    angles_path = out_dir / f"{video_stem}_angles.json"
    save_angles_json(angles_path, frame_angles_list, video_stem, fps)
    print(f"Saved angles: {angles_path}")

    # ----- Step 9: Output video with skeleton and angles (white background only) -----
    out_video_path = out_dir / f"{video_stem}_skeleton_angles.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))
    if not out_writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {out_video_path}")

    print("Step 9: Writing output video (white background, skeleton + angles)...")
    for i in range(n_frames):
        frame = np.full((height, width, 3), 255, dtype=np.uint8)
        kps = smoothed_frames[i].get("keypoints", {})
        angles = None
        if i < len(frame_angles_list) and frame_angles_list[i].get("angles"):
            angles = frame_angles_list[i]["angles"]
        _draw_skeleton_and_angles(frame, kps, angles, min_conf=args.min_conf, white_background=True)
        out_writer.write(frame)
    out_writer.release()
    print(f"Saved output video: {out_video_path}")

    # ----- Step 10: Cleanup -----
    if not args.keep_temp:
        print("Step 10: Cleaning up temporary files...")
        if video_copy.is_file():
            video_copy.unlink()
        for _, op_path in op_frame_list:
            if op_path.is_file():
                op_path.unlink()
        # Optionally remove OpenPose-generated video if present
        op_video = output_dir / f"{video_stem}.avi"
        if op_video.is_file():
            op_video.unlink(missing_ok=True)

    print("Done.")
    print(f"  Smoothed keypoints: {smoothed_path}")
    print(f"  Angles:            {angles_path}")
    print(f"  Output video:      {out_video_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
