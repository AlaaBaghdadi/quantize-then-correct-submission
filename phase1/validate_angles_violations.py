"""
Validate angle limits and (optionally) limb length drift against pipeline inputs.
Uses the same limits as run_video_correction_pipeline and pose_corrector.
Summarizes angle violations, limb length violations, and how many frames will
be passed to the pose corrector (angle violation OR length violation).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Reuse limits from correction pipeline
PITCH_MIN = 0.0
PITCH_MAX = 180.0
LEFT_YAW_MIN = -45.0
LEFT_YAW_MAX = 135.0
RIGHT_YAW_MIN = -135.0
RIGHT_YAW_MAX = 45.0

DEFAULT_LENGTH_DRIFT_THRESHOLD = 0.05   # 5%
DEFAULT_LENGTH_REF_ALPHA = 0.3

LIMB_KEYS = ("left_upper", "left_forearm", "right_upper", "right_forearm")
LIMB_JOINTS = (
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
)


def load_angles(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "frames" not in data:
        raise ValueError(f"Angles JSON must have 'frames': {path}")
    return data


def check_angle_violations(angles: dict) -> List[str]:
    """Return list of angle violation descriptions for this frame, or empty if none."""
    violations = []
    lp = angles.get("left_shoulder_pitch")
    ly = angles.get("left_shoulder_yaw")
    rp = angles.get("right_shoulder_pitch")
    ry = angles.get("right_shoulder_yaw")
    if lp is not None and (lp < PITCH_MIN or lp > PITCH_MAX):
        violations.append(f"left_shoulder_pitch={lp:.2f} (valid [{PITCH_MIN},{PITCH_MAX}])")
    if ly is not None and (ly < LEFT_YAW_MIN or ly > LEFT_YAW_MAX):
        violations.append(f"left_shoulder_yaw={ly:.2f} (valid [{LEFT_YAW_MIN},{LEFT_YAW_MAX}])")
    if rp is not None and (rp < PITCH_MIN or rp > PITCH_MAX):
        violations.append(f"right_shoulder_pitch={rp:.2f} (valid [{PITCH_MIN},{PITCH_MAX}])")
    if ry is not None and (ry < RIGHT_YAW_MIN or ry > RIGHT_YAW_MAX):
        violations.append(f"right_shoulder_yaw={ry:.2f} (valid [{RIGHT_YAW_MIN},{RIGHT_YAW_MAX}])")
    return violations


def limb_lengths_from_keypoints(keypoints: dict) -> Optional[Dict[str, float]]:
    """Compute four arm limb lengths; return None if any joint missing."""
    out: Dict[str, float] = {}
    for key, (j1, j2) in zip(LIMB_KEYS, LIMB_JOINTS):
        a = keypoints.get(j1)
        b = keypoints.get(j2)
        if not isinstance(a, dict) or not isinstance(b, dict):
            return None
        x1, y1, z1 = float(a.get("x", 0)), float(a.get("y", 0)), float(a.get("z", 0))
        x2, y2, z2 = float(b.get("x", 0)), float(b.get("y", 0)), float(b.get("z", 0))
        out[key] = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return out


def length_violates(
    ref: Dict[str, float],
    current: Dict[str, float],
    threshold: float,
) -> bool:
    """True if any limb length deviates from ref by more than threshold (fraction)."""
    for key in LIMB_KEYS:
        r = ref.get(key)
        c = current.get(key)
        if r is None or c is None or r < 1e-9:
            continue
        if abs(c - r) / r > threshold:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Validate angles (and optionally limb lengths) against pipeline limits. "
        "Reports how many frames will be passed to the pose corrector."
    )
    ap.add_argument("angles_json", type=Path, help="Path to angles JSON (e.g. kalman_out/man1_angles.json).")
    ap.add_argument("--keypoints_json", type=Path, default=None,
                    help="Path to smoothed keypoints JSON. If provided, limb length violations are computed and summary includes frames passed to corrector (angle OR length).")
    ap.add_argument("--length_drift_threshold", type=float, default=DEFAULT_LENGTH_DRIFT_THRESHOLD,
                    help=f"Limb length drift threshold as fraction of reference (default {DEFAULT_LENGTH_DRIFT_THRESHOLD}).")
    ap.add_argument("--length_ref_alpha", type=float, default=DEFAULT_LENGTH_REF_ALPHA,
                    help=f"EMA alpha for reference lengths (default {DEFAULT_LENGTH_REF_ALPHA}).")
    ap.add_argument("--quiet", action="store_true", help="Only print summary, no per-frame details.")
    args = ap.parse_args()

    path = args.angles_json.resolve()
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    data = load_angles(path)
    frames_angles = data.get("frames", [])
    n_frames = len(frames_angles)

    # ---- Angle violations ----
    angle_violated_frames: List[Tuple[int, List[str]]] = []
    for entry in frames_angles:
        idx = entry.get("frame_index")
        angles = entry.get("angles", {})
        v = check_angle_violations(angles)
        if v:
            angle_violated_frames.append((idx, v))
            if not args.quiet:
                print(f"Frame {idx} [angle]: {v}")

    angle_indices = [x[0] for x in angle_violated_frames]
    n_angle = len(angle_violated_frames)

    # ---- Limb length violations (if keypoints provided) ----
    length_violated_frames: List[int] = []
    corrector_frames: List[int] = []
    keypoints_frames: List[dict] = []

    if args.keypoints_json is not None:
        kp_path = args.keypoints_json.resolve()
        if not kp_path.is_file():
            print(f"Keypoints file not found: {kp_path}", file=sys.stderr)
            return 1
        with open(kp_path, "r", encoding="utf-8") as f:
            kp_data = json.load(f)
        keypoints_frames = kp_data.get("frames", [])
        if len(keypoints_frames) != n_frames:
            print(f"Warning: angles has {n_frames} frames, keypoints has {len(keypoints_frames)}. Using min.", file=sys.stderr)
        n_frames = min(n_frames, len(keypoints_frames))

        ref_lengths: Optional[Dict[str, float]] = None
        for i in range(n_frames):
            kps = keypoints_frames[i].get("keypoints", {})
            current_lengths = limb_lengths_from_keypoints(kps)
            length_violated = False
            if ref_lengths is not None and current_lengths is not None:
                length_violated = length_violates(ref_lengths, current_lengths, args.length_drift_threshold)
            elif current_lengths is not None:
                ref_lengths = dict(current_lengths)

            if length_violated:
                length_violated_frames.append(i)
                if not args.quiet:
                    print(f"Frame {i} [length]: drift > {args.length_drift_threshold:.0%} from reference")

            # Update ref from "output" lengths (we don't run corrector here, so use current)
            if current_lengths is not None and ref_lengths is not None:
                for k in LIMB_KEYS:
                    ref_lengths[k] = (1.0 - args.length_ref_alpha) * ref_lengths[k] + args.length_ref_alpha * current_lengths[k]

        corrector_frames = sorted(set(angle_indices) | set(length_violated_frames))
    else:
        corrector_frames = angle_indices

    n_length = len(length_violated_frames)
    n_corrector = len(corrector_frames)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total frames:                    {n_frames}")
    print(f"Angle violations:                 {n_angle}  (frames: {angle_indices if n_angle <= 20 else angle_indices[:20] + ['...']})")
    if args.keypoints_json is not None:
        print(f"Limb length violations:           {n_length}  (threshold={args.length_drift_threshold:.0%})")
        if not args.quiet and n_length > 0 and n_length <= 30:
            print(f"  Length-violated frame indices:  {length_violated_frames}")
    print(f"Frames passed to pose corrector:  {n_corrector}  (angle OR length)")
    if args.keypoints_json is not None and n_corrector <= 40:
        print(f"  Corrector frame indices:        {corrector_frames}")
    elif args.keypoints_json is None:
        print("  (Pass --keypoints_json for length-inclusive count.)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
