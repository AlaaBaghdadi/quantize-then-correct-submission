"""
Upper-body joint angles from 3D MediaPipe landmarks (NumPy implementation).

Implements hip-based body frame and shoulder (Pitch, Yaw, Roll) plus elbow angle.
No visualization; calculation only. Uses atan2-based formulas where possible
for full 360° coverage and robustness to floating-point errors.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from pose_handler.common_io import ensure_dir, read_json, write_json


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TINY = 1e-10   # Minimum norm to avoid division by zero
EPS = 1e-6     # Threshold for near-zero vectors (use fallback axis)


# ---------------------------------------------------------------------------
# Body frame and angle calculation (MATLAB logic, NumPy, robust)
# ---------------------------------------------------------------------------

def _norm(v: np.ndarray) -> np.ndarray:
    """Normalize vector; return zero vector if too small."""
    n = np.linalg.norm(v)
    if n < TINY:
        return np.zeros_like(v)
    return v / n


def _angle_between_atan2_degrees(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Angle between two vectors in degrees [0, 180] using atan2.
    Avoids NaN when dot product slightly exceeds 1.0 due to floating-point error.
    """
    v1n = _norm(v1)
    v2n = _norm(v2)
    if np.linalg.norm(v1n) < TINY or np.linalg.norm(v2n) < TINY:
        return 0.0
    cross_norm = np.linalg.norm(np.cross(v1n, v2n))
    dot_val = np.dot(v1n, v2n)
    return float(np.degrees(np.arctan2(cross_norm, dot_val)))


def body_frame_orthonormal(
    LS: np.ndarray,
    RS: np.ndarray,
    LH: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build hip-based body reference frame (Xb, Yb, Zb) with rigid orthonormality.

    Step 1 — Raw axes (right-handed):
      Zb = (LS - LH) / norm(LS - LH)   # Up from hip
      Yb = cross(Zb, RS - LS); Yb /= norm(Yb)   # Forward
      Xb = cross(Yb, Zb); Xb /= norm(Xb)        # Lateral (right)

    Step 2 — Double-cross orthorectification so the frame is perfectly rigid:
      Xb = cross(Yb, Zb); Xb /= norm(Xb)
      Yb = cross(Zb, Xb); Yb /= norm(Yb)
      Zb = cross(Xb, Yb); Zb /= norm(Zb)
    This guarantees Xb, Yb, Zb are orthonormal and right-handed.

    Returns:
        Xb: Lateral (red), Yb: Forward (green), Zb: Up from hip (blue)
    """
    Zb = _norm(LS - LH)
    if np.linalg.norm(Zb) < TINY:
        Zb = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    Xb_raw = RS - LS
    Yb = np.cross(Zb, Xb_raw)
    Yb = _norm(Yb)
    if np.linalg.norm(Yb) < TINY:
        Yb = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        Yb = _norm(Yb - np.dot(Yb, Zb) * Zb)
        if np.linalg.norm(Yb) < TINY:
            Yb = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    Xb = np.cross(Yb, Zb)
    Xb = _norm(Xb)
    if np.linalg.norm(Xb) < TINY:
        Xb = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    # Double-cross orthorectification: rebuild each axis from the other two
    # so the frame is perfectly orthonormal (no drift).
    Xb = np.cross(Yb, Zb)
    Xb = _norm(Xb)
    Yb = np.cross(Zb, Xb)
    Yb = _norm(Yb)
    Zb = np.cross(Xb, Yb)
    Zb = _norm(Zb)

    return Xb, Yb, Zb


def calc_angles_from_landmarks(
    landmarks: Dict[str, np.ndarray],
) -> Dict[str, Union[Dict[str, float], float]]:
    """
    Compute shoulder (Pitch, Yaw, Roll) and elbow angles from 3D landmarks.

    Input:
        landmarks: Dict with keys LS, RS, LE, RE, LW, RW, LH.
                   Each value is a 3-element array (x, y, z) in MediaPipe world units.

    Output:
        Structured dict:
          - left_shoulder / right_shoulder: { "pitch", "yaw", "roll" } in degrees
          - left_elbow / right_elbow: { "angle" } in degrees (180° = straight, 0° = fully closed)
    """
    LS = np.asarray(landmarks["LS"], dtype=np.float64).ravel()
    RS = np.asarray(landmarks["RS"], dtype=np.float64).ravel()
    LE = np.asarray(landmarks["LE"], dtype=np.float64).ravel()
    RE = np.asarray(landmarks["RE"], dtype=np.float64).ravel()
    LW = np.asarray(landmarks["LW"], dtype=np.float64).ravel()
    RW = np.asarray(landmarks["RW"], dtype=np.float64).ravel()
    LH = np.asarray(landmarks["LH"], dtype=np.float64).ravel()

    Xb, Yb, Zb = body_frame_orthonormal(LS, RS, LH)

    # Upper arm and forearm unit vectors
    UL = LE - LS
    UR = RE - RS
    ULn = _norm(UL)
    URn = _norm(UR)

    FL = LW - LE
    FR = RW - RE
    FLn = _norm(FL)
    FRn = _norm(FR)

    # ---- Shoulder Pitch (angle with -Zb), robust atan2 ----
    neg_Zb = -Zb
    pitch_L = _angle_between_atan2_degrees(ULn, neg_Zb)
    pitch_R = _angle_between_atan2_degrees(URn, neg_Zb)

    # ---- Shoulder Yaw: angle in plane ⊥ Zb from Yb (0° = forward), Xb = ±90° (lateral) ----
    # atan2( cross(Yb, U_proj)·Zb, Yb·U_proj ) so Forward (Yb) = 0°, Lateral (Xb) = ±90°
    def project_onto_plane(v: np.ndarray, n: np.ndarray) -> np.ndarray:
        return v - np.dot(v, n) * n

    UL_proj = project_onto_plane(ULn, Zb)
    UR_proj = project_onto_plane(URn, Zb)
    ul_proj_norm = np.linalg.norm(UL_proj)
    ur_proj_norm = np.linalg.norm(UR_proj)

    if ul_proj_norm > EPS:
        UL_proj = UL_proj / ul_proj_norm
        yaw_L = float(np.degrees(np.arctan2(
            np.dot(np.cross(Yb, UL_proj), Zb),
            np.dot(Yb, UL_proj),
        )))
    else:
        yaw_L = 0.0

    if ur_proj_norm > EPS:
        UR_proj = UR_proj / ur_proj_norm
        yaw_R = float(np.degrees(np.arctan2(
            np.dot(np.cross(Yb, UR_proj), Zb),
            np.dot(Yb, UR_proj),
        )))
    else:
        yaw_R = 0.0

    # ---- Shoulder Roll (normal of arm plane vs reference normal) ----
    N_L = np.cross(ULn, FLn)
    N_R = np.cross(URn, FRn)
    n_l_norm = np.linalg.norm(N_L)
    n_r_norm = np.linalg.norm(N_R)
    if n_l_norm > TINY:
        N_L = N_L / n_l_norm
    else:
        N_L = Xb.copy()
    if n_r_norm > TINY:
        N_R = N_R / n_r_norm
    else:
        N_R = Xb.copy()

    Nref_L = np.cross(-Zb, ULn)
    Nref_R = np.cross(-Zb, URn)
    nref_l_norm = np.linalg.norm(Nref_L)
    nref_r_norm = np.linalg.norm(Nref_R)
    if nref_l_norm < EPS:
        Nref_L = Xb.copy()
        nref_l_norm = 1.0
    else:
        Nref_L = Nref_L / nref_l_norm
    if nref_r_norm < EPS:
        Nref_R = Xb.copy()
        nref_r_norm = 1.0
    else:
        Nref_R = Nref_R / nref_r_norm

    roll_L = float(np.degrees(np.arctan2(
        np.dot(np.cross(Nref_L, N_L), ULn),
        np.dot(Nref_L, N_L),
    )))
    roll_R = float(np.degrees(np.arctan2(
        np.dot(np.cross(Nref_R, N_R), URn),
        np.dot(Nref_R, N_R),
    )))

    # ---- Elbow angle (atan2-based, 180° = straight, 0° = fully closed) ----
    v_es_L = LS - LE
    v_ew_L = LW - LE
    v_es_R = RS - RE
    v_ew_R = RW - RE
    elbow_L = _angle_between_atan2_degrees(v_es_L, v_ew_L)
    elbow_R = _angle_between_atan2_degrees(v_es_R, v_ew_R)

    def r2(x: float) -> float:
        return round(float(x), 2)

    return {
        "left_shoulder": {
            "pitch": r2(pitch_L),
            "yaw": r2(yaw_L),
            "roll": r2(roll_L),
        },
        "right_shoulder": {
            "pitch": r2(pitch_R),
            "yaw": r2(yaw_R),
            "roll": r2(roll_R),
        },
        "left_elbow": {"angle": r2(elbow_L)},
        "right_elbow": {"angle": r2(elbow_R)},
    }


def calc_angles_flat(landmarks: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Same as calc_angles_from_landmarks but returns a flat dict suitable for JSON/logging:
    left_shoulder_pitch, left_shoulder_yaw, left_shoulder_roll,
    right_shoulder_pitch, right_shoulder_yaw, right_shoulder_roll,
    left_elbow_angle, right_elbow_angle.
    """
    out = calc_angles_from_landmarks(landmarks)
    flat = {}
    for side in ("left_shoulder", "right_shoulder"):
        for key in ("pitch", "yaw", "roll"):
            flat[f"{side}_{key}"] = out[side][key]
    flat["left_elbow_angle"] = out["left_elbow"]["angle"]
    flat["right_elbow_angle"] = out["right_elbow"]["angle"]
    return flat


# ---------------------------------------------------------------------------
# MediaPipe keypoint conversion and batch entry point
# ---------------------------------------------------------------------------

def keypoints_to_arrays(
    kps: Dict,
    image_size: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Build landmark dict from keypoints (MediaPipe-style or unified).
    Keys: x, y, z_scaled or z. Returns dict with LS, RS, LE, RE, LW, RW, LH
    (and optionally RH). Returns None if any required keypoint is missing.
    """
    scale_x = 1.0
    scale_y = 1.0
    if image_size:
        w = image_size.get("width")
        h = image_size.get("height")
        if w and w > 0 and h and h > 0:
            scale_x = 1.0 / float(w)
            scale_y = 1.0 / float(h)

    def get3(name: str) -> Optional[np.ndarray]:
        if name not in kps or kps[name] is None:
            return None
        rec = kps[name]
        x = float(rec.get("x", 0.0)) * scale_x
        y = float(rec.get("y", 0.0)) * scale_y
        z = float(rec.get("z_scaled", rec.get("z", 0.0)))
        return np.array([x, y, z], dtype=np.float64).ravel()

    ls = get3("left_shoulder")
    rs = get3("right_shoulder")
    le = get3("left_elbow")
    re = get3("right_elbow")
    lw = get3("left_wrist")
    rw = get3("right_wrist")
    lh = get3("left_hip")
    if any(v is None for v in [ls, rs, le, re, lw, rw, lh]):
        return None
    result = {
        "LS": ls,
        "RS": rs,
        "LE": le,
        "RE": re,
        "LW": lw,
        "RW": rw,
        "LH": lh,
    }
    rh = get3("right_hip")
    if rh is not None:
        result["RH"] = rh
    return result


def angles_from_keypoints(
    kps: Dict,
    image_size: Optional[Dict[str, float]] = None,
    flat: bool = False,
) -> Optional[Union[Dict[str, Dict[str, float]], Dict[str, float]]]:
    """
    Compute angles from a keypoints dict (e.g. MediaPipe). Returns None if
    keypoints are incomplete. If flat=True, returns flat dict of angle names to values.
    """
    arr = keypoints_to_arrays(kps, image_size=image_size)
    if arr is None:
        return None
    if flat:
        return calc_angles_flat(arr)
    return calc_angles_from_landmarks(arr)


def _angles_from_arrays(arr: Optional[Dict[str, np.ndarray]]) -> Optional[Dict[str, float]]:
    """Compute flat angles from landmark arrays; returns None if arr is None."""
    if arr is None:
        return None
    return calc_angles_flat(arr)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compute shoulder (Pitch/Yaw/Roll) and elbow angles from keypoints JSON.",
    )
    ap.add_argument("--in_json", required=True, help="Input keypoints JSON.")
    ap.add_argument("--out_dir", default=None, help="Output directory (default: same as input).")
    ap.add_argument("--out_path", default=None, help="Explicit output path (overrides --out_dir).")
    args = ap.parse_args()

    data = read_json(args.in_json)
    in_path = Path(args.in_json)

    if args.out_path:
        out_path = Path(args.out_path)
        ensure_dir(out_path.parent)
    elif args.out_dir:
        out_root = ensure_dir(args.out_dir)
        out_path = out_root / "angles.json"
    else:
        out_path = in_path.parent / "angles.json"

    if "entries" in data:
        results = []
        for entry in data.get("entries", []):
            detected = bool(entry.get("detected", False))
            kps = entry.get("keypoints", {}) if detected else {}
            arr = keypoints_to_arrays(kps) if kps else None
            angles = _angles_from_arrays(arr)
            results.append({
                "image_name": entry.get("image_name", ""),
                "image_path": entry.get("image_path", ""),
                "detected": detected,
                "scale_factor": float(entry.get("scale_factor", 1.0)),
                "angles": angles,
            })
        write_json(out_path, {"format": "pose_handler/calc_angles_v2", "entries": results})
    else:
        kps = data.get("keypoints", {})
        image_size = data.get("image_size")
        arr = keypoints_to_arrays(kps, image_size=image_size) if kps else None
        angles = _angles_from_arrays(arr)
        write_json(out_path, {
            "format": "pose_handler/calc_angles_v2",
            "image_name": data.get("image_name", ""),
            "image_path": data.get("image_path", ""),
            "angles": angles,
        })

    print(f"Wrote: {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Compatibility: wrapper for code that expects (LS, RS, ...) and calculate_angles()
# ---------------------------------------------------------------------------

class JointVisualizer:
    """
    Compatibility wrapper: accepts LS, RS, LE, RE, LW, RW, LH (and optional RH).
    calculate_angles() returns a flat dict: left_shoulder_pitch, left_shoulder_yaw,
    left_shoulder_roll, left_elbow_angle, right_shoulder_*, right_elbow_angle.
    """

    def __init__(
        self,
        LS: np.ndarray,
        RS: np.ndarray,
        LE: np.ndarray,
        RE: np.ndarray,
        LW: np.ndarray,
        RW: np.ndarray,
        LH: np.ndarray,
        RH: Optional[np.ndarray] = None,
        neck: Optional[np.ndarray] = None,
    ):
        self._landmarks = {
            "LS": np.asarray(LS, dtype=np.float64).ravel(),
            "RS": np.asarray(RS, dtype=np.float64).ravel(),
            "LE": np.asarray(LE, dtype=np.float64).ravel(),
            "RE": np.asarray(RE, dtype=np.float64).ravel(),
            "LW": np.asarray(LW, dtype=np.float64).ravel(),
            "RW": np.asarray(RW, dtype=np.float64).ravel(),
            "LH": np.asarray(LH, dtype=np.float64).ravel(),
        }

    def calculate_angles(self) -> Dict[str, float]:
        return calc_angles_flat(self._landmarks)


if __name__ == "__main__":
    raise SystemExit(main())
