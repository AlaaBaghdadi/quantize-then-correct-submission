"""
Simple 3D skeleton + angles visualization using matplotlib.

Loads averaged_keypoints.json and averaged_angles.json (or computes angles),
draws skeleton, joints, angle labels, and optional shoulder axes.
Rotate with mouse in the interactive window.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from pose_handler.calc_angles import JointVisualizer, keypoints_to_arrays
from pose_handler.common_io import ensure_dir, read_json

CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


def _pt(kps: Dict, name: str) -> Optional[np.ndarray]:
    if name not in kps or kps[name] is None:
        return None
    rec = kps[name]
    return np.array([
        float(rec.get("x", 0)),
        float(rec.get("y", 0)),
        float(rec.get("z_scaled", rec.get("z", 0))),
    ], dtype=np.float64)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-10 else v.copy()


def _shoulder_axes(shoulder: np.ndarray, other_shoulder: np.ndarray, mid_hip: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vertical (shoulder→hip), lateral (→other shoulder), forward."""
    vert = _normalize(mid_hip - shoulder)
    lat = _normalize(other_shoulder - shoulder)
    fwd = np.cross(lat, vert)
    return vert, lat, fwd / (np.linalg.norm(fwd) + 1e-10)


def plot_skeleton_3d(
    kps: Dict,
    angles: Optional[Dict] = None,
    draw_axes: bool = True,
    title: str = "Skeleton + angles",
    image_size: Optional[Dict] = None,
) -> None:
    pts = {n: _pt(kps, n) for n in [
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ] if _pt(kps, n) is not None}
    pts = {k: v for k, v in pts.items() if v is not None}
    if not pts:
        raise ValueError("No keypoints found")

    if angles is None:
        arr = keypoints_to_arrays(kps, image_size=image_size)
        if arr is not None:
            viz = JointVisualizer(
                arr["LS"], arr["RS"], arr["LE"], arr["RE"],
                arr["LW"], arr["RW"], arr["LH"], arr.get("RH"),
                neck=arr.get("neck"),
            )
            angles = viz.calculate_angles()
        else:
            angles = {}

    xyz = np.array(list(pts.values()))
    center = xyz.mean(axis=0)
    span = float(np.ptp(xyz)) * 0.55 or 1.0
    axis_len = span * 0.12

    fig = plt.figure(figsize=(9, 7))
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    # Skeleton
    for a, b in CONNECTIONS:
        if a in pts and b in pts:
            p, q = pts[a], pts[b]
            ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], "b-", lw=2, alpha=0.8)

    # Shoulder axes (green=→hip, red=→other sh, blue=forward)
    if draw_axes and "left_hip" in pts and "right_hip" in pts:
        mid_hip = (pts["left_hip"] + pts["right_hip"]) / 2
        for sh_name, other_name in [("left_shoulder", "right_shoulder"), ("right_shoulder", "left_shoulder")]:
            if sh_name not in pts or other_name not in pts:
                continue
            sh = pts[sh_name]
            v, l, f = _shoulder_axes(sh, pts[other_name], mid_hip)
            for vec, color in [(v, "green"), (l, "red"), (f, "blue")]:
                end = sh + axis_len * vec
                ax.plot([sh[0], end[0]], [sh[1], end[1]], [sh[2], end[2]], color=color, lw=3)

    # Joints
    xs, ys, zs = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    ax.scatter(xs, ys, zs, s=25, c="orange", edgecolors="black", linewidths=0.5)

    # Angle labels (black text) - Pitch / Yaw / Roll (shoulder), Angle (elbow)
    dy = -span * 0.03
    for key in ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow"]:
        if key not in pts:
            continue
        p = pts[key]
        if "shoulder" in key:
            side = "left" if "left" in key else "right"
            pitch = angles.get(f"{side}_shoulder_pitch", 0)
            yaw = angles.get(f"{side}_shoulder_yaw", 0)
            roll = angles.get(f"{side}_shoulder_roll", 0)
            text = f"pitch={pitch:.0f}° yaw={yaw:.0f}° roll={roll:.0f}°"
        else:
            side = "left" if "left" in key else "right"
            a = angles.get(f"{side}_elbow_angle", 0)
            text = f"{a:.0f}°"
        ax.text(p[0], p[1] + dy, p[2], text, fontsize=9, color="black")

    ax.set_xlabel("x", color="black")
    ax.set_ylabel("y", color="black")
    ax.set_zlabel("z", color="black")
    ax.set_title(title, color="black")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("white")
        axis.pane.set_alpha(0.9)
    ax.tick_params(colors="black")
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    return fig, ax


def main() -> int:
    ap = argparse.ArgumentParser(description="3D skeleton + angles (matplotlib). Rotate with mouse.")
    ap.add_argument("--keypoints", default="pose_handler_out/averaged_keypoints.json")
    ap.add_argument("--angles", default="pose_handler_out/averaged_angles.json")
    ap.add_argument("--out_dir", default="pose_handler_out")
    ap.add_argument("--no_axes", action="store_true", help="Hide shoulder axes")
    ap.add_argument("--show", action="store_true", help="Show interactive window")
    args = ap.parse_args()

    kp_path = Path(args.keypoints)
    if not kp_path.exists():
        print(f"Not found: {kp_path}")
        return 1
    data = read_json(kp_path)
    kps = data.get("keypoints", {})
    if not kps:
        print("No keypoints")
        return 1

    angles = None
    if Path(args.angles).exists():
        angles = read_json(args.angles).get("angles")
    title = data.get("image_name", "Skeleton + angles")
    image_size = data.get("image_size")  # e.g. side2/averaged_keypoints.json: scale x,y by width/height

    fig, ax = plot_skeleton_3d(kps, angles, draw_axes=not args.no_axes, title=title, image_size=image_size)

    out_dir = ensure_dir(Path(args.out_dir) / "vis_3d")
    out_png = out_dir / "skeleton_angles_3d.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_png}")

    if args.show:
        plt.show()
    plt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
