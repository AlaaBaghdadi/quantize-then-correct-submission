from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from pose_handler.common_io import ensure_dir
from pose_handler.unified_keypoints_io import load_unified_keypoints


DEFAULT_CONNECTIONS = [
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
    x = float(rec.get("x", 0.0))
    y = float(rec.get("y", 0.0))
    # Try z_scaled first, then z, then default to 0
    z = float(rec.get("z_scaled", rec.get("z", 0.0)))
    return np.array([x, y, z], dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="Save 3D keypoint plots (matplotlib) from scaled MediaPipe output.")
    ap.add_argument(
        "--in_json",
        default="pose_handler_out/mediapipe_pose_scaled.json",
        help="Input unified keypoints JSON (e.g. pose_handler_out/averaged_keypoints.json).",
    )
    ap.add_argument("--out_dir", default="pose_handler_out", help="Output directory (creates vis_3d/).")
    ap.add_argument("--min_vis", type=float, default=0.3, help="Minimum visibility to include a point.")
    ap.add_argument(
        "--show",
        action="store_true",
        help="If set, open an interactive window so you can rotate the 3D pose.",
    )
    args = ap.parse_args()

    # Load unified format (works with any format)
    data = load_unified_keypoints(args.in_json)
    kps = data.get("keypoints", {})
    img_name = data.get("image_name", "pose")
    
    out_root = ensure_dir(args.out_dir)
    out_vis = ensure_dir(out_root / "vis_3d")
    min_vis = float(args.min_vis)

    # filter points by visibility/confidence
    pts = {}
    for name, rec in kps.items():
        if rec is None:
            continue
        conf = float(rec.get("confidence", rec.get("visibility", 0.0)))
        if conf < min_vis:
            continue
        p = _pt(kps, name)
        if p is not None:
            pts[name] = p

    if not pts:
        print("Warning: No keypoints above visibility threshold")
        return 0

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # bones
    for a, b in DEFAULT_CONNECTIONS:
        if a in pts and b in pts:
            pa, pb = pts[a], pts[b]
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], color="white", linewidth=2, alpha=0.8)

    # points
    xyz = np.stack(list(pts.values()), axis=0)
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=30, c="cyan", depthshade=True)

    ax.set_title(img_name)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # equal-ish limits for X/Y
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    centers = (mins + maxs) / 2.0
    span = float(np.max(maxs - mins)) * 0.6 + 1e-6
    ax.set_xlim(float(centers[0] - span), float(centers[0] + span))
    ax.set_ylim(float(centers[1] - span), float(centers[1] + span))
    # Use a fixed, tight range for Z so depth differences are visible
    ax.set_zlim(-1.0, 1.0)
    ax.view_init(elev=20, azim=45)

    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color("white")
    ax.tick_params(colors="white")
    ax.title.set_color("white")

    out_path = out_vis / f"{Path(img_name).stem}_vis3d.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())

    # Optionally show interactive window for rotating/viewing
    if args.show:
        plt.show()

    plt.close(fig)

    print(f"Wrote 3D visualizations to: {out_vis}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())