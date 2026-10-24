from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from pose_handler.common_io import ensure_dir, read_json
from pose_handler.keypoints_schema import FORMAT_OPENPOSE18_V1
from pose_handler.openpose18_mapping import PISE18_ORDER


def _to_xy_lists(openpose18: List[Optional[dict]], score_thresh: float) -> tuple[list[int], list[int]]:
    # reorder to PISE-18
    pise18 = [openpose18[i] if i < len(openpose18) else None for i in PISE18_ORDER]
    xs: list[int] = []
    ys: list[int] = []
    for kp in pise18:
        if kp is None:
            xs.append(-1)
            ys.append(-1)
            continue
        c = float(kp.get("confidence", 0.0))
        if c < score_thresh:
            xs.append(-1)
            ys.append(-1)
            continue
        xs.append(int(round(float(kp["x"]))))
        ys.append(int(round(float(kp["y"]))))
    return ys, xs


def main() -> int:
    ap = argparse.ArgumentParser(description="Export OpenPose-18 JSON into PISE CSV format.")
    ap.add_argument("--in_json", default="pose_handler_out/openpose18.json", help="Input OpenPose18 JSON.")
    ap.add_argument("--out_dir", default="pose_handler_out", help="Output directory.")
    ap.add_argument("--score_thresh", type=float, default=0.15, help="Keypoints below this confidence are set to -1.")
    ap.add_argument("--out_csv", default=None, help="Optional output CSV name (default: pise18_keypoints.csv).")
    args = ap.parse_args()

    in_path = Path(args.in_json)
    data = read_json(in_path)
    if data.get("format") != FORMAT_OPENPOSE18_V1:
        raise ValueError(f"Unexpected format in {in_path}: {data.get('format')}")

    out_dir = ensure_dir(args.out_dir)
    out_csv = Path(args.out_csv) if args.out_csv else out_dir / "pise18_keypoints.csv"

    lines = ["name:keypoints_y:keypoints_x"]
    for entry in data.get("entries", []):
        name = entry.get("image_name", "image")
        ys, xs = _to_xy_lists(entry.get("keypoints", []), float(args.score_thresh))
        lines.append(f"{name}:{json.dumps(ys)}:{json.dumps(xs)}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

