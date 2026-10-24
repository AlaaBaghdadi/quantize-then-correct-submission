"""
SmoothNet baseline runner — batch or single video / Kalman JSON.

Usage:
  python -m experiments.smoothnet_baseline.run_smoothnet --kalman_dir experiments/reviewer_r5_r6/kalman
  python -m experiments.smoothnet_baseline.run_smoothnet --video media/clip.mp4
  python -m experiments.smoothnet_baseline.run_smoothnet --kalman_json kalman_out/foo_smoothed_keypoints.json
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from experiments.smoothnet_baseline.aggregate_table import (
    format_summary,
    write_benchmark_csv,
)
from experiments.smoothnet_baseline.bridge import (
    apply_smoothed_h36m_to_frames,
    build_output_payload,
    load_keypoints_json,
    save_keypoints_json,
)
from experiments.smoothnet_baseline.config import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SMOOTHNET_ROOT,
    DEFAULT_WORK_DIR,
    KALMAN_JSON_SUFFIX,
    PROJECT_ROOT,
    VIDEO_EXTENSIONS,
    WINDOW_SIZE,
)
from experiments.smoothnet_baseline.evaluate import evaluate_pair
from experiments.smoothnet_baseline.inference import (
    load_smoothnet_model,
    smooth_keypoints_data,
)


def _ensure_project_root_on_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _video_stem_from_kalman_json(path: Path) -> str:
    name = path.name
    if name.endswith(KALMAN_JSON_SUFFIX):
        return name[: -len(KALMAN_JSON_SUFFIX)]
    return path.stem


def _kalman_json_for_video(video_path: Path, kalman_dir: Path) -> Path:
    return kalman_dir / f"{video_path.stem}{KALMAN_JSON_SUFFIX}"


def _list_videos(videos_dir: Path) -> List[Path]:
    out = []
    for p in sorted(videos_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            out.append(p.resolve())
    return out


def _list_kalman_jsons(kalman_dir: Path) -> List[Path]:
    return sorted(kalman_dir.glob(f"*{KALMAN_JSON_SUFFIX}"))


def run_kalman_pipeline(video_path: Path, out_dir: Path) -> Path:
    """Run Kalman via subprocess (no import side effects)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = _kalman_json_for_video(video_path, out_dir)
    cmd = [
        sys.executable,
        "-m",
        "kalman.run_video_pipeline",
        "--video",
        str(video_path.resolve()),
        "--out_dir",
        str(out_dir.resolve()),
    ]
    print(f"[kalman] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"Kalman pipeline failed (exit {result.returncode}) for {video_path}")
    if not expected.is_file():
        raise FileNotFoundError(f"Expected Kalman output not found: {expected}")
    return expected


def resolve_kalman_json(
    video_path: Optional[Path],
    kalman_json: Optional[Path],
    kalman_dir: Path,
    skip_kalman: bool,
) -> Tuple[Path, str]:
    """
    Returns (kalman_json_path, video_stem).
    """
    if kalman_json is not None:
        path = kalman_json.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Kalman JSON not found: {path}")
        return path, _video_stem_from_kalman_json(path)

    if video_path is None:
        raise ValueError("Either video_path or kalman_json must be provided")

    expected = _kalman_json_for_video(video_path, kalman_dir)
    if expected.is_file():
        return expected, video_path.stem

    if skip_kalman:
        raise FileNotFoundError(
            f"Kalman JSON missing ({expected}) and --skip_kalman set. "
            f"Run Kalman first or omit --skip_kalman."
        )

    return run_kalman_pipeline(video_path, kalman_dir), video_path.stem


def process_one(
    kalman_json: Path,
    video_stem: str,
    model: torch.nn.Module,
    device: torch.device,
    work_dir: Path,
    skip_eval: bool,
    window_size: int,
) -> Dict[str, Any]:
    """SmoothNet + optional eval for one video."""
    conditions_dir = work_dir / "conditions" / "smoothnet"
    reports_dir = work_dir / "reports"
    conditions_dir.mkdir(parents=True, exist_ok=True)

    out_json = conditions_dir / f"{video_stem}_smoothnet.json"

    print(f"[smoothnet] Processing {video_stem} ...")
    data = load_keypoints_json(str(kalman_json))
    smoothed_pixels = smooth_keypoints_data(
        data, model, device, window_size=window_size,
    )
    frames = data.get("frames", [])
    smoothed_frames = apply_smoothed_h36m_to_frames(frames, smoothed_pixels)
    output = build_output_payload(data, smoothed_frames)
    save_keypoints_json(str(out_json), output)
    print(f"[smoothnet] Wrote {out_json}")

    row: Dict[str, Any] = {
        "video": video_stem,
        "method": "smoothnet",
        "kalman_json": str(kalman_json),
        "smoothnet_json": str(out_json),
    }

    if skip_eval:
        return row

    report_md = reports_dir / f"{video_stem}_smoothnet.md"
    _, metrics, success = evaluate_pair(
        kalman_json,
        out_json,
        report_md=report_md,
    )
    row.update(metrics)
    row["success"] = success
    print(f"[eval] Report: {report_md}")
    return row


def collect_jobs(args: argparse.Namespace, work_dir: Path) -> List[Tuple[Optional[Path], Optional[Path], str]]:
    """
    Returns list of (video_path or None, kalman_json or None, stem hint).
    kalman_json resolved later in main loop.
    """
    jobs: List[Tuple[Optional[Path], Optional[Path], str]] = []

    if args.kalman_json:
        p = Path(args.kalman_json)
        jobs.append((None, p, _video_stem_from_kalman_json(p)))

    if args.kalman_dir:
        kalman_dir = Path(args.kalman_dir)
        for p in _list_kalman_jsons(kalman_dir):
            jobs.append((None, p, _video_stem_from_kalman_json(p)))

    if args.video:
        jobs.append((Path(args.video), None, Path(args.video).stem))

    if args.videos_dir:
        for v in _list_videos(Path(args.videos_dir)):
            jobs.append((v, None, v.stem))

    # Deduplicate by stem
    seen = set()
    unique: List[Tuple[Optional[Path], Optional[Path], str]] = []
    for video, kj, stem in jobs:
        key = str(kj.resolve()) if kj else f"video:{stem}"
        if key in seen:
            continue
        seen.add(key)
        unique.append((video, kj, stem))

    return unique


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run SmoothNet baseline on Kalman-smoothed keypoints (isolated experiment module).",
    )
    src = ap.add_argument_group("Input (pick one or combine batch flags)")
    src.add_argument("--video", type=str, default=None, help="Single input video (runs Kalman if needed).")
    src.add_argument("--videos_dir", type=str, default=None, help="Directory of videos (*.mp4, ...).")
    src.add_argument("--kalman_json", type=str, default=None, help="Single Kalman smoothed keypoints JSON.")
    src.add_argument("--kalman_dir", type=str, default=None, help="Directory of *_smoothed_keypoints.json files.")

    ap.add_argument("--work_dir", type=str, default=str(DEFAULT_WORK_DIR), help="Output root for all artifacts.")
    ap.add_argument("--skip_kalman", action="store_true", help="Do not run Kalman; error if JSON missing.")
    ap.add_argument("--skip_eval", action="store_true", help="Only produce SmoothNet JSONs, skip evaluation.")

    ap.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="SmoothNet checkpoint path.")
    ap.add_argument("--smoothnet_root", type=str, default=str(DEFAULT_SMOOTHNET_ROOT), help="SmoothNet repo root.")
    ap.add_argument("--device", type=str, default=None, help="cuda or cpu (default: auto).")
    ap.add_argument("--window_size", type=int, default=WINDOW_SIZE, help="SmoothNet sliding window size.")
    ap.add_argument("--batch_size", type=int, default=64, help="Inference batch size (windows).")

    return ap.parse_args()


def main() -> int:
    _ensure_project_root_on_path()
    args = parse_args()

    work_dir = Path(args.work_dir).resolve()
    kalman_dir = work_dir / "kalman"
    kalman_dir.mkdir(parents=True, exist_ok=True)

    jobs = collect_jobs(args, work_dir)
    if not jobs:
        print(
            "No input specified. Use --video, --videos_dir, --kalman_json, or --kalman_dir.\n"
            "Example:\n"
            "  python -m experiments.smoothnet_baseline.run_smoothnet --kalman_dir kalman_out",
            file=sys.stderr,
        )
        return 1

    checkpoint = Path(args.checkpoint).resolve()
    smoothnet_root = Path(args.smoothnet_root).resolve()
    if not checkpoint.is_file():
        print(f"Checkpoint not found: {checkpoint}", file=sys.stderr)
        return 1
    if not smoothnet_root.is_dir():
        print(f"SmoothNet root not found: {smoothnet_root}", file=sys.stderr)
        return 1

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[smoothnet] Loading model from {checkpoint} on {device} ...")
    model = load_smoothnet_model(
        checkpoint,
        smoothnet_root,
        device,
        window_size=args.window_size,
    )

    rows: List[Dict[str, Any]] = []
    for video_path, kalman_json_in, stem_hint in jobs:
        try:
            kalman_path, video_stem = resolve_kalman_json(
                video_path,
                kalman_json_in,
                kalman_dir,
                skip_kalman=args.skip_kalman,
            )
            row = process_one(
                kalman_path,
                video_stem,
                model,
                device,
                work_dir,
                skip_eval=args.skip_eval,
                window_size=args.window_size,
            )
            rows.append(row)
        except Exception as exc:
            print(f"[error] Failed on {stem_hint}: {exc}", file=sys.stderr)
            rows.append({
                "video": stem_hint,
                "method": "smoothnet",
                "error": str(exc),
            })

    if not args.skip_eval and rows:
        table_path = work_dir / "tables" / "smoothnet_benchmark.csv"
        csv_rows = [
            {k: r.get(k) for k in ("video", "method", "accel_var", "viol_frac", "limb_std", "jitter",
                                   "accel_var_orig", "viol_frac_orig", "limb_std_orig", "success")}
            for r in rows if "error" not in r
        ]
        write_benchmark_csv(csv_rows, table_path)
        print("")
        print(format_summary(csv_rows))
        print("")
        print(f"[done] Table written to {table_path}")

    failed = sum(1 for r in rows if r.get("error"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
