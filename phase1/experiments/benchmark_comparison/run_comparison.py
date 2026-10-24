"""
Run KAPT vs SmoothNet benchmark: correct, evaluate, aggregate, Wilcoxon.

Usage:
  python -m experiments.benchmark_comparison.run_comparison
  python -m experiments.benchmark_comparison.run_comparison --kalman_dir kalman_out
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from experiments.benchmark_comparison.config import (
    DEFAULT_KALMAN_DIR,
    DEFAULT_VIDEO_STEMS,
    DEFAULT_WORK_DIR,
    KALMAN_JSON_SUFFIX,
    METRIC_COLUMNS,
    PROJECT_ROOT,
)
from experiments.benchmark_comparison.kapt_runner import run_kapt_correction
from experiments.benchmark_comparison.stats import (
    format_summary_markdown,
    summarize_by_method,
    wilcoxon_kapt_vs_smoothnet,
)
from experiments.smoothnet_baseline.config import DEFAULT_CHECKPOINT, DEFAULT_SMOOTHNET_ROOT
from experiments.smoothnet_baseline.evaluate import evaluate_pair, metrics_dict_from_report
from experiments.smoothnet_baseline.inference import load_smoothnet_model, smooth_keypoints_data
from experiments.smoothnet_baseline.bridge import (
    apply_smoothed_h36m_to_frames,
    build_output_payload,
    load_keypoints_json,
    save_keypoints_json,
)


def _ensure_project_root() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _kalman_path(kalman_dir: Path, stem: str) -> Path:
    return kalman_dir / f"{stem}{KALMAN_JSON_SUFFIX}"


def _run_smoothnet_if_needed(
    kalman_json: Path,
    out_json: Path,
    model,
    device: torch.device,
    window_size: int,
) -> None:
    if out_json.is_file():
        return
    out_json.parent.mkdir(parents=True, exist_ok=True)
    data = load_keypoints_json(str(kalman_json))
    smoothed_pixels = smooth_keypoints_data(data, model, device, window_size=window_size)
    frames = data.get("frames", [])
    smoothed_frames = apply_smoothed_h36m_to_frames(frames, smoothed_pixels)
    output = build_output_payload(data, smoothed_frames)
    save_keypoints_json(str(out_json), output)


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def process_video(
    stem: str,
    kalman_dir: Path,
    work_dir: Path,
    model,
    device: torch.device,
    skip_kapt: bool,
    skip_smoothnet: bool,
    window_size: int,
) -> List[Dict[str, Any]]:
    kalman_json = _kalman_path(kalman_dir, stem)
    if not kalman_json.is_file():
        raise FileNotFoundError(f"Kalman JSON not found: {kalman_json}")

    smoothnet_json = work_dir / "conditions" / "smoothnet" / f"{stem}_smoothnet.json"
    kapt_json = work_dir / "conditions" / "kapt_full" / f"{stem}_corrected.json"
    reports_dir = work_dir / "reports"

    if not skip_smoothnet:
        _run_smoothnet_if_needed(kalman_json, smoothnet_json, model, device, window_size)

    if not skip_kapt:
        if not kapt_json.is_file():
            print(f"[kapt] Running correction for {stem} ...")
            run_kapt_correction(kalman_json, kapt_json)
        else:
            print(f"[kapt] Reusing {kapt_json.name}")

    rows: List[Dict[str, Any]] = []

    for method, corrected_json in [
        ("smoothnet", smoothnet_json),
        ("kapt_full", kapt_json),
    ]:
        if not corrected_json.is_file():
            raise FileNotFoundError(f"Missing {method} output: {corrected_json}")

        report_md = reports_dir / f"{stem}_{method}.md"
        _, metrics, success = evaluate_pair(
            kalman_json,
            corrected_json,
            report_md=report_md,
        )
        row: Dict[str, Any] = {
            "video": stem,
            "method": method,
            "success": success,
            "kalman_json": str(kalman_json),
            "corrected_json": str(corrected_json),
        }
        row.update(metrics)
        rows.append(row)
        print(
            f"[eval] {stem} / {method}: "
            f"accel_var={metrics.get('accel_var', float('nan')):.4f}, "
            f"viol_frac={metrics.get('viol_frac', float('nan')):.4f}, "
            f"limb_std={metrics.get('limb_std', float('nan')):.4f}, "
            f"jitter={metrics.get('jitter', float('nan')):.4f}"
        )

    return rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="KAPT vs SmoothNet benchmark comparison.")
    ap.add_argument("--kalman_dir", type=str, default=str(DEFAULT_KALMAN_DIR))
    ap.add_argument("--work_dir", type=str, default=str(DEFAULT_WORK_DIR))
    ap.add_argument(
        "--videos",
        type=str,
        default=",".join(DEFAULT_VIDEO_STEMS),
        help="Comma-separated video stems (default: man1,man2,...,woman3).",
    )
    ap.add_argument("--skip_kapt", action="store_true")
    ap.add_argument("--skip_smoothnet", action="store_true")
    ap.add_argument("--skip_eval", action="store_true", help="Only generate JSONs, no table.")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--window_size", type=int, default=32)
    ap.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--smoothnet_root", type=str, default=str(DEFAULT_SMOOTHNET_ROOT))
    return ap.parse_args()


def main() -> int:
    _ensure_project_root()
    args = parse_args()

    kalman_dir = Path(args.kalman_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    stems = [s.strip() for s in args.videos.split(",") if s.strip()]

    if not kalman_dir.is_dir():
        print(f"Kalman dir not found: {kalman_dir}", file=sys.stderr)
        return 1

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = None
    if not args.skip_smoothnet:
        checkpoint = Path(args.checkpoint).resolve()
        smoothnet_root = Path(args.smoothnet_root).resolve()
        print(f"[smoothnet] Loading model on {device} ...")
        model = load_smoothnet_model(
            checkpoint, smoothnet_root, device, window_size=args.window_size,
        )

    all_rows: List[Dict[str, Any]] = []
    errors: List[str] = []

    for stem in stems:
        try:
            rows = process_video(
                stem,
                kalman_dir,
                work_dir,
                model,
                device,
                skip_kapt=args.skip_kapt,
                skip_smoothnet=args.skip_smoothnet,
                window_size=args.window_size,
            )
            all_rows.extend(rows)
        except Exception as exc:
            msg = f"{stem}: {exc}"
            errors.append(msg)
            print(f"[error] {msg}", file=sys.stderr)

    if args.skip_eval or not all_rows:
        if errors:
            return 1
        return 0

    tables_dir = work_dir / "tables"
    per_video_fields = [
        "video", "method", "accel_var", "viol_frac", "limb_std", "jitter",
        "accel_var_orig", "viol_frac_orig", "limb_std_orig", "success",
    ]
    _write_csv(tables_dir / "benchmark_ablation.csv", all_rows, per_video_fields)

    method_summary = summarize_by_method(all_rows)
    summary_fields = ["method", "n_videos"] + [
        f"{c}_mean" for c in METRIC_COLUMNS
    ] + [f"{c}_std" for c in METRIC_COLUMNS]
    _write_csv(tables_dir / "kapt_vs_smoothnet_summary.csv", method_summary, summary_fields)

    wilcoxon_rows = wilcoxon_kapt_vs_smoothnet(all_rows)
    wilcoxon_fields = [
        "comparison", "metric", "n_pairs", "kapt_mean", "smoothnet_mean",
        "statistic", "p_value", "winner", "significant_p05",
    ]
    _write_csv(tables_dir / "kapt_vs_smoothnet_wilcoxon.csv", wilcoxon_rows, wilcoxon_fields)

    report_md = format_summary_markdown(all_rows, method_summary, wilcoxon_rows)
    report_path = work_dir / "reports" / "kapt_vs_smoothnet_final.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md, encoding="utf-8")

    print("")
    print(report_md)
    print(f"[done] Per-video table: {tables_dir / 'benchmark_ablation.csv'}")
    print(f"[done] Summary:         {tables_dir / 'kapt_vs_smoothnet_summary.csv'}")
    print(f"[done] Wilcoxon:        {tables_dir / 'kapt_vs_smoothnet_wilcoxon.csv'}")
    print(f"[done] Final report:    {report_path}")

    if errors:
        print(f"\n[warn] {len(errors)} video(s) failed:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1 if not all_rows else 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
