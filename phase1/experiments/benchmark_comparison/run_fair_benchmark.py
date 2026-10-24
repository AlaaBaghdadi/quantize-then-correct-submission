"""
Fair KAPT power benchmark — separate from kapt_vs_smoothnet_final.md.

Uses angle-limit violations and improvement-vs-Kalman deltas.
Reuses existing corrected JSONs; does not overwrite original reports.

Usage:
  python -m experiments.benchmark_comparison.run_fair_benchmark
  python -m experiments.benchmark_comparison.run_fair_benchmark --kalman_dir kalman_out
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

from experiments.benchmark_comparison.config import (
    DEFAULT_KALMAN_DIR,
    DEFAULT_VIDEO_STEMS,
    DEFAULT_WORK_DIR,
    KALMAN_JSON_SUFFIX,
    PROJECT_ROOT,
)
from experiments.benchmark_comparison.fair_evaluate import (
    evaluate_kalman_baseline_need,
    evaluate_method_vs_kalman,
)
from experiments.benchmark_comparison.fair_stats import (
    DELTA_COLUMNS,
    FAIR_METRIC_COLUMNS,
    format_fair_report,
    summarize_fair,
    wilcoxon_fair,
)
from experiments.benchmark_comparison.kapt_runner import run_kapt_correction
from experiments.smoothnet_baseline.config import DEFAULT_CHECKPOINT, DEFAULT_SMOOTHNET_ROOT
from experiments.smoothnet_baseline.inference import load_smoothnet_model, smooth_keypoints_data
from experiments.smoothnet_baseline.bridge import (
    apply_smoothed_h36m_to_frames,
    build_output_payload,
    load_keypoints_json,
    save_keypoints_json,
)

import torch


def _ensure_project_root() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _ensure_smoothnet(
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
    smoothed = smooth_keypoints_data(data, model, device, window_size=window_size)
    frames = data.get("frames", [])
    out_frames = apply_smoothed_h36m_to_frames(frames, smoothed)
    save_keypoints_json(str(out_json), build_output_payload(data, out_frames))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fair KAPT power benchmark (separate report).")
    ap.add_argument("--kalman_dir", type=str, default=str(DEFAULT_KALMAN_DIR))
    ap.add_argument("--work_dir", type=str, default=str(DEFAULT_WORK_DIR))
    ap.add_argument("--videos", type=str, default=",".join(DEFAULT_VIDEO_STEMS))
    ap.add_argument("--ensure_outputs", action="store_true",
                    help="Run KAPT/SmoothNet if JSONs missing.")
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
    if args.ensure_outputs:
        model = load_smoothnet_model(
            Path(args.checkpoint).resolve(),
            Path(args.smoothnet_root).resolve(),
            device,
            window_size=args.window_size,
        )

    all_rows: List[Dict[str, Any]] = []
    baseline_need: List[Dict[str, Any]] = []

    for stem in stems:
        kalman_json = kalman_dir / f"{stem}{KALMAN_JSON_SUFFIX}"
        if not kalman_json.is_file():
            print(f"[skip] Missing {kalman_json}", file=sys.stderr)
            continue

        smoothnet_json = work_dir / "conditions" / "smoothnet" / f"{stem}_smoothnet.json"
        kapt_json = work_dir / "conditions" / "kapt_full" / f"{stem}_corrected.json"

        if args.ensure_outputs:
            if model is not None:
                _ensure_smoothnet(kalman_json, smoothnet_json, model, device, args.window_size)
            if not kapt_json.is_file():
                run_kapt_correction(kalman_json, kapt_json)

        need = evaluate_kalman_baseline_need(kalman_json)
        baseline_need.append({"video": stem, **need})

        for method_name, method_json in [
            ("smoothnet", smoothnet_json),
            ("kapt_full", kapt_json),
        ]:
            if not method_json.is_file():
                print(f"[skip] Missing {method_json} for {stem}", file=sys.stderr)
                continue
            row = evaluate_method_vs_kalman(kalman_json, method_json, method_name, stem)
            all_rows.append(row)
            print(
                f"[fair] {stem}/{method_name}: "
                f"angle_viol={row['angle_viol_frac']:.4f} "
                f"(delta {row.get('angle_viol_frac_delta', float('nan')):+.4f}), "
                f"limb_std={row['limb_std']:.4f} "
                f"(delta {row.get('limb_std_delta', float('nan')):+.4f})"
            )

    if not all_rows:
        print("No results.", file=sys.stderr)
        return 1

    tables_dir = work_dir / "tables"
    row_fields = (
        ["video", "method"]
        + FAIR_METRIC_COLUMNS
        + ["kalman_angle_viol_frac", "kalman_limb_std", "kalman_jitter", "kalman_accel_var"]
        + ["frames_needing_kapt_frac", "frames_needing_kapt_count"]
        + DELTA_COLUMNS
    )
    _write_csv(tables_dir / "benchmark_fair.csv", all_rows, row_fields)
    _write_csv(tables_dir / "kapt_baseline_need.csv", baseline_need,
               ["video", "frames_needing_kapt_frac", "frames_needing_kapt_count"])

    summary = summarize_fair(all_rows)
    summary_fields = ["method", "n_videos"]
    for col in FAIR_METRIC_COLUMNS + DELTA_COLUMNS:
        summary_fields.extend([f"{col}_mean", f"{col}_std"])
    _write_csv(tables_dir / "kapt_power_summary.csv", summary, summary_fields)

    wilcoxon_rows: List[Dict[str, Any]] = []
    for metric, higher in [
        ("angle_viol_frac", False),
        ("limb_std", False),
        ("jitter", False),
        ("accel_var", False),
        ("angle_viol_frac_delta", True),
        ("limb_std_delta", True),
        ("jitter_delta", True),
        ("accel_var_delta", True),
    ]:
        wilcoxon_rows.append(wilcoxon_fair(all_rows, metric, higher_is_better=higher))

    wilcoxon_fields = [
        "metric", "n_pairs", "kapt_mean", "smoothnet_mean", "statistic",
        "p_value", "winner", "significant_p05", "higher_is_better",
    ]
    _write_csv(tables_dir / "kapt_power_wilcoxon.csv", wilcoxon_rows, wilcoxon_fields)

    report = format_fair_report(all_rows, summary, wilcoxon_rows, baseline_need)
    report_path = work_dir / "reports" / "kapt_power_benchmark.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print("")
    print(report)
    print(f"[done] Fair table:   {tables_dir / 'benchmark_fair.csv'}")
    print(f"[done] Power report: {report_path}")
    print("(Original report unchanged: reports/kapt_vs_smoothnet_final.md)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
