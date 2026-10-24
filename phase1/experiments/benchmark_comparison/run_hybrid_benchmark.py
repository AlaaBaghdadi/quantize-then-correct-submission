"""
KAPT + SmoothNet hybrid benchmark.

Pipeline: Kalman → KAPT → SmoothNet (kapt_then_smoothnet)
Compares against Kalman → SmoothNet and Kalman → KAPT alone.

Usage:
  python -m experiments.benchmark_comparison.run_hybrid_benchmark
  python -m experiments.benchmark_comparison.run_hybrid_benchmark --force
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

from experiments.benchmark_comparison.config import (
    DEFAULT_KALMAN_DIR,
    DEFAULT_VIDEO_STEMS,
    DEFAULT_WORK_DIR,
    KALMAN_JSON_SUFFIX,
    PROJECT_ROOT,
)
from experiments.benchmark_comparison.fair_evaluate import evaluate_method_vs_kalman
from experiments.benchmark_comparison.fair_stats import DELTA_COLUMNS, FAIR_METRIC_COLUMNS
from experiments.benchmark_comparison.hybrid_stats import (
    HYBRID_METHODS,
    format_hybrid_report,
    summarize_hybrid,
    wilcoxon_pair,
)
from experiments.benchmark_comparison.kapt_runner import run_kapt_correction
from experiments.smoothnet_baseline.bridge import (
    apply_smoothed_h36m_to_frames,
    build_output_payload,
    load_keypoints_json,
    save_keypoints_json,
)
from experiments.smoothnet_baseline.config import DEFAULT_CHECKPOINT, DEFAULT_SMOOTHNET_ROOT
from experiments.smoothnet_baseline.inference import load_smoothnet_model, smooth_keypoints_data


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


def _run_smoothnet(
    in_json: Path,
    out_json: Path,
    model,
    device: torch.device,
    window_size: int,
    *,
    method_label: str = "SmoothNet",
    force: bool = False,
) -> None:
    if out_json.is_file() and not force:
        return
    out_json.parent.mkdir(parents=True, exist_ok=True)
    data = load_keypoints_json(str(in_json))
    smoothed_pixels = smooth_keypoints_data(data, model, device, window_size=window_size)
    frames = data.get("frames", [])
    smoothed_frames = apply_smoothed_h36m_to_frames(frames, smoothed_pixels)
    output = build_output_payload(data, smoothed_frames)
    output["refinement"] = {
        "method": method_label,
        "reference": "Zeng et al., ECCV 2022",
        "joint_layout": "H36M-17 bridged from OpenPose-18",
        "input": in_json.name,
    }
    save_keypoints_json(str(out_json), output)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="KAPT + SmoothNet hybrid benchmark.")
    ap.add_argument("--kalman_dir", type=str, default=str(DEFAULT_KALMAN_DIR))
    ap.add_argument("--work_dir", type=str, default=str(DEFAULT_WORK_DIR))
    ap.add_argument("--videos", type=str, default=",".join(DEFAULT_VIDEO_STEMS))
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--window_size", type=int, default=32)
    ap.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--smoothnet_root", type=str, default=str(DEFAULT_SMOOTHNET_ROOT))
    ap.add_argument("--force", action="store_true", help="Regenerate all JSON outputs.")
    ap.add_argument("--skip_generation", action="store_true", help="Evaluate existing JSONs only.")
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
    if not args.skip_generation:
        checkpoint = Path(args.checkpoint).resolve()
        smoothnet_root = Path(args.smoothnet_root).resolve()
        print(f"[smoothnet] Loading model on {device} ...")
        model = load_smoothnet_model(
            checkpoint, smoothnet_root, device, window_size=args.window_size,
        )

    all_rows: List[Dict[str, Any]] = []
    errors: List[str] = []

    for stem in stems:
        kalman_json = kalman_dir / f"{stem}{KALMAN_JSON_SUFFIX}"
        if not kalman_json.is_file():
            errors.append(f"{stem}: missing {kalman_json}")
            continue

        kapt_json = work_dir / "conditions" / "kapt_full" / f"{stem}_corrected.json"
        smoothnet_json = work_dir / "conditions" / "smoothnet" / f"{stem}_smoothnet.json"
        hybrid_json = work_dir / "conditions" / "kapt_then_smoothnet" / f"{stem}_kapt_smoothnet.json"

        try:
            if not args.skip_generation:
                if not kapt_json.is_file() or args.force:
                    print(f"[kapt] {stem} ...")
                    run_kapt_correction(kalman_json, kapt_json, force=args.force)

                if model is not None:
                    print(f"[smoothnet] {stem} on Kalman ...")
                    _run_smoothnet(
                        kalman_json, smoothnet_json, model, device, args.window_size,
                        method_label="SmoothNet", force=args.force,
                    )
                    print(f"[hybrid] {stem}: KAPT -> SmoothNet ...")
                    _run_smoothnet(
                        kapt_json, hybrid_json, model, device, args.window_size,
                        method_label="KAPT+SmoothNet", force=args.force,
                    )

            for method_name, method_json in [
                ("smoothnet", smoothnet_json),
                ("kapt_full", kapt_json),
                ("kapt_then_smoothnet", hybrid_json),
            ]:
                if not method_json.is_file():
                    raise FileNotFoundError(f"Missing {method_json}")

                row = evaluate_method_vs_kalman(kalman_json, method_json, method_name, stem)
                all_rows.append(row)
                print(
                    f"[eval] {stem}/{method_name}: "
                    f"limb_std={row['limb_std']:.4f} "
                    f"jitter={row['jitter']:.4f} "
                    f"accel_var={row['accel_var']:.4f}"
                )
        except Exception as exc:
            errors.append(f"{stem}: {exc}")
            print(f"[error] {stem}: {exc}", file=sys.stderr)

    if not all_rows:
        print("No results.", file=sys.stderr)
        return 1

    tables_dir = work_dir / "tables"
    row_fields = (
        ["video", "method"]
        + FAIR_METRIC_COLUMNS
        + ["kalman_angle_viol_frac", "kalman_limb_std", "kalman_jitter", "kalman_accel_var"]
        + DELTA_COLUMNS
    )
    _write_csv(tables_dir / "hybrid_benchmark.csv", all_rows, row_fields)

    summary = summarize_hybrid(all_rows)
    summary_fields = ["method", "n_videos"]
    for col in FAIR_METRIC_COLUMNS + DELTA_COLUMNS:
        summary_fields.extend([f"{col}_mean", f"{col}_std"])
    _write_csv(tables_dir / "hybrid_summary.csv", summary, summary_fields)

    wilcoxon_rows: List[Dict[str, Any]] = []
    comparisons = [
        ("kapt_then_smoothnet", "kapt_full"),
        ("kapt_then_smoothnet", "smoothnet"),
        ("kapt_full", "smoothnet"),
    ]
    for metric, higher in [
        ("limb_std", False),
        ("jitter", False),
        ("accel_var", False),
        ("limb_std_delta", True),
        ("jitter_delta", True),
        ("accel_var_delta", True),
    ]:
        for method_a, method_b in comparisons:
            wilcoxon_rows.append(
                wilcoxon_pair(all_rows, method_a, method_b, metric, higher_is_better=higher)
            )

    wilcoxon_fields = [
        "comparison", "metric", "n_pairs",
        "kapt_then_smoothnet_mean", "kapt_full_mean", "smoothnet_mean",
        "statistic", "p_value", "winner", "significant_p05", "higher_is_better",
    ]
    _write_csv(tables_dir / "hybrid_wilcoxon.csv", wilcoxon_rows, wilcoxon_fields)

    report = format_hybrid_report(all_rows, summary, wilcoxon_rows)
    report_path = work_dir / "reports" / "kapt_smoothnet_hybrid.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print("")
    print(report)
    print(f"[done] Per-video:  {tables_dir / 'hybrid_benchmark.csv'}")
    print(f"[done] Summary:    {tables_dir / 'hybrid_summary.csv'}")
    print(f"[done] Wilcoxon:   {tables_dir / 'hybrid_wilcoxon.csv'}")
    print(f"[done] Report:     {report_path}")

    if errors:
        print(f"\n[warn] {len(errors)} error(s):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1 if len(all_rows) < len(stems) * len(HYBRID_METHODS) else 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
