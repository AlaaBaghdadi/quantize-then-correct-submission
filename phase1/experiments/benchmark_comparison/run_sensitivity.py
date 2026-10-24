"""
Run KAPT hyperparameter sensitivity sweeps (alpha, max_drift, process_noise).

Usage:
  python -m experiments.benchmark_comparison.run_sensitivity
  python -m experiments.benchmark_comparison.run_sensitivity --force
  python -m experiments.benchmark_comparison.run_sensitivity --skip_generation
  python -m experiments.benchmark_comparison.run_sensitivity --params alpha,max_drift
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from experiments.benchmark_comparison.config import (
    DEFAULT_KALMAN_DIR,
    DEFAULT_VIDEO_STEMS,
    DEFAULT_WORK_DIR,
    KALMAN_JSON_SUFFIX,
    KAPT_ALPHA,
    PROJECT_ROOT,
)
from experiments.benchmark_comparison.fair_evaluate import evaluate_method_vs_kalman
from experiments.benchmark_comparison.kapt_runner import run_kapt_correction
from experiments.benchmark_comparison.run_ablation import (
    _ensure_project_root,
    _kalman_path,
    _write_csv,
    ensure_fused_json,
    validate_frame_counts,
)
from experiments.benchmark_comparison.sensitivity_config import (
    DEFAULT_MAX_DRIFT,
    SENSITIVITY_SWEEPS,
    SensitivityParam,
    condition_dir_name,
)
from experiments.benchmark_comparison.sensitivity_stats import (
    format_sensitivity_report,
    plot_sensitivity_figures,
    summarize_sensitivity,
)
from kalman.resmooth_fused import resmooth_fused_json


METRIC_COLUMNS = [
    "accel_var",
    "angle_viol_frac",
    "limb_std",
    "jitter",
    "accel_var_delta",
    "angle_viol_frac_delta",
    "limb_std_delta",
    "jitter_delta",
]


def _corrected_path(work_dir: Path, param: SensitivityParam, value: float, stem: str) -> Path:
    cond = condition_dir_name(param, value)
    return work_dir / "conditions" / cond / f"{stem}_corrected.json"


def _smoothed_path(work_dir: Path, param: SensitivityParam, value: float, stem: str) -> Path:
    cond = condition_dir_name(param, value)
    return work_dir / "conditions" / cond / f"{stem}_smoothed_keypoints.json"


def _method_id(param: SensitivityParam, value: float) -> str:
    return condition_dir_name(param, value)


def run_alpha_sweep(
    stem: str,
    kalman_json: Path,
    work_dir: Path,
    *,
    force: bool,
    skip_generation: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for value in next(s.values for s in SENSITIVITY_SWEEPS if s.param == "alpha"):
        out_path = _corrected_path(work_dir, "alpha", value, stem)
        if not skip_generation:
            run_kapt_correction(
                kalman_json,
                out_path,
                alpha=value,
                max_drift=DEFAULT_MAX_DRIFT,
                force=force,
                variant_id=_method_id("alpha", value),
            )
        if not out_path.is_file():
            raise FileNotFoundError(f"Missing alpha={value} output: {out_path}")
        validate_frame_counts(kalman_json, out_path)
        row = evaluate_method_vs_kalman(
            kalman_json,
            out_path,
            _method_id("alpha", value),
            stem,
        )
        row.update({
            "parameter": "alpha",
            "value": value,
            "output_json": str(out_path),
            "kalman_json": str(kalman_json),
        })
        rows.append(row)
    return rows


def run_max_drift_sweep(
    stem: str,
    kalman_json: Path,
    work_dir: Path,
    *,
    force: bool,
    skip_generation: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for value in next(s.values for s in SENSITIVITY_SWEEPS if s.param == "max_drift"):
        out_path = _corrected_path(work_dir, "max_drift", value, stem)
        if not skip_generation:
            run_kapt_correction(
                kalman_json,
                out_path,
                alpha=KAPT_ALPHA,
                max_drift=value,
                force=force,
                variant_id=_method_id("max_drift", value),
            )
        if not out_path.is_file():
            raise FileNotFoundError(f"Missing max_drift={value} output: {out_path}")
        validate_frame_counts(kalman_json, out_path)
        row = evaluate_method_vs_kalman(
            kalman_json,
            out_path,
            _method_id("max_drift", value),
            stem,
        )
        row.update({
            "parameter": "max_drift",
            "value": value,
            "output_json": str(out_path),
            "kalman_json": str(kalman_json),
        })
        rows.append(row)
    return rows


def run_process_noise_sweep(
    stem: str,
    kalman_dir: Path,
    work_dir: Path,
    *,
    force: bool,
    skip_generation: bool,
    refresh_fused: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    fused_json = ensure_fused_json(
        stem, kalman_dir, work_dir, refresh_fused=refresh_fused,
    )

    for value in next(s.values for s in SENSITIVITY_SWEEPS if s.param == "process_noise"):
        smoothed_path = _smoothed_path(work_dir, "process_noise", value, stem)
        corrected_path = _corrected_path(work_dir, "process_noise", value, stem)

        if not skip_generation:
            resmooth_fused_json(
                fused_json,
                smoothed_path,
                process_noise=value,
                force=force,
            )
            run_kapt_correction(
                smoothed_path,
                corrected_path,
                alpha=KAPT_ALPHA,
                max_drift=DEFAULT_MAX_DRIFT,
                force=force,
                variant_id=_method_id("process_noise", value),
            )

        if not smoothed_path.is_file():
            raise FileNotFoundError(f"Missing re-smoothed JSON: {smoothed_path}")
        if not corrected_path.is_file():
            raise FileNotFoundError(f"Missing corrected JSON: {corrected_path}")

        validate_frame_counts(smoothed_path, corrected_path)
        row = evaluate_method_vs_kalman(
            smoothed_path,
            corrected_path,
            _method_id("process_noise", value),
            stem,
        )
        row.update({
            "parameter": "process_noise",
            "value": value,
            "output_json": str(corrected_path),
            "kalman_json": str(smoothed_path),
            "fused_json": str(fused_json),
        })
        rows.append(row)
    return rows


def process_video(
    stem: str,
    kalman_dir: Path,
    work_dir: Path,
    params: Set[SensitivityParam],
    *,
    force: bool,
    skip_generation: bool,
    refresh_fused: bool,
) -> tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    kalman_json = _kalman_path(kalman_dir, stem)
    if not kalman_json.is_file():
        raise FileNotFoundError(f"Kalman JSON not found: {kalman_json}")

    entry: Dict[str, Any] = {"kalman_json": str(kalman_json), "runs": []}

    if "alpha" in params:
        try:
            alpha_rows = run_alpha_sweep(
                stem, kalman_json, work_dir, force=force, skip_generation=skip_generation,
            )
            rows.extend(alpha_rows)
            entry["runs"].append({"parameter": "alpha", "n": len(alpha_rows)})
        except Exception as exc:
            errors.append(f"{stem}/alpha: {exc}")

    if "max_drift" in params:
        try:
            drift_rows = run_max_drift_sweep(
                stem, kalman_json, work_dir, force=force, skip_generation=skip_generation,
            )
            rows.extend(drift_rows)
            entry["runs"].append({"parameter": "max_drift", "n": len(drift_rows)})
        except Exception as exc:
            errors.append(f"{stem}/max_drift: {exc}")

    if "process_noise" in params:
        try:
            q_rows = run_process_noise_sweep(
                stem,
                kalman_dir,
                work_dir,
                force=force,
                skip_generation=skip_generation,
                refresh_fused=refresh_fused,
            )
            rows.extend(q_rows)
            entry["runs"].append({"parameter": "process_noise", "n": len(q_rows)})
        except Exception as exc:
            errors.append(f"{stem}/process_noise: {exc}")

    return rows, entry, errors


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="KAPT hyperparameter sensitivity (alpha, max_drift, process_noise).",
    )
    ap.add_argument("--kalman_dir", type=str, default=str(DEFAULT_KALMAN_DIR))
    ap.add_argument("--work_dir", type=str, default=str(DEFAULT_WORK_DIR))
    ap.add_argument(
        "--videos",
        type=str,
        default=",".join(DEFAULT_VIDEO_STEMS),
        help="Comma-separated video stems.",
    )
    ap.add_argument(
        "--params",
        type=str,
        default="alpha,max_drift,process_noise",
        help="Comma-separated parameters to sweep.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing variant JSON outputs.",
    )
    ap.add_argument(
        "--skip_generation",
        action="store_true",
        help="Evaluate only (variant JSONs must already exist).",
    )
    ap.add_argument(
        "--refresh_fused",
        action="store_true",
        help="Re-run Kalman pipeline to refresh fused JSONs (process_noise sweep).",
    )
    ap.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip PDF figure generation.",
    )
    return ap.parse_args()


def main() -> int:
    _ensure_project_root()
    args = parse_args()

    kalman_dir = Path(args.kalman_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    stems = [s.strip() for s in args.videos.split(",") if s.strip()]
    params: Set[SensitivityParam] = set()
    for p in args.params.split(","):
        p = p.strip()
        if p:
            params.add(p)  # type: ignore[arg-type]

    valid = {"alpha", "max_drift", "process_noise"}
    unknown = params - valid  # type: ignore[operator]
    if unknown:
        print(f"Unknown params: {unknown}. Valid: {valid}", file=sys.stderr)
        return 1

    if not kalman_dir.is_dir():
        print(f"Kalman dir not found: {kalman_dir}", file=sys.stderr)
        return 1

    all_rows: List[Dict[str, Any]] = []
    all_errors: List[str] = []
    manifest: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kalman_dir": str(kalman_dir),
        "work_dir": str(work_dir),
        "videos": stems,
        "params": sorted(params),
        "force": args.force,
        "skip_generation": args.skip_generation,
        "per_video": {},
    }

    for stem in stems:
        try:
            rows, entry, errors = process_video(
                stem,
                kalman_dir,
                work_dir,
                params,
                force=args.force,
                skip_generation=args.skip_generation,
                refresh_fused=args.refresh_fused,
            )
            all_rows.extend(rows)
            manifest["per_video"][stem] = entry
            all_errors.extend(errors)
        except Exception as exc:
            msg = f"{stem}: {exc}"
            all_errors.append(msg)
            print(f"[error] {msg}", file=sys.stderr)

    manifest_path = work_dir / "conditions" / "sensitivity_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not all_rows:
        print("No sensitivity results.", file=sys.stderr)
        return 1

    tables_dir = work_dir / "tables"
    figures_dir = work_dir / "figures"
    row_fields = (
        ["video", "parameter", "value", "method", "display_name"]
        + METRIC_COLUMNS
        + ["kalman_angle_viol_frac", "kalman_limb_std", "kalman_jitter", "kalman_accel_var"]
        + ["output_json", "kalman_json"]
    )
    _write_csv(tables_dir / "sensitivity_per_video.csv", all_rows, row_fields)

    summary = summarize_sensitivity(all_rows)
    summary_fields = ["parameter", "display_name", "value", "is_default", "n_videos"]
    for col in METRIC_COLUMNS:
        summary_fields.extend([f"{col}_mean", f"{col}_std"])
    _write_csv(tables_dir / "sensitivity_summary.csv", summary, summary_fields)

    report = format_sensitivity_report(all_rows, summary, manifest)
    report_path = work_dir / "reports" / "sensitivity_study.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    figure_paths: List[Path] = []
    if not args.no_plots:
        try:
            figure_paths = plot_sensitivity_figures(summary, figures_dir)
        except Exception as exc:
            all_errors.append(f"plotting: {exc}")
            print(f"[warn] Figure generation failed: {exc}", file=sys.stderr)

    print("")
    try:
        print(report)
    except UnicodeEncodeError:
        print("[done] Report written (console encoding skipped non-ASCII characters).")
    print(f"[done] Per-video:  {tables_dir / 'sensitivity_per_video.csv'}")
    print(f"[done] Summary:    {tables_dir / 'sensitivity_summary.csv'}")
    print(f"[done] Report:     {report_path}")
    print(f"[done] Manifest:   {manifest_path}")
    for fig in figure_paths:
        print(f"[done] Figure:     {fig}")

    if all_errors:
        print(f"\n[warn] {len(all_errors)} issue(s):", file=sys.stderr)
        for e in all_errors[:20]:
            print(f"  - {e}", file=sys.stderr)
        if len(all_errors) > 20:
            print(f"  ... and {len(all_errors) - 20} more", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
