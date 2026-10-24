"""
Run KAPT ablation study (A0–A4): generate variants, evaluate, aggregate.

Usage:
  python -m experiments.benchmark_comparison.run_ablation
  python -m experiments.benchmark_comparison.run_ablation --refresh_fused
  python -m experiments.benchmark_comparison.run_ablation --force
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from experiments.benchmark_comparison.ablation_config import (
    ABLATION_VARIANTS,
    AblationVariant,
    REFERENCE_METHOD,
)
from experiments.benchmark_comparison.ablation_stats import (
    ABLATION_DELTA_COLUMNS,
    ABLATION_METRIC_COLUMNS,
    FUSED_BASELINE_COLUMNS,
    build_wilcoxon_table,
    format_ablation_report,
    summarize_ablation,
)
from experiments.benchmark_comparison.config import (
    DEFAULT_KALMAN_DIR,
    DEFAULT_VIDEO_STEMS,
    DEFAULT_WORK_DIR,
    KALMAN_JSON_SUFFIX,
    KAPT_ALPHA,
    PROJECT_ROOT,
)
from experiments.benchmark_comparison.fair_evaluate import (
    evaluate_method_vs_kalman,
    evaluate_track,
)
from experiments.benchmark_comparison.kapt_runner import run_kapt_correction
from pose_handler.common_io import read_json, write_json


FUSED_SUFFIX = "_fused_keypoints.json"
FRAME_COUNT_TOLERANCE = 0


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


def _kalman_path(kalman_dir: Path, stem: str) -> Path:
    return kalman_dir / f"{stem}{KALMAN_JSON_SUFFIX}"


def _fused_work_path(work_dir: Path, stem: str) -> Path:
    return work_dir / "fused_raw" / f"{stem}{FUSED_SUFFIX}"


def _fused_kalman_path(kalman_dir: Path, stem: str) -> Path:
    return kalman_dir / f"{stem}{FUSED_SUFFIX}"


def resolve_video_path(stem: str) -> Optional[Path]:
    """Locate source video for Kalman pipeline re-run."""
    media = PROJECT_ROOT / "media"
    candidates = [
        media / f"{stem}.mp4",
        media / f"{stem}_original.mp4",
        media / f"{stem}-original.mp4",
        PROJECT_ROOT / f"{stem}.mp4",
    ]
    for p in candidates:
        if p.is_file():
            return p
    if media.is_dir():
        matches = sorted(media.glob(f"{stem}*.mp4"))
        if matches:
            return matches[0]
    return None


def run_kalman_pipeline(video_path: Path, out_dir: Path) -> None:
    """Re-run fusion + Kalman to produce smoothed and fused JSONs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "kalman.run_video_pipeline",
        "--video",
        str(video_path.resolve()),
        "--out_dir",
        str(out_dir.resolve()),
    ]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"Kalman pipeline failed for {video_path.name}: {err}")


def export_fused_from_smoothed(smoothed_json: Path, fused_out: Path) -> bool:
    """Fallback: extract keypoints_averaged from smoothed JSON if present."""
    data = read_json(smoothed_json)
    frames_in = data.get("frames", [])
    if not frames_in:
        return False
    fused_frames: List[Dict[str, Any]] = []
    for frame in frames_in:
        avg = frame.get("keypoints_averaged")
        if not isinstance(avg, dict):
            return False
        fused_frames.append({"keypoints": avg})
    fused_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "kalman/fused_keypoints_v1",
        "video_stem": data.get("video_stem", smoothed_json.stem),
        "fps": data.get("fps"),
        "image_size": data.get("image_size"),
        "num_frames": len(fused_frames),
        "frames": fused_frames,
        "source": str(smoothed_json),
    }
    write_json(fused_out, payload)
    return True


def ensure_fused_json(
    stem: str,
    kalman_dir: Path,
    work_dir: Path,
    *,
    refresh_fused: bool,
) -> Path:
    """
    Return path to fused pre-Kalman JSON.

    Order: work_dir/fused_raw → kalman_dir fused file → extract from smoothed → re-run pipeline.
    """
    work_fused = _fused_work_path(work_dir, stem)
    kalman_fused = _fused_kalman_path(kalman_dir, stem)
    kalman_json = _kalman_path(kalman_dir, stem)

    if refresh_fused:
        video = resolve_video_path(stem)
        if video is None:
            raise FileNotFoundError(
                f"Cannot refresh fused JSON for {stem}: no video found under media/"
            )
        print(f"[fused] Re-running Kalman pipeline for {stem} ({video.name}) ...")
        run_kalman_pipeline(video, kalman_dir)
        if kalman_fused.is_file():
            work_fused.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(kalman_fused, work_fused)
            return work_fused

    if work_fused.is_file():
        return work_fused
    if kalman_fused.is_file():
        work_fused.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(kalman_fused, work_fused)
        return work_fused
    if kalman_json.is_file() and export_fused_from_smoothed(kalman_json, work_fused):
        print(f"[fused] Extracted keypoints_averaged -> {work_fused.name}")
        return work_fused

    video = resolve_video_path(stem)
    if video is not None:
        print(f"[fused] Running Kalman pipeline for {stem} ({video.name}) ...")
        run_kalman_pipeline(video, kalman_dir)
        if kalman_fused.is_file():
            work_fused.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(kalman_fused, work_fused)
            return work_fused

    raise FileNotFoundError(
        f"No fused keypoints for {stem}. Run with --refresh_fused and place "
        f"{stem}.mp4 under media/, or re-run kalman pipeline to create {kalman_fused.name}."
    )


def validate_frame_counts(reference: Path, candidate: Path) -> None:
    ref = read_json(reference)
    cand = read_json(candidate)
    n_ref = len(ref.get("frames", []))
    n_cand = len(cand.get("frames", []))
    if n_ref != n_cand:
        raise ValueError(
            f"Frame count mismatch: {reference.name} has {n_ref}, "
            f"{candidate.name} has {n_cand}"
        )


def variant_output_path(work_dir: Path, variant: AblationVariant, stem: str) -> Path:
    return work_dir / "conditions" / variant.method_id / variant.output_filename(stem)


def generate_variant(
    stem: str,
    variant: AblationVariant,
    kalman_json: Path,
    fused_json: Path,
    work_dir: Path,
    *,
    force: bool,
    alpha: float,
) -> Path:
    """Generate one ablation output JSON; return output path."""
    out_path = variant_output_path(work_dir, variant, stem)

    if variant.input_kind == "passthrough":
        if out_path.is_file() and not force:
            return out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(kalman_json, out_path)
        return out_path

    if variant.input_kind == "kalman":
        in_json = kalman_json
    elif variant.input_kind == "fused":
        in_json = fused_json
    else:
        raise ValueError(f"Unknown input_kind: {variant.input_kind}")

    if not variant.run_correction:
        if out_path.is_file() and not force:
            return out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(in_json, out_path)
        return out_path

    print(
        f"[gen] {stem}/{variant.method_id}: "
        f"in={in_json.name} hard_clamp={variant.hard_clamp} max_drift={variant.max_drift}"
    )
    run_kapt_correction(
        in_json,
        out_path,
        alpha=alpha,
        hard_clamp=variant.hard_clamp,
        max_drift=variant.max_drift,
        force=force,
        variant_id=variant.method_id,
    )
    validate_frame_counts(kalman_json, out_path)
    return out_path


def sanity_check_no_correction(row: Dict[str, Any], tol: float = 1e-6) -> None:
    """A2 should match Kalman baseline metrics."""
    if row.get("method") != "ablation_no_correction":
        return
    for key in ABLATION_METRIC_COLUMNS:
        val = row.get(key, float("nan"))
        kalman_key = f"kalman_{key}" if key != "angle_viol_frac" else "kalman_angle_viol_frac"
        if kalman_key not in row:
            kalman_key = {
                "angle_viol_frac": "kalman_angle_viol_frac",
                "limb_std": "kalman_limb_std",
                "jitter": "kalman_jitter",
                "accel_var": "kalman_accel_var",
            }.get(key, f"kalman_{key}")
        kval = row.get(kalman_key, float("nan"))
        try:
            if abs(float(val) - float(kval)) > tol and float(val) > 0:
                print(
                    f"[warn] sanity A2 {row.get('video')}: {key}={val} vs kalman={kval}"
                )
        except (TypeError, ValueError):
            pass


def process_video(
    stem: str,
    kalman_dir: Path,
    work_dir: Path,
    *,
    refresh_fused: bool,
    force: bool,
    skip_generation: bool,
    alpha: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
    """Generate all variants and evaluate for one video stem."""
    errors: List[str] = []
    kalman_json = _kalman_path(kalman_dir, stem)
    if not kalman_json.is_file():
        raise FileNotFoundError(f"Kalman JSON not found: {kalman_json}")

    fused_json: Optional[Path] = None
    manifest_entry: Dict[str, Any] = {
        "video": stem,
        "kalman_json": str(kalman_json),
        "fused_json": None,
        "variants": {},
    }

    if not skip_generation:
        for variant in ABLATION_VARIANTS:
            try:
                in_fused = fused_json
                if variant.input_kind == "fused":
                    if in_fused is None:
                        in_fused = ensure_fused_json(
                            stem, kalman_dir, work_dir, refresh_fused=refresh_fused,
                        )
                        fused_json = in_fused
                        manifest_entry["fused_json"] = str(fused_json)
                out_path = generate_variant(
                    stem,
                    variant,
                    kalman_json,
                    in_fused if in_fused is not None else kalman_json,
                    work_dir,
                    force=force,
                    alpha=alpha,
                )
                manifest_entry["variants"][variant.method_id] = str(out_path)
            except Exception as exc:
                msg = f"{stem}/{variant.method_id}: {exc}"
                errors.append(msg)
                print(f"[error] {msg}", file=sys.stderr)

    rows: List[Dict[str, Any]] = []
    fused_metrics: Dict[str, float] = {}

    for variant in ABLATION_VARIANTS:
        out_path = variant_output_path(work_dir, variant, stem)
        if not out_path.is_file():
            msg = f"{stem}/{variant.method_id}: missing output {out_path}"
            if variant.method_id == "ablation_no_kalman":
                print(f"[skip] {msg} (need fused JSON: run with --refresh_fused)", file=sys.stderr)
            else:
                errors.append(msg)
                print(f"[skip] {msg}", file=sys.stderr)
            continue

        try:
            validate_frame_counts(kalman_json, out_path)
            row = evaluate_method_vs_kalman(
                kalman_json,
                out_path,
                variant.method_id,
                stem,
            )
            row["display_name"] = variant.display_name
            row["hard_clamp"] = variant.hard_clamp
            row["max_drift"] = variant.max_drift
            row["input_kind"] = variant.input_kind
            row["output_json"] = str(out_path)

            if variant.method_id == "ablation_no_kalman":
                if fused_json is None:
                    try:
                        fused_json = ensure_fused_json(
                            stem, kalman_dir, work_dir, refresh_fused=False,
                        )
                    except FileNotFoundError:
                        pass
                if fused_json is not None and fused_json.is_file():
                    fused_metrics = evaluate_track(fused_json)
                    row["fused_angle_viol_frac"] = fused_metrics["angle_viol_frac"]
                    row["fused_limb_std"] = fused_metrics["limb_std"]
                    row["fused_jitter"] = fused_metrics["jitter"]
                    row["fused_accel_var"] = fused_metrics["accel_var"]

            sanity_check_no_correction(row)
            rows.append(row)
            print(
                f"[eval] {stem}/{variant.method_id}: "
                f"limb_std={row['limb_std']:.4f} (d {row.get('limb_std_delta', float('nan')):+.4f}), "
                f"jitter={row['jitter']:.4f}, "
                f"angle_viol={row['angle_viol_frac']:.4f}"
            )
        except Exception as exc:
            msg = f"{stem}/{variant.method_id} eval: {exc}"
            errors.append(msg)
            print(f"[error] {msg}", file=sys.stderr)

    return rows, manifest_entry, errors


def merge_benchmark_table(
    work_dir: Path,
    ablation_rows: List[Dict[str, Any]],
) -> None:
    """Append ablation rows to benchmark_ablation.csv if it exists (keeps SmoothNet rows)."""
    bench_path = work_dir / "tables" / "benchmark_ablation.csv"
    existing: List[Dict[str, Any]] = []
    if bench_path.is_file():
        with bench_path.open(encoding="utf-8", newline="") as f:
            existing = list(csv.DictReader(f))

    ablation_methods = {v.method_id for v in ABLATION_VARIANTS}
    kept = [r for r in existing if r.get("method") not in ablation_methods]

    new_rows: List[Dict[str, Any]] = []
    for r in ablation_rows:
        new_rows.append({
            "video": r["video"],
            "method": r["method"],
            "accel_var": r.get("accel_var"),
            "viol_frac": r.get("angle_viol_frac"),
            "limb_std": r.get("limb_std"),
            "jitter": r.get("jitter"),
            "accel_var_orig": r.get("kalman_accel_var"),
            "viol_frac_orig": r.get("kalman_angle_viol_frac"),
            "limb_std_orig": r.get("kalman_limb_std"),
            "success": True,
        })

    fieldnames = [
        "video", "method", "accel_var", "viol_frac", "limb_std", "jitter",
        "accel_var_orig", "viol_frac_orig", "limb_std_orig", "success",
    ]
    _write_csv(bench_path, kept + new_rows, fieldnames)
    print(f"[done] Merged ablation rows into {bench_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="KAPT ablation study (A0–A4).")
    ap.add_argument("--kalman_dir", type=str, default=str(DEFAULT_KALMAN_DIR))
    ap.add_argument("--work_dir", type=str, default=str(DEFAULT_WORK_DIR))
    ap.add_argument(
        "--videos",
        type=str,
        default=",".join(DEFAULT_VIDEO_STEMS),
        help="Comma-separated video stems.",
    )
    ap.add_argument(
        "--refresh_fused",
        action="store_true",
        help="Re-run Kalman pipeline per video to create fused JSON exports.",
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
        "--merge_benchmark_table",
        action="store_true",
        help="Append ablation rows to tables/benchmark_ablation.csv.",
    )
    ap.add_argument("--alpha", type=float, default=KAPT_ALPHA)
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

    all_rows: List[Dict[str, Any]] = []
    all_errors: List[str] = []
    manifest: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kalman_dir": str(kalman_dir),
        "work_dir": str(work_dir),
        "videos": stems,
        "variants": [v.method_id for v in ABLATION_VARIANTS],
        "alpha": args.alpha,
        "refresh_fused": args.refresh_fused,
        "per_video": {},
    }

    for stem in stems:
        try:
            rows, entry, errors = process_video(
                stem,
                kalman_dir,
                work_dir,
                refresh_fused=args.refresh_fused,
                force=args.force,
                skip_generation=args.skip_generation,
                alpha=args.alpha,
            )
            all_rows.extend(rows)
            manifest["per_video"][stem] = entry
            all_errors.extend(errors)
        except Exception as exc:
            msg = f"{stem}: {exc}"
            all_errors.append(msg)
            print(f"[error] {msg}", file=sys.stderr)

    manifest_path = work_dir / "conditions" / "ablation_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not all_rows:
        print("No ablation results.", file=sys.stderr)
        return 1

    tables_dir = work_dir / "tables"
    row_fields = (
        ["video", "method", "display_name", "input_kind", "hard_clamp", "max_drift", "output_json"]
        + ABLATION_METRIC_COLUMNS
        + ["kalman_angle_viol_frac", "kalman_limb_std", "kalman_jitter", "kalman_accel_var"]
        + ["frames_needing_kapt_frac", "frames_needing_kapt_count"]
        + ABLATION_DELTA_COLUMNS
        + FUSED_BASELINE_COLUMNS
    )
    _write_csv(tables_dir / "ablation_per_video.csv", all_rows, row_fields)

    summary = summarize_ablation(all_rows)
    summary_fields = ["method", "display_name", "n_videos"]
    for col in ABLATION_METRIC_COLUMNS + ABLATION_DELTA_COLUMNS:
        summary_fields.extend([f"{col}_mean", f"{col}_std"])
    _write_csv(tables_dir / "ablation_summary.csv", summary, summary_fields)

    wilcoxon_rows = build_wilcoxon_table(all_rows)
    wilcoxon_fields = [
        "metric", "ablation_method", "n_pairs", "kapt_full_mean", "ablation_mean",
        "statistic", "p_value", "winner", "significant_p05", "higher_is_better",
    ]
    _write_csv(tables_dir / "ablation_wilcoxon.csv", wilcoxon_rows, wilcoxon_fields)

    report = format_ablation_report(all_rows, summary, wilcoxon_rows, manifest)
    report_path = work_dir / "reports" / "ablation_study.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    if args.merge_benchmark_table:
        merge_benchmark_table(work_dir, all_rows)

    print("")
    try:
        print(report)
    except UnicodeEncodeError:
        print("[done] Report written (console encoding skipped non-ASCII characters).")
    print(f"[done] Per-video:  {tables_dir / 'ablation_per_video.csv'}")
    print(f"[done] Summary:    {tables_dir / 'ablation_summary.csv'}")
    print(f"[done] Wilcoxon:   {tables_dir / 'ablation_wilcoxon.csv'}")
    print(f"[done] Report:     {report_path}")
    print(f"[done] Manifest:   {manifest_path}")

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
