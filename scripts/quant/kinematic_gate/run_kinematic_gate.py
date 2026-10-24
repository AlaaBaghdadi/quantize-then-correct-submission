#!/usr/bin/env python3
"""
Kinematic delta gate: compare FP16 vs quantized generated videos on pose metrics.

Extracts poses via phase1 (OpenPose + MediaPipe + Kalman), computes the same
metric vector on both videos, and gates on *deltas* (not absolute violations).

Manual smoke test (requires OpenPose binary + phase1 deps):
  python scripts/quant/kinematic_gate/run_kinematic_gate.py \\
    --phase1-root C:/path/to/phase1 \\
    --fp16-video outputs/fp16/demo.mp4 \\
    --quant-video outputs/vidit_w8a8/demo.mp4 \\
    --clip-id demo

Chain after quant benchmark:
  python scripts/quant/apply_and_benchmark_vidit.py ... && \\
  python scripts/quant/kinematic_gate/run_kinematic_gate.py --config configs/quant/vidit_ptq.json \\
    --fp16-video <fp16_mp4> --quant-video <quant_mp4> --clip-id human2_motion1

All grid clips (fp16_grid vs vidit_w8a8_grid):
  python scripts/quant/kinematic_gate/run_grid_kinematic_gate.py \\
    --config configs/quant/vidit_ptq.json --phase1-root ../phase1

Exit codes: 0 = pass, 1 = gate fail, 2 = setup/preflight error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
QUANT_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = QUANT_DIR.parent.parent

if str(QUANT_DIR) not in sys.path:
    sys.path.insert(0, str(QUANT_DIR))

from kinematic_gate.compare import GateThresholds, compute_deltas, evaluate_gate
from kinematic_gate.metrics import (
    compute_joint_movement_summary_between,
    compute_metric_vector,
    load_keypoints_json,
    strip_internal_fields,
)
from kinematic_gate.phase1_bridge import (
    extract_keypoints,
    resolve_openpose_root,
    resolve_phase1_root,
)
from kinematic_gate.security import safe_child_path, sanitize_clip_id, validate_video_path
from vidit_ptq_config import load_kinematic_gate_config


def _resolve_path(base: Path, p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def build_report(
    *,
    clip_id: str,
    fp16_video: Path,
    quant_video: Path,
    fp16_keypoints: Path,
    quant_keypoints: Path,
    metrics_fp16: Dict[str, Any],
    metrics_quant: Dict[str, Any],
    movement_summary: Dict[str, float],
    deltas: Dict[str, float],
    passed: bool,
    checks: Dict[str, bool],
    thresholds: GateThresholds,
) -> Dict[str, Any]:
    return {
        "clip_id": clip_id,
        "passed": passed,
        "fp16_video": str(fp16_video.resolve()),
        "quant_video": str(quant_video.resolve()),
        "fp16_keypoints_json": str(fp16_keypoints.resolve()),
        "quant_keypoints_json": str(quant_keypoints.resolve()),
        "metrics_fp16": strip_internal_fields(metrics_fp16),
        "metrics_quant": strip_internal_fields(metrics_quant),
        "movement_summary": movement_summary,
        "deltas": deltas,
        "checks": checks,
        "thresholds": {
            "max_rel_delta_accel_variance": thresholds.max_rel_delta_accel_variance,
            "max_rel_delta_limb_std": thresholds.max_rel_delta_limb_std,
            "max_rel_delta_jitter": thresholds.max_rel_delta_jitter,
            "max_abs_delta_violation_fraction": thresholds.max_abs_delta_violation_fraction,
            "max_mean_npe": thresholds.max_mean_npe,
            "max_p95_npe": thresholds.max_p95_npe,
            "max_under_motion_violations": thresholds.max_under_motion_violations,
        },
    }


def run_gate(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config).resolve() if args.config else None

    kg_cfg = None
    if config_path is not None:
        kg_cfg = load_kinematic_gate_config(config_path, repo_root=repo_root)

    phase1_root = resolve_phase1_root(
        Path(args.phase1_root) if args.phase1_root else (kg_cfg.phase1_root if kg_cfg else None),
        repo_root=repo_root,
    )
    openpose_root = resolve_openpose_root(
        Path(args.openpose_root) if args.openpose_root else (kg_cfg.openpose_root if kg_cfg else None),
        phase1_root=phase1_root,
    )
    phase1_python = (
        Path(args.phase1_python).resolve()
        if getattr(args, "phase1_python", None)
        else None
    )

    fp16_video = validate_video_path(_resolve_path(repo_root, args.fp16_video))
    quant_video = validate_video_path(_resolve_path(repo_root, args.quant_video))
    clip_id = sanitize_clip_id(
        args.clip_id or f"{fp16_video.stem}_vs_{quant_video.stem}"
    )

    cache_dir = _resolve_path(
        repo_root,
        args.cache_dir or (kg_cfg.cache_dir if kg_cfg else "./kinematic_cache"),
    )
    clip_cache = safe_child_path(cache_dir, "keypoints", clip_id)

    skip_if_cached = args.skip_if_cached
    if kg_cfg is not None and not args.no_cache:
        skip_if_cached = kg_cfg.skip_if_cached

    keep_staging = args.keep_openpose_temp

    # Sequential extraction (OpenPose shared output folder)
    fp16_keypoints = extract_keypoints(
        fp16_video,
        phase1_root=phase1_root,
        openpose_root=openpose_root,
        out_dir=clip_cache,
        clip_id=clip_id,
        role="fp16",
        skip_if_cached=skip_if_cached,
        keep_staging=keep_staging,
        phase1_python=phase1_python,
        repo_root=repo_root,
    )
    quant_keypoints = extract_keypoints(
        quant_video,
        phase1_root=phase1_root,
        openpose_root=openpose_root,
        out_dir=clip_cache,
        clip_id=clip_id,
        role="quant",
        skip_if_cached=skip_if_cached,
        keep_staging=keep_staging,
        phase1_python=phase1_python,
        repo_root=repo_root,
    )

    metrics_fp16 = compute_metric_vector(fp16_keypoints, phase1_root=phase1_root)
    metrics_quant = compute_metric_vector(quant_keypoints, phase1_root=phase1_root)

    n_fp16 = int(metrics_fp16.get("num_frames", 0))
    n_quant = int(metrics_quant.get("num_frames", 0))
    if n_fp16 != n_quant:
        raise RuntimeError(
            f"Frame count mismatch: FP16={n_fp16}, quant={n_quant}. "
            "Videos must have the same number of frames."
        )
    if n_fp16 == 0:
        raise RuntimeError("No frames in keypoints JSON.")

    data_fp16 = load_keypoints_json(fp16_keypoints)
    data_quant = load_keypoints_json(quant_keypoints)
    movement_summary = compute_joint_movement_summary_between(data_fp16, data_quant)

    if kg_cfg is not None:
        thresholds = GateThresholds(
            max_rel_delta_accel_variance=kg_cfg.max_rel_delta_accel_variance,
            max_rel_delta_limb_std=kg_cfg.max_rel_delta_limb_std,
            max_rel_delta_jitter=kg_cfg.max_rel_delta_jitter,
            max_abs_delta_violation_fraction=kg_cfg.max_abs_delta_violation_fraction,
            max_mean_npe=kg_cfg.max_mean_npe,
            max_p95_npe=kg_cfg.max_p95_npe,
            max_under_motion_violations=kg_cfg.max_under_motion_violations,
        )
    else:
        thresholds = GateThresholds()

    deltas = compute_deltas(
        strip_internal_fields(metrics_fp16),
        strip_internal_fields(metrics_quant),
        movement_summary=movement_summary,
    )
    passed, checks = evaluate_gate(
        strip_internal_fields(metrics_fp16),
        strip_internal_fields(metrics_quant),
        deltas,
        thresholds,
    )

    report = build_report(
        clip_id=clip_id,
        fp16_video=fp16_video,
        quant_video=quant_video,
        fp16_keypoints=fp16_keypoints,
        quant_keypoints=quant_keypoints,
        metrics_fp16=metrics_fp16,
        metrics_quant=metrics_quant,
        movement_summary=movement_summary,
        deltas=deltas,
        passed=passed,
        checks=checks,
        thresholds=thresholds,
    )

    report_path = _resolve_path(
        repo_root,
        args.report_json or (kg_cfg.report_json if kg_cfg else "./logs/kinematic_gate.json"),
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[kinematic_gate] report: {report_path}")
    print(f"[kinematic_gate] passed={passed}")
    for name, ok in sorted(checks.items()):
        status = "OK" if ok else "FAIL"
        print(f"  {name}: {status}")

    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Kinematic delta gate: extract poses (OpenPose+MediaPipe+Kalman) from FP16 and "
            "quantized videos, compare kinematic metrics, exit 0 on pass."
        ),
        epilog=(
            "Requires phase1 deps (mediapipe==0.10.21, opencv) and OpenPoseDemo.exe. "
            "Auto-uses .venv_kinematic_gate when present (see scripts/quant/kinematic_gate/requirements.txt). "
            "Do not run two gate processes in parallel on the same OpenPose root."
        ),
    )
    parser.add_argument("--repo-root", default=str(PROJECT_ROOT), help="phase2 project root")
    parser.add_argument(
        "--config",
        default=None,
        help="vidit_ptq.json (optional; loads kinematic_gate section)",
    )
    parser.add_argument("--phase1-root", default=None, help="Path to phase1 repo (default: ../phase1)")
    parser.add_argument(
        "--phase1-python",
        default=None,
        help="Python executable for kalman.run_video_pipeline (default: current interpreter)",
    )
    parser.add_argument("--openpose-root", default=None, help="OpenPose install root (default: under phase1)")
    parser.add_argument("--fp16-video", required=True, help="FP16 baseline generated MP4")
    parser.add_argument("--quant-video", required=True, help="Quantized generated MP4")
    parser.add_argument("--clip-id", default=None, help="Clip identifier for cache keys")
    parser.add_argument("--cache-dir", default=None, help="Kinematic cache root (default from config)")
    parser.add_argument("--report-json", default=None, help="Output report JSON path")
    parser.add_argument(
        "--skip-if-cached",
        action="store_true",
        default=True,
        help="Reuse keypoints JSON when newer than source MP4 (default: true)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force re-extraction even if cache exists",
    )
    parser.add_argument(
        "--keep-openpose-temp",
        action="store_true",
        help="Keep staged video copies under cache/_staging",
    )
    args = parser.parse_args()
    if args.no_cache:
        args.skip_if_cached = False

    try:
        return run_gate(args)
    except (FileNotFoundError, RuntimeError, ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
