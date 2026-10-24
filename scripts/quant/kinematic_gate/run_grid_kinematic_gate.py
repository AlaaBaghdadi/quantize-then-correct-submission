#!/usr/bin/env python3
"""
Run kinematic delta gate on all fp16_grid vs vidit_w8a8_grid clip pairs.

Compares symmetric videos under outputs/fp16_grid/<clip_id>/ and
outputs/vidit_w8a8_grid/<clip_id>/ using the same thresholds as
run_kinematic_gate.py. Clips are processed sequentially (OpenPose is not
safe to parallelize on a shared install).

Example:
  python scripts/quant/kinematic_gate/run_grid_kinematic_gate.py \\
    --config configs/quant/vidit_ptq.json \\
    --phase1-root ../phase1

Uses .venv_kinematic_gate (mediapipe==0.10.21) automatically when present.

Exit codes: 0 = all clips pass, 1 = one or more clips fail gate, 2 = setup error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
QUANT_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = QUANT_DIR.parent.parent

if str(QUANT_DIR) not in sys.path:
    sys.path.insert(0, str(QUANT_DIR))


def resolve_repo_path(repo_root: str | Path, path: str | Path) -> Path:
    """Resolve repo-relative paths against --repo-root."""
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(repo_root).resolve() / p


def add_repo_to_path(repo_root: str | Path) -> Path:
    """Add repo root to sys.path for downstream imports."""
    repo = Path(repo_root).resolve()
    sys.path.insert(0, str(repo))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return repo


from run_kinematic_gate import run_gate
from vidit_ptq_config import load_grid_config, load_vidit_ptq_config


def clip_id(image_path: str, motion_path: str) -> str:
    return f"{Path(image_path).stem}_{Path(motion_path).stem}"


def iter_grid_pairs(images: list[str], motions: list[str]) -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []
    for image in images:
        for motion in motions:
            pairs.append((clip_id(image, motion), image, motion))
    return pairs


def fp16_video_path(out_dir: Path, cid: str, seed: int) -> Path:
    return out_dir / f"fp16_{cid}_seed{seed}.mp4"


def quant_video_path(out_dir: Path, cid: str, seed: int, prefix: str) -> Path:
    return out_dir / f"{prefix}_{cid}_seed{seed}.mp4"


def vidit_video_path(out_dir: Path, cid: str, seed: int) -> Path:
    return quant_video_path(out_dir, cid, seed, "vidit_w8a8")


def discover_clip_ids(quant_root: Path) -> list[str]:
    """Clip IDs from subdirectories of the quant grid output that contain an MP4."""
    clip_ids: list[str] = []
    if not quant_root.is_dir():
        return clip_ids
    for child in sorted(quant_root.iterdir()):
        if child.is_dir() and list(child.glob("*.mp4")):
            clip_ids.append(child.name)
    return clip_ids


def resolve_grid_clips(
    *,
    repo: Path,
    config_path: Path,
    fp16_root: Path | None,
    quant_root: Path | None,
    clip_filter: set[str] | None,
    discover: bool,
) -> tuple[list[str], Path, Path, int]:
    cfg = load_vidit_ptq_config(config_path)
    grid = load_grid_config(config_path, repo_root=repo)

    resolved_fp16 = fp16_root or resolve_repo_path(repo, grid.fp16_output_dir)
    resolved_quant = quant_root or resolve_repo_path(repo, grid.vidit_output_dir)
    seed = cfg.SEED

    if discover:
        clip_ids = discover_clip_ids(resolved_quant)
        if not clip_ids:
            raise FileNotFoundError(
                f"No clip subdirectories with MP4 files under {resolved_quant}"
            )
    else:
        clip_ids = [cid for cid, _, _ in iter_grid_pairs(grid.images, grid.motions)]

    if clip_filter:
        clip_ids = [cid for cid in clip_ids if cid in clip_filter]
        missing = clip_filter - set(clip_ids)
        if missing:
            raise ValueError(f"Unknown or missing clip id(s): {', '.join(sorted(missing))}")

    return clip_ids, resolved_fp16, resolved_quant, seed


def build_clip_gate_args(
    *,
    base: Namespace,
    repo: Path,
    clip_id: str,
    fp16_mp4: Path,
    quant_mp4: Path,
    report_path: Path,
) -> Namespace:
    return Namespace(
        repo_root=str(repo),
        config=base.config,
        phase1_root=base.phase1_root,
        openpose_root=base.openpose_root,
        phase1_python=base.phase1_python,
        fp16_video=str(fp16_mp4.resolve()),
        quant_video=str(quant_mp4.resolve()),
        clip_id=clip_id,
        cache_dir=base.cache_dir,
        report_json=str(report_path.resolve()),
        skip_if_cached=base.skip_if_cached,
        no_cache=base.no_cache,
        keep_openpose_temp=base.keep_openpose_temp,
    )


def run_grid_gate(args: argparse.Namespace) -> int:
    repo = add_repo_to_path(args.repo_root)
    config_path = (
        Path(args.config).resolve()
        if args.config
        else resolve_repo_path(repo, "configs/quant/vidit_ptq.json")
    )

    clip_filter = set(args.clip_id) if args.clip_id else None
    fp16_override = Path(args.fp16_root).resolve() if args.fp16_root else None
    quant_override = None
    if args.quant_root:
        quant_override = Path(args.quant_root).resolve()
    elif args.vidit_root:
        quant_override = Path(args.vidit_root).resolve()

    clip_ids, fp16_root, quant_root, seed = resolve_grid_clips(
        repo=repo,
        config_path=config_path,
        fp16_root=fp16_override,
        quant_root=quant_override,
        clip_filter=clip_filter,
        discover=args.discover,
    )

    quant_prefix = args.quant_prefix

    report_dir = resolve_repo_path(repo, args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = resolve_repo_path(repo, args.summary_json)

    print(
        f"Grid kinematic gate: {len(clip_ids)} clip(s), seed={seed}\n"
        f"  fp16_root={fp16_root}\n"
        f"  quant_root={quant_root}\n"
        f"  quant_prefix={quant_prefix}\n"
        f"  report_dir={report_dir}"
    )

    results: list[dict] = []
    failed: list[str] = []
    skipped: list[str] = []
    setup_errors: list[str] = []

    for index, clip_id in enumerate(clip_ids, start=1):
        fp16_mp4 = fp16_video_path(fp16_root / clip_id, clip_id, seed)
        quant_mp4 = quant_video_path(quant_root / clip_id, clip_id, seed, quant_prefix)

        if not fp16_mp4.is_file() or not quant_mp4.is_file():
            message = []
            if not fp16_mp4.is_file():
                message.append(f"missing FP16 {fp16_mp4}")
            if not quant_mp4.is_file():
                message.append(f"missing quant {quant_mp4}")
            detail = "; ".join(message)
            if args.skip_missing:
                print(f"[{index}/{len(clip_ids)}] SKIP {clip_id}: {detail}")
                skipped.append(clip_id)
                results.append({"clip_id": clip_id, "skipped": True, "reason": detail})
                continue
            raise FileNotFoundError(f"{clip_id}: {detail}")

        print(f"\n[{index}/{len(clip_ids)}] === {clip_id} ===")
        clip_report = report_dir / f"{clip_id}.json"
        clip_args = build_clip_gate_args(
            base=args,
            repo=repo,
            clip_id=clip_id,
            fp16_mp4=fp16_mp4,
            quant_mp4=quant_mp4,
            report_path=clip_report,
        )

        try:
            rc = run_gate(clip_args)
        except (FileNotFoundError, RuntimeError, ValueError, TypeError) as exc:
            print(f"ERROR [{clip_id}]: {exc}", file=sys.stderr)
            setup_errors.append(clip_id)
            results.append({"clip_id": clip_id, "passed": False, "error": str(exc)})
            if not args.continue_on_error:
                break
            continue

        passed = rc == 0
        if not passed:
            failed.append(clip_id)
        results.append({"clip_id": clip_id, "passed": passed, "report_json": str(clip_report)})

        if not passed and not args.continue_on_error:
            break

    passed_count = sum(1 for r in results if r.get("passed") is True)
    evaluated = passed_count + len(failed)
    summary = {
        "passed_all": evaluated > 0 and not failed and not setup_errors,
        "total_clips": len(clip_ids),
        "evaluated": evaluated,
        "passed": passed_count,
        "failed": failed,
        "skipped": skipped,
        "setup_errors": setup_errors,
        "fp16_root": str(fp16_root),
        "quant_root": str(quant_root),
        "quant_prefix": quant_prefix,
        "vidit_root": str(quant_root),
        "seed": seed,
        "clips": results,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[kinematic_gate_grid] summary: {summary_path}")
    print(
        f"[kinematic_gate_grid] passed={passed_count} "
        f"failed={len(failed)} skipped={len(skipped)} setup_errors={len(setup_errors)}"
    )

    if setup_errors:
        return 2
    if failed:
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run kinematic delta gate on all fp16_grid vs vidit_w8a8_grid symmetric pairs."
        ),
        epilog=(
            "Processes clips sequentially. Do not run multiple gate processes in parallel "
            "against the same OpenPose install."
        ),
    )
    parser.add_argument("--repo-root", default=str(PROJECT_ROOT), help="phase2 project root")
    parser.add_argument(
        "--config",
        default=None,
        help="vidit_ptq.json (default: repo configs/quant/vidit_ptq.json)",
    )
    parser.add_argument("--phase1-root", default=None, help="Path to phase1 repo")
    parser.add_argument(
        "--phase1-python",
        default=None,
        help="Python executable for kalman.run_video_pipeline (default: current interpreter)",
    )
    parser.add_argument("--openpose-root", default=None, help="OpenPose install root")
    parser.add_argument(
        "--fp16-root",
        default=None,
        help="Override fp16 grid root (default: grid.fp16_output_dir from config)",
    )
    parser.add_argument(
        "--vidit-root",
        default=None,
        help="Override quant grid root (alias for --quant-root; default: grid.vidit_output_dir)",
    )
    parser.add_argument(
        "--quant-root",
        default=None,
        help="Override quantized video grid root (default: grid.vidit_output_dir from config)",
    )
    parser.add_argument(
        "--quant-prefix",
        default="vidit_w8a8",
        help="MP4 filename prefix for quant videos (default: vidit_w8a8; use hybrid_w8a8 for hybrid map)",
    )
    parser.add_argument(
        "--clip-id",
        action="append",
        default=None,
        help="Run gate for specific clip id only (repeatable, e.g. human2_motion1)",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Discover clip ids from vidit grid subdirs instead of config image×motion list",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip clips whose FP16 or ViDiT MP4 is missing instead of failing",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep evaluating remaining clips after a failure (default: true)",
    )
    parser.add_argument(
        "--report-dir",
        default="./logs/kinematic_gate_grid",
        help="Directory for per-clip gate JSON reports",
    )
    parser.add_argument(
        "--summary-json",
        default="./logs/kinematic_gate_grid.json",
        help="Aggregate summary report path",
    )
    parser.add_argument("--cache-dir", default=None, help="Kinematic cache root")
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
        return run_grid_gate(args)
    except (FileNotFoundError, RuntimeError, ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
