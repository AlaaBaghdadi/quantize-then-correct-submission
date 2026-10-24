#!/usr/bin/env python3
"""Generate hybrid block-map W8A8 videos (FP16 blocks 1-13 + INT8 blocks 14-40)."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from quant_common import (
    add_repo_to_path,
    apply_w8a8,
    args_namespace,
    load_quantize_names,
    parse_dtype,
    prepare_control_video,
    prepare_fp16_pipeline,
    resolve_hybrid_fp16_layers,
    resolve_repo_path,
    run_inference,
)
from vidit_ptq_config import load_grid_config, load_hybrid_map_config, load_vidit_ptq_config
from vidit_quant import apply_vidit_w8a8


def clip_id(image_path: str, motion_path: str) -> str:
    return f"{Path(image_path).stem}_{Path(motion_path).stem}"


def iter_grid_pairs(images: list[str], motions: list[str]) -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []
    for image in images:
        for motion in motions:
            pairs.append((clip_id(image, motion), image, motion))
    return pairs


def hybrid_video_path(out_dir: Path, cid: str, seed: int, prefix: str) -> Path:
    return out_dir / f"{prefix}_{cid}_seed{seed}.mp4"


def make_clip_args(base: SimpleNamespace, image: str, motion: str) -> SimpleNamespace:
    clip_args = copy.copy(base)
    clip_args.reference_image = image
    clip_args.driving_video = motion
    return clip_args


def build_base_args(cfg: SimpleNamespace) -> SimpleNamespace:
    return args_namespace(
        prompt=cfg.PROMPT,
        num_inference_steps=cfg.NUM_INFERENCE_STEPS,
        width=cfg.WIDTH,
        height=cfg.HEIGHT,
        num_frames=cfg.NUM_FRAMES,
        pose_control_function=cfg.POSE_CONTROL_FUNCTION,
        guidance_scale=cfg.GUIDANCE_SCALE,
        seed=cfg.SEED,
        fps=cfg.FPS,
        reference_image=cfg.REFERENCE_IMAGE,
        driving_video=cfg.DRIVING_VIDEO,
    )


def resolve_sweep_assets(
    hybrid_cfg: SimpleNamespace,
    repo: Path,
    *,
    full_grid: bool,
    grid: SimpleNamespace,
) -> tuple[list[str], list[str]]:
    if full_grid:
        images = [str(resolve_repo_path(repo, p)) for p in grid.images]
        motions = [str(resolve_repo_path(repo, p)) for p in grid.motions]
        return images, motions
    image = str(resolve_repo_path(repo, hybrid_cfg.sweep_image))
    motions = [str(resolve_repo_path(repo, p)) for p in hybrid_cfg.sweep_motions]
    return [image], motions


def print_hybrid_map_stats(
    *,
    block_fp16: tuple[int, int],
    block_int8: tuple[int, int],
    fp16_layers: set[str],
    actual_int8_layers: set[str],
    quantize_names: set[str],
) -> None:
    block_coverage = len(range(block_int8[0], block_int8[1] + 1))
    total_middle_blocks = len(range(block_fp16[0], block_int8[1] + 1))
    layer_pct = 100.0 * len(actual_int8_layers) / len(quantize_names)
    print(
        f"Hybrid map: blocks {block_fp16[0]}-{block_fp16[1]} FP16, "
        f"blocks {block_int8[0]}-{block_int8[1]} INT8"
    )
    print(
        f"  FP16 skip: {len(fp16_layers)} layers, INT8 targets: "
        f"{len(actual_int8_layers)} ({layer_pct:.1f}%), "
        f"block coverage: {block_coverage}/{total_middle_blocks}"
    )


def run_hybrid_algorithm_grid(
    *,
    algorithm: str,
    model_path: str,
    dtype: torch.dtype,
    quantize_names: set[str],
    fp16_layers: set[str],
    actual_int8_layers: set[str],
    block_fp16: tuple[int, int],
    block_int8: tuple[int, int],
    base_args: SimpleNamespace,
    pairs: list[tuple[str, str, str]],
    output_root: Path,
    quant_prefix: str,
    calib_root: Path | None,
    smoothquant_alpha: float,
    skip_existing: bool,
) -> tuple[int, int]:
    from diffusers.utils import export_to_video

    if algorithm not in ("naive", "vidit"):
        raise ValueError(f"Unknown algorithm: {algorithm}")

    output_root.mkdir(parents=True, exist_ok=True)
    generated = 0
    skipped = 0
    meta_name = "hybrid_meta.json" if algorithm == "naive" else "hybrid_vidit_meta.json"
    algo_label = "hybrid_naive_w8a8" if algorithm == "naive" else "hybrid_vidit_w8a8"

    print(f"\n=== Algorithm: {algorithm} -> {output_root} (prefix={quant_prefix}) ===")

    for cid, image, motion in pairs:
        out_dir = output_root / cid
        out_dir.mkdir(parents=True, exist_ok=True)
        out_mp4 = hybrid_video_path(out_dir, cid, base_args.seed, quant_prefix)
        meta_path = out_dir / meta_name

        if skip_existing and out_mp4.is_file():
            print(f"Skip existing: {out_mp4.name}")
            skipped += 1
            continue

        print(f"\n--- Hybrid {algorithm.upper()} {cid} ---")
        clip_args = make_clip_args(base_args, image, motion)
        pipe = prepare_fp16_pipeline(model_path, dtype)
        try:
            if algorithm == "naive":
                meta = apply_w8a8(
                    pipe.transformer,
                    quantize_names,
                    fp16_layers=fp16_layers,
                )
            else:
                if calib_root is None:
                    raise ValueError("calib_root is required for ViDiT algorithm")
                calib_path = calib_root / f"{cid}_vidit.pt"
                if not calib_path.is_file():
                    raise FileNotFoundError(
                        f"Missing {calib_path}. Run ViDiT calib for {cid} first "
                        "(ViDiT notebook batch cell 2 or sweep mini-calib)."
                    )
                meta = apply_vidit_w8a8(
                    pipe.transformer,
                    quantize_names,
                    calib_path,
                    smoothquant_alpha=smoothquant_alpha,
                    actual_int8_layers=actual_int8_layers,
                )

            print(
                f"  Quantized {meta['quantized_layers']} Linear layers "
                f"({len(meta.get('early_fp16_layers', fp16_layers))} FP16 skip)"
            )
            image_obj, control = prepare_control_video(
                image,
                motion,
                base_args.width,
                base_args.height,
                base_args.num_frames,
            )
            frames = run_inference(pipe, clip_args, image_obj, control)
            export_to_video(
                frames[: base_args.num_frames],
                str(out_mp4),
                fps=base_args.fps,
            )
            hybrid_meta = {
                "clip_id": cid,
                "seed": base_args.seed,
                "algorithm": algo_label,
                "hybrid_block_fp16": list(block_fp16),
                "hybrid_block_int8": list(block_int8),
                "smoothquant": algorithm == "vidit",
                **meta,
            }
            meta_path.write_text(json.dumps(hybrid_meta, indent=2), encoding="utf-8")
            print(f"  -> {out_mp4}")
            generated += 1
        finally:
            del pipe
            gc.collect()
            torch.cuda.empty_cache()

    print(
        f"\n[{algorithm}] complete: {generated} generated, {skipped} skipped, "
        f"{len(pairs)} total clips"
    )
    return generated, skipped


def run_hybrid_grid(
    *,
    algorithms: list[str],
    model_path: str,
    dtype: torch.dtype,
    quantize_layers: Path,
    base_args: SimpleNamespace,
    images: list[str],
    motions: list[str],
    hybrid_cfg: SimpleNamespace,
    block_fp16: tuple[int, int],
    block_int8: tuple[int, int],
    skip_existing: bool,
    clip_filter: set[str] | None = None,
    smoothquant_alpha: float = 0.5,
) -> dict[str, tuple[int, int]]:
    quantize_names = load_quantize_names(quantize_layers)
    fp16_layers, actual_int8_layers = resolve_hybrid_fp16_layers(
        quantize_names, block_fp16, block_int8
    )
    print_hybrid_map_stats(
        block_fp16=block_fp16,
        block_int8=block_int8,
        fp16_layers=fp16_layers,
        actual_int8_layers=actual_int8_layers,
        quantize_names=quantize_names,
    )

    pairs = iter_grid_pairs(images, motions)
    if clip_filter:
        pairs = [(cid, img, mot) for cid, img, mot in pairs if cid in clip_filter]
        missing = clip_filter - {cid for cid, _, _ in pairs}
        if missing:
            raise ValueError(f"Unknown clip id(s): {', '.join(sorted(missing))}")

    calib_root = Path(hybrid_cfg.calib_dir) if hybrid_cfg.calib_dir else None
    results: dict[str, tuple[int, int]] = {}

    output_by_algo = {
        "naive": (Path(hybrid_cfg.naive_output_dir), hybrid_cfg.naive_quant_prefix),
        "vidit": (Path(hybrid_cfg.vidit_output_dir), hybrid_cfg.vidit_quant_prefix),
    }

    for algorithm in algorithms:
        output_root, quant_prefix = output_by_algo[algorithm]
        generated, skipped = run_hybrid_algorithm_grid(
            algorithm=algorithm,
            model_path=model_path,
            dtype=dtype,
            quantize_names=quantize_names,
            fp16_layers=fp16_layers,
            actual_int8_layers=actual_int8_layers,
            block_fp16=block_fp16,
            block_int8=block_int8,
            base_args=base_args,
            pairs=pairs,
            output_root=output_root,
            quant_prefix=quant_prefix,
            calib_root=calib_root,
            smoothquant_alpha=smoothquant_alpha,
            skip_existing=skip_existing,
        )
        results[algorithm] = (generated, skipped)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate hybrid block-map W8A8 videos (FP16 blocks 1-13 + INT8 14-40). "
            "Default: human2 x 3 motions, naive + ViDiT."
        )
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to vidit_ptq.json (default: repo configs/quant/vidit_ptq.json)",
    )
    parser.add_argument(
        "--algorithm",
        choices=("naive", "vidit", "both"),
        default="both",
        help="Quantization algorithm (default: both)",
    )
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Use full 4x3 grid from config instead of hybrid_map sweep (human2 x 3 motions)",
    )
    parser.add_argument(
        "--block-fp16-start",
        type=int,
        default=None,
        help="First FP16 block index (default: hybrid_map.block_fp16[0])",
    )
    parser.add_argument(
        "--block-fp16-end",
        type=int,
        default=None,
        help="Last FP16 block index (default: hybrid_map.block_fp16[1])",
    )
    parser.add_argument(
        "--block-int8-start",
        type=int,
        default=None,
        help="First INT8 block index (default: hybrid_map.block_int8[0])",
    )
    parser.add_argument(
        "--block-int8-end",
        type=int,
        default=None,
        help="Last INT8 block index (default: hybrid_map.block_int8[1])",
    )
    parser.add_argument(
        "--clip-id",
        action="append",
        default=None,
        help="Generate specific clip id only (repeatable, e.g. human2_motion1)",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip clips whose output MP4 already exists (default: true)",
    )
    args = parser.parse_args()

    repo = add_repo_to_path(args.repo_root)
    config_path = (
        Path(args.config)
        if args.config
        else resolve_repo_path(repo, "configs/quant/vidit_ptq.json")
    )
    cfg = load_vidit_ptq_config(config_path)
    grid = load_grid_config(config_path, repo_root=repo)
    hybrid_cfg = load_hybrid_map_config(config_path, repo_root=repo)

    block_fp16 = hybrid_cfg.block_fp16
    block_int8 = hybrid_cfg.block_int8
    if args.block_fp16_start is not None or args.block_fp16_end is not None:
        start = args.block_fp16_start if args.block_fp16_start is not None else block_fp16[0]
        end = args.block_fp16_end if args.block_fp16_end is not None else block_fp16[1]
        block_fp16 = (start, end)
    if args.block_int8_start is not None or args.block_int8_end is not None:
        start = args.block_int8_start if args.block_int8_start is not None else block_int8[0]
        end = args.block_int8_end if args.block_int8_end is not None else block_int8[1]
        block_int8 = (start, end)

    algorithms = ["naive", "vidit"] if args.algorithm == "both" else [args.algorithm]

    model_path = str(resolve_repo_path(repo, cfg.MODEL_PATH))
    quantize_layers = resolve_repo_path(repo, cfg.QUANTIZE_LAYERS)
    if not quantize_layers.is_file():
        raise FileNotFoundError(
            f"Missing {quantize_layers}. Run list_layers.py (Step 2) first."
        )

    dtype = parse_dtype(cfg.DTYPE)
    base_args = build_base_args(cfg)
    images, motions = resolve_sweep_assets(
        hybrid_cfg, repo, full_grid=args.full_grid, grid=grid
    )
    clip_filter = set(args.clip_id) if args.clip_id else None

    grid_label = "full 4x3" if args.full_grid else "sweep human2 x 3 motions"
    total_pairs = len(images) * len(motions)
    print(
        f"Grid ({grid_label}): {len(images)} image(s) x {len(motions)} motion(s) = "
        f"{total_pairs} clips, algorithms={algorithms}, skip_existing={args.skip_existing}"
    )

    try:
        run_hybrid_grid(
            algorithms=algorithms,
            model_path=model_path,
            dtype=dtype,
            quantize_layers=quantize_layers,
            base_args=base_args,
            images=images,
            motions=motions,
            hybrid_cfg=hybrid_cfg,
            block_fp16=block_fp16,
            block_int8=block_int8,
            skip_existing=args.skip_existing,
            clip_filter=clip_filter,
            smoothquant_alpha=cfg.SMOOTHQUANT_ALPHA,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
