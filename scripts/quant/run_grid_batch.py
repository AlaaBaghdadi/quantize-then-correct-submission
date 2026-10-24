#!/usr/bin/env python3
"""Optimized 4×3 grid batch: FP16, ViDiT calib, ViDiT-Q W8A8, and naive W8A8 video generation."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from collect_calib_vidit import run_vidit_calib_pass, save_vidit_calib
from quant_common import (
    add_repo_to_path,
    apply_w8a8,
    args_namespace,
    load_quantize_names,
    parse_dtype,
    prepare_control_video,
    prepare_fp16_pipeline,
    resolve_repo_path,
    run_inference,
    wrap_transformer_timestep,
)
from vidit_eval import export_vidit_outputs, export_w8a8_outputs, load_reference_frames_from_dir
from vidit_ptq_config import load_grid_config, load_vidit_ptq_config
from vidit_quant import apply_vidit_w8a8, resolve_matched_quantize_plan


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


def vidit_video_path(out_dir: Path, cid: str, seed: int) -> Path:
    return out_dir / f"vidit_w8a8_{cid}_seed{seed}.mp4"


def w8a8_video_path(out_dir: Path, cid: str, seed: int) -> Path:
    return out_dir / f"w8a8_{cid}_seed{seed}.mp4"


def make_clip_args(base: SimpleNamespace, image: str, motion: str) -> SimpleNamespace:
    clip_args = copy.copy(base)
    clip_args.reference_image = image
    clip_args.driving_video = motion
    return clip_args


def run_fp16_grid(
    *,
    model_path: str,
    dtype: torch.dtype,
    base_args: SimpleNamespace,
    images: list[str],
    motions: list[str],
    fp16_root: Path,
    skip_existing: bool,
) -> int:
    from diffusers.utils import export_to_video

    print("Loading FP16 pipeline once for grid...")
    pipe = prepare_fp16_pipeline(model_path, dtype)
    generated = 0
    try:
        for cid, image, motion in iter_grid_pairs(images, motions):
            out_dir = fp16_root / cid
            out_dir.mkdir(parents=True, exist_ok=True)
            out_mp4 = fp16_video_path(out_dir, cid, base_args.seed)
            if skip_existing and out_mp4.exists():
                print(f"Skip existing FP16: {out_mp4}")
                continue

            print(f"\n=== FP16 {cid} ===")
            clip_args = make_clip_args(base_args, image, motion)
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
            print(f"  -> {out_mp4}")
            generated += 1
    finally:
        del pipe
        torch.cuda.empty_cache()
    return generated


def run_calib_grid(
    *,
    model_path: str,
    dtype: torch.dtype,
    quantize_layers: Path,
    base_args: SimpleNamespace,
    images: list[str],
    motions: list[str],
    calib_root: Path,
    skip_existing: bool,
) -> int:
    quantize_names = load_quantize_names(quantize_layers)
    print(f"Loaded {len(quantize_names)} quantize layer names")
    print("Loading FP16 pipeline once for calibration grid...")
    pipe = prepare_fp16_pipeline(model_path, dtype)
    wrap_transformer_timestep(pipe.transformer)
    collected = 0
    try:
        for cid, image, motion in iter_grid_pairs(images, motions):
            calib_out = calib_root / f"{cid}_vidit.pt"
            if skip_existing and calib_out.exists():
                print(f"Skip existing calib: {calib_out}")
                continue

            print(f"\n=== Calib {cid} ===")
            clip_args = make_clip_args(base_args, image, motion)
            stats = run_vidit_calib_pass(pipe, quantize_names, clip_args)
            save_vidit_calib(stats, clip_args, calib_out)
            collected += 1
    finally:
        del pipe
        torch.cuda.empty_cache()
    return collected


def run_vidit_grid(
    *,
    model_path: str,
    dtype: torch.dtype,
    quantize_layers: Path,
    base_args: SimpleNamespace,
    images: list[str],
    motions: list[str],
    fp16_root: Path,
    vidit_root: Path,
    calib_root: Path,
    smoothquant_alpha: float,
    early_ratio: float,
    max_early_fp16_layers: int,
    skip_existing: bool,
) -> int:
    quantize_names = load_quantize_names(quantize_layers)
    generated = 0

    for cid, image, motion in iter_grid_pairs(images, motions):
        out_dir = vidit_root / cid
        out_dir.mkdir(parents=True, exist_ok=True)
        out_mp4 = vidit_video_path(out_dir, cid, base_args.seed)
        if skip_existing and out_mp4.exists():
            print(f"Skip existing ViDiT: {out_mp4}")
            continue

        calib_path = calib_root / f"{cid}_vidit.pt"
        if not calib_path.exists():
            raise FileNotFoundError(
                f"Missing {calib_path}. Run calib phase first or reuse an existing calib file."
            )

        fp16_ref_dir = fp16_root / cid
        if not fp16_ref_dir.is_dir() or not list(fp16_ref_dir.glob("*.mp4")):
            raise FileNotFoundError(
                f"Missing FP16 reference video in {fp16_ref_dir}. Run fp16 phase first."
            )

        print(f"\n=== ViDiT-Q W8A8 {cid} ===")
        clip_args = make_clip_args(base_args, image, motion)
        ref_arrays = load_reference_frames_from_dir(fp16_ref_dir, base_args.num_frames)

        # Quantization mutates weights in-place — reload a fresh pipeline per clip.
        pipe = prepare_fp16_pipeline(model_path, dtype)
        try:
            vidit_meta = apply_vidit_w8a8(
                pipe.transformer,
                quantize_names,
                calib_path,
                smoothquant_alpha=smoothquant_alpha,
                early_ratio_threshold=early_ratio,
                max_early_fp16_layers=max_early_fp16_layers,
            )
            image_obj, control = prepare_control_video(
                image,
                motion,
                base_args.width,
                base_args.height,
                base_args.num_frames,
            )
            quant_frames = run_inference(pipe, clip_args, image_obj, control)
            export_vidit_outputs(
                out_dir,
                quant_frames,
                ref_arrays,
                vidit_meta,
                clip_args,
                compiled=False,
            )
            generated += 1
        finally:
            del pipe
            torch.cuda.empty_cache()

    return generated


def run_w8a8_grid(
    *,
    model_path: str,
    dtype: torch.dtype,
    quantize_layers: Path,
    base_args: SimpleNamespace,
    images: list[str],
    motions: list[str],
    fp16_root: Path,
    w8a8_root: Path,
    calib_root: Path,
    early_ratio: float,
    max_early_fp16_layers: int,
    skip_existing: bool,
) -> int:
    quantize_names = load_quantize_names(quantize_layers)
    generated = 0

    for cid, image, motion in iter_grid_pairs(images, motions):
        out_dir = w8a8_root / cid
        out_dir.mkdir(parents=True, exist_ok=True)
        out_mp4 = w8a8_video_path(out_dir, cid, base_args.seed)
        if skip_existing and out_mp4.exists():
            print(f"Skip existing W8A8: {out_mp4}")
            continue

        calib_path = calib_root / f"{cid}_vidit.pt"
        if not calib_path.exists():
            raise FileNotFoundError(
                f"Missing {calib_path}. Run calib phase first or reuse an existing calib file."
            )

        fp16_ref_dir = fp16_root / cid
        if not fp16_ref_dir.is_dir() or not list(fp16_ref_dir.glob("*.mp4")):
            raise FileNotFoundError(
                f"Missing FP16 reference video in {fp16_ref_dir}. Run fp16 phase first."
            )

        print(f"\n=== Matched naive W8A8 {cid} ===")
        clip_args = make_clip_args(base_args, image, motion)
        ref_arrays = load_reference_frames_from_dir(fp16_ref_dir, base_args.num_frames)
        quantize_plan = resolve_matched_quantize_plan(
            quantize_names,
            calib_path,
            early_ratio_threshold=early_ratio,
            max_early_fp16_layers=max_early_fp16_layers,
        )

        # Quantization mutates weights in-place — reload a fresh pipeline per clip.
        pipe = prepare_fp16_pipeline(model_path, dtype)
        try:
            w8a8_meta = apply_w8a8(
                pipe.transformer,
                quantize_names,
                quantize_plan=quantize_plan,
                calib_path=calib_path,
            )
            print(
                f"Quantized {w8a8_meta['quantized_layers']} Linear layers "
                f"({len(quantize_plan['early_fp16_layers'])} calib-selected FP16 layers skipped)"
            )
            image_obj, control = prepare_control_video(
                image,
                motion,
                base_args.width,
                base_args.height,
                base_args.num_frames,
            )
            quant_frames = run_inference(pipe, clip_args, image_obj, control)
            export_w8a8_outputs(
                out_dir,
                quant_frames,
                ref_arrays,
                clip_args,
                clip_id=cid,
                w8a8_meta=w8a8_meta,
            )
            generated += 1
        finally:
            del pipe
            torch.cuda.empty_cache()

    return generated


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


def main():
    parser = argparse.ArgumentParser(
        description="Optimized FP16 / calib / ViDiT-Q grid batch (loads pipeline once where safe)."
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to vidit_ptq.json (default: repo configs/quant/vidit_ptq.json)",
    )
    parser.add_argument(
        "--phase",
        choices=("fp16", "calib", "vidit", "w8a8", "all"),
        default="all",
        help="Which grid phase to run (w8a8 is not included in all)",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip clips whose output file already exists (default: true)",
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

    model_path = str(resolve_repo_path(repo, cfg.MODEL_PATH))
    quantize_layers = resolve_repo_path(repo, cfg.QUANTIZE_LAYERS)
    dtype = parse_dtype(cfg.DTYPE)
    base_args = build_base_args(cfg)

    images = [str(resolve_repo_path(repo, p)) for p in grid.images]
    motions = [str(resolve_repo_path(repo, p)) for p in grid.motions]
    fp16_root = resolve_repo_path(repo, grid.fp16_output_dir)
    vidit_root = resolve_repo_path(repo, grid.vidit_output_dir)
    w8a8_root = resolve_repo_path(repo, grid.w8a8_output_dir)
    calib_root = resolve_repo_path(repo, grid.calib_dir)

    fp16_root.mkdir(parents=True, exist_ok=True)
    vidit_root.mkdir(parents=True, exist_ok=True)
    w8a8_root.mkdir(parents=True, exist_ok=True)
    calib_root.mkdir(parents=True, exist_ok=True)

    total_pairs = len(images) * len(motions)
    print(
        f"Grid: {len(images)} images x {len(motions)} motions = {total_pairs} clips "
        f"(phase={args.phase}, skip_existing={args.skip_existing})"
    )

    if args.phase in ("fp16", "all"):
        n = run_fp16_grid(
            model_path=model_path,
            dtype=dtype,
            base_args=base_args,
            images=images,
            motions=motions,
            fp16_root=fp16_root,
            skip_existing=args.skip_existing,
        )
        print(f"FP16 phase: generated {n} new video(s)")

    if args.phase in ("calib", "all"):
        n = run_calib_grid(
            model_path=model_path,
            dtype=dtype,
            quantize_layers=quantize_layers,
            base_args=base_args,
            images=images,
            motions=motions,
            calib_root=calib_root,
            skip_existing=args.skip_existing,
        )
        print(f"Calib phase: collected {n} new calib file(s)")

    if args.phase in ("vidit", "all"):
        n = run_vidit_grid(
            model_path=model_path,
            dtype=dtype,
            quantize_layers=quantize_layers,
            base_args=base_args,
            images=images,
            motions=motions,
            fp16_root=fp16_root,
            vidit_root=vidit_root,
            calib_root=calib_root,
            smoothquant_alpha=cfg.SMOOTHQUANT_ALPHA,
            early_ratio=cfg.EARLY_SENSITIVITY_RATIO,
            max_early_fp16_layers=cfg.MAX_EARLY_FP16_LAYERS,
            skip_existing=args.skip_existing,
        )
        print(f"ViDiT phase: generated {n} new video(s)")

    if args.phase == "w8a8":
        n = run_w8a8_grid(
            model_path=model_path,
            dtype=dtype,
            quantize_layers=quantize_layers,
            base_args=base_args,
            images=images,
            motions=motions,
            fp16_root=fp16_root,
            w8a8_root=w8a8_root,
            calib_root=calib_root,
            early_ratio=cfg.EARLY_SENSITIVITY_RATIO,
            max_early_fp16_layers=cfg.MAX_EARLY_FP16_LAYERS,
            skip_existing=args.skip_existing,
        )
        print(f"W8A8 phase: generated {n} new video(s)")

    print("Grid batch complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
