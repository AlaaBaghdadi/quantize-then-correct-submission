#!/usr/bin/env python3
"""Step 4 (ViDiT-Q): SmoothQuant + W8A8 with static FP16 for sensitive layers."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from quant_common import (
    add_repo_to_path,
    load_quantize_names,
    parse_dtype,
    prepare_control_video,
    prepare_fp16_pipeline,
    resolve_repo_path,
    run_inference,
)
from vidit_eval import (
    export_vidit_outputs,
    frames_to_arrays,
    load_reference_frames_from_dir,
)
from vidit_quant import apply_vidit_w8a8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--model-path", default="./checkpoints/DynamiCtrl")
    parser.add_argument(
        "--quantize-layers", default="./configs/quant/quantize_layers.txt"
    )
    parser.add_argument("--calib", required=True, help="ViDiT calibration .pt from collect_calib_vidit.py")
    parser.add_argument("--reference-image", required=True)
    parser.add_argument("--driving-video", required=True)
    parser.add_argument("--prompt", default="Input the test prompt here.")
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1360)
    parser.add_argument("--num-frames", type=int, default=37)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--pose-control-function", default="padaln")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--fp16-reference-dir",
        default=None,
        help="Directory with FP16 baseline MP4; skips re-running FP16 inference",
    )
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--smoothquant-alpha", type=float, default=0.5)
    parser.add_argument("--early-ratio", type=float, default=2.0)
    parser.add_argument("--max-early-fp16-layers", type=int, default=24)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for the transformer")
    args = parser.parse_args()

    repo = add_repo_to_path(args.repo_root)
    dtype = parse_dtype(args.dtype)
    quantize_names = load_quantize_names(
        resolve_repo_path(repo, args.quantize_layers)
    )
    calib_path = resolve_repo_path(repo, args.calib)
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    model_path = str(resolve_repo_path(repo, args.model_path))
    args.reference_image = str(resolve_repo_path(repo, args.reference_image))
    args.driving_video = str(resolve_repo_path(repo, args.driving_video))
    out_dir = resolve_repo_path(repo, args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    image, control = prepare_control_video(
        args.reference_image,
        args.driving_video,
        args.width,
        args.height,
        args.num_frames,
    )

    if args.fp16_reference_dir:
        ref_arrays = load_reference_frames_from_dir(
            Path(args.fp16_reference_dir), args.num_frames
        )
    else:
        fp16_pipe = prepare_fp16_pipeline(model_path, dtype)
        print("Running FP16 reference for PSNR...")
        ref_frames = run_inference(fp16_pipe, args, image, control)
        ref_arrays = frames_to_arrays(ref_frames[: args.num_frames])
        del fp16_pipe
        torch.cuda.empty_cache()

    pipe = prepare_fp16_pipeline(model_path, dtype)
    vidit_meta = apply_vidit_w8a8(
        pipe.transformer,
        quantize_names,
        calib_path,
        smoothquant_alpha=args.smoothquant_alpha,
        early_ratio_threshold=args.early_ratio,
        max_early_fp16_layers=args.max_early_fp16_layers,
    )
    print(f"ViDiT-Q W8A8 applied to {vidit_meta['quantized_layers']} layers")
    print(f"SmoothQuant folded: {len(vidit_meta['smoothquant_layers'])} layers")
    print(
        f"Static FP16 layers (all timesteps): {len(vidit_meta['early_fp16_layers'])} layers"
    )

    compiled = False
    if args.compile:
        from quant_common import compile_model

        try:
            pipe.transformer = compile_model(pipe.transformer)
            compiled = True
        except Exception as exc:
            print(f"WARN: torch.compile failed ({exc}); running eager.")

    print("Running ViDiT-Q W8A8 inference...")
    quant_frames = run_inference(pipe, args, image, control)

    export_vidit_outputs(
        out_dir,
        quant_frames,
        ref_arrays,
        vidit_meta,
        args,
        compiled=compiled,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
