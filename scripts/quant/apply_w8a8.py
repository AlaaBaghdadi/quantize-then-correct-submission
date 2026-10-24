#!/usr/bin/env python3
"""Step 4: apply naive W8A8 quantization and run inference."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from skimage.metrics import peak_signal_noise_ratio
except ImportError:
    peak_signal_noise_ratio = None

from quant_common import (
    add_repo_to_path,
    apply_w8a8,
    load_quantize_names,
    parse_dtype,
    prepare_control_video,
    prepare_fp16_pipeline,
    resolve_repo_path,
    run_inference,
)
from vidit_quant import resolve_matched_quantize_plan


def frames_to_arrays(frames) -> list[np.ndarray]:
    return [np.array(frame.convert("RGB"), dtype=np.float32) for frame in frames]


def load_reference_frames_from_dir(ref_dir: Path, num_frames: int) -> list[np.ndarray]:
    import imageio.v3 as iio

    mp4s = sorted(ref_dir.glob("*.mp4"))
    if not mp4s:
        raise FileNotFoundError(f"No MP4 files in FP16 reference dir: {ref_dir}")
    frames = []
    for frame in iio.imiter(str(mp4s[0])):
        frames.append(np.array(frame, dtype=np.float32))
        if len(frames) >= num_frames:
            break
    if not frames:
        raise RuntimeError(f"Could not decode frames from {mp4s[0]}")
    return frames


def compute_psnr(ref_frames, quant_frames) -> float:
    if peak_signal_noise_ratio is None:
        raise RuntimeError("scikit-image is required for PSNR (pip install scikit-image)")
    n = min(len(ref_frames), len(quant_frames))
    psnrs = [
        peak_signal_noise_ratio(ref_frames[i], quant_frames[i], data_range=255.0)
        for i in range(n)
    ]
    return float(np.mean(psnrs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--model-path", default="./checkpoints/DynamiCtrl")
    parser.add_argument(
        "--quantize-layers", default="./configs/quant/quantize_layers.txt"
    )
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
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for the transformer")
    parser.add_argument(
        "--calib",
        default=None,
        help="ViDiT calibration .pt; when set, skip the same early_fp16 layers as ViDiT",
    )
    parser.add_argument("--early-ratio", type=float, default=2.0)
    parser.add_argument("--max-early-fp16-layers", type=int, default=24)
    args = parser.parse_args()

    add_repo_to_path(args.repo_root)
    dtype = parse_dtype(args.dtype)
    quantize_names = load_quantize_names(Path(args.quantize_layers))
    out_dir = Path(args.output_path)
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
        fp16_pipe = prepare_fp16_pipeline(args.model_path, dtype)
        print("Running FP16 reference for PSNR...")
        ref_frames = run_inference(fp16_pipe, args, image, control)
        ref_arrays = frames_to_arrays(ref_frames[: args.num_frames])
        del fp16_pipe
        torch.cuda.empty_cache()

    pipe = prepare_fp16_pipeline(args.model_path, dtype)
    fp16_layers: set[str] | None = None
    calib_path = None
    quantize_plan = None
    if args.calib:
        calib_path = resolve_repo_path(args.repo_root, args.calib)
        quantize_plan = resolve_matched_quantize_plan(
            quantize_names,
            calib_path,
            early_ratio_threshold=args.early_ratio,
            max_early_fp16_layers=args.max_early_fp16_layers,
        )
        fp16_layers = set(quantize_plan["early_fp16_layers"])
        print(
            f"Matched naive W8A8: skipping {len(fp16_layers)} calib-selected FP16 layers; "
            f"INT8 targets={quantize_plan['quantized_layers']}"
        )
    w8a8_meta = apply_w8a8(
        pipe.transformer,
        quantize_names,
        fp16_layers=fp16_layers,
        calib_path=calib_path,
        quantize_plan=quantize_plan,
    )
    print(f"Quantized {w8a8_meta['quantized_layers']} Linear layers with W8A8")

    if args.compile:
        from quant_common import compile_model
        pipe.transformer = compile_model(pipe.transformer)

    from diffusers.utils import export_to_video

    print("Running W8A8 inference...")
    quant_frames = run_inference(pipe, args, image, control)

    image_name = Path(args.reference_image).stem
    video_name = Path(args.driving_video).stem
    out_mp4 = out_dir / f"w8a8_{image_name}_{video_name}_seed{args.seed}.mp4"
    export_to_video(quant_frames[: args.num_frames], str(out_mp4), fps=args.fps)
    print(f"W8A8 output: {out_mp4}")

    psnr = compute_psnr(ref_arrays, frames_to_arrays(quant_frames[: args.num_frames]))
    print(f"Mean PSNR vs FP16: {psnr:.2f} dB")
    (out_dir / "psnr.txt").write_text(f"mean_psnr={psnr:.4f}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
