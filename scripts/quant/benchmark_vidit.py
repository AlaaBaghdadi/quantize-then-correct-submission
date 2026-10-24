#!/usr/bin/env python3
"""Step 5 (ViDiT-Q): benchmark FP16 vs ViDiT-Q W8A8 DiT forward and end-to-end inference."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from quant_common import (
    add_repo_to_path,
    capture_dit_forward_kwargs,
    compile_model,
    load_quantize_names,
    microbenchmark_dit,
    parse_dtype,
    prepare_fp16_pipeline,
    resolve_repo_path,
    run_timed_inference,
    run_timed_vidit_inference,
)
from vidit_eval import build_kernel_conclusion
from vidit_quant import apply_vidit_w8a8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--model-path", default="./checkpoints/DynamiCtrl")
    parser.add_argument(
        "--quantize-layers", default="./configs/quant/quantize_layers.txt"
    )
    parser.add_argument("--calib", required=True, help="ViDiT calibration .pt from collect_calib_vidit.py")
    parser.add_argument("--reference-image", default="./assets/human2.jpg")
    parser.add_argument("--driving-video", default="./assets/motion1.mp4")
    parser.add_argument("--prompt", default="Input the test prompt here.")
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1360)
    parser.add_argument("--num-frames", type=int, default=37)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--pose-control-function", default="padaln")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--log-json", default="./logs/vidit_benchmark.json")
    parser.add_argument("--micro-repeats", type=int, default=10)
    parser.add_argument("--warmup-repeats", type=int, default=3)
    parser.add_argument("--skip-e2e", action="store_true")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for microbenchmark and e2e (both FP16 and ViDiT-Q W8A8)",
    )
    parser.add_argument(
        "--e2e-warmup",
        type=int,
        default=None,
        help="Untimed full-pipeline warmup runs before e2e timing (default: 1 when --compile, else 0)",
    )
    parser.add_argument("--smoothquant-alpha", type=float, default=0.5)
    parser.add_argument("--early-ratio", type=float, default=2.0)
    parser.add_argument("--max-early-fp16-layers", type=int, default=24)
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

    print("Benchmarking FP16 baseline...")
    fp16_pipe = prepare_fp16_pipeline(model_path, dtype)
    forward_kwargs = capture_dit_forward_kwargs(fp16_pipe, args)
    fp16_micro = microbenchmark_dit(
        fp16_pipe.transformer,
        forward_kwargs,
        repeats=args.micro_repeats,
        warmup=args.warmup_repeats,
        compile=args.compile,
    )
    del fp16_pipe
    torch.cuda.empty_cache()

    print("\nBenchmarking ViDiT-Q W8A8 model...")
    vidit_pipe = prepare_fp16_pipeline(model_path, dtype)
    vidit_meta = apply_vidit_w8a8(
        vidit_pipe.transformer,
        quantize_names,
        calib_path,
        smoothquant_alpha=args.smoothquant_alpha,
        early_ratio_threshold=args.early_ratio,
        max_early_fp16_layers=args.max_early_fp16_layers,
    )
    vidit_micro_compiled = args.compile
    if args.compile:
        try:
            vidit_pipe.transformer = compile_model(vidit_pipe.transformer)
        except Exception as exc:
            print(f"WARN: torch.compile failed on ViDiT-Q microbenchmark ({exc}); running eager.")
            vidit_micro_compiled = False
    vidit_micro = microbenchmark_dit(
        vidit_pipe.transformer,
        forward_kwargs,
        repeats=args.micro_repeats,
        warmup=args.warmup_repeats,
        compile=False,
    )
    vidit_micro["compiled"] = vidit_micro_compiled
    del vidit_pipe
    torch.cuda.empty_cache()

    speedup, kernel_conclusion = build_kernel_conclusion(fp16_micro, vidit_micro)

    report = {
        "fp16_micro": fp16_micro,
        "vidit_w8a8_micro": vidit_micro,
        "dit_forward_speedup": speedup,
        "kernel_conclusion": kernel_conclusion,
        "vidit_meta": vidit_meta,
        "calib_path": str(calib_path),
    }

    if not args.skip_e2e:
        e2e_warmup = args.e2e_warmup if args.e2e_warmup is not None else (1 if args.compile else 0)
        if args.compile and e2e_warmup > 0:
            print(
                f"\nE2E: each variant runs {e2e_warmup} untimed warmup(s) before timing "
                "(compile cost is excluded from wall_sec)."
            )

        fp16_wall, fp16_peak, _, fp16_meta = run_timed_inference(
            args.model_path,
            dtype,
            args,
            quantize_names=None,
            compile=args.compile,
            e2e_warmup=e2e_warmup,
        )
        vidit_wall, vidit_peak, _, vidit_e2e_meta = run_timed_vidit_inference(
            args.model_path,
            dtype,
            args,
            quantize_names,
            calib_path,
            smoothquant_alpha=args.smoothquant_alpha,
            early_ratio_threshold=args.early_ratio,
            max_early_fp16_layers=args.max_early_fp16_layers,
            compile=args.compile,
            e2e_warmup=e2e_warmup,
        )
        report["fp16_e2e"] = {
            "wall_sec": fp16_wall,
            "peak_vram_gb": fp16_peak,
            "sec_per_step": fp16_wall / args.num_inference_steps,
            **fp16_meta,
        }
        report["vidit_w8a8_e2e"] = {
            "wall_sec": vidit_wall,
            "peak_vram_gb": vidit_peak,
            "sec_per_step": vidit_wall / args.num_inference_steps,
            **vidit_e2e_meta,
        }
        report["e2e_speedup"] = fp16_wall / max(vidit_wall, 1e-9)

    log_path = Path(args.log_json)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {log_path}")
    print(f"Conclusion: {kernel_conclusion}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
