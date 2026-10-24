#!/usr/bin/env python3
"""Step 5: benchmark FP16 vs W8A8 DiT forward and end-to-end inference."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from quant_common import (
    add_repo_to_path,
    apply_w8a8,
    capture_dit_forward_kwargs,
    load_quantize_names,
    microbenchmark_dit,
    parse_dtype,
    prepare_fp16_pipeline,
    resolve_repo_path,
    run_timed_inference,
)
from vidit_quant import resolve_matched_quantize_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--model-path", default="./checkpoints/DynamiCtrl")
    parser.add_argument(
        "--quantize-layers", default="./configs/quant/quantize_layers.txt"
    )
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
    parser.add_argument("--log-json", default="./logs/step5_benchmark.json")
    parser.add_argument("--micro-repeats", type=int, default=10)
    parser.add_argument("--warmup-repeats", type=int, default=3)
    parser.add_argument("--skip-e2e", action="store_true")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for microbenchmark and e2e (both FP16 and W8A8)",
    )
    parser.add_argument(
        "--e2e-warmup",
        type=int,
        default=None,
        help="Untimed full-pipeline warmup runs before e2e timing (default: 1 when --compile, else 0)",
    )
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

    print("Benchmarking FP16 baseline...")
    fp16_pipe = prepare_fp16_pipeline(args.model_path, dtype)
    forward_kwargs = capture_dit_forward_kwargs(fp16_pipe, args)
    fp16_micro = microbenchmark_dit(
        fp16_pipe.transformer, forward_kwargs, repeats=args.micro_repeats, warmup=args.warmup_repeats, compile=args.compile
    )
    del fp16_pipe
    torch.cuda.empty_cache()

    print("\nBenchmarking W8A8 quantized model...")
    w8_pipe = prepare_fp16_pipeline(args.model_path, dtype)
    w8a8_meta = apply_w8a8(
        w8_pipe.transformer,
        quantize_names,
        fp16_layers=fp16_layers,
        calib_path=calib_path,
        quantize_plan=quantize_plan,
    )
    w8_micro = microbenchmark_dit(
        w8_pipe.transformer, forward_kwargs, repeats=args.micro_repeats, warmup=args.warmup_repeats, compile=args.compile
    )
    del w8_pipe
    torch.cuda.empty_cache()

    speedup = fp16_micro["sec_per_forward"] / max(w8_micro["sec_per_forward"], 1e-9)
    if speedup >= 1.15:
        kernel_conclusion = "INT8 GEMM likely working (DiT forward speedup >= 1.15x)"
    elif w8_micro["peak_vram_gb"] < fp16_micro["peak_vram_gb"] * 0.95:
        kernel_conclusion = "Memory-only win; no meaningful latency gain detected"
    else:
        kernel_conclusion = "Inconclusive; inspect profiler or weight types"

    report = {
        "fp16_micro": fp16_micro,
        "w8a8_micro": w8_micro,
        "dit_forward_speedup": speedup,
        "kernel_conclusion": kernel_conclusion,
    }
    if w8a8_meta is not None:
        report["w8a8_meta"] = w8a8_meta

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
        w8_wall, w8_peak, _, w8_meta = run_timed_inference(
            args.model_path,
            dtype,
            args,
            quantize_names=quantize_names,
            fp16_layers=fp16_layers,
            calib_path=calib_path,
            compile=args.compile,
            e2e_warmup=e2e_warmup,
        )
        report["fp16_e2e"] = {
            "wall_sec": fp16_wall,
            "peak_vram_gb": fp16_peak,
            "sec_per_step": fp16_wall / args.num_inference_steps,
            **fp16_meta,
        }
        report["w8a8_e2e"] = {
            "wall_sec": w8_wall,
            "peak_vram_gb": w8_peak,
            "sec_per_step": w8_wall / args.num_inference_steps,
            **w8_meta,
        }
        report["e2e_speedup"] = fp16_wall / max(w8_wall, 1e-9)

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
