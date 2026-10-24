#!/usr/bin/env python3
"""Step 4 (ViDiT-Q): apply W8A8, PSNR quality gate, and FP16 vs ViDiT benchmark in one run."""
from __future__ import annotations

import argparse
import sys

from quant_common import add_repo_to_path
from vidit_eval import run_combined_apply_and_benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--model-path", default="./checkpoints/DynamiCtrl")
    parser.add_argument(
        "--quantize-layers", default="./configs/quant/quantize_layers.txt"
    )
    parser.add_argument(
        "--calib",
        required=True,
        help="ViDiT calibration .pt from collect_calib_vidit.py",
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
        required=True,
        help="Directory with FP16 baseline MP4 from setup F",
    )
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument(
        "--log-json",
        default="./logs/vidit_benchmark.json",
        help="Benchmark report JSON path",
    )
    parser.add_argument("--micro-repeats", type=int, default=10)
    parser.add_argument("--warmup-repeats", type=int, default=3)
    parser.add_argument(
        "--skip-fp16-e2e",
        action="store_true",
        help="Skip FP16 end-to-end timing (no e2e_speedup in report)",
    )
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

    add_repo_to_path(args.repo_root)
    run_combined_apply_and_benchmark(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
