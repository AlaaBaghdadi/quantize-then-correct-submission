#!/usr/bin/env python3
"""Step 3: collect activation calibration stats with pose-conditioned pipeline."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

from quant_common import (
    add_repo_to_path,
    get_current_timestep,
    load_quantize_names,
    make_generator,
    parse_dtype,
    prepare_control_video,
    prepare_fp16_pipeline,
    timestep_bucket,
    wrap_transformer_timestep,
)


def aggregate_stats(raw: dict) -> dict:
    out: dict[str, dict[str, dict]] = {}
    for layer_name, buckets in raw.items():
        out[layer_name] = {}
        for bucket, samples in buckets.items():
            if not samples:
                continue
            absmax = max(s["absmax"] for s in samples)
            shapes = list({s["shape"] for s in samples})
            out[layer_name][bucket] = {
                "absmax": absmax,
                "samples": len(samples),
                "shapes": shapes,
            }
    return out


def register_hooks(dit: nn.Module, quantize_names: set[str]) -> tuple[list, dict]:
    raw: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    handles = []

    def make_hook(name: str):
        def hook(_module, inp, _out):
            t = get_current_timestep()
            if t is None:
                return
            if not inp or inp[0] is None:
                return
            x = inp[0].detach().float()
            bucket = timestep_bucket(t)
            raw[name][bucket].append(
                {"absmax": x.abs().max().item(), "shape": tuple(x.shape)}
            )

        return hook

    hooked = 0
    for name, module in dit.named_modules():
        if name in quantize_names and isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(make_hook(name)))
            hooked += 1
    if hooked == 0:
        raise RuntimeError("No forward hooks registered — quantize layer names mismatch?")
    return handles, raw


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
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    add_repo_to_path(args.repo_root)
    dtype = parse_dtype(args.dtype)
    quantize_names = load_quantize_names(Path(args.quantize_layers))
    print(f"Loaded {len(quantize_names)} quantize layer names")

    pipe = prepare_fp16_pipeline(args.model_path, dtype)
    dit = pipe.transformer
    wrap_transformer_timestep(dit)
    handles, raw = register_hooks(dit, quantize_names)

    image, control_video = prepare_control_video(
        args.reference_image,
        args.driving_video,
        args.width,
        args.height,
        args.num_frames,
    )

    generator = make_generator(args.seed)
    print("Running calibration inference (activations recorded via hooks)...")
    with torch.no_grad():
        pipe(
            height=args.height,
            width=args.width,
            prompt=args.prompt,
            image=image,
            control_video=control_video,
            pose_control_function=args.pose_control_function,
            num_inference_steps=args.num_inference_steps,
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale,
            use_dynamic_cfg=False,
            generator=generator,
        )

    for h in handles:
        h.remove()

    stats = aggregate_stats(raw)
    if not stats:
        raise RuntimeError("Calibration produced empty stats")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "reference_image": args.reference_image,
            "driving_video": args.driving_video,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "pose_control_function": args.pose_control_function,
            "seed": args.seed,
        },
        "stats": stats,
    }
    torch.save(payload, out_path)
    print(f"Saved calibration stats to {out_path}")

    diffs = 0
    for layer_stats in stats.values():
        early = layer_stats.get("early", {}).get("absmax")
        late = layer_stats.get("late", {}).get("absmax")
        if early is not None and late is not None and abs(early - late) > 1e-6:
            diffs += 1
    print(f"Layers with differing early/late absmax: {diffs}/{len(stats)}")
    if diffs == 0:
        raise RuntimeError(
            "Calibration sanity check failed: early and late buckets identical"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
