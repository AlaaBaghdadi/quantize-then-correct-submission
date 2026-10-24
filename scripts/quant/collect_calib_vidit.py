#!/usr/bin/env python3
"""Step 3 (ViDiT-Q): collect per-channel activation stats per timestep bucket."""
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
    resolve_repo_path,
    timestep_bucket,
    wrap_transformer_timestep,
)


def _channel_absmax(x: torch.Tensor) -> torch.Tensor:
    if x.dim() < 2:
        raise ValueError(f"Expected activation rank >= 2, got shape {tuple(x.shape)}")
    reduce_dims = tuple(range(x.dim() - 1))
    return x.abs().amax(dim=reduce_dims).cpu()


def aggregate_stats_vidit(raw: dict) -> dict:
    out: dict[str, dict[str, dict]] = {}
    for layer_name, buckets in raw.items():
        out[layer_name] = {}
        for bucket, samples in buckets.items():
            if not samples:
                continue
            channel_absmax = samples[0]["channel_absmax"].clone()
            tensor_absmax = samples[0]["tensor_absmax"]
            shapes = {samples[0]["shape"]}
            for sample in samples[1:]:
                channel_absmax = torch.maximum(channel_absmax, sample["channel_absmax"])
                tensor_absmax = max(tensor_absmax, sample["tensor_absmax"])
                shapes.add(sample["shape"])
            out[layer_name][bucket] = {
                "channel_absmax": channel_absmax,
                "tensor_absmax": tensor_absmax,
                "samples": len(samples),
                "shapes": [list(s) for s in sorted(shapes)],
            }
    return out


def validate_vidit_calib_stats(stats: dict) -> None:
    diffs = 0
    empty_channels = 0
    for layer_stats in stats.values():
        early = layer_stats.get("early", {})
        late = layer_stats.get("late", {})
        e_abs = early.get("tensor_absmax")
        l_abs = late.get("tensor_absmax")
        if e_abs is not None and l_abs is not None and abs(e_abs - l_abs) > 1e-6:
            diffs += 1
        for bucket_stats in layer_stats.values():
            ch = bucket_stats.get("channel_absmax")
            if ch is None or ch.numel() == 0:
                empty_channels += 1

    print(f"Layers with differing early/late tensor_absmax: {diffs}/{len(stats)}")
    if diffs == 0:
        raise RuntimeError(
            "Calibration sanity check failed: early and late buckets identical"
        )
    if empty_channels > 0:
        raise RuntimeError(
            f"Calibration sanity check failed: {empty_channels} empty channel_absmax entries"
        )


def save_vidit_calib(stats: dict, args, out_path: Path) -> None:
    validate_vidit_calib_stats(stats)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "reference_image": args.reference_image,
            "driving_video": args.driving_video,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "pose_control_function": args.pose_control_function,
            "seed": args.seed,
            "format": "vidit_v1",
        },
        "stats": stats,
    }
    torch.save(payload, out_path)
    print(f"Saved ViDiT calibration stats to {out_path}")


def run_vidit_calib_pass(pipe, quantize_names: set[str], args) -> dict:
    """Collect stats for one clip. Call wrap_transformer_timestep(dit) once per pipeline."""
    dit = pipe.transformer
    handles, raw = register_hooks_vidit(dit, quantize_names)
    try:
        image, control_video = prepare_control_video(
            args.reference_image,
            args.driving_video,
            args.width,
            args.height,
            args.num_frames,
        )
        generator = make_generator(args.seed)
        print("Running ViDiT calibration inference (per-channel stats via hooks)...")
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
    finally:
        for handle in handles:
            handle.remove()

    stats = aggregate_stats_vidit(raw)
    if not stats:
        raise RuntimeError("Calibration produced empty stats")
    return stats


def register_hooks_vidit(dit: nn.Module, quantize_names: set[str]) -> tuple[list, dict]:
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
                {
                    "channel_absmax": _channel_absmax(x),
                    "tensor_absmax": x.abs().max().item(),
                    "shape": tuple(x.shape),
                }
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

    repo = add_repo_to_path(args.repo_root)
    dtype = parse_dtype(args.dtype)
    quantize_names = load_quantize_names(
        resolve_repo_path(repo, args.quantize_layers)
    )
    print(f"Loaded {len(quantize_names)} quantize layer names")

    model_path = str(resolve_repo_path(repo, args.model_path))
    args.reference_image = str(resolve_repo_path(repo, args.reference_image))
    args.driving_video = str(resolve_repo_path(repo, args.driving_video))
    out_path = resolve_repo_path(repo, args.out)

    pipe = prepare_fp16_pipeline(model_path, dtype)
    wrap_transformer_timestep(pipe.transformer)
    stats = run_vidit_calib_pass(pipe, quantize_names, args)
    save_vidit_calib(stats, args, out_path)
    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
