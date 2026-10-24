#!/usr/bin/env python3
"""Step 1: verify DynamiCtrl DiT loads outside the notebook."""
import argparse
import sys
from pathlib import Path

import torch

from quant_common import add_repo_to_path, ensure_dynamictrl_config_files, parse_dtype, resolve_repo_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="./checkpoints/DynamiCtrl")
    parser.add_argument("--repo-root", default=".", help="DynamiCtrl git clone root")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--dit-only", action="store_true")
    args = parser.parse_args()

    repo = add_repo_to_path(args.repo_root)
    dtype = parse_dtype(args.dtype)
    model_path = resolve_repo_path(repo, args.model_path)
    ensure_dynamictrl_config_files(model_path)

    if args.dit_only:
        from dynamictrl.models.model_dynamictrl import DynamiCtrlTransformerModel

        dit = DynamiCtrlTransformerModel.from_pretrained_2d(
            str(model_path), subfolder="transformer", torch_dtype=dtype
        )
    else:
        from dynamictrl.pipelines.dynamictrl_pipeline import DynamiCtrlAnimationPipeline

        pipe = DynamiCtrlAnimationPipeline.from_pretrained(str(model_path), torch_dtype=dtype)
        dit = pipe.transformer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Step 1")

    dit.to("cuda")
    n_params = sum(p.numel() for p in dit.parameters())
    print(f"Model class: {type(dit).__name__}")
    print(f"Parameters:  {n_params / 1e9:.2f} B")
    print(f"Dtype:       {dtype}")
    torch.cuda.reset_peak_memory_stats()
    _ = next(dit.parameters()).device
    print(f"GPU:         {torch.cuda.get_device_name()}")
    print(f"Peak VRAM:   {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    if n_params < 5.5e9:
        raise RuntimeError(f"Unexpected param count: {n_params / 1e9:.2f} B (expected ~5.77B)")
    print("OK: DiT loaded successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
