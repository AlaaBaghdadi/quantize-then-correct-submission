#!/usr/bin/env python3
"""Step 2: inventory nn.Linear modules and classify quantize vs protect."""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

from quant_common import (
    add_repo_to_path,
    build_protected_patterns,
    classify_layer,
    ensure_dynamictrl_config_files,
    resolve_repo_path,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--model-path", default="./checkpoints/DynamiCtrl")
    parser.add_argument("--out-dir", default="./configs/quant")
    args = parser.parse_args()

    repo = add_repo_to_path(args.repo_root)
    from dynamictrl.models.model_dynamictrl import DynamiCtrlTransformerModel

    model_path = resolve_repo_path(repo, args.model_path)
    ensure_dynamictrl_config_files(model_path)
    dit = DynamiCtrlTransformerModel.from_pretrained_2d(
        str(model_path), subfolder="transformer", torch_dtype=torch.bfloat16
    )
    protected_patterns = build_protected_patterns(len(dit.transformer_blocks))

    quantize, protect = [], []
    quant_params, protect_params = 0, 0

    for name, module in dit.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        n = module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
        tag = classify_layer(name, protected_patterns)
        entry = {
            "name": name,
            "in": module.in_features,
            "out": module.out_features,
            "params": n,
        }
        if tag == "QUANTIZE":
            quantize.append(entry)
            quant_params += n
        else:
            protect.append(entry)
            protect_params += n

    out = resolve_repo_path(repo, args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "quantize_layers.json").write_text(json.dumps(quantize, indent=2))
    (out / "protected_layers.json").write_text(json.dumps(protect, indent=2))
    (out / "quantize_layers.txt").write_text("\n".join(e["name"] for e in quantize))
    (out / "protected_layers.txt").write_text("\n".join(e["name"] for e in protect))

    print(f"Quantize: {len(quantize)} Linear layers, {quant_params / 1e9:.2f}B params")
    print(f"Protect:  {len(protect)} Linear layers, {protect_params / 1e9:.2f}B params")
    print(f"Wrote configs to {out}/")

    prot_text = (out / "protected_layers.txt").read_text()
    if "control_norm1" not in prot_text:
        raise RuntimeError("Validation failed: control_norm1 not in protected list")
    if len(quantize) < 200:
        raise RuntimeError(
            f"Validation failed: only {len(quantize)} quantize layers "
            "(expected ~240 with AdaLN + edge blocks protected)"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
