"""ViDiT-Q-style W8A8: SmoothQuant folding + static FP16 for calib-sensitive layers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class SmoothQuantActivationWrapper(nn.Module):
    """Runtime input-side smoothing: linear(x / s) with s folded into linear weights."""

    def __init__(self, module: nn.Module, scales: torch.Tensor):
        super().__init__()
        self.module = module
        device = next(module.parameters()).device
        self.register_buffer("scales", scales.detach().float().to(device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        view_shape = [1] * (x.dim() - 1) + [-1]
        s = self.scales.to(device=x.device, dtype=x.dtype).view(*view_shape)
        return self.module(x / s)


def get_module_by_name(root: nn.Module, name: str) -> nn.Module:
    module = root
    for part in name.split("."):
        module = getattr(module, part)
    return module


def replace_module_by_name(root: nn.Module, name: str, new_module: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def load_calib_stats(calib_path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    payload = torch.load(calib_path, map_location="cpu")
    stats = payload.get("stats")
    if not stats:
        raise RuntimeError(f"Calibration file missing stats: {calib_path}")
    return stats


def _max_channel_absmax_across_buckets(layer_stats: dict) -> torch.Tensor | None:
    merged: torch.Tensor | None = None
    for bucket_stats in layer_stats.values():
        ch = bucket_stats.get("channel_absmax")
        if ch is None:
            continue
        ch = ch.float()
        merged = ch if merged is None else torch.maximum(merged, ch)
    return merged


def compute_smoothquant_scales(
    weight: torch.Tensor,
    channel_absmax: torch.Tensor,
    alpha: float = 0.5,
    eps: float = 1e-5,
) -> torch.Tensor:
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D weight, got shape {tuple(weight.shape)}")
    if channel_absmax.numel() != weight.shape[1]:
        raise ValueError(
            f"channel_absmax length {channel_absmax.numel()} != in_features {weight.shape[1]}"
        )
    # Calib stats are stored on CPU; weights may live on GPU during apply.
    w = weight.detach().float().cpu()
    act = channel_absmax.float().clamp(min=eps)
    w_per_channel = w.abs().amax(dim=0).clamp(min=eps)
    return (act.pow(alpha) / w_per_channel.pow(1.0 - alpha)).clamp(min=eps)


def fold_smoothquant_into_linear(linear: nn.Linear, scales: torch.Tensor) -> None:
    device = linear.weight.device
    dtype = linear.weight.dtype
    s = scales.to(device=device, dtype=dtype)
    linear.weight.data.mul_(s)
    if linear.bias is not None:
        pass  # input-side smoothing; bias unchanged


def classify_early_sensitive_layers(
    stats: dict[str, dict[str, dict[str, Any]]],
    *,
    ratio_threshold: float = 2.0,
    max_layers: int = 24,
) -> list[str]:
    scored: list[tuple[float, str]] = []
    for layer_name, layer_stats in stats.items():
        early = layer_stats.get("early", {}).get("tensor_absmax")
        late = layer_stats.get("late", {}).get("tensor_absmax")
        if early is None or late is None:
            continue
        ratio = float(early) / max(float(late), 1e-6)
        if ratio >= ratio_threshold:
            scored.append((ratio, layer_name))
    scored.sort(reverse=True)
    return [name for _, name in scored[:max_layers]]


def resolve_early_fp16_layers(
    calib_path: Path,
    *,
    early_ratio_threshold: float = 2.0,
    max_early_fp16_layers: int = 24,
    stats: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> list[str]:
    """Return calib-selected layer names to keep in FP16 (shared by ViDiT and matched naive W8A8)."""
    if stats is None:
        stats = load_calib_stats(calib_path)
    return classify_early_sensitive_layers(
        stats,
        ratio_threshold=early_ratio_threshold,
        max_layers=max_early_fp16_layers,
    )


def resolve_matched_quantize_plan(
    quantize_names: set[str],
    calib_path: Path,
    *,
    early_ratio_threshold: float = 2.0,
    max_early_fp16_layers: int = 24,
    stats: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Single source of truth for matched naive vs ViDiT layer coverage."""
    if stats is None:
        stats = load_calib_stats(calib_path)
    early_fp16_layers = resolve_early_fp16_layers(
        calib_path,
        early_ratio_threshold=early_ratio_threshold,
        max_early_fp16_layers=max_early_fp16_layers,
        stats=stats,
    )
    early_set = set(early_fp16_layers)
    actual_int8_layers = sorted(quantize_names - early_set)
    return {
        "early_fp16_layers": early_fp16_layers,
        "actual_int8_layers": actual_int8_layers,
        "quantized_layers": len(actual_int8_layers),
        "calib_path": str(calib_path),
        "early_ratio_threshold": early_ratio_threshold,
        "max_early_fp16_layers": max_early_fp16_layers,
    }


def assert_matched_quantize_parity(w8a8_meta: dict[str, Any], vidit_meta: dict[str, Any]) -> None:
    """Runtime guard: naive and ViDiT must skip and quantize the same Linear layers."""
    w8_early = set(w8a8_meta.get("early_fp16_layers", []))
    v_early = set(vidit_meta.get("early_fp16_layers", []))
    if w8_early != v_early:
        only_w8 = sorted(w8_early - v_early)
        only_v = sorted(v_early - w8_early)
        raise RuntimeError(
            "Matched naive vs ViDiT early_fp16_layers mismatch: "
            f"only_in_naive={only_w8[:5]}, only_in_vidit={only_v[:5]}"
        )
    w8_int8 = set(w8a8_meta.get("actual_int8_layers", []))
    v_int8 = set(vidit_meta.get("actual_int8_layers", vidit_meta.get("smoothquant_layers", [])))
    if w8_int8 != v_int8:
        only_w8 = sorted(w8_int8 - v_int8)
        only_v = sorted(v_int8 - w8_int8)
        raise RuntimeError(
            "Matched naive vs ViDiT INT8 layer set mismatch: "
            f"only_in_naive={only_w8[:5]}, only_in_vidit={only_v[:5]}"
        )
    if w8a8_meta.get("quantized_layers") != vidit_meta.get("quantized_layers"):
        raise RuntimeError(
            "Matched naive vs ViDiT quantized_layers count mismatch: "
            f"naive={w8a8_meta.get('quantized_layers')}, vidit={vidit_meta.get('quantized_layers')}"
        )


def apply_vidit_w8a8(
    dit: nn.Module,
    quantize_names: set[str],
    calib_path: Path,
    *,
    smoothquant_alpha: float = 0.5,
    early_ratio_threshold: float = 2.0,
    max_early_fp16_layers: int = 24,
    actual_int8_layers: set[str] | None = None,
) -> dict[str, Any]:
    from torchao.quantization import int8_dynamic_activation_int8_weight, quantize_

    stats = load_calib_stats(calib_path)
    if actual_int8_layers is not None:
        override_set = set(actual_int8_layers)
        unknown = override_set - quantize_names
        if unknown:
            raise ValueError(
                f"actual_int8_layers contains {len(unknown)} name(s) not in quantize_names: "
                f"{sorted(unknown)[:5]}"
            )
        actual_quantize_names = override_set
        early_sensitive = sorted(quantize_names - actual_quantize_names)
        plan: dict[str, Any] = {
            "early_fp16_layers": early_sensitive,
            "actual_int8_layers": sorted(actual_quantize_names),
            "quantized_layers": len(actual_quantize_names),
            "calib_path": str(calib_path),
            "early_ratio_threshold": early_ratio_threshold,
            "max_early_fp16_layers": max_early_fp16_layers,
            "layer_override": True,
        }
    else:
        plan = resolve_matched_quantize_plan(
            quantize_names,
            calib_path,
            early_ratio_threshold=early_ratio_threshold,
            max_early_fp16_layers=max_early_fp16_layers,
            stats=stats,
        )
        actual_quantize_names = set(plan["actual_int8_layers"])
        early_sensitive = plan["early_fp16_layers"]

    smoothquant_layers: list[str] = []
    layer_scales: dict[str, torch.Tensor] = {}

    for name in sorted(actual_quantize_names):
        linear = get_module_by_name(dit, name)
        if not isinstance(linear, nn.Linear):
            continue
        layer_stats = stats.get(name)
        if layer_stats is None:
            raise RuntimeError(f"Calibration missing stats for quantize layer: {name}")
        channel_absmax = _max_channel_absmax_across_buckets(layer_stats)
        if channel_absmax is None:
            raise RuntimeError(f"Calibration missing channel_absmax for layer: {name}")
        scales = compute_smoothquant_scales(
            linear.weight.detach().float(),
            channel_absmax,
            alpha=smoothquant_alpha,
        )
        fold_smoothquant_into_linear(linear, scales)
        layer_scales[name] = scales
        smoothquant_layers.append(name)

    quant_count = 0
    for name in sorted(actual_quantize_names):
        linear = get_module_by_name(dit, name)
        if not isinstance(linear, nn.Linear):
            continue
        quantize_(linear, int8_dynamic_activation_int8_weight())
        quant_module = SmoothQuantActivationWrapper(linear, layer_scales[name])
        quant_count += 1
        replace_module_by_name(dit, name, quant_module)

    if quant_count == 0:
        raise RuntimeError("No Linear layers were quantized — check quantize_layers.txt")

    meta: dict[str, Any] = {
        "quantized_layers": quant_count,
        "actual_int8_layers": plan["actual_int8_layers"],
        "smoothquant_layers": smoothquant_layers,
        "early_fp16_layers": early_sensitive,
        "smoothquant_alpha": smoothquant_alpha,
        "early_ratio_threshold": early_ratio_threshold,
        "max_early_fp16_layers": max_early_fp16_layers,
        "calib_path": str(calib_path),
    }
    if plan.get("layer_override"):
        meta["layer_override"] = True
    return meta


