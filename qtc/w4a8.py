"""W4A8 post-training quantization for the DynamiCtrl DiT (simulated, accuracy-faithful).

Weights  : symmetric INT4, per output channel and per group of `group_size` input channels (RTN),
           levels {-8..7} * scale with scale = max|w_group| / 7 (the -8 level is never produced, so the
           grid is symmetric; this is the usual RTN convention for g128 W4 PTQ).
Activations: symmetric INT8, per-token dynamic scale = max|x_token| / 127 (identical to the torchao
           int8_dynamic_activation path used for W8A8 in Article 2).
SmoothQuant: the same channel folding as ViDiT-Q (alpha = 0.5) is applied before weight quantization,
           so W8A8 and W4A8 differ only in the weight bit-width and grouping.

The quantized weight is stored de-quantized in the compute dtype (fake quantization). The numerics equal a
real W4A8 kernel up to accumulation order; latency is NOT representative and is reported as simulated.
The packed-weight memory a real kernel would use is returned by `packed_weight_bytes`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_weight_int4_groupwise(w: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (dequantized weight in w.dtype, per-group scales [out, n_groups]) for a 2-D weight."""
    if w.dim() != 2:
        raise ValueError(f"expected 2-D weight, got {tuple(w.shape)}")
    out_f, in_f = w.shape
    g = group_size if group_size > 0 else in_f
    pad = (-in_f) % g
    wf = w.detach().float()
    if pad:
        wf = F.pad(wf, (0, pad))
    wg = wf.view(out_f, -1, g)
    scale = wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 7.0
    q = torch.clamp(torch.round(wg / scale), -8, 7)
    deq = (q * scale).view(out_f, -1)[:, :in_f]
    return deq.to(w.dtype), scale.squeeze(-1)


def fake_quant_act_int8_per_token(x: torch.Tensor) -> torch.Tensor:
    """Symmetric per-token (last-dim) dynamic INT8 fake quantization."""
    xf = x.float()
    s = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    return (torch.clamp(torch.round(xf / s), -127, 127) * s).to(x.dtype)


class W4A8Linear(nn.Module):
    """Linear layer with SmoothQuant input scaling, INT8 per-token activations and INT4 g-wise weights."""

    def __init__(self, linear: nn.Linear, smooth_scales: torch.Tensor | None, group_size: int = 128):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.group_size = group_size
        w = linear.weight.data
        if smooth_scales is not None:
            s = smooth_scales.to(device=w.device, dtype=torch.float32)
            w = (w.float() * s.view(1, -1)).to(w.dtype)               # fold: W' = W diag(s)
            self.register_buffer("smooth", s)
        else:
            self.smooth = None
        deq, scales = quantize_weight_int4_groupwise(w, group_size)
        self.weight = nn.Parameter(deq, requires_grad=False)
        self.register_buffer("w_scales", scales.to(torch.float16))
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.data, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.smooth is not None:
            x = x / self.smooth.to(dtype=x.dtype).view(*([1] * (x.dim() - 1)), -1)
        x = fake_quant_act_int8_per_token(x)
        return F.linear(x, self.weight, self.bias)

    def packed_weight_bytes(self) -> int:
        """Bytes a packed INT4 kernel would store: 0.5 B/weight + fp16 scale per group."""
        return self.weight.numel() // 2 + self.w_scales.numel() * 2


def apply_w4a8(dit: nn.Module, layer_names: Iterable[str], calib_path: Path | None, *,
               smoothquant_alpha: float = 0.5, group_size: int = 128) -> dict[str, Any]:
    """Replace every nn.Linear named in `layer_names` by a W4A8Linear. Returns a meta dict."""
    from vidit_quant import (_max_channel_absmax_across_buckets, compute_smoothquant_scales,  # Article 2 code
                             get_module_by_name, load_calib_stats, replace_module_by_name)
    stats = load_calib_stats(calib_path) if calib_path is not None else None
    done, dense_bytes, packed = [], 0, 0
    for name in sorted(set(layer_names)):
        lin = get_module_by_name(dit, name)
        if not isinstance(lin, nn.Linear):
            continue
        s = None
        if stats is not None:
            if name not in stats:
                raise RuntimeError(f"calibration missing stats for {name}")
            ch = _max_channel_absmax_across_buckets(stats[name])
            s = compute_smoothquant_scales(lin.weight.detach().float(), ch, alpha=smoothquant_alpha)
        q = W4A8Linear(lin, s, group_size)
        dense_bytes += lin.weight.numel() * lin.weight.element_size()
        packed += q.packed_weight_bytes()
        replace_module_by_name(dit, name, q)
        done.append(name)
    if not done:
        raise RuntimeError("apply_w4a8: no Linear layer was quantized")
    return {"w4a8_layers": len(done), "group_size": group_size, "smoothquant_alpha": smoothquant_alpha,
            "w4a8_dense_weight_bytes": dense_bytes, "w4a8_packed_weight_bytes": packed,
            "simulated": True}
