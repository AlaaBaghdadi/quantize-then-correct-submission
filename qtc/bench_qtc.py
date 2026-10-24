#!/usr/bin/env python3
"""Latency / memory census for FP16, W8A8 (real torchao INT8 GEMM) and W4A8 (simulated) on one clip.

Per configuration: resident DiT weight memory after quantization, per-forward DiT latency on captured real
inputs (3 warm-up + N timed), end-to-end latency (1 warm-up + R timed) and peak VRAM. W4A8 latency is
recorded but is not a speed claim (fake-quantized weights run in bf16).
Writes $W/runs/qtc/bench/<gpu>__<variant>.json
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "revision"))
import revision.gen as G  # noqa: E402
import torch  # noqa: E402

from qtc import units as QU  # noqa: E402
from qtc.gen_qtc import build_quantized  # noqa: E402
from quant_common import capture_dit_forward_kwargs, prepare_control_video, run_inference  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=QU.PILOT_CLIP)
    ap.add_argument("--variants", default="fp16,vidit,w4a8")
    ap.add_argument("--micro", type=int, default=10)
    ap.add_argument("--runs", type=int, default=2)
    a = ap.parse_args()
    out = QU.QTC / "bench"
    out.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(0).replace(" ", "_")
    pipe = G.prepare_fp16_pipeline(G.MODEL, G.DT)
    master = copy.deepcopy(pipe.transformer).to("cpu")
    img, mot = G.clip_assets(a.clip)
    image_obj, control = prepare_control_video(img, mot, G.INF["width"], G.INF["height"], G.INF["num_frames"])
    for v in a.variants.split(","):
        f = out / f"{gpu}__{v}.json"
        if f.is_file():
            continue
        u = QU.Unit(v, a.clip, 42)
        pipe.transformer = None
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        m0 = torch.cuda.memory_allocated()
        dit, meta = build_quantized(master, u)
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        rec = {"gpu": torch.cuda.get_device_name(0), "variant": v, "torch": torch.__version__,
               "resident_dit_weight_gb": (torch.cuda.memory_allocated() - m0) / 1e9,
               **{k: meta.get(k) for k in ("int8_param_fraction", "int8_layers", "int4_layers",
                                           "w4a8_packed_weight_bytes", "w4a8_dense_weight_bytes", "dit_params")},
               "latency_is_simulated": v.startswith("w4a8")}
        pipe.transformer = dit
        args = G.args_for(u)
        kw = capture_dit_forward_kwargs(pipe, args)
        kw["return_dict"] = False
        with torch.no_grad():
            for _ in range(3):
                pipe.transformer(**kw)
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(a.micro):
                pipe.transformer(**kw)
            torch.cuda.synchronize()
        rec["dit_forward_s"] = (time.perf_counter() - t) / a.micro
        run_inference(pipe, args, image_obj, control)
        torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(a.runs):
            torch.cuda.synchronize(); t = time.perf_counter()
            run_inference(pipe, args, image_obj, control)
            torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
        rec.update(e2e_s=ts, e2e_mean_s=sum(ts) / len(ts), peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9)
        f.write_text(json.dumps(rec, indent=1))
        print(json.dumps(rec), flush=True)
        del kw
        pipe.transformer = None
        del dit
        gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
