#!/usr/bin/env python3
"""[REV-R1] WI-8 cost census: INT8 parameter fraction, weight memory, peak VRAM, per-step DiT latency
and end-to-end latency for FP16, full, legacy-hybrid and nested-hybrid maps (both variants), eager and
torch.compile. One clip (human2_motion1, seed 42, 25 steps), 1 warm-up + N timed runs each.
Writes $W/runs/bench/<gpu>__<config>__<compile>.json
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import time
from pathlib import Path

REV = Path(__file__).resolve().parent
sys.path.insert(0, str(REV))
import gen as G  # noqa: E402  (sets cwd, paths, config)
import units as U  # noqa: E402
import torch  # noqa: E402
from quant_common import apply_w8a8, capture_dit_forward_kwargs, compile_model, prepare_fp16_pipeline, run_inference  # noqa: E402
from vidit_quant import apply_vidit_w8a8  # noqa: E402

CONFIGS = [("fp16", "none"), ("vidit", "full"), ("naive", "full"), ("vidit", "lhyb"), ("naive", "lhyb"),
           ("vidit", "nhyb"), ("naive", "nhyb")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="human2_motion1")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--micro", type=int, default=10)
    ap.add_argument("--compile", default="0,1")
    a = ap.parse_args()
    out = U.RUNS / "bench"
    out.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(0).replace(" ", "_")
    pipe = prepare_fp16_pipeline(G.MODEL, G.DT)
    master = copy.deepcopy(pipe.transformer).to("cpu")
    img, mot = G.clip_assets(a.clip)
    image_obj, control = G.prepare_control_video(img, mot, G.INF["width"], G.INF["height"], G.INF["num_frames"])
    for comp in [int(x) for x in a.compile.split(",")]:
        for variant, mp in CONFIGS:
            f = out / f"{gpu}__{variant}-{mp}__compile{comp}.json"
            if f.is_file():
                continue
            u = U.Unit("WI8", variant, mp, "na" if variant == "fp16" else "own", a.clip)
            pipe.transformer = None
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            m0 = torch.cuda.memory_allocated()
            dit = copy.deepcopy(master).to("cuda")
            rec = {"gpu": torch.cuda.get_device_name(0), "variant": variant, "map": mp, "compile": comp,
                   "torch": torch.__version__}
            if variant != "fp16":
                cp = G.calib_path(u)
                s8, K = G.int8_set(u, cp)
                rec.update(G.census(dit, s8))
                if variant == "vidit":
                    apply_vidit_w8a8(dit, G.qnames(), cp, smoothquant_alpha=G.ALPHA, actual_int8_layers=s8)
                else:
                    apply_w8a8(dit, G.qnames(), fp16_layers=G.qnames() - s8)
            else:
                rec.update(G.census(dit, set()))
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
            rec["dit_weight_mem_gb"] = (torch.cuda.memory_allocated() - m0) / 1e9  # resident DiT weights after quantization
            pipe.transformer = compile_model(dit) if comp else dit
            args = G.args_for(u)
            # per-step DiT latency on captured real inputs
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
            run_inference(pipe, args, image_obj, control)  # warm-up
            ts = []
            torch.cuda.reset_peak_memory_stats()
            for _ in range(a.runs):
                torch.cuda.synchronize(); t = time.perf_counter()
                run_inference(pipe, args, image_obj, control)
                torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
            rec["e2e_s"] = ts
            rec["e2e_mean_s"] = sum(ts) / len(ts)
            rec["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
            rec["throughput_videos_per_h"] = 3600.0 / rec["e2e_mean_s"]
            f.write_text(json.dumps(rec, indent=1))
            print(json.dumps(rec), flush=True)
            del kw
            pipe.transformer = None
            del dit
            gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
