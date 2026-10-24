#!/usr/bin/env python3
"""[REV-R1] GPU generation worker. Runs in $W/venv_gen with cwd = /content/DynamiCtrl.

Loads the DynamiCtrl pipeline once, keeps a pristine CPU copy of the transformer, and for every
quantization configuration (quant_key) builds a fresh quantized copy, then renders all seeds that
share it. Every unit is idempotent (skipped when its mp4 + meta exist) and writes a meta JSON with
the exact INT8 layer set, the parameter census, timings and peak memory.

Usage: python gen.py --wi WI2,WI3 --shard 0/1
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import socket
import sys
import time
import traceback
import zlib
from pathlib import Path

REV = Path(__file__).resolve().parent
sys.path.insert(0, str(REV))
REPO = Path(os.environ.get("REPO", "/content/DynamiCtrl"))
sys.path.insert(0, str(REPO / "scripts" / "quant"))
os.chdir(REPO)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import units as U  # noqa: E402
from quant_common import (  # noqa: E402
    apply_w8a8,
    layers_for_block_group,
    load_quantize_names,
    make_generator,
    prepare_control_video,
    prepare_fp16_pipeline,
    run_inference,
    timestep_bucket,
)
from vidit_quant import apply_vidit_w8a8, classify_early_sensitive_layers, load_calib_stats  # noqa: E402

CFG = json.loads((REPO / "configs" / "quant" / "vidit_ptq.json").read_text())
INF = CFG["inference"]
VQ = CFG["vidit_quantization"]
ALPHA = float(VQ["smoothquant_alpha"])          # 0.5
RHO = float(VQ["early_sensitivity_ratio"])      # 1.25
KMAX = int(VQ["max_early_fp16_layers"])         # 24
MODEL = str(REPO / "checkpoints" / "DynamiCtrl")
DT = torch.bfloat16
CALIB_OWN = U.DRIVE / "calib_data"
CALIB_NEW = U.RUNS / "calib"
CALIB_LOSO = U.RUNS / "calib_loso"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def asset(name: str) -> str:
    return str(REPO / "assets" / name)


def clip_assets(clip: str):
    subj, mot = clip.split("_")
    return asset(f"{subj}.jpg"), asset(f"{mot}.mp4")


def args_for(u: U.Unit):
    from types import SimpleNamespace
    img, mot = clip_assets(u.clip)
    return SimpleNamespace(reference_image=img, driving_video=mot, prompt=INF["prompt"],
                           num_inference_steps=u.steps, width=INF["width"], height=INF["height"],
                           num_frames=INF["num_frames"], pose_control_function=INF["pose_control_function"],
                           guidance_scale=INF["guidance_scale"], seed=u.seed, fps=INF["fps"])


# ------------------------------------------------------------------ calibration files
def calib_path(u: U.Unit) -> Path | None:
    if u.calib == "own":
        for root in (CALIB_OWN, CALIB_NEW):
            p = root / f"{u.clip}_vidit.pt"
            if p.is_file():
                return p
        raise FileNotFoundError(f"no calibration file for {u.clip}")
    if u.calib == "loso":
        return loso_calib(u.clip.split("_")[0])
    return None


def loso_calib(held: str) -> Path:
    """Merge (elementwise max) the calibration statistics of every clip of the other subjects.
    Max-merging is the same reduction the published calibration applies across timesteps."""
    out = CALIB_LOSO / f"loso_{held}.pt"
    if out.is_file():
        return out
    CALIB_LOSO.mkdir(parents=True, exist_ok=True)
    srcs = [CALIB_OWN / f"{c}_vidit.pt" for c in U.clips([s for s in U.SUBJECTS if s != held])]
    merged: dict = {}
    for p in srcs:
        st = load_calib_stats(p)
        for layer, buckets in st.items():
            ml = merged.setdefault(layer, {})
            for b, d in buckets.items():
                if b not in ml:
                    ml[b] = {"channel_absmax": d["channel_absmax"].clone().float(),
                             "tensor_absmax": float(d["tensor_absmax"]), "samples": d.get("samples", 0)}
                else:
                    ml[b]["channel_absmax"] = torch.maximum(ml[b]["channel_absmax"], d["channel_absmax"].float())
                    ml[b]["tensor_absmax"] = max(ml[b]["tensor_absmax"], float(d["tensor_absmax"]))
                    ml[b]["samples"] += d.get("samples", 0)
    torch.save({"meta": {"format": "vidit_v1", "loso_heldout": held, "sources": [str(s) for s in srcs]},
                "stats": merged}, out)
    log("wrote", out)
    return out


# ------------------------------------------------------------------ layer maps
QNAMES = None


def qnames() -> set[str]:
    global QNAMES
    if QNAMES is None:
        QNAMES = load_quantize_names(REPO / "configs" / "quant" / "quantize_layers.txt")
    return QNAMES


def k_layers(cp: Path | None) -> set[str]:
    if cp is None:
        return set()
    return set(classify_early_sensitive_layers(load_calib_stats(cp), ratio_threshold=RHO, max_layers=KMAX))


def int8_set(u: U.Unit, cp: Path | None):
    allq = qnames()
    K = k_layers(cp) if (u.map in ("full", "nhyb") or (u.map.startswith("c") and u.keep_k)) else set()
    if u.map == "full":
        s = allq - K
    elif u.map == "lhyb":
        s = layers_for_block_group(allq, 14, 40)
    elif u.map == "nhyb":
        s = layers_for_block_group(allq, 14, 40) - K
    elif u.map in U.GROUPS:
        a, b = U.GROUPS[u.map]
        s = layers_for_block_group(allq, a, b)
    elif u.map.startswith("c"):
        fp = set()
        for a, b in u.fp16_blocks:
            fp |= layers_for_block_group(allq, a, b)
        s = allq - fp - K
    else:
        raise ValueError(u.map)
    return s, K


def census(dit: nn.Module, int8: set[str]) -> dict:
    total = sum(p.numel() for p in dit.parameters())
    n8 = 0
    for name, m in dit.named_modules():
        if name in int8 and isinstance(m, nn.Linear):
            n8 += m.weight.numel()
    return {"dit_params": int(total), "int8_params": int(n8), "int8_param_fraction": n8 / total,
            "int8_layers": len(int8)}


# ------------------------------------------------------------------ calibration hooks (FP16 units)
def hook_calib(dit, names):
    from collections import defaultdict
    from quant_common import get_current_timestep, wrap_transformer_timestep
    wrap_transformer_timestep(dit)
    raw = defaultdict(lambda: defaultdict(list))
    hs = []

    def mk(n):
        def h(_m, inp, _o):
            t = get_current_timestep()
            if t is None or not inp or inp[0] is None:
                return
            x = inp[0].detach().float()
            raw[n][timestep_bucket(t)].append({"channel_absmax": x.abs().amax(dim=tuple(range(x.dim() - 1))).cpu(),
                                              "tensor_absmax": x.abs().max().item(), "shape": tuple(x.shape)})
        return h
    for n, m in dit.named_modules():
        if n in names and isinstance(m, nn.Linear):
            hs.append(m.register_forward_hook(mk(n)))
    return hs, raw


def save_calib(raw, u: U.Unit):
    from collect_calib_vidit import aggregate_stats_vidit, validate_vidit_calib_stats
    stats = aggregate_stats_vidit(raw)
    validate_vidit_calib_stats(stats)
    CALIB_NEW.mkdir(parents=True, exist_ok=True)
    out = CALIB_NEW / f"{u.clip}_vidit.pt"
    torch.save({"meta": {"clip": u.clip, "seed": u.seed, "steps": u.steps, "format": "vidit_v1",
                         "source": "revision gen.py FP16 hook"}, "stats": stats}, out)
    log("calib ->", out)


# ------------------------------------------------------------------ main loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wi", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--only", default="", help="comma list of uids (debug)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    k, n = map(int, a.shard.split("/"))

    units = U.all_units(a.wi.split(","))
    units += U.required_fp16(units)
    units.sort(key=lambda u: (u.variant != "fp16", u.quant_key(), u.seed))
    if a.only:
        keep = set(a.only.split(","))
        units = [u for u in units if u.uid in keep]
    # shard by quant_key so seeds sharing a quantized model stay on one worker
    keys = sorted({u.quant_key() for u in units})
    mine = {kk for i, kk in enumerate(keys) if i % n == k}
    units = [u for u in units if (u.variant != "fp16" and u.quant_key() in mine)
             or (u.variant == "fp16" and zlib.crc32(u.uid.encode()) % n == k)]
    todo = [u for u in units if not (u.video.is_file() and u.meta.is_file())]
    if a.limit:
        todo = todo[: a.limit]
    log(f"{len(units)} units in shard {a.shard}, {len(todo)} to do")
    if not todo:
        return
    (U.RUNS / "videos").mkdir(parents=True, exist_ok=True)
    (U.RUNS / "meta").mkdir(parents=True, exist_ok=True)
    (U.RUNS / "errors").mkdir(parents=True, exist_ok=True)

    from diffusers.utils import export_to_video
    t0 = time.time()
    pipe = prepare_fp16_pipeline(MODEL, DT)
    master = copy.deepcopy(pipe.transformer).to("cpu")
    log(f"pipeline loaded in {time.time() - t0:.0f}s")
    controls: dict = {}
    cur_key, cur_meta = None, None
    env = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "host": socket.gethostname(),
           "pod": os.environ.get("RUNPOD_POD_ID", "")}
    try:
        import torchao
        env["torchao"] = torchao.__version__
    except Exception:
        pass

    for u in todo:
        try:
            if u.clip not in controls:
                img, mot = clip_assets(u.clip)
                controls[u.clip] = prepare_control_video(img, mot, INF["width"], INF["height"], INF["num_frames"])
            image_obj, control = controls[u.clip]
            t_q = time.time()
            hooks = None
            if u.variant == "fp16" or u.quant_key() != cur_key or u.collect_calib:
                pipe.transformer = None
                gc.collect(); torch.cuda.empty_cache()
                dit = copy.deepcopy(master).to("cuda")
                cur_meta = {}
                if u.variant != "fp16":
                    cp = calib_path(u)
                    s8, K = int8_set(u, cp)
                    cur_meta.update(census(dit, s8))
                    cur_meta["k_layers"] = sorted(K)
                    cur_meta["calib_path"] = str(cp) if cp else None
                    if u.variant == "vidit":
                        m = apply_vidit_w8a8(dit, qnames(), cp, smoothquant_alpha=ALPHA, actual_int8_layers=s8)
                    else:
                        m = apply_w8a8(dit, qnames(), fp16_layers=qnames() - s8)
                    cur_meta["quantized_layers"] = m["quantized_layers"]
                    cur_meta["int8_layer_names"] = sorted(s8)
                    cur_key = u.quant_key()
                else:
                    cur_key = None
                    if u.collect_calib:
                        hooks = hook_calib(dit, qnames())
                pipe.transformer = dit
            t_quant = time.time() - t_q
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t1 = time.time()
            frames = run_inference(pipe, args_for(u), image_obj, control)
            torch.cuda.synchronize()
            wall = time.time() - t1
            peak = torch.cuda.max_memory_allocated() / 1e9
            if hooks is not None:
                for h in hooks[0]:
                    h.remove()
                save_calib(hooks[1], u)
                cur_key = None
            export_to_video(frames[: INF["num_frames"]], str(u.video), fps=INF["fps"])
            meta = {**u.to_json(), **{kk: vv for kk, vv in cur_meta.items() if kk != "int8_layer_names"},
                    "gen_seconds": wall, "quantize_seconds": t_quant, "peak_vram_gb": peak,
                    "n_frames": len(frames), "env": env, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            if "int8_layer_names" in cur_meta:
                lp = U.RUNS / "meta" / "layersets"
                lp.mkdir(exist_ok=True)
                (lp / f"{cur_key or u.uid}.txt".replace("/", "_")).write_text("\n".join(cur_meta["int8_layer_names"]))
            u.meta.write_text(json.dumps(meta, indent=1))
            log(f"done {u.uid} gen={wall:.0f}s quant={t_quant:.0f}s peak={peak:.1f}GB")
        except Exception as e:  # keep going; record the defect
            (U.RUNS / "errors" / f"{u.uid}.txt").write_text(traceback.format_exc())
            log("ERROR", u.uid, repr(e))
            cur_key = None
            gc.collect(); torch.cuda.empty_cache()
    (U.RUNS / f"GEN_DONE_{a.wi.replace(',', '_')}_{k}of{n}").write_text(time.strftime("%c"))


if __name__ == "__main__":
    main()
