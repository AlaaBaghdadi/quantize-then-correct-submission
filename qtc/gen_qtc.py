#!/usr/bin/env python3
"""GPU generation worker for the Quantize-then-Correct study. Runs in $W/venv_gen.

Reuses the Article 2 revision generator (revision/gen.py): pipeline loading, per-clip calibration files,
protected-layer set K, INT8 layer maps and the parameter census. Adds
  * variant w4a8   : INT4 g128 weights + INT8 per-token activations with SmoothQuant folding (qtc/w4a8.py)
  * variant w4a8mp : blocks 1-13 real W8A8 (ViDiT-Q), blocks 14-40 W4A8 (fallback map)
  * drive = pre    : KAPT-corrected driving pose (qtc/precorrect.py)
Every unit is idempotent (skipped when its mp4 + meta exist).

Usage: python -m qtc.gen_qtc --sets pilot            (from the repository root)
       python -m qtc.gen_qtc --sets core,optional --shard 0/1
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "revision"))

import revision.gen as G  # noqa: E402  (chdir to the DynamiCtrl clone, loads the PTQ config)
import torch  # noqa: E402

from qtc import units as QU  # noqa: E402
from qtc.w4a8 import apply_w4a8  # noqa: E402
from qtc.precorrect import FIELD_VERSION  # noqa: E402
from quant_common import layers_for_block_group, prepare_control_video, run_inference  # noqa: E402
from vidit_quant import apply_vidit_w8a8  # noqa: E402

CORR_DIR = QU.QTC / "driving_correction"
GROUP_SIZE = int(os.environ.get("QTC_W4_GROUP", "128"))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def quant_sets(u: QU.Unit, cp: Path):
    """(int8 layer set, int4 layer set, protected K) for a unit."""
    s_all, K = G.int8_set(u, cp)                       # full map: every quantizable layer minus K
    if u.variant == "vidit":
        return s_all, set(), K
    if u.variant == "w4a8":
        return set(), s_all, K
    if u.variant == "w4a8mp":
        early = layers_for_block_group(G.qnames(), 1, 13) & s_all
        return early, s_all - early, K
    raise ValueError(u.variant)


def heldout_calib(u: QU.Unit) -> Path:
    """Held-out calibration file for the subject of `u` (R1.2): elementwise max over timestep buckets and
    clips of the per-clip statistics of the three OTHER subjects. Uses the same per-clip files as the
    submitted runs (G.calib_path for calib='own'); the same reduction as revision/gen.py:loso_calib."""
    held = u.subject()
    out = QU.RUNS / "calib_heldout" / f"heldout_{held}.pt"
    if out.is_file():
        return out
    from vidit_quant import load_calib_stats
    srcs = [G.calib_path(QU.Unit("vidit", c)) for c in QU.clips([s for s in QU.SUBJECTS if s != held])]
    merged: dict = {}
    for p in srcs:
        for layer, buckets in load_calib_stats(p).items():
            ml = merged.setdefault(layer, {})
            for b, d in buckets.items():
                if b not in ml:
                    ml[b] = {"channel_absmax": d["channel_absmax"].clone().float(),
                             "tensor_absmax": float(d["tensor_absmax"]), "samples": d.get("samples", 0)}
                else:
                    ml[b]["channel_absmax"] = torch.maximum(ml[b]["channel_absmax"], d["channel_absmax"].float())
                    ml[b]["tensor_absmax"] = max(ml[b]["tensor_absmax"], float(d["tensor_absmax"]))
                    ml[b]["samples"] += d.get("samples", 0)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": {"format": "vidit_v1", "heldout_subject": held, "sources": [str(s) for s in srcs]},
                "stats": merged}, out)
    (out.with_suffix(".json")).write_text(json.dumps({"heldout_subject": held, "sources": [str(s) for s in srcs]}))
    log("wrote", out)
    return out


def calib_for(u: QU.Unit) -> Path:
    return heldout_calib(u) if u.calib == QU.HELDOUT else G.calib_path(u)


def build_quantized(master, u: QU.Unit):
    dit = copy.deepcopy(master).to("cuda")
    meta = {}
    if u.variant == "fp16":
        return dit, meta
    cp = calib_for(u)
    s8, s4, K = quant_sets(u, cp)
    meta.update(G.census(dit, s8 | s4))
    meta.update({"k_layers": sorted(K), "calib_path": str(cp), "int8_layers": len(s8), "int4_layers": len(s4)})
    if s4:
        meta.update(apply_w4a8(dit, s4, cp, smoothquant_alpha=G.ALPHA, group_size=GROUP_SIZE))
    if s8:
        m = apply_vidit_w8a8(dit, G.qnames(), cp, smoothquant_alpha=G.ALPHA, actual_int8_layers=s8)
        meta["w8a8_layers"] = m["quantized_layers"]
    lp = QU.RUNS / "meta" / "layersets"
    lp.mkdir(parents=True, exist_ok=True)
    (lp / f"{u.quant_key()}.json").write_text(json.dumps({"int8": sorted(s8), "int4": sorted(s4), "K": sorted(K)}))
    return dit, meta


def calib_available(clip: str) -> bool:
    return any((root / f"{clip}_vidit.pt").is_file() for root in (G.CALIB_OWN, G.CALIB_NEW))


def control_for(u: QU.Unit, cache: dict):
    key = (u.clip, u.drive)
    if key not in cache:
        img, mot = G.clip_assets(u.clip)
        W, H, N = G.INF["width"], G.INF["height"], G.INF["num_frames"]
        if u.drive == "raw":
            cache[key] = prepare_control_video(img, mot, W, H, N)
        else:
            from qtc.precorrect import prepare_control_video_corrected
            cj = CORR_DIR / f"{u.motion}_correction_{FIELD_VERSION}.json"
            if not cj.is_file():
                raise FileNotFoundError(f"driving correction missing: {cj} (run qtc.precorrect build first)")
            cache[key] = prepare_control_video_corrected(img, mot, W, H, N, cj)
    return cache[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="pilot")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-pre", action="store_true", help="defer drive=pre units (correction not ready)")
    a = ap.parse_args()
    k, n = map(int, a.shard.split("/"))
    units = QU.units_for(a.sets)
    if a.only:
        keep = set(a.only.split(","))
        units = [u for u in units if u.uid in keep]
    keys = sorted({u.quant_key() for u in units if u.variant != "fp16"})
    mine = {kk for i, kk in enumerate(keys) if i % n == k}
    units = [u for u in units if (u.variant != "fp16" and u.quant_key() in mine)
             or (u.variant == "fp16" and zlib.crc32(u.uid.encode()) % n == k)]
    todo = [u for u in units if not (u.video.is_file() and u.meta.is_file())]
    if a.skip_pre:
        todo = [u for u in todo if u.drive == "raw"]
    if a.limit:
        todo = todo[: a.limit]
    log(f"{len(units)} units in shard {a.shard}, {len(todo)} to do")
    if not todo:
        return 0
    for d in ("videos", "meta", "errors"):
        (QU.RUNS / d).mkdir(parents=True, exist_ok=True)

    from diffusers.utils import export_to_video
    t0 = time.time()
    pipe = G.prepare_fp16_pipeline(G.MODEL, G.DT)
    master = copy.deepcopy(pipe.transformer).to("cpu")
    log(f"pipeline loaded in {time.time() - t0:.0f}s")
    env = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "host": socket.gethostname(),
           "pod": os.environ.get("RUNPOD_POD_ID", ""), "study": "qtc"}
    controls: dict = {}
    cur_key, cur_meta = None, {}
    for u in todo:
        try:
            image_obj, control = control_for(u, controls)
            t_q = time.time()
            want = u.quant_key() if u.variant != "fp16" else "fp16"
            if want != cur_key:
                pipe.transformer = None
                gc.collect(); torch.cuda.empty_cache()
                dit, cur_meta = build_quantized(master, u)
                pipe.transformer = dit
                cur_key = want
            t_quant = time.time() - t_q
            hooks = None
            if u.collect_calib and not calib_available(u.clip):
                hooks = G.hook_calib(pipe.transformer, G.qnames())      # Article 2 calibration hooks
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t1 = time.time()
            frames = run_inference(pipe, G.args_for(u), image_obj, control)
            torch.cuda.synchronize()
            wall = time.time() - t1
            peak = torch.cuda.max_memory_allocated() / 1e9
            if hooks is not None:
                for h in hooks[0]:
                    h.remove()
                G.save_calib(hooks[1], u)                               # -> runs/calib/<clip>_vidit.pt
                del hooks
            export_to_video(frames[: G.INF["num_frames"]], str(u.video), fps=G.INF["fps"])
            meta = {**u.to_json(), **cur_meta, "gen_seconds": wall, "quantize_seconds": t_quant,
                    "peak_vram_gb": peak, "n_frames": len(frames), "env": env,
                    "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            if u.collect_calib:
                meta["calib_source"] = ("published (Drive)" if (G.CALIB_OWN / f"{u.clip}_vidit.pt").is_file()
                                        else "recorded in this run")
            if u.drive == "pre":
                meta["driving_correction"] = str(CORR_DIR / f"{u.motion}_correction_{FIELD_VERSION}.json")
            u.meta.write_text(json.dumps(meta, indent=1))
            log(f"done {u.uid} gen={wall:.0f}s quant={t_quant:.0f}s peak={peak:.1f}GB")
        except Exception as e:
            (QU.RUNS / "errors" / f"{u.uid}.txt").write_text(traceback.format_exc())
            log("ERROR", u.uid, repr(e))
            cur_key = None
            pipe.transformer = None
            gc.collect(); torch.cuda.empty_cache()
    (QU.QTC).mkdir(parents=True, exist_ok=True)
    (QU.QTC / f"GEN_DONE_{a.sets.replace(',', '_')}_{k}of{n}").write_text(time.strftime("%c"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
