#!/usr/bin/env python3
"""Scoring for the Quantize-then-Correct study.

Mode `kin` (CPU, $W/venv_gate, which additionally holds a CPU torch for SmoothNet):
  for every generated video
    * skeleton extraction with the unmodified phase-1 pipeline (OpenPose 1.7 BODY_25 + MediaPipe + Kalman),
      cached by the video's SHA-256 (the same cache as the Article 2 gate worker);
    * post-correction "kapt_smoothnet": Article 1 KAPT (alpha 0.7, max_drift 0.05) followed by SmoothNet
      (H36M FCN checkpoint, window 32), cached per skeleton;
    * absolute kinematic metric vector for post in {none, kapt_smoothnet}  -> runs/qtc/metrics/<uid>__<post>.json
  for every pair (candidate, FP16 reference of the same clip and seed) and post in {none, kapt_smoothnet}
    * gate deltas, NPE movement summary and per-check verdicts with the Article 2 thresholds, computed on
      skeletons that received the SAME post-processing (matched processing)
                                                          -> runs/qtc/gate/<cand>__vs__<ref>__<post>.json
Mode `recon` ($W/venv_gen, GPU if available): PSNR, SSIM, LPIPS(alex), tLPIPS, tMSE per pair
                                                          -> runs/qtc/recon/<cand>__vs__<ref>.json
Both modes are idempotent and can follow a running generator (--follow).
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT / "phase1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "quant"))

from qtc import units as QU  # noqa: E402

METRICS = QU.QTC / "metrics"
GATE = QU.QTC / "gate"
RECON = QU.QTC / "recon"
POST = QU.QTC / "post"
ERR = QU.QTC / "score_errors"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ready(uid: str) -> bool:
    return (QU.RUNS / "videos" / f"{uid}.mp4").is_file() and (QU.RUNS / "meta" / f"{uid}.json").is_file()


def video(uid: str) -> Path:
    return QU.RUNS / "videos" / f"{uid}.mp4"


# ----------------------------------------------------------------------------- skeletons
def skeleton(uid: str, worker: int) -> Path:
    sys.path.insert(0, str(ROOT / "revision"))
    import gate_worker as GW                       # Article 2 extractor + SHA cache (runs/keypoints)
    return GW.keypoints(video(uid), worker)


def post_json(kp: Path, post: str) -> Path:
    """Return the skeleton JSON for a post-processing mode (cached under runs/qtc/post)."""
    if post == "none":
        return kp
    if post != "kapt_smoothnet":
        raise ValueError(post)
    from qtc.refine import kapt_smoothnet
    return kapt_smoothnet(kp, POST / f"{kp.stem}__kapt_smoothnet.json", tag="qtc_post")


def metric_vector(path: Path) -> dict:
    from kinematic_gate.metrics import compute_metric_vector, strip_internal_fields
    return strip_internal_fields(compute_metric_vector(path, phase1_root=PHASE1))


def absolute(uid: str, post: str, worker: int) -> dict:
    out = METRICS / f"{uid}__{post}.json"
    if out.is_file():
        return json.loads(out.read_text())
    kp = post_json(skeleton(uid, worker), post)
    rec = {"uid": uid, "post": post, "skeleton": str(kp), **metric_vector(kp)}
    METRICS.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    return rec


def gate_pair(cand: str, ref: str, tag: str, post: str, worker: int) -> Path:
    from kinematic_gate.compare import GateThresholds, compute_deltas, evaluate_gate
    from kinematic_gate.metrics import compute_joint_movement_summary_between, load_keypoints_json
    out = GATE / f"{cand}__vs__{ref}__{post}.json"
    if out.is_file():
        return out
    mq, mr = absolute(cand, post, worker), absolute(ref, post, worker)
    kq, kr = Path(mq["skeleton"]), Path(mr["skeleton"])
    mov = compute_joint_movement_summary_between(load_keypoints_json(kr), load_keypoints_json(kq))
    d = compute_deltas(mr, mq, movement_summary=mov)
    d["signed_delta_violation_fraction"] = float(mq["violation_fraction"]) - float(mr["violation_fraction"])
    for key in ("acceleration_variance", "mean_limb_length_std", "var_accel_norm"):
        d[f"signed_rel_delta_{key}"] = (float(mq[key]) - float(mr[key])) / max(abs(float(mr[key])), 1e-9)
    th = GateThresholds()
    passed, checks = evaluate_gate(mr, mq, d, th)
    rep = {"cand": cand, "ref": ref, "tag": tag, "post": post, "passed": passed, "checks": checks, "deltas": d,
           "metrics_ref": {k: v for k, v in mr.items() if k != "skeleton"},
           "metrics_cand": {k: v for k, v in mq.items() if k != "skeleton"},
           "movement": mov, "thresholds": th.__dict__, "gated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    GATE.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    return out


def _work(job):
    cand, ref, tag, post, worker = job
    try:
        gate_pair(cand, ref, tag, post, worker)
        return job, None
    except Exception:
        ERR.mkdir(parents=True, exist_ok=True)
        (ERR / f"{cand}__vs__{ref}__{post}.txt").write_text(traceback.format_exc())
        return job, traceback.format_exc().splitlines()[-1]


def _abs_work(job):
    uid, post, worker = job
    try:
        absolute(uid, post, worker)
        return job, None
    except Exception:
        ERR.mkdir(parents=True, exist_ok=True)
        (ERR / f"{uid}__{post}.txt").write_text(traceback.format_exc())
        return job, traceback.format_exc().splitlines()[-1]


def run_kin(sets: str, procs: int, follow: bool, done_flag: Path):
    while True:
        vids = [uid for uid in QU.all_videos(sets) if ready(uid)]
        jobs_a = [(uid, p, i % procs) for i, uid in enumerate(vids) for p in QU.POSTS
                  if not (METRICS / f"{uid}__{p}.json").is_file() and not (ERR / f"{uid}__{p}.txt").is_file()]
        jobs_g = [(c, r, t, p, 0) for c, r, t in QU.pairs(sets) for p in QU.POSTS
                  if ready(c) and ready(r) and not (GATE / f"{c}__vs__{r}__{p}.json").is_file()
                  and not (ERR / f"{c}__vs__{r}__{p}.txt").is_file()]
        if jobs_a:
            # post="none" first: one skeleton extraction per video, parallel across videos
            jobs_a.sort(key=lambda j: j[1] != "none")
            with mp.Pool(procs) as pool:
                for job, e in pool.imap_unordered(_abs_work, jobs_a):
                    log("metrics" if e is None else "ERROR", job[0], job[1], e or "")
            continue
        if jobs_g:
            for job in jobs_g:
                _, e = _work(job)
                log("gated" if e is None else "ERROR", job[0], "vs", job[1], job[3], e or "")
            continue
        if not follow or done_flag.is_file():
            break
        time.sleep(60)


# ----------------------------------------------------------------------------- pixel metrics
def run_recon(sets: str, follow: bool, done_flag: Path):
    import numpy as np
    import torch
    import imageio.v3 as iio
    import lpips
    from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = lpips.LPIPS(net="alex", version="0.1", verbose=False).to(dev).eval()

    def frames(p):
        return np.stack([f[..., :3] for f in iio.imiter(str(p))])

    def lp(x, y):
        tx = torch.from_numpy(x).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(dev)
        ty = torch.from_numpy(y).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(dev)
        with torch.no_grad():
            return torch.cat([net(tx[i:i + 8], ty[i:i + 8]).flatten().cpu() for i in range(0, len(tx), 8)]).numpy()

    RECON.mkdir(parents=True, exist_ok=True)
    while True:
        did = 0
        for c, r, t in QU.pairs(sets):
            o = RECON / f"{c}__vs__{r}.json"
            if o.is_file() or not (ready(c) and ready(r)):
                continue
            q, ref = frames(video(c)), frames(video(r))
            n = min(len(q), len(ref))
            q, ref = q[:n], ref[:n]
            ps = [psnr(ref[i], q[i], data_range=255) for i in range(n)]
            ss = [ssim(ref[i], q[i], channel_axis=2, data_range=255) for i in range(n)]
            l = lp(q, ref)
            tl = np.abs(lp(q[1:], q[:-1]) - lp(ref[1:], ref[:-1]))
            tm = float(np.mean((np.diff(q.astype(np.float32), axis=0) - np.diff(ref.astype(np.float32), axis=0)) ** 2)
                       / 255.0 ** 2)
            fin = [x for x in ps if np.isfinite(x)]
            rec = {"cand": c, "ref": r, "tag": t, "frames": int(n), "psnr_db": float(np.mean(fin)) if fin else float("inf"),
                   "ssim": float(np.mean(ss)), "lpips": float(np.mean(l)), "tlpips": float(np.mean(tl)), "tmse": tm,
                   "per_frame": {"psnr": [float(x) for x in ps], "ssim": [float(x) for x in ss], "lpips": l.tolist()}}
            o.write_text(json.dumps(rec))
            did += 1
            log("recon", c, "vs", r, f"psnr={rec['psnr_db']:.2f} lpips={rec['lpips']:.4f}")
        if did:
            continue
        if not follow or done_flag.is_file():
            break
        time.sleep(120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["kin", "recon"])
    ap.add_argument("--sets", default="pilot,core,optional")
    ap.add_argument("--procs", type=int, default=int(os.environ.get("GATE_PROCS", "4")))
    ap.add_argument("--follow", action="store_true")
    a = ap.parse_args()
    done = Path(os.environ.get("DONE_FLAG", str(QU.QTC / "GEN_ALL_DONE")))
    if a.mode == "kin":
        run_kin(a.sets, a.procs, a.follow, done)
    else:
        run_recon(a.sets, a.follow, done)


if __name__ == "__main__":
    main()
