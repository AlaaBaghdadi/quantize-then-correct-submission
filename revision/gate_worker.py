#!/usr/bin/env python3
"""[REV-R1] CPU kinematic-gate worker (runs in $W/venv_gate).

For every pair in units.pairs() whose two videos exist, extracts keypoints with the unmodified
phase-1 pipeline (OpenPose 1.7.0 BODY_25 + MediaPipe 0.10.21 + Kalman), caches them by video SHA-256,
and writes the published gate report (metric vectors, deltas, per-check verdicts) with the published
thresholds of configs/quant/vidit_ptq.json. Several processes can run in parallel (--procs): each gets
a private OpenPose working directory so OpenPose outputs never collide.

Loop mode (--follow) keeps polling until the file $W/runs/GEN_ALL_DONE exists and nothing is left.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

REV = Path(__file__).resolve().parent
ROOT = REV.parent
sys.path.insert(0, str(REV))
sys.path.insert(0, str(ROOT / "scripts" / "quant"))
PHASE1 = ROOT / "phase1"

import units as U  # noqa: E402

W = U.W
KP = U.RUNS / "keypoints"
GATE = U.RUNS / "gate"
OPENPOSE = W / "openpose"


def video_path(uid: str) -> Path:
    if uid.startswith("L-"):
        return U.legacy_items()[uid]
    return U.RUNS / "videos" / f"{uid}.mp4"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def op_root(worker: int) -> Path:
    r = W / "op_work" / f"w{worker}"
    (r / "media_videos").mkdir(parents=True, exist_ok=True)
    (r / "output").mkdir(parents=True, exist_ok=True)
    if not (r / "models").exists():
        (r / "models").symlink_to(OPENPOSE / "models")
    return r


def keypoints(video: Path, worker: int) -> Path:
    """Cached phase-1 keypoints JSON for one video (keyed by content hash)."""
    import subprocess
    import fcntl
    h = sha(video)
    out = KP / f"{h}.json"
    if out.is_file():
        return out
    KP.mkdir(parents=True, exist_ok=True)
    lock = open(KP / f"{h}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)  # one extraction per video even across processes/pods
    if out.is_file():
        return out
    worker = os.getpid()
    tmp = W / "op_work" / f"w{worker}" / "stage"
    tmp.mkdir(parents=True, exist_ok=True)
    staged = tmp / f"v_{h[:16]}.mp4"
    shutil.copy2(video, staged)
    env = {**os.environ, "OPENPOSE_BIN": str(OPENPOSE / "build/examples/openpose/openpose.bin"),
           "OMP_NUM_THREADS": os.environ.get("GATE_THREADS", "4"),
           "OPENBLAS_NUM_THREADS": os.environ.get("GATE_THREADS", "4")}
    cmd = [sys.executable, "-m", "kalman.run_video_pipeline", "--video", str(staged),
           "--out_dir", str(tmp), "--openpose_root", str(op_root(worker))]
    r = subprocess.run(cmd, cwd=str(PHASE1), env=env, capture_output=True, text=True, timeout=3600)
    res = tmp / f"{staged.stem}_smoothed_keypoints.json"
    if r.returncode != 0 or not res.is_file():
        raise RuntimeError(f"phase1 failed ({r.returncode}): {(r.stderr or r.stdout)[-3000:]}")
    shutil.move(str(res), out)
    (KP / f"{h}.src.json").write_text(json.dumps({"video": str(video), "sha256": h}))
    for f in tmp.glob(f"{staged.stem}*"):
        f.unlink(missing_ok=True)
    for f in (op_root(worker) / "output").glob(f"{staged.stem}_*"):
        f.unlink(missing_ok=True)
    staged.unlink(missing_ok=True)
    return out


def gate_pair(cand: str, ref: str, tag: str, worker: int) -> Path:
    from kinematic_gate.compare import GateThresholds, compute_deltas, evaluate_gate
    from kinematic_gate.metrics import (compute_joint_movement_summary_between, compute_metric_vector,
                                        load_keypoints_json, strip_internal_fields)
    out = GATE / f"{cand}__vs__{ref}.json"
    if out.is_file():
        return out
    vq, vr = video_path(cand), video_path(ref)
    kq, kr = keypoints(vq, worker), keypoints(vr, worker)
    mq = compute_metric_vector(kq, phase1_root=PHASE1)
    mr = compute_metric_vector(kr, phase1_root=PHASE1)
    mov = compute_joint_movement_summary_between(load_keypoints_json(kr), load_keypoints_json(kq))
    d = compute_deltas(mr, mq, movement_summary=mov)
    # signed violation delta as printed in the manuscript tables
    d["signed_delta_violation_fraction"] = float(mq["violation_fraction"]) - float(mr["violation_fraction"])
    th = GateThresholds()
    passed, checks = evaluate_gate(mr, mq, d, th)
    rep = {"cand": cand, "ref": ref, "tag": tag, "passed": passed, "checks": checks, "deltas": d,
           "metrics_ref": strip_internal_fields(mr), "metrics_cand": strip_internal_fields(mq),
           "movement": mov, "cand_video": str(vq), "ref_video": str(vr),
           "cand_sha256": kq.stem, "ref_sha256": kr.stem, "thresholds": th.__dict__,
           "gated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    GATE.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    return out


def ready(uid: str) -> bool:
    p = video_path(uid)
    if not p.is_file():
        return False
    if uid.startswith("L-"):
        return True
    return (U.RUNS / "meta" / f"{uid}.json").is_file()


def work(args):
    cand, ref, tag, worker = args
    try:
        gate_pair(cand, ref, tag, worker)
        return cand, ref, None
    except Exception:
        err = traceback.format_exc()
        (U.RUNS / "gate_errors").mkdir(parents=True, exist_ok=True)
        (U.RUNS / "gate_errors" / f"{cand}__vs__{ref}.txt").write_text(err)
        return cand, ref, err.splitlines()[-1]


def pending(tags=None):
    todo = []
    for c, r, t in U.pairs():
        if tags and t not in tags:
            continue
        if (GATE / f"{c}__vs__{r}.json").is_file():
            continue
        if (U.RUNS / "gate_errors" / f"{c}__vs__{r}.txt").is_file():
            continue
        if ready(c) and ready(r):
            todo.append((c, r, t))
    return todo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--tags", default="")
    ap.add_argument("--follow", action="store_true")
    a = ap.parse_args()
    tags = set(a.tags.split(",")) if a.tags else None
    with mp.Pool(a.procs) as pool:
        while True:
            todo = pending(tags)
            if todo:
                jobs = [(c, r, t, i % a.procs) for i, (c, r, t) in enumerate(todo)]
                # a video appears in many pairs: process one pair per distinct candidate first
                for c, r, e in pool.imap_unordered(work, jobs):
                    print(time.strftime("%H:%M:%S"), "gated" if e is None else "ERROR", c, "vs", r, e or "",
                          flush=True)
                continue
            if not a.follow or Path(os.environ.get("DONE_FLAG", str(U.RUNS / "GEN_ALL_DONE"))).is_file():
                break
            time.sleep(60)


if __name__ == "__main__":
    main()
