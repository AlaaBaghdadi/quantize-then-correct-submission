#!/usr/bin/env python3
"""[REV-R1] Reconstruction metrics for every gated pair (runs in $W/venv_gen, light GPU use).

Per pair (candidate vs FP16 reference, 37 frames, 1360x768):
  PSNR (dB, per frame then mean), SSIM (per frame, RGB, data_range 255), LPIPS (AlexNet v0.1),
  tLPIPS = mean_t |LPIPS(c_t, c_t+1) - LPIPS(r_t, r_t+1)|  (temporal-consistency deviation),
  tMSE   = mean_t mean|(c_t+1 - c_t) - (r_t+1 - r_t)|^2     (frame-difference deviation, /255^2).
The same definitions as the Sep-13 WI-1 job for PSNR/SSIM/LPIPS, so the 54 overlapping pairs cross-check.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REV = Path(__file__).resolve().parent
sys.path.insert(0, str(REV))
import units as U  # noqa: E402
from gate_worker import ready, video_path  # noqa: E402

OUT = U.RUNS / "recon"


def read_frames(p: Path):
    import imageio.v3 as iio
    return np.stack([f[..., :3] for f in iio.imiter(str(p))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--follow", action="store_true")
    a = ap.parse_args()
    import torch
    import lpips
    from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
    torch.manual_seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = lpips.LPIPS(net="alex", version="0.1", verbose=False).to(dev).eval()

    def lp(x, y):
        tx = torch.from_numpy(x).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(dev)
        ty = torch.from_numpy(y).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(dev)
        out = []
        with torch.no_grad():
            for i in range(0, len(tx), 8):
                out.append(net(tx[i:i + 8], ty[i:i + 8]).flatten().cpu())
        return torch.cat(out).numpy()

    OUT.mkdir(parents=True, exist_ok=True)
    while True:
        did = 0
        for c, r, t in U.pairs():
            o = OUT / f"{c}__vs__{r}.json"
            if o.is_file() or not (ready(c) and ready(r)):
                continue
            try:
                q, ref = read_frames(video_path(c)), read_frames(video_path(r))
                n = min(len(q), len(ref))
                q, ref = q[:n], ref[:n]
                ps = [psnr(ref[i], q[i], data_range=255) for i in range(n)]
                ss = [ssim(ref[i], q[i], channel_axis=2, data_range=255) for i in range(n)]
                l = lp(q, ref)
                tl = np.abs(lp(q[1:], q[:-1]) - lp(ref[1:], ref[:-1]))
                dq = np.diff(q.astype(np.float32), axis=0)
                dr = np.diff(ref.astype(np.float32), axis=0)
                tm = float(np.mean((dq - dr) ** 2) / 255.0 ** 2)
                fin = [x for x in ps if np.isfinite(x)]
                rec = {"cand": c, "ref": r, "tag": t, "frames": int(n),
                       "psnr_db": float(np.mean(fin)) if fin else float("inf"), "psnr_db_min": float(np.min(fin)) if fin else float("inf"),
                       "ssim": float(np.mean(ss)), "ssim_min": float(np.min(ss)),
                       "lpips": float(np.mean(l)), "lpips_max": float(np.max(l)),
                       "tlpips": float(np.mean(tl)), "tmse": tm, "device": dev,
                       "per_frame": {"psnr": [float(x) for x in ps], "ssim": [float(x) for x in ss], "lpips": l.tolist()}}
                o.write_text(json.dumps(rec))
                did += 1
                print(time.strftime("%H:%M:%S"), "recon", c, "vs", r, f"psnr={rec['psnr_db']:.2f} lpips={rec['lpips']:.4f}", flush=True)
            except Exception as e:
                print("recon ERROR", c, r, repr(e), flush=True)
        if did:
            continue
        if not a.follow or Path(os.environ.get("DONE_FLAG", str(U.RUNS / "GEN_ALL_DONE"))).is_file():
            break
        time.sleep(120)


if __name__ == "__main__":
    main()
