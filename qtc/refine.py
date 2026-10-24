"""Article 1 skeleton refinement used by both correction arms: Kalman track -> KAPT -> SmoothNet.

Article 1 showed that KAPT alone enforces limb-length and joint-limit constraints but raises jitter and
acceleration variance, and that SmoothNet applied after KAPT gives the best result on all three metrics
(hybrid table). The same hybrid is therefore used for pre-correction of the driving pose and for
post-correction of generated skeletons. Hyper-parameters are frozen at the Article 1 values.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT / "phase1"
SMOOTHNET_ROOT = Path(os.environ.get("SMOOTHNET_ROOT", str(PHASE1 / "third_party" / "SmoothNet")))
SMOOTHNET_CKPT = Path(os.environ.get("SMOOTHNET_CKPT", str(SMOOTHNET_ROOT / "data" / "checkpoints" / "h36m_fcn_3D"
                                                          / "checkpoint_32.pth.tar")))
KAPT_ALPHA, KAPT_MAX_DRIFT, SN_WINDOW = 0.7, 0.05, 32

_SN = None


def smoothnet_model():
    global _SN
    if _SN is None:
        import torch
        if str(PHASE1) not in sys.path:
            sys.path.insert(0, str(PHASE1))
        from experiments.smoothnet_baseline.inference import load_smoothnet_model
        torch.set_num_threads(int(os.environ.get("GATE_THREADS", "4")))
        _SN = load_smoothnet_model(SMOOTHNET_CKPT, SMOOTHNET_ROOT, torch.device("cpu"), window_size=SN_WINDOW)
    return _SN


def kapt_smoothnet(in_json: Path, out_json: Path, *, model=None, tag: str = "qtc") -> Path:
    """Write the KAPT -> SmoothNet refinement of a Kalman keypoints JSON; returns out_json."""
    if out_json.is_file():
        return out_json
    if str(PHASE1) not in sys.path:
        sys.path.insert(0, str(PHASE1))
    import torch
    from experiments.benchmark_comparison.kapt_runner import run_kapt_correction
    from experiments.smoothnet_baseline.bridge import (apply_smoothed_h36m_to_frames, build_output_payload,
                                                       load_keypoints_json, save_keypoints_json)
    from experiments.smoothnet_baseline.inference import smooth_keypoints_data
    out_json.parent.mkdir(parents=True, exist_ok=True)
    kapt = out_json.with_name(out_json.stem + "__kapt_only.json")
    run_kapt_correction(Path(in_json), kapt, alpha=KAPT_ALPHA, max_drift=KAPT_MAX_DRIFT, force=True, variant_id=tag)
    data = load_keypoints_json(str(kapt))
    sm = smooth_keypoints_data(data, model or smoothnet_model(), torch.device("cpu"), window_size=SN_WINDOW)
    frames = apply_smoothed_h36m_to_frames(data.get("frames", []), sm)
    payload = build_output_payload(data, frames)
    payload["refinement"] = {"method": "Kalman -> KAPT -> SmoothNet (Article 1 hybrid)", "input": Path(in_json).name,
                             "kapt": {"alpha": KAPT_ALPHA, "max_drift": KAPT_MAX_DRIFT},
                             "smoothnet": {"checkpoint": str(SMOOTHNET_CKPT), "window": SN_WINDOW},
                             "kapt_stats": data.get("correction_params", {}).get("stats")}
    tmp = out_json.with_suffix(".tmp")
    save_keypoints_json(str(tmp), payload)
    tmp.replace(out_json)
    return out_json
