"""Pre-correction of the DynamiCtrl driving pose with the Article 1 KAPT pipeline.

Stage A (CPU, venv_gate; `python -m qtc.precorrect build --motion motion1`):
    1. Trim the driving video to the frames DynamiCtrl consumes (first `num_frames - 1`, plus a margin).
    2. Run the unmodified phase-1 extractor (OpenPose BODY_25 + MediaPipe fusion + Kalman).
    3. Refine the Kalman track with the Article 1 hybrid (qtc/refine.py): KAPT (soft conical shoulder
       constraints, EMA, velocity limit, limb-length stabiliser; alpha = 0.7, max_drift = 0.05) followed by
       SmoothNet (window 32).
    4. Store the correction field  delta[t, j] = refined[t, j] - fused_raw[t, j]  in normalised image
       coordinates (x / W, y / H) for the 18 OpenPose joints. The field therefore carries the Kalman
       smoothing, the biomechanical correction and the SmoothNet denoising of Article 1. Joints missing or
       below confidence in either track receive delta = 0.

Stage B (GPU, venv_gen; used by qtc/gen_qtc.py):
    `prepare_control_video_corrected` reproduces DynamiCtrl's utils.pre_process.preprocess and
    dwpose.preprocess.get_video_pose line by line, adding the correction field to each detected
    DWPose body before the reference-rescale fit and the rendering. Hands move with their wrist
    and the face moves with the nose, so the rendered pose stays connected.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PHASE1 = ROOT / "phase1"

OPENPOSE18 = ("nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow",
              "left_wrist", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
              "right_eye", "left_eye", "right_ear", "left_ear")
MIN_CONF = 0.3
# v2 (after the pilot): the field is restricted to the joints KAPT constrains (shoulder-elbow-wrist chains) and each
# joint's shift is capped at 3 % of the frame; v1 moved every joint (p95 shift 8-12 % of the frame), which altered
# the intended motion and raised violations in the pilot. Hands follow the wrists; the face is left untouched.
FIELD_VERSION = "v3"
# v3 (after pilot v2): the hard 3 % cap switched joints between capped and uncapped values from frame to frame and
# injected jitter (W4A8 acceleration variance 766 vs 268 without pre-correction). v3 low-pass filters the field over
# time (Gaussian, sigma = 2 frames) and replaces the hard cap by a soft one, |d| -> CAP * tanh(|d| / CAP).
SMOOTH_SIGMA = 2.0
CORRECTED_JOINTS = ("right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow", "left_wrist")
CAP = 0.03
MAX_DELTA = 0.15          # guard: a single-frame correction larger than 15 % of the frame is treated as a
                          # tracking failure and set to zero (logged in the stats)


# ----------------------------------------------------------------------------- stage A
def _kp_xy(kps: dict, name: str, w: float, h: float):
    r = kps.get(name) if isinstance(kps, dict) else None
    if not r or r.get("x") is None or r.get("y") is None:
        return None
    if float(r.get("confidence", 1.0)) < MIN_CONF:
        return None
    return float(r["x"]) / w, float(r["y"]) / h


def correction_field(smoothed_json: dict, kapt_json: dict, n_frames: int) -> tuple[np.ndarray, dict]:
    """delta (n_frames, 18, 2) in normalised coordinates; stats dict."""
    w = float(smoothed_json["image_size"]["width"])
    h = float(smoothed_json["image_size"]["height"])
    raw_frames = smoothed_json["frames"]
    cor_frames = kapt_json["frames"]
    T = min(n_frames, len(raw_frames), len(cor_frames))
    delta = np.zeros((n_frames, 18, 2), dtype=np.float64)
    n_set = n_guard = n_capped = 0
    for t in range(T):
        raw = raw_frames[t].get("keypoints_averaged") or raw_frames[t].get("keypoints") or {}
        cor = cor_frames[t].get("keypoints") or {}
        for j, name in enumerate(OPENPOSE18):
            if name not in CORRECTED_JOINTS:
                continue
            a, b = _kp_xy(raw, name, w, h), _kp_xy(cor, name, w, h)
            if a is None or b is None:
                continue
            d = np.array([b[0] - a[0], b[1] - a[1]])
            if not np.all(np.isfinite(d)):
                continue
            if np.hypot(*d) > MAX_DELTA:
                n_guard += 1
                continue
            delta[t, j] = d
            n_set += 1
    # temporal low-pass per joint and axis (reflect padding), then soft cap
    if T > 1 and SMOOTH_SIGMA > 0:
        r = int(3 * SMOOTH_SIGMA)
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / SMOOTH_SIGMA) ** 2)
        k /= k.sum()
        seg = delta[:T]
        pad = np.pad(seg, ((r, r), (0, 0), (0, 0)), mode="reflect" if T > r else "edge")
        delta[:T] = np.stack([np.tensordot(k, pad[i:i + 2 * r + 1], axes=(0, 0)) for i in range(T)])
    m = np.hypot(delta[..., 0], delta[..., 1])
    n_capped = int((m > CAP).sum())
    scale = np.where(m > 1e-12, CAP * np.tanh(m / CAP) / np.maximum(m, 1e-12), 1.0)
    delta *= scale[..., None]
    mag = np.hypot(delta[..., 0], delta[..., 1])
    stats = {"frames": int(T), "joints_set": int(n_set), "guard_zeroed": int(n_guard), "capped": int(n_capped),
             "field_version": FIELD_VERSION, "joints": list(CORRECTED_JOINTS), "cap": CAP,
             "smooth_sigma_frames": SMOOTH_SIGMA, "soft_cap": "CAP*tanh(|d|/CAP)",
             "mean_abs_delta": float(mag[mag > 0].mean()) if (mag > 0).any() else 0.0,
             "p95_abs_delta": float(np.percentile(mag[mag > 0], 95)) if (mag > 0).any() else 0.0,
             "image_size": [w, h]}
    return delta, stats


def build(motion_mp4: Path, out_dir: Path, *, n_frames: int, openpose_root: Path) -> Path:
    """Stage A. Writes <out_dir>/<motion>_correction_<FIELD_VERSION>.json and returns its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = motion_mp4.stem
    out = out_dir / f"{stem}_correction_{FIELD_VERSION}.json"
    if out.is_file():
        return out
    work = out_dir / f"work_{stem}"
    work.mkdir(parents=True, exist_ok=True)
    trimmed = work / f"{stem}_first{n_frames + 4}.mp4"
    if not trimmed.is_file():
        # visually lossless trim; frame i of the trim is frame i of the original (CFR source)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(motion_mp4), "-frames:v", str(n_frames + 4),
                        "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", str(trimmed)], check=True)
    env = {**os.environ}
    r = subprocess.run([sys.executable, "-m", "kalman.run_video_pipeline", "--video", str(trimmed),
                        "--out_dir", str(work), "--openpose_root", str(openpose_root)],
                       cwd=str(PHASE1), env=env, capture_output=True, text=True)
    sm = work / f"{trimmed.stem}_smoothed_keypoints.json"
    if r.returncode != 0 or not sm.is_file():
        raise RuntimeError(f"phase-1 extraction failed: {(r.stderr or r.stdout)[-3000:]}")
    sys.path.insert(0, str(ROOT))
    from qtc.refine import kapt_smoothnet                      # Article 1 hybrid: KAPT -> SmoothNet
    kp = kapt_smoothnet(sm, work / f"{trimmed.stem}_kapt_smoothnet.json", tag="qtc_precorrect")
    smoothed, kapt = json.loads(sm.read_text()), json.loads(kp.read_text())
    delta, stats = correction_field(smoothed, kapt, n_frames)
    payload = {"motion": stem, "source": str(motion_mp4), "n_frames": n_frames, "joints": list(OPENPOSE18),
               "coords": "normalised (x/W, y/H) of the driving video", "refinement": kapt.get("refinement"),
               "stats": stats, "delta": delta.round(6).tolist(),
               }
    out.write_text(json.dumps(payload))
    return out


def load_delta(path: Path) -> np.ndarray:
    return np.asarray(json.loads(Path(path).read_text())["delta"], dtype=np.float64)


# ----------------------------------------------------------------------------- stage B
def apply_delta_to_pose(pose: dict, d: np.ndarray) -> dict:
    """Add an (18, 2) normalised correction to person 0 of one DWPose detection (in place, returns pose)."""
    cand = pose["bodies"]["candidate"]
    if cand.shape[0] < 18:
        return pose
    cand[:18, :2] += d
    # hands: [left hand of person 0.., right hand of person 0..] stacked per person; person 0 only
    hands = pose.get("hands")
    if hands is not None and len(hands) >= 2:
        n_people = len(hands) // 2
        hands[0][:, :2] += d[OPENPOSE18.index("left_wrist")]
        hands[n_people][:, :2] += d[OPENPOSE18.index("right_wrist")]
    faces = pose.get("faces")
    if faces is not None and len(faces) >= 1:
        faces[0][:, :2] += d[OPENPOSE18.index("nose")]          # zero under the v2 field (nose not corrected)
    return pose


def get_video_pose_corrected(video_path: str, ref_image: np.ndarray, delta: np.ndarray, max_frame_num=None,
                             sample_stride: int = 1):
    """dwpose.preprocess.get_video_pose with the correction applied to every detected body."""
    import decord
    from tqdm import tqdm
    from dwpose.dwpose_detector import dwpose_detector as dwprocessor
    from dwpose.util import draw_pose

    ref_pose = dwprocessor(ref_image)
    ref_keypoint_id = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    ref_keypoint_id = [i for i in ref_keypoint_id
                       if len(ref_pose['bodies']['subset']) > 0 and ref_pose['bodies']['subset'][0][i] >= .0]
    ref_body = ref_pose['bodies']['candidate'][ref_keypoint_id]
    height, width, _ = ref_image.shape
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    max_frame_num = len(vr) if max_frame_num is None else min(len(vr), max_frame_num * sample_stride)
    frames = vr.get_batch(list(range(0, max_frame_num * sample_stride, sample_stride))).asnumpy()
    detected_poses = [dwprocessor(frm) for frm in tqdm(frames, desc="DWPose(corrected)")]
    dwprocessor.release_memory()
    # ---- the only change w.r.t. DynamiCtrl: add the KAPT correction field -----------------------
    for i, p in enumerate(detected_poses):
        if i < len(delta):
            apply_delta_to_pose(p, delta[i])
    # --------------------------------------------------------------------------------------------
    detected_bodies = np.stack([p['bodies']['candidate'] for p in detected_poses
                                if p['bodies']['candidate'].shape[0] == 18])[:, ref_keypoint_id]
    ay, by = np.polyfit(detected_bodies[:, :, 1].flatten(), np.tile(ref_body[:, 1], len(detected_bodies)), 1)
    if ref_body[:, 1][0] > detected_bodies[0, :, 1][0]:
        by = ref_body[:, 1][0] - detected_bodies[0, :, 1][0]
    elif ref_body[:, 1][0] < detected_bodies[0, :, 1][0]:
        by = -(detected_bodies[0, :, 1][0] - ref_body[:, 1][0])
    fh, fw, _ = vr[0].shape
    ax = ay / (fh / fw / height * width)
    bx = np.mean(np.tile(ref_body[:, 0], len(detected_bodies)) - detected_bodies[:, :, 0].flatten() * ax)
    a = np.array([ax, ay])
    b = np.array([bx, by])
    re_ensure = detected_poses[0]['bodies']['candidate'] * a + b
    if ref_body[:, 1][0] > re_ensure[0][1]:
        by = by + ref_body[:, 1][0] - re_ensure[0][1]
    elif ref_body[:, 1][0] < re_ensure[0][1]:
        by = by - (re_ensure[0][1] - ref_body[:, 1][0])
    b = np.array([bx, by])
    out = []
    for p in detected_poses:
        p['bodies']['candidate'] = p['bodies']['candidate'] * a + b
        p['faces'] = p['faces'] * a + b
        p['hands'] = p['hands'] * a + b
        out.append(np.array(draw_pose(p, height, width)))
    return np.stack(out)


def prepare_control_video_corrected(reference_image_path: str, driving_video: str, width: int, height: int,
                                    num_frames: int, correction_json: Path):
    """quant_common.prepare_control_video with the corrected driving pose."""
    from quant_common import (ensure_dwpose_weights, ensure_dynami_ctrl_on_path, ensure_onnxruntime_cuda_compat,
                              validate_clip_assets, verify_torch_stack)
    ensure_onnxruntime_cuda_compat()
    verify_torch_stack()
    ensure_dynami_ctrl_on_path(reference_image_path, driving_video)
    validate_clip_assets(reference_image_path, driving_video)
    ensure_dwpose_weights(repo_root=Path(reference_image_path).parent.parent)
    from diffusers.utils import load_image
    from torchvision.datasets.folder import pil_loader
    from torchvision.transforms.functional import center_crop, pil_to_tensor, resize
    from PIL import Image
    from dwpose.preprocess import get_image_pose
    from utils.load_validation_control import load_contorl_video_from_Image

    delta = load_delta(correction_json)
    image = load_image(image=reference_image_path)
    # --- utils.pre_process.preprocess, verbatim except for the video-pose call ---
    px = pil_to_tensor(pil_loader(reference_image_path))
    h, w = px.shape[-2:]
    h_w_ratio = float(h) / float(w)
    if h_w_ratio < height / width:
        h_resize, w_resize = height, math.ceil(height / h_w_ratio)
    else:
        h_resize, w_resize = math.ceil(width * h_w_ratio), width
    px = resize(px, [h_resize, w_resize], antialias=None)
    px = center_crop(px, [height, width]).permute((1, 2, 0)).numpy()
    image_pose = get_image_pose(px)
    video_pose = get_video_pose_corrected(driving_video, px, delta, max_frame_num=num_frames - 1, sample_stride=1)
    pose_pixels = np.concatenate([np.expand_dims(image_pose, 0), video_pose])
    pose_images = [Image.fromarray(p.transpose((1, 2, 0))) for p in pose_pixels]
    control = load_contorl_video_from_Image(list(pose_images), height, width).unsqueeze(0)
    return image, control


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--motion", required=True, help="motion1 | motion2 | motion3 or a path to an .mp4")
    b.add_argument("--assets", default=os.environ.get("QTC_ASSETS", "/workspace/content/DynamiCtrl/assets"))
    b.add_argument("--out-dir", default=os.environ.get("QTC_CORR", "/workspace/runs/qtc/driving_correction"))
    b.add_argument("--num-frames", type=int, default=37)
    b.add_argument("--openpose-root", default=os.environ.get("QTC_OP_ROOT", "/workspace/op_work/precorrect"))
    a = ap.parse_args()
    mp4 = Path(a.motion) if a.motion.endswith(".mp4") else Path(a.assets) / f"{a.motion}.mp4"
    opr = Path(a.openpose_root)
    (opr / "media_videos").mkdir(parents=True, exist_ok=True)
    (opr / "output").mkdir(parents=True, exist_ok=True)
    models = Path(os.environ.get("W", "/workspace")) / "openpose" / "models"
    if not (opr / "models").exists() and models.exists():
        (opr / "models").symlink_to(models)
    p = build(mp4, Path(a.out_dir), n_frames=a.num_frames - 1, openpose_root=opr)
    print(json.dumps(json.loads(p.read_text())["stats"]))


if __name__ == "__main__":
    main()
