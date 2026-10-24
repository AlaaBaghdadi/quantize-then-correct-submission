# Pose Correction Pipeline

A pipeline for **video pose estimation**, **temporal smoothing** (Kalman), **biomechanical pose correction**, and **evaluation** (temporal stability, FID, PSNR/MSE). It uses OpenPose + MediaPipe for keypoints, then corrects shoulder/elbow kinematics and optionally runs PISE for pose-transfer comparison and metrics.

---

## Setup after cloning from GitHub

Large assets (virtual environments, OpenPose binaries, model weights, generated outputs, test videos) are **not in Git**. After clone, run one command from the project root:

```powershell
.\scripts\setup_all.ps1
```

This downloads OpenPose (~1.3 GB), PGN + PISE weights, and creates three Python environments.

**Prerequisites:** install **Python 3.9** and **Python 3.7** on your system (both are needed — they cannot share one venv because TensorFlow 1.15 requires 3.7).

| Environment | Path | Python | Requirements | Used by |
|-------------|------|--------|--------------|---------|
| Main pipeline | `diffusion/` | **3.9** | [requirements.txt](requirements.txt) | `kalman/`, `pose_handler/`, `pose-corrector/`, `run_video_correction_pipeline.py` |
| PGN parser | `parser/parserv/` | **3.7** | [parser/requirements.txt](parser/requirements.txt) | `run.py` (human parsing step) |
| PISE | `PISE-main/pise/` | **3.9** | [PISE-main/requirements.txt](PISE-main/requirements.txt) | `run.py`, PISE `test.py` |

Individual setup scripts (if you prefer step-by-step):

| Script | Purpose |
|--------|---------|
| [scripts/setup_openpose.ps1](scripts/setup_openpose.ps1) | Download OpenPose 1.7.0 Windows binaries |
| [scripts/setup_models.ps1](scripts/setup_models.ps1) | Download PGN checkpoint + PISE fashion weights |
| [scripts/setup_envs.ps1](scripts/setup_envs.ps1) | Create all three virtual environments |

**Test videos** are not in Git. Add your own `.mp4` (see [media/README.md](media/README.md)) and pass `--video path\to\your.mp4` to the pipeline.

**OpenPose custom path:** set `OPENPOSE_ROOT` to your `openpose/` folder, or use `--openpose_root` on the Kalman pipeline.

---

## Repository tree (what lives where)

```
Diffusion/
├── README.md                    ← You are here (overview + full pipeline)
├── SCRIPTS.md                   ← How to run each root-level script
│
├── kalman/                      ← Kalman smoothing + video pose pipeline
│   ├── README.md
│   ├── run_video_pipeline.py     # Video → OpenPose + MediaPipe → average → Kalman → output
│   ├── keypoint_smoother.py      # Kalman filter per keypoint
│   ├── video_pose_io.py         # OpenPose/MediaPipe I/O, averaging
│   └── config.py                # Paths (OpenPose, output dir)
│
├── kalman_out/                  ← Default output of Kalman pipeline (smoothed keypoints, videos)
│   └── README.md
│
├── pose-corrector/              ← Biomechanical pose correction (shoulder angles, kinematic chain)
│   ├── README.md
│   └── pose_corrector.py        # Constrained correction; CLI and Python API
│
├── pose_handler/                ← Pose I/O, formats, angles, visualization
│   ├── README.md
│   └── run.md                   # Detailed commands for each pose_handler script
│
├── pose_handler_out/            ← Default output of pose_handler scripts
│   └── README.md
│
├── run.py                       ← Single-image pose transfer (OpenPose → PISE); used by pipeline
├── run_video_correction_pipeline.py  ← Full correction pipeline (corrected vs original, metrics)
├── plot_smoothed_vs_corrected_keypoints.py  ← Visualize smoothed vs corrected on video frames
├── pose_evaluator.py            ← Temporal stability + FID evaluation
└── video_metrics.py             ← Extract frames, FID between dirs, temporal consistency
```

---

## Full pipeline: video → metrics

End-to-end flow:

1. **Video → smoothed keypoints** (Kalman pipeline)  
2. **Smoothed keypoints → corrected keypoints** (pose corrector; can be inside step 3)  
3. **Corrected keypoints + video → comparison videos + metrics** (run_video_correction_pipeline)  
4. **Optional: evaluation report** (pose_evaluator) and **visualization** (plot_smoothed_vs_corrected_keypoints)

### Step 1: Run Kalman video pipeline (video → smoothed keypoints)

From project root:

```bash
python -m kalman.run_video_pipeline --video path/to/your_video.mp4 --out_dir kalman_out
```

- **Input:** A video file.  
- **Output (in `kalman_out/`):**
  - `your_video_smoothed_keypoints.json` — per-frame smoothed keypoints (and metadata).
  - Optional rendered video with skeleton/angles if the script writes it.
- **Requires:** OpenPose installed; path set in `kalman/config.py` or via `--openpose_root`.

See [kalman/README.md](kalman/README.md) for details.

---

### Step 2 (optional standalone): Correct keypoints only

If you already have a smoothed keypoints JSON and only want corrected keypoints (no PISE):

```bash
python pose-corrector/pose_corrector.py --in_json kalman_out/your_video_smoothed_keypoints.json --out_dir kalman_out
```

- **Output:** e.g. `kalman_out/your_video_smoothed_keypoints_corrected.json` (name depends on `--out_path`/`--out_dir`).

See [pose-corrector/README.md](pose-corrector/README.md) and [SCRIPTS.md](SCRIPTS.md).

---

### Step 3: Run full correction pipeline (corrected vs original, videos, metrics)

Uses the **smoothed** keypoints JSON from Step 1; runs pose correction internally, then for each “corrected” frame generates two PISE images (corrected vs original keypoints), builds two videos, and computes metrics:

```bash
python run_video_correction_pipeline.py --video path/to/your_video.mp4 --keypoints_json kalman_out/your_video_smoothed_keypoints.json --out_dir kalman_out
```

- **Inputs:**
  - `--video`: same (or matching) video.
  - `--keypoints_json`: output of Step 1 (`*_smoothed_keypoints.json`).
- **Output (in `--out_dir`, e.g. `kalman_out/`):**
  - `your_video_smoothed_keypoints_corrected.json` — corrected keypoints (and `keypoints_smoothed` per frame).
  - `comparison_frames/` — side-by-side frames (corrected vs original).
  - `your_video_corrected_keypoints.mp4` and `your_video_original_keypoints.mp4`.
  - `your_video_metrics.json` — per-frame PSNR/MSE, optional FID.

**Useful flags:**  
`--skip_fid` to skip FID; `--skip_verify_plots` to skip verification images; `--workspace_root` for PISE/OpenPose root; `--alpha` for pose corrector smoothing.

See [SCRIPTS.md](SCRIPTS.md) for all options.

---

### Step 4 (optional): Evaluation report (temporal stability + FID)

```bash
python pose_evaluator.py --keypoints_original kalman_out/your_video_smoothed_keypoints.json --keypoints_corrected kalman_out/your_video_smoothed_keypoints_corrected.json --path_ground_truth path/to/real_frames --path_baseline path/to/baseline_frames --path_corrected path/to/corrected_frames
```

- Compares original vs corrected keypoints (temporal stability) and can compute FID between baseline vs corrected generated frames.

See [SCRIPTS.md](SCRIPTS.md).

---

### Step 5 (optional): Visualize smoothed vs corrected on video

```bash
python plot_smoothed_vs_corrected_keypoints.py --video path/to/your_video.mp4 --corrected_json kalman_out/your_video_smoothed_keypoints_corrected.json --out_dir kalman_out/smoothed_vs_corrected_plots
```

- If the corrected JSON has `keypoints_smoothed` per frame (as from `run_video_correction_pipeline`), you don’t need `--smoothed_json`. Otherwise pass both.

See [SCRIPTS.md](SCRIPTS.md).

---

## Dependencies

This project uses **three separate virtual environments** (TensorFlow 1.15 needs Python 3.7; main pipeline and PISE use Python 3.9). See [Setup after cloning](#setup-after-cloning-from-github) above.

| Component | Python | Requirements | Not in Git | Redownload |
|-----------|--------|--------------|------------|------------|
| Main venv | **3.9** | [requirements.txt](requirements.txt) | `diffusion/` | `.\scripts\setup_envs.ps1` |
| Parser venv | **3.7** | [parser/requirements.txt](parser/requirements.txt) | `parser/parserv/` | `.\scripts\setup_envs.ps1` |
| PISE venv | **3.9** | [PISE-main/requirements.txt](PISE-main/requirements.txt) | `PISE-main/pise/` | `.\scripts\setup_envs.ps1` |
| OpenPose binaries | — | — | `openpose-1.7.0-binaries-win64-cpu-python3.7-flir-3d/` | `.\scripts\setup_openpose.ps1` |
| PGN checkpoint | — | — | `parser/checkpoint/CIHP_pgn/` | `.\scripts\setup_models.ps1` |
| PISE weights | — | — | `PISE-main/results/fashion/` | `.\scripts\setup_models.ps1` |

Paths to OpenPose are in `kalman/config.py` (supports `OPENPOSE_ROOT` env var). PISE/parser venv paths are used by `run.py`.

---

## Where to read more

| Topic | File |
|-------|------|
| Project tree + full pipeline | This file (README.md) |
| Root-level scripts (run, pipeline, plot, evaluator, video_metrics) | [SCRIPTS.md](SCRIPTS.md) |
| Kalman smoothing + video pipeline | [kalman/README.md](kalman/README.md) |
| Kalman output folder | [kalman_out/README.md](kalman_out/README.md) |
| Pose corrector (pose_corrector.py) | [pose-corrector/README.md](pose-corrector/README.md) |
| Pose I/O, formats, angles, viz | [pose_handler/README.md](pose_handler/README.md) |
| Pose handler output folder | [pose_handler_out/README.md](pose_handler_out/README.md) |

---

## Quick reference: one-shot from video to metrics

Assuming OpenPose and PISE are set up and `kalman_out` is the desired output dir:

```bash
# 1) Video → smoothed keypoints
python -m kalman.run_video_pipeline --video path/to/video.mp4 --out_dir kalman_out

# 2) Smoothed → corrected + comparison videos + metrics (same video, same stem)
python run_video_correction_pipeline.py --video path/to/video.mp4 --keypoints_json kalman_out/video_smoothed_keypoints.json --out_dir kalman_out
```

Then open `kalman_out/video_metrics.json` for PSNR/MSE (and FID if not skipped). Optionally run `plot_smoothed_vs_corrected_keypoints.py` and `pose_evaluator.py` as above.
