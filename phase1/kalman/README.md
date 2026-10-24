# kalman — Video pose pipeline & keypoint smoothing

This folder contains the **Kalman-based keypoint smoothing** and the **full video pose pipeline**: extract keypoints (OpenPose + MediaPipe), average per frame, smooth with Kalman, optionally compute angles and render video.

---

## Setup after cloning from GitHub

1. **OpenPose** (~1.3 GB, not in Git):
   ```powershell
   .\scripts\setup_openpose.ps1
   ```
   See [openpose-1.7.0-binaries-win64-cpu-python3.7-flir-3d/README.md](../openpose-1.7.0-binaries-win64-cpu-python3.7-flir-3d/README.md).

2. **Main Python environment** (`diffusion/`, **Python 3.9**):
   ```powershell
   .\scripts\setup_envs.ps1
   # Manual (Python 3.9 required):
   py -3.9 -m venv diffusion
   diffusion\Scripts\pip install -r requirements.txt
   ```

3. **Test video** — add your own `.mp4` (see [media/README.md](../media/README.md)).

Optional: set `OPENPOSE_ROOT` or pass `--openpose_root` instead of using the default folder name.

---

## What’s in this folder

| File | Purpose |
|------|--------|
| `run_video_pipeline.py` | **Main entry:** run full pipeline (video → smoothed keypoints JSON, optional video with skeleton/angles). |
| `keypoint_smoother.py` | Kalman filter per keypoint (constant velocity model; one `cv2.KalmanFilter` per joint). |
| `video_pose_io.py` | OpenPose frame JSON discovery, MediaPipe extraction from video, scaling, OpenPose→named conversion, averaging, save/load. |
| `config.py` | Default paths: project root, OpenPose root, default output directory (e.g. `kalman_out`). |
| `ANGLES.md` | Notes on angle computation / clinical angles. |

---

## How to run the video pipeline

From the **project root** (parent of `kalman/`):

```bash
python -m kalman.run_video_pipeline --video path/to/your_video.mp4 --out_dir kalman_out
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--video` | Yes | — | Path to input video. |
| `--out_dir` | No | `kalman_out` (from config) | Directory for smoothed keypoints JSON and any rendered output. |
| `--openpose_root` | No | From `config.get_default_openpose_root()` | OpenPose installation root (e.g. `openpose-1.7.0-binaries-win64-.../openpose`). |
| `--keep_temp` | No | False | Keep temporary files (video copy, OpenPose output, etc.). |
| `--min_conf` | No | 0.3 | Minimum confidence for averaging and drawing. |
| `--person_index` | No | 0 | OpenPose person index. |

### Output

- **Smoothed keypoints JSON** in `out_dir`, e.g.  
  `your_video_smoothed_keypoints.json`  
  Structure: `frames` (list of per-frame keypoints), `image_size`, `fps`, etc. This file is the input for:
  - `run_video_correction_pipeline.py`
  - `pose-corrector/pose_corrector.py`
  - `plot_smoothed_vs_corrected_keypoints.py`
- Optional rendered video with skeleton/angles if the script writes it (see script for current behavior).

---

## Pipeline steps (what run_video_pipeline does)

1. Copy video to OpenPose `media/` (or use existing input).
2. Run OpenPose on the video to get per-frame keypoints JSON.
3. Extract MediaPipe pose from the same video (in memory or temp).
4. Scale and average OpenPose + MediaPipe keypoints per frame.
5. Run Kalman smoothing on the averaged keypoints (all frames).
6. Compute angles (e.g. shoulder/elbow) from smoothed keypoints.
7. Save smoothed keypoints JSON to `out_dir`.
8. Optionally render video with skeleton/angles and cleanup temp files (unless `--keep_temp`).

---

## Using the smoother from your own code

```python
from kalman.keypoint_smoother import KeypointSmoother

# keypoint_names: list of joint names (e.g. from pose_handler.openpose18_mapping)
smoother = KeypointSmoother(keypoint_names=keypoint_names, dt=1.0)
for frame_keypoints in frames:
    smoothed = smoother.smooth(frame_keypoints)  # same dict structure, smoothed x,y
```

---

## Dependencies

- **OpenPose** (external binary) — path set in `config.py` or via `--openpose_root`.
- **Python 3.9** + [requirements.txt](../requirements.txt) in the `diffusion/` venv: `opencv-python`, `numpy`, `mediapipe`.
- **Project:** `pose_handler` (for angles, keypoint names, I/O).

---

## See also

- [kalman_out/README.md](../kalman_out/README.md) — where the pipeline writes by default.
- [README.md](../README.md) — full pipeline from video to metrics.
