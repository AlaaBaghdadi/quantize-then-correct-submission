# Full Video Pose Correction Pipeline

This document describes the end-to-end pipeline from input video to corrected pose-transfer videos and metrics.

---

## Overview

The pipeline has **two main stages**:

1. **Kalman pipeline** (`kalman/run_video_pipeline.py`) — produces temporally smoothed keypoints from video (OpenPose + MediaPipe + Kalman). Output: `*_smoothed_keypoints.json`.

2. **Video correction pipeline** (`run_video_correction_pipeline.py`) — loads that JSON, applies **angle-based biomechanical pose correction**, then for each frame that needed correction runs **PISE pose transfer** twice (corrected vs original keypoints), builds comparison videos, and computes metrics (PSNR, MSE, FID).

---

## Stage 1: Kalman pipeline (upstream)

**Script:** `kalman/run_video_pipeline.py`  
**Input:** Video file  
**Output:** `{out_dir}/{video_stem}_smoothed_keypoints.json`

| Step | What it does |
|------|----------------|
| 1 | Copy video to OpenPose `media_videos/` |
| 2 | Run **OpenPose** on the full video → per-frame JSON keypoints |
| 3 | Run **MediaPipe** on the same video → per-frame landmarks |
| 4 | Scale MediaPipe Z (depth) for consistency |
| 5–6 | For each frame: load OpenPose JSON, convert to named keypoints, **average** with MediaPipe keypoints (same frame) |
| 7 | **Kalman smooth**: run `KeypointSmoother` on the averaged (x, y) over time; keep z and confidence from averaged |
| 8 | Save `*_smoothed_keypoints.json` (frames with `keypoints`, plus fps, image_size); optionally compute/save angles and render overlay video |

So the input to the next stage is **Kalman-smoothed, OpenPose+MediaPipe-averaged** keypoints — no pose correction yet.

---

## Stage 2: Video correction pipeline

**Script:** `run_video_correction_pipeline.py`  
**Input:** Video + `--keypoints_json` (e.g. `kalman_out/video1_smoothed_keypoints.json`)  
**Output:** Corrected keypoints JSON, comparison frames (corrected vs original), two videos, and metrics.

### 2.1 Load keypoints and run pose correction

- Load the Kalman-smoothed keypoints JSON (`frames[].keypoints`).
- If `frame_corrected` is not already present for each frame, run **pose correction** and mark which frames were corrected.

### 2.2 What pose correction does (angle-based)

**Yes — the code corrects pose according to angles.** It does **not** re-detect pose; it takes the existing keypoints and:

1. **Deconstruct:** From keypoints (shoulder, elbow, wrist) and a body frame (from LS, RS, LH), compute **shoulder angles** for left and right:
   - **Pitch**: angle of upper arm from “down” (−Zb), range 0°–180°.
   - **Yaw**: angle in the horizontal plane (0° = forward).
   - **Roll**: rotation around upper arm (passed through, not constrained).

2. **Soft conical constraints:** For each shoulder, a “feasible cone” is defined (axis ~30° forward, 45° down; max deviation ~55°). If the upper-arm direction is outside this cone:
   - In a buffer zone: smoothstep resistance toward the cone.
   - Beyond the limit: clamp direction to the cone edge.  
   So impossible shoulder orientations are pulled back into a biomechanically plausible range.

3. **Temporal smoothing:** EMA (Exponential Moving Average) on the clamped angles; yaw uses wrap-aware interpolation so 170° and −170° smooth correctly.

4. **Velocity limiting (optional):** Cap per-frame change in pitch/yaw (e.g. 25°) so large corrections don’t “teleport” in one frame.

5. **Reconstruct:** From the corrected pitch/yaw (and unchanged roll), recompute **elbow and wrist** positions using the kinematic chain:
   - New upper-arm direction from corrected angles → new elbow = shoulder + length × unit vector.
   - Forearm is rotated by the same rotation that aligns old upper arm to new (Rodrigues); new wrist = new elbow + rotated forearm.

Only **shoulder pitch/yaw** are constrained and used to reconstruct **elbow and wrist**; other keypoints are left unchanged.  
Implementation: `pose-corrector/pose_corrector.py` (`PoseCorrector.correct_frame`).

### 2.3 Mark frames and save corrected keypoints

- For each frame, `frame_corrected` is set to `True` if either shoulder was corrected (left or right).
- Original smoothed keypoints are stored as `keypoints_smoothed`; corrected keypoints as `keypoints`.
- Corrected keypoints are written to e.g. `{out_dir}/{video_stem}_smoothed_keypoints_corrected.json`.

### 2.4 For each frame where `frame_corrected` is True

For every such frame index the pipeline:

1. **Resize to 256×256:**  
   - Extract frame from video; resize frame to 256×256.  
   - Resize both **corrected** and **original (smoothed)** keypoints from original image size to 256×256 (for PISE).

2. **PISE pose-transfer pipeline** (in-process via `run.PoseTransferPipeline`), run **twice** per frame. **Source keypoints** come from the **passed smoothed keypoints** (original smoothed), written in OpenPose format — **no OpenPose** is run on the frame.
   - **Source:** `src_openpose_json` = **original smoothed** keypoints (from JSON, resized to 256×256).
   - **Target:** `tgt_openpose_json` = either **corrected** or **original smoothed** keypoints (from JSON).
   - **Run 1 (corrected):** src = original smoothed, target = **corrected** → generates the “corrected” comparison frame.
   - **Run 2 (original, for fair comparison):** src = original smoothed, target = **original smoothed** (same pose on both sides) → generates the “original” comparison frame. Using the same pose for src and target applies the same pose-transfer effect so that when comparing the two frames, the only difference is the keypoint set (corrected vs original).
   - **Step 2:** Run PISE `make_pise_fashion_sample.py` with the above src/tgt OpenPose JSONs.
   - **Step 3:** Run **PGN parser** on src and tgt images → parsing maps (e.g. 8-class segmentation).
   - **Step 4:** Copy images and parsing maps into PISE dataroot; run PISE **test** (generate pose-transferred image).

3. **Outputs per frame:**  
   - `comparison_frames/frame_{idx:05d}_corrected.jpg` — PISE result with **corrected** target keypoints.  
   - `comparison_frames/frame_{idx:05d}_original.jpg` — PISE result with **original (smoothed)** target keypoints.  
   - Optional verification overlays in `verify_keypoints_256/`.

### 2.5 Build videos and metrics

- **Videos:** Two videos are built from the original video, with corrected-frame indices replaced by the corresponding comparison frames:
  - `{video_stem}_corrected_keypoints.mp4` — corrected keypoints.
  - `{video_stem}_original_keypoints.mp4` — original keypoints.
- **Metrics:** Per-frame PSNR/MSE between corrected and original generated images; optional FID between the two sets of generated frames; saved to e.g. `{video_stem}_metrics.json`.

---

## Summary table

| Stage | Script / module | Main action |
|-------|------------------|-------------|
| 1 | `kalman/run_video_pipeline.py` | OpenPose + MediaPipe → average → Kalman smooth → `*_smoothed_keypoints.json` |
| 2a | `run_video_correction_pipeline.py` + `pose_corrector` | Load smoothed JSON → **angle-based correction** (cone constraints + EMA + velocity limit + kinematic reconstruct) → `*_smoothed_keypoints_corrected.json` |
| 2b | Same + `run.PoseTransferPipeline` | For each corrected frame: 256×256; **src keypoints = original smoothed from JSON** (no OpenPose); target = corrected then original smoothed; parser; PISE generate → comparison frames |
| 2c | Same | Build two videos from comparison frames; compute PSNR/MSE/FID |

So: **pose correction is angle-based** (shoulder pitch/yaw constrained and elbow/wrist reconstructed). **Source and target keypoints in PISE both come from your JSON** (original smoothed for src; corrected or original smoothed for target). No OpenPose is run in the video correction pipeline. Run 2 uses src = target = original smoothed for a fair comparison (same pose-transfer effect).

