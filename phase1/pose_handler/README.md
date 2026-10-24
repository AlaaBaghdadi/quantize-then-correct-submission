# pose_handler — Pose I/O, formats, angles, visualization

This package provides **keypoint I/O** (unified + legacy formats), **angle computation** (shoulder, elbow), **OpenPose/MediaPipe conversion**, and **visualization** (2D overlay, 3D keypoints). It is used by the Kalman pipeline, pose-corrector, and run_video_correction_pipeline.

---

## Setup after cloning from GitHub

Uses the **main pipeline environment** (`diffusion/`, **Python 3.9**):

```powershell
.\scripts\setup_envs.ps1
.\diffusion\Scripts\Activate.ps1
```

Install deps from [requirements.txt](../requirements.txt) (Python 3.9). Output goes to `pose_handler_out/` by default (recreated on run; not in Git).

---

## What’s in this folder

| Category | Files | Purpose |
|----------|--------|---------|
| **I/O & schema** | `common_io.py`, `keypoints_schema.py`, `unified_keypoints_io.py` | read_json, write_json, ensure_dir; unified format and validation. |
| **OpenPose** | `openpose18_mapping.py`, `openpose_json_to_named.py`, `openpose_keypoints_io.py` | COCO-18 names/connections; OpenPose JSON ↔ named keypoints; draw skeleton. |
| **MediaPipe** | `extract_mediapipe_from_folder.py`, `scale_mediapipe_z.py`, `average_mediapipe_openpose.py` | Extract pose from images; scale Z; average with OpenPose. |
| **Angles** | `calc_angles.py`, `calc_angles_from_unified.py` | Shoulder/elbow angles; used by pose_corrector. |
| **Conversion** | `convert_mediapipe_to_openpose18.py`, `export_pise18_csv.py` | MediaPipe → OpenPose COCO-18; export PISE CSV. |
| **Visualization** | `visualize_2d_overlay.py`, `visualize_3d_keypoints.py`, `visualize_openpose_named_on_image.py`, `visualize_skeleton_angles_3d.py` | 2D/3D overlays and angle visualization. |
| **Pipelines** | `run_full_pipeline.py`, `run_full_pipeline_compare.py` | Full image pipelines (extract → scale → convert → average → viz). |

---

## Unified JSON format

All keypoint JSONs can use the **unified format** (`pose_handler/unified_keypoints_v1`): pixel (x, y), confidence, optional z_scaled, and consistent names (`nose`, `left_shoulder`, `right_elbow`, etc.). Scripts here read both legacy and unified; see **run.md** for examples.

---

## How to run scripts

Run from the **project root** as a module:

```bash
python -m pose_handler.<script_name> [args]
```

Example (extract MediaPipe from a folder):

```bash
python -m pose_handler.extract_mediapipe_from_folder --input_dir path/to/images --out_dir pose_handler_out
```

**Full list of scripts and exact commands** → **[run.md](run.md)** in this folder. It documents:

1. Extract MediaPipe from folder  
2. Scale Z (mediapipe_pose → mediapipe_pose_scaled)  
3. Convert MediaPipe → OpenPose COCO-18  
4. Export PISE CSV  
5. Visualize 2D overlay  
6. Visualize 3D keypoints  
7. Compute shoulder/elbow angles  
8. OpenPose JSON → named keypoints  
9. Average MediaPipe and OpenPose  
10. Visualize openpose_named on image  

Default output directory for these scripts is **`pose_handler_out/`** (overridable with `--out_dir`).

---

## Use by other parts of the project

- **kalman** — uses `pose_handler.calc_angles`, `openpose18_mapping`, `common_io`, `video_pose_io` (which uses keypoints_schema, etc.).
- **pose-corrector** — uses `pose_handler.calc_angles` (body frame, keypoints→arrays) and `common_io`.
- **run_video_correction_pipeline** — uses `common_io`, `openpose_keypoints_io` (verify_openpose_frames, verify_openpose_overlay, write_openpose_json).

---

## Dependencies

- **Python 3.9** + [requirements.txt](../requirements.txt) in the `diffusion/` venv: `numpy`, `opencv-python`, `mediapipe` (for extraction); `matplotlib` for 3D viz.
- No external binaries; OpenPose is used by **kalman** and **run.py**, not by pose_handler scripts directly.

---

## See also

- [run.md](run.md) — detailed commands for each script.
- [pose_handler_out/README.md](../pose_handler_out/README.md) — default output folder.
- [README.md](../README.md) — project overview and full pipeline.
