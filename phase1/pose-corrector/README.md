# pose-corrector — Biomechanical pose correction

This folder contains **constrained kinematic correction** for human pose keypoints: it corrects shoulder (and related) angles to stay within biomechanical limits, applies temporal smoothing (EMA), and reconstructs the 3D kinematic chain (e.g. elbow/wrist from corrected shoulder).

The main module is **`pose_corrector.py`**. It is used by `run_video_correction_pipeline.py` and can be run standalone on any keypoints JSON (e.g. Kalman-smoothed output).

---

## Setup after cloning from GitHub

Uses the **main pipeline environment** only (`diffusion/`, **Python 3.9**):

```powershell
.\scripts\setup_envs.ps1
.\diffusion\Scripts\Activate.ps1
```

Install deps from [requirements.txt](../requirements.txt) (Python 3.9). No separate venv or large downloads for this module.

---

## What pose_corrector.py does

- **Input:** Per-frame keypoints (e.g. from Kalman or MediaPipe), in named format (`left_shoulder`, `right_elbow`, etc.).
- **Steps:**
  1. **Deconstruct:** Compute upper-arm vectors and pitch/yaw from landmarks.
  2. **Soft conical constraints:** Apply feasible cone per shoulder (axis ~30° forward, 45° down; max deviation ~55°). In a buffer zone, smoothstep resistance; beyond limit, clamp to cone edge.
  3. **Smooth:** EMA on resulting angles (yaw uses wrap-aware interpolation).
  4. **Optional:** Velocity limit (max angular change per frame).
  5. **Reconstruct:** Rotate forearm by the same rotation that aligns old upper-arm to new one (Rodrigues); recompute elbow and wrist.
- **Output:** Corrected keypoints per frame (same structure as input), plus optional stats.

Coordinate system and biomechanical limits are described in the docstring at the top of `pose_corrector.py` (pitch 0°–180°, left/right yaw ranges, etc.).

---

## How to run (CLI)

From the **project root** (parent of `pose-corrector/`):

```bash
python pose-corrector/pose_corrector.py --in_json path/to/smoothed_keypoints.json --out_dir path/to/output
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--in_json` | Yes | — | Input keypoints JSON (e.g. Kalman-smoothed). Must have `frames` with per-frame `keypoints` (named dict). |
| `--out_dir` | No | Same as input dir | Output directory for corrected JSON. |
| `--out_path` | No | None | Explicit output file path (overrides `--out_dir` naming). |
| `--alpha` | No | 0.7 | EMA smoothing factor [0, 1]. |
| `--no_smooth` | No | False | Disable temporal smoothing (only apply clamping). |
| `--stats` | No | False | Print processing statistics. |

### Example

```bash
python pose-corrector/pose_corrector.py --in_json kalman_out/my_video_smoothed_keypoints.json --out_dir kalman_out --stats
```

Output file name is derived from input (e.g. `*_corrected.json`) unless `--out_path` is set.

---

## Using from Python

The folder name is `pose-corrector` (hyphen), so it is loaded as a file, not a package. The pipeline does:

```python
import importlib.util
from pathlib import Path

pose_corrector_path = Path("pose-corrector/pose_corrector.py")
spec = importlib.util.spec_from_file_location("pose_corrector_mod", pose_corrector_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
PoseCorrector = mod.PoseCorrector

corrector = PoseCorrector(alpha=0.7)
for frame in frames:
    kps = frame["keypoints"]
    result = corrector.correct_frame(kps)
    # result.keypoints   — corrected keypoints dict
    # result.corrections_applied  — e.g. {"left": True, "right": False}
```

---

## Dependencies

- **Python 3.9** + [requirements.txt](../requirements.txt) in the `diffusion/` venv: `numpy`
- **Project:** `pose_handler.calc_angles` (body frame, keypoints→arrays), `pose_handler.common_io` (read_json, write_json, ensure_dir)

---

## See also

- [README.md](../README.md) — full pipeline; correction is also run inside `run_video_correction_pipeline.py`.
- [SCRIPTS.md](../SCRIPTS.md) — script reference.
