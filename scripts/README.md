# scripts/

> Python toolchain for DynamiCtrl post-training quantization and kinematic evaluation.

## Contents

```
scripts/
  quant/                    # Main PTQ pipeline (Steps 0–5, batch runners, shared modules)
    kinematic_gate/         # Pose-metric acceptance gate (OpenPose + MediaPipe + Kalman)
    README.md
  README.md                 # (this file)
```

All runnable code lives under `scripts/quant/`. There are no other script packages at this level.

## Prerequisites

- Cloned [DynamiCtrl](https://github.com/gulucaptain/dynamictrl) repo
- CUDA GPU for Steps 1–5 (except Step 0)
- Config: [`configs/quant/vidit_ptq.json`](../configs/quant/vidit_ptq.json)
- For kinematic gate: sibling [`../phase1`](../phase1) repo + separate venv (see kinematic_gate README)

## How to use

Start with the quant pipeline README for step-by-step commands:

```bash
# From project root
python scripts/quant/validate_offline.py --repo-root /path/to/DynamiCtrl
```

Then follow [quant/README.md](quant/README.md) for Steps 1–5 and batch experiments.

## Outputs / artifacts

Scripts write to:

- `outputs/` — generated videos
- `logs/` — benchmark and gate JSON
- `calib_data/` — calibration tensors (created by Step 3)
- `kinematic_cache/` — cached pose keypoints (gate only)

## See also

- [../README.md](../README.md) — project overview
- [quant/README.md](quant/README.md) — full pipeline reference
- [quant/kinematic_gate/README.md](quant/kinematic_gate/README.md) — gate setup and runners
- [../docs/DynamiCtrl-PTQ-integration-steps.md](../docs/DynamiCtrl-PTQ-integration-steps.md) — integration guide
