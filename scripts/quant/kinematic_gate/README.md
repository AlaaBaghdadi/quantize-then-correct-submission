# scripts/quant/kinematic_gate/

> Automatic pass/fail acceptance test: compare pose metrics of quantized vs FP16 generated videos.

Kinematics is used **only as an evaluation gate**, not as a training signal. Pose is extracted via OpenPose + MediaPipe + Kalman from the sibling **phase1** repo, then metric deltas are compared against thresholds in `vidit_ptq.json`.

## Contents

| File | Role |
|------|------|
| `run_kinematic_gate.py` | Single-clip gate: FP16 video vs quant video |
| `run_grid_kinematic_gate.py` | Full grid gate over all clips in an output directory |
| `generate_report_md.py` | Markdown gate report from JSON logs |
| `phase1_bridge.py` | Bridge to `../phase1` OpenPose + MediaPipe + Kalman pipeline |
| `metrics.py` | Pose metric vector computation |
| `compare.py` | Delta thresholds + pass/fail evaluation |
| `security.py` | Path/subprocess safety checks |
| `requirements.txt` | Gate-specific deps (`mediapipe==0.10.21`, `opencv-python`) |

## Prerequisites

### 1. Separate Python environment

The gate uses a **dedicated venv** at project root (`.venv_kinematic_gate`) because MediaPipe ≥0.10.30 breaks the phase1 API.

```bash
# From project root
python -m venv .venv_kinematic_gate
.venv_kinematic_gate\Scripts\activate        # Windows
# source .venv_kinematic_gate/bin/activate   # Linux/macOS

pip install -r scripts/quant/kinematic_gate/requirements.txt
```

### 2. phase1 repo

Clone or locate the sibling **phase1** repo. Default path in config: `../phase1`.

Set in [`configs/quant/vidit_ptq.json`](../../../configs/quant/vidit_ptq.json) → `kinematic_gate.phase1_root`.

### 3. OpenPose binaries

Required for pose extraction. Set `kinematic_gate.openpose_root` in config, or let `phase1_bridge.py` auto-detect.

### 4. Generated videos

FP16 baseline and quantized videos must exist under `outputs/` before running the gate.

## Gate thresholds

From `vidit_ptq.json` → `kinematic_gate`:

| Threshold | Default | Metric |
|-----------|---------|--------|
| `max_rel_delta_accel_variance` | 0.10 | +10% max Δ acceleration variance |
| `max_rel_delta_limb_std` | 0.10 | +10% max Δ mean limb length std |
| `max_rel_delta_jitter` | 0.15 | +15% max Δ jitter |
| `max_abs_delta_violation_fraction` | 0.02 | Max absolute Δ violation fraction |
| `max_mean_npe` | 0.15 | Max mean normalized pose error |
| `max_p95_npe` | 0.30 | Max 95th-percentile NPE |
| `max_under_motion_violations` | 0.0 | Max under-motion violations |

Full rationale: [docs/DynamiCtrl-PTQ-method.md](../../../docs/DynamiCtrl-PTQ-method.md).

## How to use

Activate `.venv_kinematic_gate` before running gate scripts.

### Single clip

```bash
python scripts/quant/kinematic_gate/run_kinematic_gate.py \
  --config configs/quant/vidit_ptq.json \
  --phase1-root ../phase1 \
  --fp16-video outputs/fp16_grid/human2_motion1/human2_motion1_seed42.mp4 \
  --quant-video outputs/vidit_w8a8_grid/human2_motion1/vidit_w8a8_human2_motion1_seed42.mp4 \
  --clip-id human2_motion1
```

**Exit codes:** `0` = pass, `1` = gate fail, `2` = setup/preflight error.

### Full grid

```bash
python scripts/quant/kinematic_gate/run_grid_kinematic_gate.py \
  --config configs/quant/vidit_ptq.json \
  --phase1-root ../phase1
```

Runs gate for all clips comparing `fp16_grid/` vs `vidit_w8a8_grid/` (or as configured).

### Chain after quant benchmark

```bash
python scripts/quant/apply_and_benchmark_vidit.py --config configs/quant/vidit_ptq.json --repo-root /path/to/DynamiCtrl
python scripts/quant/kinematic_gate/run_kinematic_gate.py \
  --config configs/quant/vidit_ptq.json \
  --fp16-video <fp16_mp4> --quant-video <quant_mp4> --clip-id human2_motion1
```

## Outputs / artifacts

| Path | Content |
|------|---------|
| `logs/kinematic_gate.json` | Single-run summary |
| `logs/kinematic_gate_grid/` | Per-clip gate logs (full grid) |
| `logs/kinematic_gate_w8a8_grid/` | Naive W8A8 grid gate logs |
| `logs/kinematic_gate_sweep_group*_*/` | Sensitivity sweep gate logs |
| `kinematic_cache/keypoints/<clip_id>/` | Cached smoothed keypoints (see [kinematic_cache/README.md](../../../kinematic_cache/README.md)) |

Set `kinematic_gate.skip_if_cached: true` in config to reuse cached keypoints.

Regenerate human2 comparison doc:

```bash
python scripts/quant/generate_human2_gate_report.py
```

## See also

- [../README.md](../README.md) — quant pipeline (run before gate)
- [../../../kinematic_cache/README.md](../../../kinematic_cache/README.md) — cache layout
- [../../../logs/README.md](../../../logs/README.md) — gate log formats
- [../../../configs/quant/README.md](../../../configs/quant/README.md) — threshold config
- [../../../docs/DynamiCtrl-PTQ-kinematic-gate-results.md](../../../docs/DynamiCtrl-PTQ-kinematic-gate-results.md) — full results
