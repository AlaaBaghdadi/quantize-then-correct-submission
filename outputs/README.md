# outputs/

> Generated videos, PSNR scores, and per-clip metadata from quantization experiments. **Do not edit manually.**

## Contents

Output directories are created by notebooks and scripts under `scripts/quant/`. Typical layout:

```
outputs/
  fp16_grid/                    # FP16 baseline (12 clips)
  vidit_w8a8_grid/              # ViDiT-Q full-model W8A8 (12 clips)
  w8a8_grid/                    # Matched naive W8A8 (12 clips)
  hybrid_w8a8_grid/             # Hybrid naive (blocks 1-13 FP16)
  hybrid_vidit_w8a8_grid/       # Hybrid ViDiT-Q
  sweep_fp16_baseline/          # FP16 baseline for sensitivity sweep (3 clips)
  sweep_group1_vidit/             # Block group 1 only INT8 (ViDiT)
  sweep_group2_vidit/
  sweep_group3_vidit/
  sweep_group1_naive/
  sweep_group2_naive/
  sweep_group3_naive/
  temporal_quant_debug/         # Temporal quantization debug metrics
  vidit_w8a8/                   # Single-clip ViDiT output
  w8a8/                         # Single-clip naive output
```

Not all directories may exist locally; some are created only when specific experiments are run.

## Naming conventions

### Video files

| Pattern | Example |
|---------|---------|
| FP16 grid | `fp16_grid/<clip_id>/<clip_id>_seed42.mp4` |
| ViDiT grid | `vidit_w8a8_grid/<clip_id>/vidit_w8a8_<clip_id>_seed42.mp4` |
| Naive grid | `w8a8_grid/<clip_id>/w8a8_<clip_id>_seed42.mp4` |
| Hybrid | `hybrid_vidit_w8a8_<clip_id>_seed42.mp4` |
| Sweep | `sweep_group{N}_{vidit,naive}/<clip_id>/...` |

Seed is from `configs/quant/vidit_ptq.json` → `inference.seed` (default: 42).

### Per-clip sidecar files

| File | Content |
|------|---------|
| `psnr.txt` | PSNR vs FP16 baseline |
| `*_meta.json` | Generation metadata (steps, resolution, timing) |

## Experiment → directory mapping

| Experiment | Directories |
|------------|-------------|
| Full grid (Exp 1) | `fp16_grid/`, `vidit_w8a8_grid/`, `w8a8_grid/` |
| Sensitivity sweep (Exp 2) | `sweep_fp16_baseline/`, `sweep_group{1,2,3}_{vidit,naive}/` |
| Hybrid block map (Exp 3) | `hybrid_w8a8_grid/`, `hybrid_vidit_w8a8_grid/` |

Full clip-by-clip results: [docs/DynamiCtrl-PTQ-kinematic-gate-results.md](../docs/DynamiCtrl-PTQ-kinematic-gate-results.md).

## How to use

These are **outputs** of the quant pipeline. To regenerate:

```bash
python scripts/quant/run_grid_batch.py --config configs/quant/vidit_ptq.json --repo-root /path/to/DynamiCtrl
```

To evaluate existing videos with the kinematic gate:

```bash
python scripts/quant/kinematic_gate/run_grid_kinematic_gate.py --config configs/quant/vidit_ptq.json --phase1-root ../phase1
```

## See also

- [../README.md](../README.md) — project overview
- [../logs/README.md](../logs/README.md) — benchmark and gate logs
- [../kinematic_cache/README.md](../kinematic_cache/README.md) — cached pose keypoints from gate
- [../scripts/quant/README.md](../scripts/quant/README.md) — pipeline that produces these files
- [../docs/DynamiCtrl-PTQ-kinematic-gate-results.md](../docs/DynamiCtrl-PTQ-kinematic-gate-results.md) — complete results
