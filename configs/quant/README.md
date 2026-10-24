# configs/quant/

> Reference for [`vidit_ptq.json`](vidit_ptq.json) — the canonical experiment configuration.

## Contents

| File | Role |
|------|------|
| `vidit_ptq.json` | All paths, inference params, grid definition, hybrid map, gate thresholds |
| `quantize_layers.txt` | *(generated)* List of quantizable Linear layers from `list_layers.py` |

## Config sections

### `paths`

Runtime paths for model, calibration, outputs, and logs.

| Key | Purpose |
|-----|---------|
| `drive_root` | Colab Google Drive root (notebooks only) |
| `repo` | DynamiCtrl repo path on Colab |
| `model_path` | Checkpoint directory (`./checkpoints/DynamiCtrl` or HF id) |
| `calib_file` | Default ViDiT calibration `.pt` file |
| `quantize_layers` | Path to layer inventory text file |
| `fp16_reference_dir` | FP16 baseline videos for PSNR comparison |
| `vidit_output_dir` | Single-clip ViDiT output directory |
| `benchmark_log_json` | Benchmark results JSON path |

**Local vs Colab:** On local runs, override Colab paths via CLI flags (`--repo-root`, `--model-path`, etc.) rather than editing `drive_root`.

### `setup`

| Key | Purpose |
|-----|---------|
| `resume_downloads` | Resume partial HF/checkpoint downloads |

### `clip`

Default single-clip inputs for one-off runs (not the full grid).

| Key | Default | Purpose |
|-----|---------|---------|
| `reference_image` | `./assets/human2.jpg` | Identity/reference frame |
| `driving_video` | `./assets/motion1.mp4` | Pose-driving motion video |

### `inference`

Generation parameters shared across scripts and notebooks.

| Key | Default | Notes |
|-----|---------|-------|
| `prompt` | test prompt | Text conditioning |
| `num_inference_steps` | 25 | Diffusion steps |
| `width` / `height` | 768 / 1360 | Output resolution |
| `num_frames` | 37 | Video length |
| `pose_control_function` | `padaln` | DynamiCtrl pose control mode |
| `guidance_scale` | 3.0 | CFG scale |
| `seed` | 42 | Reproducibility (used in output filenames) |
| `fps` | 16 | Export frame rate |
| `dtype` | `bfloat16` | Model compute dtype |

### `vidit_quantization`

ViDiT-Q W8A8 hyperparameters.

| Key | Default | Purpose |
|-----|---------|---------|
| `smoothquant_alpha` | 0.5 | SmoothQuant migration strength |
| `early_sensitivity_ratio` | 1.25 | Timestep-sensitive layer selection ratio |
| `max_early_fp16_layers` | 24 | Cap on layers kept FP16 across all blocks |
| `use_compile` | true | `torch.compile` on transformer |

Lower `early_sensitivity_ratio` keeps **more** layers in FP16 (more accurate, less compression).

### `benchmark`

| Key | Default | Purpose |
|-----|---------|---------|
| `warmup_repeats` | 10 | Warmup iterations before timing |

### `quality_gate`

PSNR acceptance thresholds (visual fidelity vs FP16).

| Key | Default | Purpose |
|-----|---------|---------|
| `min_psnr_db` | 25.0 | Hard minimum PSNR |
| `target_psnr_db` | 30.0 | Target PSNR for reporting |

### `grid`

Full 4×3 experiment grid (4 reference images × 3 driving motions = 12 clips).

| Key | Purpose |
|-----|---------|
| `images` | List of 4 reference images (`human1.jpg` … `human4.jpg`) |
| `motions` | List of 3 driving videos (`motion1.mp4` … `motion3.mp4`) |
| `fp16_output_dir` | FP16 baseline grid output (`./outputs/fp16_grid`) |
| `vidit_output_dir` | ViDiT-Q full-model grid (`./outputs/vidit_w8a8_grid`) |
| `w8a8_output_dir` | Matched naive W8A8 grid (`./outputs/w8a8_grid`) |
| `calib_dir` | Per-clip calibration data (`./calib_data`) |

Clip IDs follow the pattern `{humanN}_motion{M}` (e.g. `human2_motion1`).

### `hybrid_map`

Mixed-precision block experiment: blocks 1–13 FP16, blocks 14–40 INT8.

| Key | Purpose |
|-----|---------|
| `block_fp16` | `[1, 13]` — inclusive block range kept in FP16 |
| `block_int8` | `[14, 40]` — blocks quantized to INT8 |
| `sweep_image` / `sweep_motions` | Inputs for hybrid grid (defaults to human2 + 3 motions) |
| `naive_output_dir` | `./outputs/hybrid_w8a8_grid` |
| `vidit_output_dir` | `./outputs/hybrid_vidit_w8a8_grid` |
| `gate_report_dir` / `gate_summary_json` | Kinematic gate outputs for hybrid run |

Run via `scripts/quant/run_hybrid_grid.py`.

### `kinematic_gate`

Automatic pose-metric acceptance test (compares quantized vs FP16 output).

| Key | Default | Purpose |
|-----|---------|---------|
| `phase1_root` | `../phase1` | Sibling phase1 repo with Kalman pipeline |
| `openpose_root` | `null` | OpenPose binary root (auto-detected if null) |
| `cache_dir` | `./kinematic_cache` | Cached keypoints |
| `report_json` | `./logs/kinematic_gate.json` | Single-run gate summary |
| `max_rel_delta_accel_variance` | 0.10 | Max +10% Δ acceleration variance |
| `max_rel_delta_limb_std` | 0.10 | Max +10% Δ limb length std |
| `max_rel_delta_jitter` | 0.15 | Max +15% Δ jitter (`var_accel_norm`) |
| `max_abs_delta_violation_fraction` | 0.02 | Max absolute Δ violation fraction |
| `max_mean_npe` | 0.15 | Max mean normalized pose error |
| `max_p95_npe` | 0.30 | Max 95th-percentile NPE |
| `max_under_motion_violations` | 0.0 | Max under-motion biomechanical violations |
| `skip_if_cached` | true | Reuse cached keypoints when available |

Loaded by `scripts/quant/vidit_ptq_config.py` → `load_kinematic_gate_config()`.

## Example: minimal local overrides

When running locally, you typically only need to ensure assets and checkpoint paths exist. The seed, grid, and gate thresholds rarely need changing between runs.

```bash
python scripts/quant/run_grid_batch.py \
  --config configs/quant/vidit_ptq.json \
  --repo-root /path/to/DynamiCtrl
```

## See also

- [../README.md](../README.md) — configs overview
- [../../scripts/quant/vidit_ptq_config.py](../../scripts/quant/vidit_ptq_config.py) — loader implementation
- [../../scripts/quant/kinematic_gate/README.md](../../scripts/quant/kinematic_gate/README.md) — gate runners
- [../../docs/DynamiCtrl-PTQ-method.md](../../docs/DynamiCtrl-PTQ-method.md) — gate threshold rationale
