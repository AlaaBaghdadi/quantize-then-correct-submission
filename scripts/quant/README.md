# scripts/quant/

> Standalone DynamiCtrl PTQ pipeline: load DiT, calibrate, apply W8A8 / ViDiT-Q, benchmark, and run grid experiments.

## Contents

### Pipeline scripts (Steps 0–5)

| Step | Script | Deliverable |
|------|--------|-------------|
| 0 | `validate_offline.py` | Static checks (imports, config, file layout) — no GPU |
| 1 | `load_dit.py` | Confirms DiT loads (~5.77B params) |
| 2 | `list_layers.py` | `configs/quant/quantize_layers.txt` |
| 3 | `collect_calib.py` | Naive calibration `calib_data/*.pt` |
| 3 | `collect_calib_vidit.py` | ViDiT per-timestep calibration |
| 4 | `apply_w8a8.py` | Matched naive W8A8 + video generation |
| 4 | `apply_vidit_w8a8.py` | ViDiT-Q SmoothQuant W8A8 + video generation |
| 4+5 | `apply_and_benchmark_vidit.py` | Combined ViDiT apply + benchmark |
| 5 | `benchmark.py` | FP16 vs naive W8A8 timing |
| 5 | `benchmark_vidit.py` | FP16 vs ViDiT-Q timing |

### Batch experiment runners

| Script | Purpose |
|--------|---------|
| `run_grid_batch.py` | Full 4×3 grid: FP16 baseline, calibration, ViDiT, naive W8A8 |
| `run_hybrid_grid.py` | Hybrid block map (blocks 1–13 FP16, 14–40 INT8) |

### Shared modules

| Module | Role |
|--------|------|
| `quant_common.py` | Shared pipeline loading, inference, ONNX runtime compat |
| `vidit_quant.py` | ViDiT-Q SmoothQuant + W8A8 implementation |
| `vidit_eval.py` | PSNR gate + combined apply/benchmark logic |
| `vidit_ptq_config.py` | JSON config loader |
| `colab_paths.py` | Colab/Drive path helpers for notebooks |

### Reporting

| Script | Purpose |
|--------|---------|
| `generate_human2_gate_report.py` | Regenerates `docs/DynamiCtrl-PTQ-human2-kinematic-gate-comparison.md` from logs |

### Kinematic gate

See [kinematic_gate/README.md](kinematic_gate/README.md).

## Prerequisites

```bash
# DynamiCtrl environment (PyTorch, diffusers, torchao, etc.)
# Install from the cloned DynamiCtrl repo's requirements.txt

export PYTHONPATH="/path/to/DynamiCtrl:$PYTHONPATH"
# Or pass --repo-root /path/to/DynamiCtrl on every script
```

Model checkpoint at `./checkpoints/DynamiCtrl` or Hugging Face `gulucaptain/DynamiCtrl`.

Reference assets in [`assets/`](../../assets/README.md).

## How to use

All commands assume **project root** as working directory.

### Step 0 — Validate (no GPU)

```bash
python scripts/quant/validate_offline.py --repo-root /path/to/DynamiCtrl
```

### Step 1 — Load DiT

```bash
python scripts/quant/load_dit.py --repo-root /path/to/DynamiCtrl --model-path ./checkpoints/DynamiCtrl
```

### Step 2 — Layer inventory

```bash
python scripts/quant/list_layers.py --repo-root /path/to/DynamiCtrl
# Writes configs/quant/quantize_layers.txt
```

### Step 3 — Calibration

```bash
# ViDiT (recommended)
python scripts/quant/collect_calib_vidit.py \
  --config configs/quant/vidit_ptq.json \
  --repo-root /path/to/DynamiCtrl

# Naive baseline
python scripts/quant/collect_calib.py \
  --repo-root /path/to/DynamiCtrl \
  --reference-image ./assets/human2.jpg \
  --driving-video ./assets/motion1.mp4
```

### Step 4 — Apply quantization

```bash
# ViDiT-Q W8A8
python scripts/quant/apply_vidit_w8a8.py \
  --config configs/quant/vidit_ptq.json \
  --repo-root /path/to/DynamiCtrl

# Matched naive W8A8
python scripts/quant/apply_w8a8.py \
  --repo-root /path/to/DynamiCtrl \
  --reference-image ./assets/human2.jpg \
  --driving-video ./assets/motion1.mp4 \
  --output-path ./outputs/w8a8/human2_motion1_seed42.mp4
```

### Step 5 — Benchmark

```bash
python scripts/quant/benchmark_vidit.py \
  --config configs/quant/vidit_ptq.json \
  --repo-root /path/to/DynamiCtrl

python scripts/quant/benchmark.py \
  --repo-root /path/to/DynamiCtrl \
  --log-json ./logs/step5_benchmark.json
```

### Full grid batch (12 clips)

```bash
python scripts/quant/run_grid_batch.py \
  --config configs/quant/vidit_ptq.json \
  --repo-root /path/to/DynamiCtrl
```

Produces FP16 baselines, ViDiT grid, and naive W8A8 grid under `outputs/`.

### Hybrid block-map experiment

```bash
python scripts/quant/run_hybrid_grid.py \
  --config configs/quant/vidit_ptq.json \
  --repo-root /path/to/DynamiCtrl
```

## Two quantization algorithms

| Algorithm | Script | Output dir | Notes |
|-----------|--------|------------|-------|
| **ViDiT-Q W8A8** | `apply_vidit_w8a8.py` | `outputs/vidit_w8a8_grid/` | SmoothQuant + per-token dynamic activations; ~24 timestep-sensitive layers stay FP16 |
| **Matched naive W8A8** | `apply_w8a8.py` | `outputs/w8a8_grid/` | Same FP16 skip list, no SmoothQuant — fair baseline |

Video filename pattern: `{algo}_{clip_id}_seed42.mp4` (e.g. `vidit_w8a8_human2_motion1_seed42.mp4`).

## Outputs / artifacts

| Path | Created by |
|------|------------|
| `configs/quant/quantize_layers.txt` | Step 2 |
| `calib_data/*.pt` | Step 3 |
| `outputs/*/` | Steps 4, batch runners |
| `logs/*.json` | Step 5, benchmarks |
| `logs/step1_load_dit.txt` | Step 1 smoke log |

Per-clip outputs typically include: `*_seed42.mp4`, `psnr.txt`, `*_meta.json`.

## See also

- [../README.md](../README.md) — scripts tree overview
- [kinematic_gate/README.md](kinematic_gate/README.md) — pose-metric gate after generation
- [../../configs/quant/README.md](../../configs/quant/README.md) — config reference
- [../../docs/DynamiCtrl-PTQ-integration-steps.md](../../docs/DynamiCtrl-PTQ-integration-steps.md) — detailed integration guide
- [../../notebooks/DynamiCtrl-PTQ-ViDiT.ipynb](../../notebooks/DynamiCtrl-PTQ-ViDiT.ipynb) — Colab equivalent of grid + sweep
