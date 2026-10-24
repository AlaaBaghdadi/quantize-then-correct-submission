# assets/

> Reference images and driving motion videos for grid experiments and single-clip calibration.

## Expected files

Place these files in this directory before running grid or calibration scripts:

### Reference images (4 identities)

| File | Used in grid as |
|------|-----------------|
| `human1.jpg` | `human1_motion{1,2,3}` |
| `human2.jpg` | `human2_motion{1,2,3}` |
| `human3.jpg` | `human3_motion{1,2,3}` |
| `human4.jpg` | `human4_motion{1,2,3}` |

### Driving motion videos (3 motions)

| File | Paired with each human as |
|------|---------------------------|
| `motion1.mp4` | `{humanN}_motion1` |
| `motion2.mp4` | `{humanN}_motion2` |
| `motion3.mp4` | `{humanN}_motion3` |

## Grid layout

The full experiment grid is **4 humans × 3 motions = 12 clips**, configured in [`configs/quant/vidit_ptq.json`](../configs/quant/vidit_ptq.json) → `grid.images` and `grid.motions`.

Clip ID naming convention: `{humanN}_motion{M}` (e.g. `human2_motion1`).

Default single-clip config uses `human2.jpg` + `motion1.mp4` (`clip` section in config).

## How to use

1. Add your reference images and driving videos to this folder using the names above (or update paths in `vidit_ptq.json`).
2. Run calibration (Step 3) — one `.pt` file per clip under `calib_data/`.
3. Run grid batch or notebooks — outputs land in `outputs/fp16_grid/`, `outputs/vidit_w8a8_grid/`, etc.

```bash
python scripts/quant/run_grid_batch.py \
  --config configs/quant/vidit_ptq.json \
  --repo-root /path/to/DynamiCtrl
```

## Outputs / artifacts

This folder is **input only**. Generated videos are written to [`outputs/`](../outputs/README.md), not here.

## See also

- [../README.md](../README.md) — project overview
- [../configs/quant/README.md](../configs/quant/README.md) — `grid` and `clip` config sections
- [../scripts/quant/README.md](../scripts/quant/README.md) — calibration and grid runners
- [../notebooks/README.md](../notebooks/README.md) — Colab workflows using these assets
