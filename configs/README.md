# configs/

> JSON configuration for quantization experiments, grid runs, hybrid maps, and kinematic gate thresholds.

## Contents

```
configs/
  quant/
    vidit_ptq.json    # Central experiment config (paths, grid, hybrid map, gate thresholds)
    README.md         # Field-by-field reference for vidit_ptq.json
```

**Generated at runtime (not committed):**

- `quantize_layers.txt` — layer inventory from `scripts/quant/list_layers.py`
- `protected_layers.txt` — layers kept in FP16 (if generated separately)

## How to use

Most scripts accept `--config configs/quant/vidit_ptq.json`. Paths inside the JSON are relative to the **project root** (phase2), except Colab-specific keys like `drive_root` which notebooks override.

Before a local run:

1. Set `paths.model_path` to your DynamiCtrl checkpoint
2. Set `paths.repo` to your cloned DynamiCtrl repo (Colab) or rely on `--repo-root` CLI flag (local)
3. Ensure `grid.images` and `grid.motions` point to files in [`assets/`](../assets/README.md)
4. Set `kinematic_gate.phase1_root` to your phase1 checkout (default: `../phase1`)

Full field reference: [quant/README.md](quant/README.md).

## See also

- [../README.md](../README.md) — project overview
- [quant/README.md](quant/README.md) — `vidit_ptq.json` reference
- [../scripts/quant/vidit_ptq_config.py](../scripts/quant/vidit_ptq_config.py) — config loader
- [../docs/DynamiCtrl-PTQ-integration-steps.md](../docs/DynamiCtrl-PTQ-integration-steps.md) — suggested folder layout
