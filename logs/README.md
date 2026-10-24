# logs/

> Benchmark timing JSON and kinematic gate pass/fail logs from quantization experiments. **Do not edit manually.**

## Contents

```
logs/
  step1_load_dit.txt              # Step 1 smoke test log
  step1_load_dit_vidit.txt
  step5_benchmark.json            # FP16 vs naive W8A8 benchmark
  vidit_benchmark.json            # FP16 vs ViDiT-Q benchmark
  run_meta.json                   # Batch run metadata
  kinematic_gate.json             # Single-clip gate summary
  kinematic_gate_grid.json        # Full grid gate summary (ViDiT)
  kinematic_gate_grid/            # Per-clip gate logs (ViDiT full grid)
  kinematic_gate_w8a8_grid/       # Per-clip gate logs (naive full grid)
  kinematic_gate_w8a8_grid.json
  kinematic_gate_sweep_group1_vidit/
  kinematic_gate_sweep_group2_vidit/
  kinematic_gate_sweep_group3_vidit/
  kinematic_gate_sweep_group1_naive/
  kinematic_gate_sweep_group2_naive/
  kinematic_gate_sweep_group3_naive/
  kinematic_gate_sweep_group{1,2,3}_{vidit,naive}.json
  kinematic_gate_hybrid_grid/
  kinematic_gate_hybrid_vidit_grid/
  kinematic_gate_hybrid_grid.json
  kinematic_gate_hybrid_vidit_grid.json
  hyprid_run/                     # Hybrid run logs (legacy folder name)
```

## Log types

| Type | Files | Produced by |
|------|-------|-------------|
| **Benchmark** | `step5_benchmark.json`, `vidit_benchmark.json` | `benchmark.py`, `benchmark_vidit.py` |
| **Gate summary** | `kinematic_gate*.json` | `run_kinematic_gate.py`, `run_grid_kinematic_gate.py` |
| **Gate per-clip** | `kinematic_gate_*/` subdirs | Grid gate runners |
| **Smoke test** | `step1_load_dit*.txt` | `load_dit.py` |

## Gate log structure

Gate JSON files contain per-clip pass/fail, metric deltas, and threshold comparisons. Used to generate:

- [docs/DynamiCtrl-PTQ-kinematic-gate-results.md](../docs/DynamiCtrl-PTQ-kinematic-gate-results.md)
- [docs/DynamiCtrl-PTQ-human2-kinematic-gate-comparison.md](../docs/DynamiCtrl-PTQ-human2-kinematic-gate-comparison.md)

Regenerate human2 comparison:

```bash
python scripts/quant/generate_human2_gate_report.py
```

## How to use

Logs are written automatically by scripts. To produce gate logs from existing videos:

```bash
python scripts/quant/kinematic_gate/run_grid_kinematic_gate.py \
  --config configs/quant/vidit_ptq.json \
  --phase1-root ../phase1
```

Config paths for gate output: `configs/quant/vidit_ptq.json` → `kinematic_gate.report_json` and experiment-specific keys under `hybrid_map`.

## See also

- [../README.md](../README.md) — project overview
- [../outputs/README.md](../outputs/README.md) — video inputs for gate evaluation
- [../scripts/quant/kinematic_gate/README.md](../scripts/quant/kinematic_gate/README.md) — gate runners
- [../kinematic_cache/README.md](../kinematic_cache/README.md) — cached keypoints
- [../docs/README.md](../docs/README.md) — reports derived from these logs
