# Per-video records

| Folder | Source run | Content |
|---|---|---|
| `submitted_runs_wo0ny5hy0vo2vt/runs/` | submitted version (pod wo0ny5hy0vo2vt) | per-video metrics, gate verdicts, reconstructions, layer sets and metadata |
| `r1_runs_9ykozyhz1vzsv8/runs/` | revision R1 (pod 9ykozyhz1vzsv8) | BF16 references; W8A8 and W4A8 with held-out-by-subject calibration; extra BF16 seeds 1-3 |
| `correction_fields/runs/qtc/driving_correction/` | submitted version | driving-pose correction fields (v1-v3) used for pre-correction; checksums in `MANIFEST.sha256` |

To reproduce the revision statistics, run this from the repository root:

```bash
python -m qtc.revision_gpu_stats --new data/r1_runs_9ykozyhz1vzsv8 --sub data/submitted_runs_wo0ny5hy0vo2vt
```

It reproduces `revision/ijies-r1/stats/revision_gpu.json` and the `gpu_*.csv` files exactly.

The figures under `*/runs/qtc/analysis/` are the analysis outputs as produced at run time. Some of them label
the reference arm "FP16", the earlier name of the BF16 arm. The cost-plausibility frontier figure there is
exploratory and is not part of the manuscript. W4A8 is simulated, so its weight memory is a computed packed
footprint, and its latency and peak memory do not measure a W4A8 kernel.
