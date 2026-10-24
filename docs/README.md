# docs/

> Methodology, integration guides, and experiment reports for DynamiCtrl PTQ and the kinematic gate.

## Contents

| Document | Purpose |
|----------|---------|
| [DynamiCtrl-PTQ-method.md](DynamiCtrl-PTQ-method.md) | High-level PTQ plan: ViDiT-Q approach, protected layers, kinematic gate thresholds, 2-day schedule |
| [DynamiCtrl-PTQ-integration-steps.md](DynamiCtrl-PTQ-integration-steps.md) | Step-by-step toolchain integration (Steps 1–5 scripts, folder layout, Colab → shell migration) |
| [DynamiCtrl-PTQ-kinematic-gate-results.md](DynamiCtrl-PTQ-kinematic-gate-results.md) | Complete experiment report: full grid, sensitivity sweep, hybrid — all clips and gate results |
| [DynamiCtrl-PTQ-human2-kinematic-gate-comparison.md](DynamiCtrl-PTQ-human2-kinematic-gate-comparison.md) | Cross-experiment comparison for `human2_motion{1,2,3}` clips |
| [temporal_quantization_hypothesis.md](temporal_quantization_hypothesis.md) | Hypothesis and design for SmoothQuant vs naive temporal attention failures |

## Reading order

Read in this order if you are new to the project:

1. **DynamiCtrl-PTQ-method.md** — understand *why* (ViDiT-Q, kinematic gate as acceptance test, no training)
2. **DynamiCtrl-PTQ-integration-steps.md** — understand *how to wire* DynamiCtrl into the quant pipeline
3. **DynamiCtrl-PTQ-kinematic-gate-results.md** — see *what was measured* (full tables, pass/fail per clip)
4. **DynamiCtrl-PTQ-human2-kinematic-gate-comparison.md** — focused analysis when comparing sweep vs grid on human2
5. **temporal_quantization_hypothesis.md** — debugging doc for temporal quantization artifacts

## Which doc answers which question?

| Question | Read |
|----------|------|
| What is kinematics-gated PTQ? | [DynamiCtrl-PTQ-method.md](DynamiCtrl-PTQ-method.md) |
| How do I run Steps 1–5? | [DynamiCtrl-PTQ-integration-steps.md](DynamiCtrl-PTQ-integration-steps.md) |
| Did ViDiT pass the gate on clip X? | [DynamiCtrl-PTQ-kinematic-gate-results.md](DynamiCtrl-PTQ-kinematic-gate-results.md) |
| How does human2 compare across experiments? | [DynamiCtrl-PTQ-human2-kinematic-gate-comparison.md](DynamiCtrl-PTQ-human2-kinematic-gate-comparison.md) |
| Why might naive W8A8 fail temporally? | [temporal_quantization_hypothesis.md](temporal_quantization_hypothesis.md) |

## Source artifacts

The results doc is generated from notebook runs and log JSON under `logs/` and `outputs/`. Primary source notebook: `notebooks/DynamiCtrl-PTQ-ViDiT.ipynb`.

To regenerate the human2 comparison doc:

```bash
python scripts/quant/generate_human2_gate_report.py
```

## See also

- [../README.md](../README.md) — project overview
- [../configs/README.md](../configs/README.md) — experiment configuration
- [../scripts/quant/README.md](../scripts/quant/README.md) — runnable pipeline
- [../report/README.md](../report/README.md) — formal LaTeX paper
