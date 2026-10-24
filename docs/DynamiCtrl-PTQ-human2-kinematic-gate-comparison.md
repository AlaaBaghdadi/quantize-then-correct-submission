# DynamiCtrl PTQ — Human2 Kinematic Gate Comparison

Complete kinematic gate results for **`human2_motion1`**, **`human2_motion2`**, and **`human2_motion3`** — the only clips shared across every quantization experiment (full grid, group sweep, and hybrid).

**Related docs:** `docs/DynamiCtrl-PTQ-kinematic-gate-results.md` (full 12-clip grid + sweep), `docs/DynamiCtrl-PTQ-method.md`
**Config:** `configs/quant/vidit_ptq.json` (seed 42)

---

## 1. Why human2 only

These three clips appear in **all 10 quantization configurations**:

| Experiment | Clips | FP16 baseline |
|------------|-------|---------------|
| Full grid | 12 (4 humans × 3 motions) | `outputs/fp16_grid/` |
| Group sweep | 3 (`human2` only) | `outputs/sweep_fp16_baseline/` |
| Hybrid (G1 FP16, G2+G3 INT8) | 3 (`human2` only) | `outputs/sweep_fp16_baseline/` |

**Important:** Full-grid runs compare against `fp16_grid`; sweep and hybrid compare against `sweep_fp16_baseline`. FP16 metric values differ slightly between baselines, so **absolute deltas are not directly comparable** across experiment families. Pass/fail, relative ranking within the same baseline, and failure-mode analysis remain valid.

---

## 2. All quantization configurations

| # | Experiment | Algorithm | Blocks INT8 | Log directory | Quant output |
|---|------------|-----------|-------------|---------------|--------------|

| 1 | Full grid | ViDiT | All non-protected (~full DiT) | `logs/kinematic_gate_grid/` | `outputs/vidit_w8a8_grid/` |
| 2 | Full grid | Naive | All non-protected (~full DiT) | `logs/kinematic_gate_w8a8_grid/` | `outputs/w8a8_grid/` |
| 3 | Sweep group1 (blocks 1-13) | ViDiT | 1–13 only | `logs/kinematic_gate_sweep_group1_vidit/` | `outputs/sweep_group1_vidit/` |
| 4 | Sweep group1 (blocks 1-13) | Naive | 1–13 only | `logs/kinematic_gate_sweep_group1_naive/` | `outputs/sweep_group1_naive/` |
| 5 | Sweep group2 (blocks 14-27) | ViDiT | 14–27 only | `logs/kinematic_gate_sweep_group2_vidit/` | `outputs/sweep_group2_vidit/` |
| 6 | Sweep group2 (blocks 14-27) | Naive | 14–27 only | `logs/kinematic_gate_sweep_group2_naive/` | `outputs/sweep_group2_naive/` |
| 7 | Sweep group3 (blocks 28-40) | ViDiT | 28–40 only | `logs/kinematic_gate_sweep_group3_vidit/` | `outputs/sweep_group3_vidit/` |
| 8 | Sweep group3 (blocks 28-40) | Naive | 28–40 only | `logs/kinematic_gate_sweep_group3_naive/` | `outputs/sweep_group3_naive/` |
| 9 | Hybrid (G1 FP16, G2+G3 INT8) | ViDiT | 14–40 (G1 stays FP16) | `logs/kinematic_gate_hybrid_vidit_grid/` | `outputs/hybrid_vidit_w8a8_grid/` |
| 10 | Hybrid (G1 FP16, G2+G3 INT8) | Naive | 14–40 (G1 stays FP16) | `logs/kinematic_gate_hybrid_grid/` | `outputs/hybrid_w8a8_grid/` |

---

## 3. Kinematic gate thresholds

| Check | Threshold | Meaning |
|-------|-----------|---------|
| `acceleration_variance` | rel Δ ≤ **0.10** | Temporal smoothness of joint motion |
| `mean_limb_length_std` | rel Δ ≤ **0.10** | Limb length stability across frames |
| `var_accel_norm` (jitter) | rel Δ ≤ **0.15** | Normalized acceleration jitter |
| `violation_fraction` | quant ≤ FP16 + **0.02** | No new biomechanical constraint breaks |
| `mean_NPE` | ≤ **0.15** | Mean normalized pose error vs FP16 skeleton |
| `p95_NPE` | ≤ **0.30** | 95th-percentile pose error |
| `under_motion_violations` | ≤ **0.0** | Frames under-moving vs driving motion |

A clip **passes** only if every check is within tolerance.

---

## 4. Executive summary

### 4a. Pass/fail matrix

| Config | motion1 | motion2 | motion3 | Total |
|--------|---------|---------|---------|-------|

| Full ViDiT | FAIL | FAIL | FAIL | **0/3** |
| Full Naive | FAIL | **PASS** | **PASS** | **2/3** |
| G1 ViDiT | FAIL | **PASS** | FAIL | **1/3** |
| G1 Naive | FAIL | **PASS** | **PASS** | **2/3** |
| G2 ViDiT | FAIL | **PASS** | **PASS** | **2/3** |
| G2 Naive | **PASS** | **PASS** | FAIL | **2/3** |
| G3 ViDiT | FAIL | FAIL | **PASS** | **1/3** |
| G3 Naive | FAIL | **PASS** | FAIL | **1/3** |
| Hybrid ViDiT | FAIL | **PASS** | **PASS** | **2/3** |
| Hybrid Naive | FAIL | FAIL | **PASS** | **1/3** |

### 4b. Key findings

- **Best pass rate on human2:** Full naive and Hybrid ViDiT tie at **2/3** clips.
- **`human2_motion1` is the bottleneck** — only **G2 naive** (group 2 isolated) passes all checks.
- **Naive vs ViDiT:** Naive **8/15** clip-passes vs ViDiT **6/15** across all configs.
- **Different failure modes:** Naive fails on **violation_fraction** (new constraint breaks); ViDiT fails on **acceleration_variance** and **var_accel_norm** (motion smoothness).
- **Group 1 (blocks 1–13) is most sensitive** when isolated — mean rel_accel ViDiT **0.353** vs group 3 **0.060**.
- **Hybrid ViDiT** improves full ViDiT from **0/3 → 2/3**; **Hybrid naive** regresses from **2/3 → 1/3** vs full naive.
- **Pose error (NPE) never fails** on human2 — gate is effectively a smoothness + violations test for this subject.
- **`mean_NPE` and `p95_NPE`:** 0 failures across all 30 runs on human2.

---

## 5. Master comparison tables


### `human2_motion1`

| Config | Pass | rel_accel | rel_limb | rel_jitter | viol_frac Δ | mean_NPE | p95_NPE | Failed checks |
|--------|------|-----------|----------|------------|-------------|----------|---------|---------------|
| Full ViDiT | FAIL | 0.2983 | 0.0200 | 0.3595 | 0.0270 | 0.0139 | 0.0384 | acceleration_variance, var_accel_norm, violation_fraction |
| Full Naive | FAIL | 0.0298 | 0.0228 | 0.0302 | 0.0811 | 0.0122 | 0.0319 | violation_fraction |
| G1 ViDiT | FAIL | 0.6967 | 0.0171 | 0.7309 | 0.0000 | 0.0150 | 0.0346 | acceleration_variance, var_accel_norm |
| G1 Naive | FAIL | 0.5943 | 0.0444 | 0.5721 | 0.0270 | 0.0132 | 0.0307 | acceleration_variance, var_accel_norm, violation_fraction |
| G2 ViDiT | FAIL | 0.3133 | 0.0173 | 0.2965 | 0.0541 | 0.0140 | 0.0333 | acceleration_variance, var_accel_norm, violation_fraction |
| G2 Naive | **PASS** | 0.0649 | 0.0211 | 0.0459 | 0.0000 | 0.0139 | 0.0318 | *(none)* |
| G3 ViDiT | FAIL | 0.1442 | 0.0041 | 0.1838 | 0.0811 | 0.0138 | 0.0322 | acceleration_variance, var_accel_norm, violation_fraction |
| G3 Naive | FAIL | 0.0237 | 0.0138 | 0.0388 | 0.0270 | 0.0132 | 0.0292 | violation_fraction |
| Hybrid ViDiT | FAIL | 0.3004 | 0.0198 | 0.2819 | 0.0000 | 0.0134 | 0.0311 | acceleration_variance, var_accel_norm |
| Hybrid Naive | FAIL | 0.3255 | 0.0396 | 0.3515 | 0.0270 | 0.0145 | 0.0362 | acceleration_variance, var_accel_norm, violation_fraction |

### `human2_motion2`

| Config | Pass | rel_accel | rel_limb | rel_jitter | viol_frac Δ | mean_NPE | p95_NPE | Failed checks |
|--------|------|-----------|----------|------------|-------------|----------|---------|---------------|
| Full ViDiT | FAIL | 0.0040 | 0.1017 | 0.0045 | 0.0270 | 0.0142 | 0.0340 | mean_limb_length_std |
| Full Naive | **PASS** | 0.0886 | 0.0240 | 0.1126 | 0.0270 | 0.0129 | 0.0335 | *(none)* |
| G1 ViDiT | **PASS** | 0.0766 | 0.0236 | 0.0296 | 0.0270 | 0.0135 | 0.0317 | *(none)* |
| G1 Naive | **PASS** | 0.0214 | 0.0273 | 0.1229 | 0.0000 | 0.0144 | 0.0382 | *(none)* |
| G2 ViDiT | **PASS** | 0.0244 | 0.0019 | 0.0712 | 0.0000 | 0.0131 | 0.0348 | *(none)* |
| G2 Naive | **PASS** | 0.0293 | 0.0023 | 0.0012 | 0.0000 | 0.0115 | 0.0265 | *(none)* |
| G3 ViDiT | FAIL | 0.0209 | 0.0190 | 0.0281 | 0.0270 | 0.0131 | 0.0359 | violation_fraction |
| G3 Naive | **PASS** | 0.0301 | 0.0427 | 0.0419 | 0.0270 | 0.0120 | 0.0294 | *(none)* |
| Hybrid ViDiT | **PASS** | 0.0248 | 0.0335 | 0.0911 | 0.0000 | 0.0137 | 0.0378 | *(none)* |
| Hybrid Naive | FAIL | 0.0004 | 0.0250 | 0.0486 | 0.0270 | 0.0101 | 0.0250 | violation_fraction |

### `human2_motion3`

| Config | Pass | rel_accel | rel_limb | rel_jitter | viol_frac Δ | mean_NPE | p95_NPE | Failed checks |
|--------|------|-----------|----------|------------|-------------|----------|---------|---------------|
| Full ViDiT | FAIL | 0.1491 | 0.0067 | 0.1527 | 0.0000 | 0.0120 | 0.0386 | acceleration_variance, var_accel_norm |
| Full Naive | **PASS** | 2.03e-05 | 0.0263 | 0.0299 | 0.0000 | 0.0132 | 0.0368 | *(none)* |
| G1 ViDiT | FAIL | 0.2847 | 0.0343 | 0.2928 | 0.0000 | 0.0128 | 0.0376 | acceleration_variance, var_accel_norm |
| G1 Naive | **PASS** | 0.0652 | 0.0478 | 0.0341 | 0.0000 | 0.0126 | 0.0366 | *(none)* |
| G2 ViDiT | **PASS** | 0.0295 | 0.0051 | 0.0123 | 0.0000 | 0.0104 | 0.0274 | *(none)* |
| G2 Naive | FAIL | 0.1093 | 0.0380 | 0.1118 | 0.0000 | 0.0125 | 0.0349 | acceleration_variance |
| G3 ViDiT | **PASS** | 0.0149 | 0.0165 | 0.0077 | 0.0000 | 0.0108 | 0.0282 | *(none)* |
| G3 Naive | FAIL | 0.1539 | 0.0285 | 0.1661 | 0.0000 | 0.0120 | 0.0322 | acceleration_variance, var_accel_norm |
| Hybrid ViDiT | **PASS** | 0.0777 | 0.0021 | 0.1145 | 0.0000 | 0.0107 | 0.0278 | *(none)* |
| Hybrid Naive | **PASS** | 0.0085 | 0.0374 | 0.0340 | 0.0000 | 0.0100 | 0.0289 | *(none)* |

---

## 6. Per-configuration detail (all metrics)


### Full grid — ViDiT-Q W8A8 — `human2_motion1` — **FAIL**

- **FP16 video:** `outputs/fp16_grid/human2_motion1/fp16_human2_motion1_seed42.mp4`
- **Quant video:** `outputs/vidit_w8a8_grid/human2_motion1/vidit_w8a8_human2_motion1_seed42.mp4`
- **Report:** `logs/kinematic_gate_grid/human2_motion1.json`

| Metric | FP16 | Quant | Delta | Check |
|--------|------|-------|-------|-------|

| acceleration_variance | 172.61 | 224.09 | rel=0.2983 | FAIL |
| mean_limb_length_std | 15.26 | 15.57 | rel=0.0200 | PASS |
| var_accel_norm (jitter) | 158.30 | 215.20 | rel=0.3595 | FAIL |
| violation_fraction | 0.00 | 0.03 | abs=0.0270 | FAIL |
| total_violations | 0.00 | 1.00 | abs=1 | — |
| mean_NPE | — | — | 0.0139 | PASS |
| p95_NPE | — | — | 0.0384 | PASS |
| under_motion_violations | — | — | 0 | PASS |

**Failed checks:** acceleration_variance, var_accel_norm, violation_fraction


### Full grid — ViDiT-Q W8A8 — `human2_motion2` — **FAIL**

- **FP16 video:** `outputs/fp16_grid/human2_motion2/fp16_human2_motion2_seed42.mp4`
- **Quant video:** `outputs/vidit_w8a8_grid/human2_motion2/vidit_w8a8_human2_motion2_seed42.mp4`
- **Report:** `logs/kinematic_gate_grid/human2_motion2.json`

| Metric | FP16 | Quant | Delta | Check |
|--------|------|-------|-------|-------|

| acceleration_variance | 40.93 | 40.76 | rel=0.0040 | PASS |
| mean_limb_length_std | 10.58 | 9.50 | rel=0.1017 | FAIL |
| var_accel_norm (jitter) | 28.42 | 28.29 | rel=0.0045 | PASS |
| violation_fraction | 0.05 | 0.03 | abs=0.0270 | PASS |
| total_violations | 2.00 | 1.00 | abs=1 | — |
| mean_NPE | — | — | 0.0142 | PASS |
| p95_NPE | — | — | 0.0340 | PASS |
| under_motion_violations | — | — | 0 | PASS |

**Failed checks:** mean_limb_length_std


### Full grid — ViDiT-Q W8A8 — `human2_motion3` — **FAIL**

- **FP16 video:** `outputs/fp16_grid/human2_motion3/fp16_human2_motion3_seed42.mp4`
- **Quant video:** `outputs/vidit_w8a8_grid/human2_motion3/vidit_w8a8_human2_motion3_seed42.mp4`
- **Report:** `logs/kinematic_gate_grid/human2_motion3.json`

| Metric | FP16 | Quant | Delta | Check |
|--------|------|-------|-------|-------|

| acceleration_variance | 217.44 | 185.03 | rel=0.1491 | FAIL |
| mean_limb_length_std | 12.68 | 12.76 | rel=0.0067 | PASS |
| var_accel_norm (jitter) | 203.68 | 172.57 | rel=0.1527 | FAIL |
| violation_fraction | 0.00 | 0.00 | abs=0.0000 | PASS |
| total_violations | 0.00 | 0.00 | abs=0 | — |
| mean_NPE | — | — | 0.0120 | PASS |
| p95_NPE | — | — | 0.0386 | PASS |
| under_motion_violations | — | — | 0 | PASS |

**Failed checks:** acceleration_variance, var_accel_norm


### Full grid — Matched naive W8A8 — `human2_motion1` — **FAIL**

- **FP16 video:** `outputs/fp16_grid/human2_motion1/fp16_human2_motion1_seed42.mp4`
- **Quant video:** `outputs/w8a8_grid/human2_motion1/w8a8_human2_motion1_seed42.mp4`
- **Report:** `logs/kinematic_gate_w8a8_grid/human2_motion1.json`

| Metric | FP16 | Quant | Delta | Check |
|--------|------|-------|-------|-------|

| acceleration_variance | 172.61 | 167.47 | rel=0.0298 | PASS |
| mean_limb_length_std | 15.26 | 14.91 | rel=0.0228 | PASS |
| var_accel_norm (jitter) | 158.30 | 153.52 | rel=0.0302 | PASS |
| violation_fraction | 0.00 | 0.08 | abs=0.0811 | FAIL |
| total_violations | 0.00 | 3.00 | abs=3 | — |
| mean_NPE | — | — | 0.0122 | PASS |
| p95_NPE | — | — | 0.0319 | PASS |
| under_motion_violations | — | — | 0 | PASS |

**Failed checks:** violation_fraction


### Full grid — Matched naive W8A8 — `human2_motion2` — **PASS**

- **FP16 video:** `outputs/fp16_grid/human2_motion2/fp16_human2_motion2_seed42.mp4`
- **Quant video:** `outputs/w8a8_grid/human2_motion2/w8a8_human2_motion2_seed42.mp4`
- **Report:** `logs/kinematic_gate_w8a8_grid/human2_motion2.json`

| Metric | FP16 | Quant | Delta | Check |
|--------|------|-------|-------|-------|

| acceleration_variance | 40.93 | 37.30 | rel=0.0886 | PASS |
| mean_limb_length_std | 10.58 | 10.32 | rel=0.0240 | PASS |
| var_accel_norm (jitter) | 28.42 | 25.22 | rel=0.1126 | PASS |
| violation_fraction | 0.05 | 0.03 | abs=0.0270 | PASS |
| total_violations | 2.00 | 1.00 | abs=1 | — |
| mean_NPE | — | — | 0.0129 | PASS |
| p95_NPE | — | — | 0.0335 | PASS |
| under_motion_violations | — | — | 0 | PASS |

**Failed checks:** *(none)*


### Full grid — Matched naive W8A8 — `human2_motion3` — **PASS**

- **FP16 video:** `outputs/fp16_grid/human2_motion3/fp16_human2_motion3_seed42.mp4`
- **Quant video:** `outputs/w8a8_grid/human2_motion3/w8a8_human2_motion3_seed42.mp4`
- **Report:** `logs/kinematic_gate_w8a8_grid/human2_motion3.json`

| Metric | FP16 | Quant | Delta | Check |
|--------|------|-------|-------|-------|

| acceleration_variance | 217.44 | 217.44 | rel=2.03e-05 | PASS |
| mean_limb_length_std | 12.68 | 13.01 | rel=0.0263 | PASS |
| var_accel_norm (jitter) | 203.68 | 209.76 | rel=0.0299 | PASS |
| violation_fraction | 0.00 | 0.00 | abs=0.0000 | PASS |
| total_violations | 0.00 | 0.00 | abs=0 | — |
| mean_NPE | — | — | 0.0132 | PASS |
| p95_NPE | — | — | 0.0368 | PASS |
| under_motion_violations | — | — | 0 | PASS |

**Failed checks:** *(none)*


### Sweep group1 — ViDiT-Q W8A8 — `human2_motion1` — **FAIL**

- **FP16 video:** `outputs/sweep_fp16_baseline/human2_motion1/fp16_human2_motion1_seed42.mp4`
- **Quant video:** `outputs/sweep_group1_vidit/human2_motion1/vidit_w8a8_human2_motion1_seed42.mp4`
- **Report:** `logs/kinematic_gate_sweep_group1_vidit/human2_motion1.json`

| Metric | FP16 | Quant | Delta | Check |
|--------|------|-------|-------|-------|

| acceleration_variance | 203.43 | 61.69 | rel=0.6967 | FAIL |
| mean_limb_length_std | 15.41 | 15.67 | rel=0.0171 | PASS |
| var_accel_norm (jitter) | 193.45 | 52.06 | rel=0.7309 | FAIL |
| violation_fraction | 0.00 | 0.00 | abs=0.0000 | PASS |
| total_violations | 0.00 | 0.00 | abs=0 | — |
| mean_NPE | — | — | 0.0150 | PASS |
| p95_NPE | — | — | 0.0346 | PASS |
| under_motion_violations | — | — | 0 | PASS |

**Failed checks:** acceleration_variance, var_accel_norm


### Sweep group1 — ViDiT-Q W8A8 — `human2_motion2` — **PASS**

- **FP16 video:** `outputs/sweep_fp16_baseline/human2_motion2/fp16_human2_motion2_seed42.mp4`
- **Quant video:** `outputs/sweep_group1_vidit/human2_motion2/vidit_w8a8_human2_motion2_seed42.mp4`
- **Report:** `logs/kinematic_gate_sweep_group1_vidit/human2_motion2.json`

| Metric | FP16 | Quant | Delta | Check |
|--------|------|-------|-------|-------|

| acceleration_variance | 40.93 | 44.06 | rel=0.0766 | PASS |
| mean_limb_length_std | 10.58 | 10.33 | rel=0.0236 | PASS |
| var_accel_norm (jitter) | 28.42 | 27.58 | rel=0.0296 | PASS |
| violation_fraction | 0.05 | 0.03 | abs=0.0270 | PASS |
| total_violations | 2.00 | 1.00 | abs=1 | — |
| mean_NPE | — | — | 0.0135 | PASS |
| p95_NPE | — | — | 0.0317 | PASS |
| under_motion_violations | — | — | 0 | PASS |

**Failed checks:** *(none)*


### Sweep group1 — ViDiT-Q W8A8 — `human2_motion3` — **FAIL**

- **FP16 video:** `outputs/sweep_fp16_baseline/human2_motion3/fp16_human2_motion3_seed42.mp4`
- **Quant video:** `outputs/sweep_group1_vidit/human2_motion3/vidit_w8a8_human2_motion3_seed42.mp4`
- **Report:** `logs/kinematic_gate_sweep_group1_vidit/human2_motion3.json`

| Metric | FP16 | Quant | Delta | Check |
|--------|------|-------|-------|-------|

| acceleration_variance | 208.87 | 149.41 | rel=0.2847 | FAIL |
| mean_limb_length_std | 12.88 | 13.32 | rel=0.0343 | PASS |
| var_accel_norm (jitter) | 191.23 | 135.24 | rel=0.2928 | FAIL |
| violation_fraction | 0.00 | 0.00 | abs=0.0000 | PASS |
| total_violations | 0.00 | 0.00 | abs=0 | — |
| mean_NPE | — | — | 0.0128 | PASS |
| p95_NPE | — | — | 0.0376 | PASS |
| under_motion_violations | — | — | 0 | PASS |

**Failed checks:** acceleration_variance, var_accel_norm


### Sweep group1 — Matched naive W8A8 — `human2_motion1` — **FAIL**

- **FP16 video:** `outputs/sweep_fp16_baseline/human2_motion1/fp16_human2_motion1_seed42.mp4`
- **Quant video:** `outputs/sweep_group1_naive/human2_motion1/w8a8_human2_motion1_seed42.mp4`
- **Report:** `logs/kinematic_gate_sweep_group1_naive/human2_motion1.json`

| Metric | FP16 | Quant | Delta | Check |
|--------|------|-------|-------|-------|

| acceleration_variance | 203.43 | 82.52 | rel=0.5943 | FAIL |
| mean_limb_length_std | 15.41 | 16.09 | rel=0.0444 | PASS |
| var_accel_norm (jitter) | 193.45 | 82.77 | rel=0.5721 | FAIL |
| violation_fraction | 0.00 | 0.03 | abs=0.0270 | FAIL |
| total_violations | 0.00 | 1.00 | abs=1 | — |
| mean_NPE | — | — | 0.0132 | PASS |
| p95_NPE | — | — | 0.0307 | PASS |
| under_motion_violations | — | — | 0 | PASS |

**Failed checks:** acceleration_variance, var_accel_norm, violation_fraction


### Sweep group1 — Matched naive W8A8 — `human2_motion2` — **PASS**

- **FP16 video:** `outputs/sweep_fp16_baseline/human2_motion2/fp16_human2_motion2_seed42.mp4`
- **Quant video:** `outputs/sweep_group1_naive/human2_motion2/w8a8_human2_motion2_seed42.mp4`
- **Report:** `logs/kinematic_gate_sweep_group1_naive/human2_motion2.json`

| Metric | FP16 | Quant | Delta | Check |
|--------|------|-------|-------|-------|

| acceleration_variance | 40.93 | 40.05 | rel=0.0214 | PASS |
| mean_limb_length_std | 10.58 | 10.29 | rel=0.0273 | PASS |
| var_accel_norm (jitter) | 28.42 | 24.93 | rel=0.1229 | PASS |
| violation_fraction | 0.05 | 0.05 | abs=0.0000 | PASS |
| total_violations | 2.00 | 2.00 | abs=0 | — |
| mean_NPE | — | — | 0.0144 | PASS |
| p95_NPE | — | — | 0.0382 | PASS |
| under_motion_violations | — | — | 0 | PASS |

**Failed checks:** *(none)*


### Sweep group1 — Matched naive W8A8 — `human2_motion3` — **PASS**

- **FP16 video:** `outputs/sweep_fp16_baseline/human2_motion3/fp16_human2_motion3_seed42.mp4`
- **Quant video:** `outputs/sweep_group1_naive/human2_motion3/w8a8_human2_motion3_seed42.mp4`
- **Report:** `logs/kinematic_gate_sweep_group1_naive/human2_motion3.json`

| Metric | FP16 | Quant | Delta | Check |
|--------|------|-------|-------|-------|

| acceleration_variance | 208.87 | 195.25 | rel=0.0652 | PASS |
| mean_limb_length_std | 12.88 | 12.26 | rel=0.0478 | PASS |
| var_accel_norm (jitter) | 191.23 | 184.70 | rel=0.0341 | PASS |
| violation_fraction | 0.00 | 0.00 | abs=0.0000 | PASS |
| total_violations | 0.00 | 0.00 | abs=0 | — |
| mean_NPE | — | — | 0.0126 | PASS |
| p95_NPE | — | — | 0.0366 | PASS |
| under_motion_violations | — | — | 0 | PASS |

**Failed checks:** *(none)*


### Sweep group2 — ViDiT-Q W8A8 — `human2_motion1` — **FAIL**

- **FP16 video:** `outputs/sweep_fp16_baseline/human2_motion1/fp16_human2_motion1_seed42.mp4`
- **Quant video:** `outputs/sweep_group2_vidit/human2_motion1/vidit_w8a8_human2_motion1_seed42.mp4`
- **Report:** `logs/kinematic_gate_sweep_group2_vidit/human2_motion1.json`

| Metric | FP16 | Quant | Delta | Check |
|--------|------|-------|-------|-------|

| acceleration_variance | 203.43 | 139.71 | rel=0.3133 | FAIL |
| mean_limb_length_std | 15.41 | 15.67 | rel=0.0173 | PASS |
| var_accel_norm (jitter) | 193.45 | 136.09 | rel=0.2965 | FAIL |
| violation_fraction | 0.00 | 0.05 | abs=0.0541 | FAIL |
| total_violations | 0.00 | 2.00 | abs=2 | — |
| mean_NPE | — | — | 0.0140 | PASS |
| p95_NPE | — | — | 0.0333 | PASS |
| under_motion_violations | — | — | 0 | PASS |

**Failed checks:** acceleration_variance, var_accel_norm, violation_fraction


### Sweep group2 — ViDiT-Q W8A8 — `human2_motion2` — **PASS**

- **FP16 video:** `outputs/sweep_fp16_baseline/human2_motion2/fp16_human2_motion2_seed42.mp4`
- **Quant video:** `outputs/sweep_group2_vidit/human2_motion2/vidit_w8a8_human2_motion2_seed42.mp4`
- **Report:** `logs/kinematic_gate_sweep_group2_vidit/human2_motion2.json`

| Metric | FP16 | Quant | Delta | Check |
|--------|------|-------|-------|-------|

| acceleration_variance | 40.93 | 39.93 | rel=0.0244 | PASS |
| mean_limb_length_std | 10.58 | 10.56 | rel=0.0019 | PASS |
| var_accel_norm (jitter) | 28.42 | 26.40 | rel=0.0712 | PASS |
| violation_fraction | 0.05 | 0.05 | abs=0.0000 | PASS |
| total_violations | 2.00 | 2.00 | abs=0 | — |
| mean_NPE | — | — | 0.0131 | PASS |
| p95_NPE | — | — | 0.0348 | PASS |
| under_motion_violations | — | — | 0 | PASS |

**Failed checks:** *(none)*


### Sweep group2 — ViDiT-Q W8A8 — `human2_motion3` — **PASS**

- **FP16 video:** `outputs/sweep_fp16_baseline/human2_motion3/fp16_human2_motion3_seed42.mp4`
- **Quant video:** `outputs/sweep_group2_vidit/human2_motion3/vidit_w8a8_human2_motion3_seed42.mp4`
- **Report:** `logs/kinematic_gate_sweep_group2_vidit/human2_motion3.json`

| Metric | FP16 | Quant | Delta | Check |
|--------|------|-------|-------|-------|

| acceleration_variance | 208.87 | 202.70 | rel=0.0295 | PASS |
| mean_limb_length_std | 12.88 | 12.81 | rel=0.0051 | PASS |
| var_accel_norm (jitter) | 191.23 | 188.89 | rel=0.0123 | PASS |
| violation_fraction | 0.00 | 0.00 | abs=0.0000 | PASS |
| total_violations | 0.00 | 0.00 | abs=0 | — |
| mean_NPE | — | — | 0.0104 | PASS |
| p95_NPE | — | — | 0.0274 | PASS |
| under_motion_violations | — | — | 0 | PASS |

**Failed checks:** *(none)*


### Sweep group2 — Matched naive W8A8 — `human2_motion1` — **PASS**

- **FP16 video:** `outputs/sweep_fp16_baseline/human2_motion1/fp16_human2_motion1_seed42.mp4`
- **Quant video:** `outputs/sweep_group2_naive/human2_motion1/w8a8_human2_motion1_seed42.mp4`
- **Report:** `logs/kinematic_gate_sweep_group2_naive/human2_motion1.json`

| Metric | FP16 | Quant | Delta | Check |
|--------|------|-------|-------|-------|

| acceleration_variance | 203.43 | 216.63 | rel=0.0649 | PASS |
| mean_limb_length_std | 15.41 | 15.08 | rel=0.0211 | PASS |
| var_accel_norm (jitter) | 193.45 | 202.32 | rel=0.0459 | PASS |
| violation_fraction | 0.00 | 0.00 | abs=0.0000 | PASS |
| total_violations | 0.00 | 0.00 | abs=0 | — |
| mean_NPE | — | — | 0.0139 | PASS |
| p95_NPE | — | — | 0.0318 | PASS |
| under_motion_violations | — | — | 0 | PASS |

**Failed checks:** *(none)*


### Sweep group2 — Matched naive W8A8 — `human2_motion2` — **PASS**

- **FP16 video:** `outputs/sweep_fp16_baseline/human2_motion2/fp16_human2_motion2_seed42.mp4`
- **Quant video:** `outputs/sweep_group2_naive/human2_motion2/w8a8_human2_motion2_seed42.mp4`
- **Report:** `logs/kinematic_gate_sweep_group2_naive/human2_motion2.json`

| Metric | FP16 | Quant | Delta | Check |
|--------|------|-------|-------|-------|

| acceleration_variance | 40.93 | 39.73 | rel=0.0293 | PASS |
| mean_limb_length_std | 10.58 | 10.55 | rel=0.0023 | PASS |
| var_accel_norm (jitter) | 28.42 | 28.46 | rel=0.0012 | PASS |
| violation_fraction | 0.05 | 0.05 | abs=0.0000 | PASS |
| total_violations | 2.00 | 2.00 | abs=0 | — |
| mean_NPE | — | — | 0.0115 | PASS |
| p95_NPE | — | — | 0.0265 | PASS |
| under_motion_violations | — | — | 0 | PASS |

**Failed checks:** *(none)*


### Sweep group2 — Matched naive W8A8 — `human2_motion3` — **FAIL**

- **FP16 video:** `outputs/sweep_fp16_baseline/human2_motion3/fp16_human2_motion3_seed42.mp4`
- **Quant video:** `outputs/sweep_group2_naive/human2_motion3/w8a8_human2_motion3_seed42.mp4`
- **Report:** `logs/kinematic_gate_sweep_group2_naive/human2_motion3.json`

| Metric | FP16 | Quant | Delta | Check |
|--------|------|-------|-------|-------|

| acceleration_variance | 208.87 | 186.04 | rel=0.1093 | FAIL |
| mean_limb_length_std | 12.88 | 12.39 | rel=0.0380 | PASS |
| var_accel_norm (jitter) | 191.23 | 169.85 | rel=0.1118 | PASS |
| violation_fraction | 0.00 | 0.00 | abs=0.0000 | PASS |
| total_violations | 0.00 | 0.00 | abs=0 | — |
| mean_NPE | — | — | 0.0125 | PASS |
| p95_NPE | — | — | 0.0349 | PASS |
| under_motion_violations | — | — | 0 | PASS |

**Failed checks:** acceleration_variance


### Sweep group3 — ViDiT-Q W8A8 — `human2_motion1` — **FAIL**

- **FP16 video:** `outputs/sweep_fp16_baseline/human2_motion1/fp16_human2_motion1_seed42.mp4`
- **Quant video:** `outputs/sweep_group3_vidit/human2_motion1/vidit_w8a8_human2_motion1_seed42.mp4`
- **Report:** `logs/kinematic_gate_sweep_group3_vidit/human2_motion1.json`

| Metric | FP16 | Quant | Delta | Check |
|--------|------|-------|-------|-------|

| acceleration_variance | 203.43 | 232.76 | rel=0.1442 | FAIL |
| mean_limb_length_std | 15.41 | 15.34 | rel=0.0041 | PASS |
| var_accel_norm (jitter) | 193.45 | 229.00 | rel=0.1838 | FAIL |
| violation_fraction | 0.00 | 0.08 | abs=0.0811 | FAIL |
| total_violations | 0.00 | 3.00 | abs=3 | — |
| mean_NPE | — | — | 0.0138 | PASS |
| p95_NPE | — | — | 0.0322 | PASS |
| under_motion_violations | — | — | 0 | PASS |

**Failed checks:** acceleration_variance, var_accel_norm, violation_fraction


### Sweep group3 — ViDiT-Q W8A8 — `human2_motion2` — **FAIL**

- **FP16 video:** `outputs/sweep_fp16_baseline/human2_motion2/fp16_human2_motion2_seed42.mp4`
- **Quant video:** `outputs/sweep_group3_vidit/human2_motion2/vidit_w8a8_human2_motion2_seed42.mp4`
- **Report:** `logs/kinematic_gate_sweep_group3_vidit/human2_motion2.json`

| Metric | FP16 | Quant | Delta | Check |
|--------|------|-------|-------|-------|

| acceleration_variance | 40.93 | 41.78 | rel=0.0209 | PASS |
| mean_limb_length_std | 10.58 | 10.38 | rel=0.0190 | PASS |
| var_accel_norm (jitter) | 28.42 | 29.22 | rel=0.0281 | PASS |
| violation_fraction | 0.05 | 0.08 | abs=0.0270 | FAIL |
| total_violations | 2.00 | 3.00 | abs=1 | — |
| mean_NPE | — | — | 0.0131 | PASS |
| p95_NPE | — | — | 0.0359 | PASS |
| under_motion_violations | — | — | 0 | PASS |

**Failed checks:** violation_fraction


### Sweep group3 — ViDiT-Q W8A8 — `human2_motion3` — **PASS**

- **FP16 video:** `outputs/sweep_fp16_baseline/human2_motion3/fp16_human2_motion3_seed42.mp4`
- **Quant video:** `outputs/sweep_group3_vidit/human2_motion3/vidit_w8a8_human2_motion3_seed42.mp4`
- **Report:** `logs/kinematic_gate_sweep_group3_vidit/human2_motion3.json`

| Metric | FP16 | Quant | Delta | Check |
|--------|------|-------|-------|-------|

| acceleration_variance | 208.87 | 211.98 | rel=0.0149 | PASS |
| mean_limb_length_std | 12.88 | 13.09 | rel=0.0165 | PASS |
| var_accel_norm (jitter) | 191.23 | 189.76 | rel=0.0077 | PASS |
| violation_fraction | 0.00 | 0.00 | abs=0.0000 | PASS |
| total_violations | 0.00 | 0.00 | abs=0 | — |
| mean_NPE | — | — | 0.0108 | PASS |
| p95_NPE | — | — | 0.0282 | PASS |
| under_motion_violations | — | — | 0 | PASS |

**Failed checks:** *(none)*


### Sweep group3 — Matched naive W8A8 — `human2_motion1` — **FAIL**

- **FP16 video:** `outputs/sweep_fp16_baseline/human2_motion1/fp16_human2_motion1_seed42.mp4`
- **Quant video:** `outputs/sweep_group3_naive/human2_motion1/w8a8_human2_motion1_seed42.mp4`
- **Report:** `logs/kinematic_gate_sweep_group3_naive/human2_motion1.json`

| Metric | FP16 | Quant | Delta | Check |
|--------|------|-------|-------|-------|

| acceleration_variance | 203.43 | 198.61 | rel=0.0237 | PASS |
| mean_limb_length_std | 15.41 | 15.19 | rel=0.0138 | PASS |
| var_accel_norm (jitter) | 193.45 | 200.96 | rel=0.0388 | PASS |
| violation_fraction | 0.00 | 0.03 | abs=0.0270 | FAIL |
| total_violations | 0.00 | 1.00 | abs=1 | — |
| mean_NPE | — | — | 0.0132 | PASS |
| p95_NPE | — | — | 0.0292 | PASS |
| under_motion_violations | — | — | 0 | PASS |

**Failed checks:** violation_fraction


### Sweep group3 — Matched naive W8A8 — `human2_motion2` — **PASS**

- **FP16 video:** `outputs/sweep_fp16_baseline/human2_motion2/fp16_human2_motion2_seed42.mp4`
- **Quant video:** `outputs/sweep_group3_naive/human2_motion2/w8a8_human2_motion2_seed42.mp4`
- **Report:** `logs/kinematic_gate_sweep_group3_naive/human2_motion2.json`

| Metric | FP16 | Quant | Delta | Check |
|--------|------|-------|-------|-------|

| acceleration_variance | 40.93 | 39.69 | rel=0.0301 | PASS |
| mean_limb_length_std | 10.58 | 10.13 | rel=0.0427 | PASS |
| var_accel_norm (jitter) | 28.42 | 27.23 | rel=0.0419 | PASS |
| violation_fraction | 0.05 | 0.03 | abs=0.0270 | PASS |
| total_violations | 2.00 | 1.00 | abs=1 | — |
| mean_NPE | — | — | 0.0120 | PASS |
| p95_NPE | — | — | 0.0294 | PASS |
| under_motion_violations | — | — | 0 | PASS |

**Failed checks:** *(none)*


### Sweep group3 — Matched naive W8A8 — `human2_motion3` — **FAIL**

- **FP16 video:** `outputs/sweep_fp16_baseline/human2_motion3/fp16_human2_motion3_seed42.mp4`
- **Quant video:** `outputs/sweep_group3_naive/human2_motion3/w8a8_human2_motion3_seed42.mp4`
- **Report:** `logs/kinematic_gate_sweep_group3_naive/human2_motion3.json`

| Metric | FP16 | Quant | Delta | Check |
|--------|------|-------|-------|-------|

| acceleration_variance | 208.87 | 241.01 | rel=0.1539 | FAIL |
| mean_limb_length_std | 12.88 | 12.51 | rel=0.0285 | PASS |
| var_accel_norm (jitter) | 191.23 | 222.99 | rel=0.1661 | FAIL |
| violation_fraction | 0.00 | 0.00 | abs=0.0000 | PASS |
| total_violations | 0.00 | 0.00 | abs=0 | — |
| mean_NPE | — | — | 0.0120 | PASS |
| p95_NPE | — | — | 0.0322 | PASS |
| under_motion_violations | — | — | 0 | PASS |

**Failed checks:** acceleration_variance, var_accel_norm


### Hybrid — ViDiT-Q W8A8 — `human2_motion1` — **FAIL**

- **FP16 video:** `outputs/sweep_fp16_baseline/human2_motion1/fp16_human2_motion1_seed42.mp4`
- **Quant video:** `outputs/hybrid_vidit_w8a8_grid/human2_motion1/hybrid_vidit_w8a8_human2_motion1_seed42.mp4`
- **Report:** `logs/kinematic_gate_hybrid_vidit_grid/human2_motion1.json`

| Metric | FP16 | Quant | Delta | Check |
|--------|------|-------|-------|-------|

| acceleration_variance | 203.43 | 142.32 | rel=0.3004 | FAIL |
| mean_limb_length_std | 15.41 | 15.10 | rel=0.0198 | PASS |
| var_accel_norm (jitter) | 193.45 | 138.92 | rel=0.2819 | FAIL |
| violation_fraction | 0.00 | 0.00 | abs=0.0000 | PASS |
| total_violations | 0.00 | 0.00 | abs=0 | — |
| mean_NPE | — | — | 0.0134 | PASS |
| p95_NPE | — | — | 0.0311 | PASS |
| under_motion_violations | — | — | 0 | PASS |

**Failed checks:** acceleration_variance, var_accel_norm


### Hybrid — ViDiT-Q W8A8 — `human2_motion2` — **PASS**

- **FP16 video:** `outputs/sweep_fp16_baseline/human2_motion2/fp16_human2_motion2_seed42.mp4`
- **Quant video:** `outputs/hybrid_vidit_w8a8_grid/human2_motion2/hybrid_vidit_w8a8_human2_motion2_seed42.mp4`
- **Report:** `logs/kinematic_gate_hybrid_vidit_grid/human2_motion2.json`

| Metric | FP16 | Quant | Delta | Check |
|--------|------|-------|-------|-------|

| acceleration_variance | 40.93 | 39.91 | rel=0.0248 | PASS |
| mean_limb_length_std | 10.58 | 10.22 | rel=0.0335 | PASS |
| var_accel_norm (jitter) | 28.42 | 25.83 | rel=0.0911 | PASS |
| violation_fraction | 0.05 | 0.05 | abs=0.0000 | PASS |
| total_violations | 2.00 | 2.00 | abs=0 | — |
| mean_NPE | — | — | 0.0137 | PASS |
| p95_NPE | — | — | 0.0378 | PASS |
| under_motion_violations | — | — | 0 | PASS |

**Failed checks:** *(none)*


### Hybrid — ViDiT-Q W8A8 — `human2_motion3` — **PASS**

- **FP16 video:** `outputs/sweep_fp16_baseline/human2_motion3/fp16_human2_motion3_seed42.mp4`
- **Quant video:** `outputs/hybrid_vidit_w8a8_grid/human2_motion3/hybrid_vidit_w8a8_human2_motion3_seed42.mp4`
- **Report:** `logs/kinematic_gate_hybrid_vidit_grid/human2_motion3.json`

| Metric | FP16 | Quant | Delta | Check |
|--------|------|-------|-------|-------|

| acceleration_variance | 208.87 | 225.10 | rel=0.0777 | PASS |
| mean_limb_length_std | 12.88 | 12.85 | rel=0.0021 | PASS |
| var_accel_norm (jitter) | 191.23 | 213.12 | rel=0.1145 | PASS |
| violation_fraction | 0.00 | 0.00 | abs=0.0000 | PASS |
| total_violations | 0.00 | 0.00 | abs=0 | — |
| mean_NPE | — | — | 0.0107 | PASS |
| p95_NPE | — | — | 0.0278 | PASS |
| under_motion_violations | — | — | 0 | PASS |

**Failed checks:** *(none)*


### Hybrid — Matched naive W8A8 — `human2_motion1` — **FAIL**

- **FP16 video:** `outputs/sweep_fp16_baseline/human2_motion1/fp16_human2_motion1_seed42.mp4`
- **Quant video:** `outputs/hybrid_w8a8_grid/human2_motion1/hybrid_w8a8_human2_motion1_seed42.mp4`
- **Report:** `logs/kinematic_gate_hybrid_grid/human2_motion1.json`

| Metric | FP16 | Quant | Delta | Check |
|--------|------|-------|-------|-------|

| acceleration_variance | 203.43 | 137.21 | rel=0.3255 | FAIL |
| mean_limb_length_std | 15.41 | 16.02 | rel=0.0396 | PASS |
| var_accel_norm (jitter) | 193.45 | 125.46 | rel=0.3515 | FAIL |
| violation_fraction | 0.00 | 0.03 | abs=0.0270 | FAIL |
| total_violations | 0.00 | 1.00 | abs=1 | — |
| mean_NPE | — | — | 0.0145 | PASS |
| p95_NPE | — | — | 0.0362 | PASS |
| under_motion_violations | — | — | 0 | PASS |

**Failed checks:** acceleration_variance, var_accel_norm, violation_fraction


### Hybrid — Matched naive W8A8 — `human2_motion2` — **FAIL**

- **FP16 video:** `outputs/sweep_fp16_baseline/human2_motion2/fp16_human2_motion2_seed42.mp4`
- **Quant video:** `outputs/hybrid_w8a8_grid/human2_motion2/hybrid_w8a8_human2_motion2_seed42.mp4`
- **Report:** `logs/kinematic_gate_hybrid_grid/human2_motion2.json`

| Metric | FP16 | Quant | Delta | Check |
|--------|------|-------|-------|-------|

| acceleration_variance | 40.93 | 40.91 | rel=0.0004 | PASS |
| mean_limb_length_std | 10.58 | 10.31 | rel=0.0250 | PASS |
| var_accel_norm (jitter) | 28.42 | 29.80 | rel=0.0486 | PASS |
| violation_fraction | 0.05 | 0.08 | abs=0.0270 | FAIL |
| total_violations | 2.00 | 3.00 | abs=1 | — |
| mean_NPE | — | — | 0.0101 | PASS |
| p95_NPE | — | — | 0.0250 | PASS |
| under_motion_violations | — | — | 0 | PASS |

**Failed checks:** violation_fraction


### Hybrid — Matched naive W8A8 — `human2_motion3` — **PASS**

- **FP16 video:** `outputs/sweep_fp16_baseline/human2_motion3/fp16_human2_motion3_seed42.mp4`
- **Quant video:** `outputs/hybrid_w8a8_grid/human2_motion3/hybrid_w8a8_human2_motion3_seed42.mp4`
- **Report:** `logs/kinematic_gate_hybrid_grid/human2_motion3.json`

| Metric | FP16 | Quant | Delta | Check |
|--------|------|-------|-------|-------|

| acceleration_variance | 208.87 | 207.10 | rel=0.0085 | PASS |
| mean_limb_length_std | 12.88 | 12.40 | rel=0.0374 | PASS |
| var_accel_norm (jitter) | 191.23 | 197.73 | rel=0.0340 | PASS |
| violation_fraction | 0.00 | 0.00 | abs=0.0000 | PASS |
| total_violations | 0.00 | 0.00 | abs=0 | — |
| mean_NPE | — | — | 0.0100 | PASS |
| p95_NPE | — | — | 0.0289 | PASS |
| under_motion_violations | — | — | 0 | PASS |

**Failed checks:** *(none)*


---

## 7. Aggregate analysis

### 7a. Mean deltas per configuration (3 clips averaged)

| Config | Pass | Mean rel_accel | Mean rel_limb | Mean rel_jitter | Mean viol Δ | Mean mean_NPE | Mean p95_NPE |
|--------|------|----------------|---------------|-----------------|-------------|---------------|--------------|

| Full ViDiT | 0/3 | 0.1505 | 0.0428 | 0.1722 | 0.0180 | 0.0134 | 0.0370 |
| Full Naive | 2/3 | 0.0395 | 0.0244 | 0.0576 | 0.0360 | 0.0128 | 0.0341 |
| G1 ViDiT | 1/3 | 0.3527 | 0.0250 | 0.3511 | 0.0090 | 0.0138 | 0.0346 |
| G1 Naive | 2/3 | 0.2270 | 0.0398 | 0.2430 | 0.0090 | 0.0134 | 0.0352 |
| G2 ViDiT | 2/3 | 0.1224 | 0.0081 | 0.1267 | 0.0180 | 0.0125 | 0.0318 |
| G2 Naive | 2/3 | 0.0678 | 0.0205 | 0.0530 | 0.0000 | 0.0126 | 0.0311 |
| G3 ViDiT | 1/3 | 0.0600 | 0.0132 | 0.0732 | 0.0360 | 0.0126 | 0.0321 |
| G3 Naive | 1/3 | 0.0692 | 0.0283 | 0.0822 | 0.0180 | 0.0124 | 0.0303 |
| Hybrid ViDiT | 2/3 | 0.1343 | 0.0185 | 0.1625 | 0.0000 | 0.0126 | 0.0322 |
| Hybrid Naive | 1/3 | 0.1115 | 0.0340 | 0.1447 | 0.0180 | 0.0115 | 0.0300 |

### 7b. Per-check failure counts (30 runs = 10 configs × 3 clips)

| Check | Failures | Rate | Primary cause on human2 |
|-------|----------|------|-------------------------|

| `acceleration_variance` | 11/30 | 37% | ViDiT full/hybrid motion1; G1/G3 isolates |
| `mean_limb_length_std` | 1/30 | 3% | Full ViDiT motion2 only (0.102 vs 0.10 limit) |
| `var_accel_norm` | 10/30 | 33% | Tracks acceleration failures |
| `violation_fraction` | 9/30 | 30% | Full naive motion1; hybrid naive motion1–2 |
| `mean_NPE` | 0/30 | 0% | Never fails on human2 |
| `p95_NPE` | 0/30 | 0% | Never fails on human2 |
| `under_motion_violations` | 0/30 | 0% | Never fails on human2 |

### 7c. Group sensitivity (sweep, mean rel_accel)

| Group | Blocks | ViDiT mean rel_accel | Naive mean rel_accel | ViDiT pass | Naive pass |
|-------|--------|----------------------|----------------------|------------|------------|

| group1 | 1–13 | 0.3527 | 0.2270 | 1/3 | 2/3 |
| group2 | 14–27 | 0.1224 | 0.0678 | 2/3 | 2/3 |
| group3 | 28–40 | 0.0600 | 0.0692 | 1/3 | 1/3 |

---

## 8. Hybrid vs full quant (human2)

| Clip | Full ViDiT | Hybrid ViDiT | Full Naive | Hybrid Naive |
|------|------------|--------------|------------|--------------|

| `human2_motion1` | FAIL | FAIL | FAIL | FAIL |
| `human2_motion2` | FAIL | PASS | PASS | FAIL |
| `human2_motion3` | FAIL | PASS | PASS | PASS |

### 8a. Hybrid ViDiT improvements

| Clip | Full ViDiT issue | Hybrid ViDiT result |
|------|------------------|---------------------|
| `human2_motion1` | accel + jitter + viol fail | Still fails accel/jitter; **viol fixed** (0.000) |
| `human2_motion2` | limb_std 0.102 (limit 0.10) | **PASS** — limb_std 0.034 |
| `human2_motion3` | accel 0.149, jitter 0.153 | **PASS** — accel 0.078, jitter 0.114 |

### 8b. Hybrid naive regressions

| Clip | Full Naive | Hybrid Naive | Notes |
|------|------------|--------------|-------|
| `human2_motion1` | FAIL (viol only) | FAIL (accel + jitter + viol) | Hybrid **worse** — smoothness degraded |
| `human2_motion2` | PASS | FAIL (viol only, Δ=0.027) | Near-miss: 0.007 over viol threshold |
| `human2_motion3` | PASS | PASS | Hybrid naive accel 0.008 — excellent |

### 8c. Hybrid vs best isolate

| Clip | Best isolate pass | Hybrid result |
|------|-------------------|---------------|
| `human2_motion1` | **G2 naive PASS** | Hybrid naive + ViDiT both FAIL |
| `human2_motion2` | G2 naive PASS (best metrics) | Hybrid ViDiT PASS; hybrid naive near-miss |
| `human2_motion3` | Full naive PASS (best accel) | Both hybrids PASS |

---

## 9. Naive vs ViDiT (human2, paired)

| Clip | ViDiT strength | Naive strength | Full ViDiT | Full Naive |
|------|----------------|----------------|------------|------------|
| `human2_motion1` | Hybrid: no new violations | Best accel/jitter (0.03) | FAIL | FAIL (viol only) |
| `human2_motion2` | Best accel/jitter (0.004) | Passes gate | FAIL (limb 0.102) | PASS |
| `human2_motion3` | — | Near-zero accel delta | FAIL (accel/jitter) | PASS |

**Totals across 5 ViDiT vs 5 naive configs (15 runs each):** ViDiT **6/15** passes, Naive **8/15** passes.

---

## 10. Practical ranking (human2)

| Rank | Config | Pass | Best for |
|------|--------|------|----------|
| 1 | Full naive W8A8 | 2/3 | motion2 + motion3; motion1 fails only on violations |
| 1 | Hybrid ViDiT | 2/3 | motion2 + motion3; fixes full ViDiT failures |
| 3 | G2 naive (sweep) | 2/3 | **Only config that passes motion1** |
| 3 | G1 naive / G2 ViDiT | 2/3 each | Different clips fail |
| 5 | Hybrid naive | 1/3 | motion3 excellent; motion2 near-miss |
| 6 | Full ViDiT | 0/3 | — |

**No single configuration passes all 3 human2 clips.**

---

## 11. Conclusions

### Was the kinematic gate useful on human2?

| Role | Verdict | Evidence |
|------|---------|----------|
| Diagnostic tool | **Yes** | Separated accel/jitter failures (ViDiT) from violation failures (naive) |
| Layer prioritization | **Yes** | Group1 isolate mean rel_accel 3–5× worse than group2/3 |
| Hybrid validation | **Partial** | ViDiT hybrid 0/3→2/3; naive hybrid 2/3→1/3 |
| Ship-ready guarantee | **No** | Best configs still 2/3; motion1 unsolved except G2 isolate |

### Recommended next steps

1. **Re-gate full 12-clip grid** with hybrid ViDiT and hybrid naive (G1 FP16, G2+G3 INT8).
2. **Sub-sweep group1** (blocks 1–6 vs 7–13) to fix `human2_motion1` without giving up G2+G3 compression.
3. **Investigate G2 naive motion1 pass** — consider G1 partial FP16 + G2 INT8 + G3 INT8 variant.
4. **Tune hybrid naive motion2** — violation Δ=0.027 is only 0.007 over threshold.
5. Keep **`human2_motion1`** as a permanent stress clip in the gate set.

---

## 12. Log file index

| Config | Log path | Summary JSON |
|--------|----------|--------------|
| Full ViDiT | `logs/kinematic_gate_grid/human2_motion*.json` | `logs/kinematic_gate_grid.json` |
| Full Naive | `logs/kinematic_gate_w8a8_grid/human2_motion*.json` | — |
| G1 ViDiT | `logs/kinematic_gate_sweep_group1_vidit/` | `logs/kinematic_gate_sweep_group1_vidit.json` |
| G1 Naive | `logs/kinematic_gate_sweep_group1_naive/` | — |
| G2 ViDiT | `logs/kinematic_gate_sweep_group2_vidit/` | `logs/kinematic_gate_sweep_group2_vidit.json` |
| G2 Naive | `logs/kinematic_gate_sweep_group2_naive/` | — |
| G3 ViDiT | `logs/kinematic_gate_sweep_group3_vidit/` | `logs/kinematic_gate_sweep_group3_vidit.json` |
| G3 Naive | `logs/kinematic_gate_sweep_group3_naive/` | — |
| Hybrid ViDiT | `logs/kinematic_gate_hybrid_vidit_grid/` | `logs/kinematic_gate_hybrid_vidit_grid.json` |
| Hybrid Naive | `logs/kinematic_gate_hybrid_grid/` | `logs/kinematic_gate_hybrid_grid.json` |

Pose keypoint cache: `kinematic_cache/keypoints/human2_motion{1,2,3}/`

*Report generated from kinematic gate JSON outputs under `logs/kinematic_gate_*/human2_motion*.json`.*
*Regenerate with: `python scripts/quant/generate_human2_gate_report.py`*
