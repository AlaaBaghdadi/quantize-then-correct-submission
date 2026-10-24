# Temporal Quantization Hypothesis — Experimental Design

This document defines the hypothesis, controlled setups, expected outcomes, and evaluation metrics for isolating a kinematic performance discrepancy between **ViDiT-Q (SmoothQuant-style W8A8)** and **matched naive W8A8** during early-block spatiotemporal attention in DynamiCtrl.

**Related artifacts:**
- Debug notebook: [`notebooks/debug_temporal_quantization.ipynb`](../notebooks/debug_temporal_quantization.ipynb)
- Kinematic gate results: [`DynamiCtrl-PTQ-kinematic-gate-results.md`](DynamiCtrl-PTQ-kinematic-gate-results.md)
- Quantization implementation: [`scripts/quant/vidit_quant.py`](../scripts/quant/vidit_quant.py)

---

## 1. Problem Statement

### SmoothQuant activation folding

ViDiT-Q applies SmoothQuant before W8A8 quantization. For each quantizable linear layer with input activation \(X \in \mathbb{R}^{N \times C}\) and weight \(W \in \mathbb{R}^{C_{\text{out}} \times C}\), a per-input-channel scale vector \(s \in \mathbb{R}^{C}_{+}\) is computed from calibration statistics:

\[
s_j = \frac{\mathrm{act}_j^{\alpha}}{w_j^{1-\alpha}}, \quad \alpha = 0.5
\]

where \(\mathrm{act}_j\) is the calibrated channel absmax and \(w_j\) is the per-channel weight magnitude. The fold is:

\[
Y = (X \operatorname{diag}(s)^{-1}) \cdot (\operatorname{diag}(s)\, W)
\]

At runtime, [`SmoothQuantActivationWrapper`](../scripts/quant/vidit_quant.py) implements this as `linear(x / s)` after `s` has been multiplied into `W`. Naive W8A8 skips this step and quantizes `(X, W)` directly with token-wise dynamic activation scales.

### Mathematical tension with temporal signals

DynamiCtrl does not expose a separate `TemporalAttention` module. Spatiotemporal mixing occurs in **`DynamiCtrlBlock.attn1`** — a joint MMDiT-style `Attention` block (`CogVideoXAttnProcessor2_0`) over concatenated text and video tokens, with 3D RoPE on the video stream.

Video tokens encode **space × time**. Channels that carry **high-frequency temporal transients** — sudden pose changes, rapid limb acceleration, frame-to-frame motion discontinuities — often exhibit **large, sparse activation spikes** on a subset of input channels during early denoising steps and early transformer blocks (Group 1, blocks 1–13).

SmoothQuant calibrates \(s_j\) from **channel absmax aggregated across clips and timestep buckets**. Channels dominated by rare temporal spikes receive **large \(s_j\)**. Dividing activations by \(s_j\) at runtime **compresses those spikes toward the representable INT8 range**, even though the fold is mathematically exact in FP32.

The hypothesis is that this **dynamic dampening**:

1. Preserves **spatial** structure in lower-variance channels (appearance, local texture).
2. **Flattens temporal activation peaks** that drive sharp frame-to-frame attention routing.
3. Causes **temporal attention smudging** (higher entropy, lower peak contrast) relative to FP16 and naive W8A8.
4. Manifests downstream as **frame-to-frame jitter** and elevated **acceleration variance** in kinematic gate metrics.

### Empirical motivation

In the Group-1 sensitivity sweep (`human2` × 3 motions, blocks 1–13 only INT8), ViDiT-Q underperformed matched naive W8A8 on the kinematic gate (1/3 vs 2/3 passes). **`human2_motion1`** is among the hardest clips in the full 12-clip grid. This experiment targets that clip to test whether SmoothQuant-specific temporal flattening in early `attn1` layers is a plausible mechanism.

---

## 2. Experimental Setup

### Target clip

| Field | Value |
|-------|-------|
| Clip ID | `human2_motion1` |
| Reference image | `./assets/human2.jpg` |
| Driving video | `./assets/motion1.mp4` |
| Calibration | `calib_data/human2_motion1_vidit.pt` |

### Inference conditions (fixed across all setups)

From [`configs/quant/vidit_ptq.json`](../configs/quant/vidit_ptq.json):

| Parameter | Value |
|-----------|-------|
| Resolution | 768 × 1360 |
| Frames | 37 |
| Pose control | `padaln` |
| Guidance scale | 3.0 |
| Seed | 42 |
| Dtype | bfloat16 |
| Denoising steps | **10** (debug budget; same noise init via `make_generator(42)`) |

### Quantization scope

Only **Group 1** blocks **1–13** are quantized; all other quantizable layers remain FP16. Blocks **0** and **41** are always protected.

| Setup | Label | Application |
|-------|-------|-------------|
| **A** | FP16 ground truth | No quantization |
| **B** | Group-1 naive W8A8 | `apply_w8a8(dit, all_quantize_names, fp16_layers=all_quantize_names - group1_layers)` |
| **C** | Group-1 ViDiT-Q W8A8 | `apply_vidit_w8a8(dit, all_quantize_names, calib_path, smoothquant_alpha=0.5, actual_int8_layers=group1_layers)` |

Naive W8A8 uses `torchao` `int8_dynamic_activation_int8_weight` without SmoothQuant. ViDiT-Q adds per-channel folding on the same Group-1 layer set.

### Instrumentation scope

Hooks and custom attention processors target **blocks 1–5** only (subset of Group 1):

| Target | Module path |
|--------|-------------|
| Query projection | `transformer_blocks.{i}.attn1.to_q` |
| Key projection | `transformer_blocks.{i}.attn1.to_k` |
| Value projection | `transformer_blocks.{i}.attn1.to_v` |
| Post-softmax attention | Custom `InstrumentedCogVideoXAttnProcessor2_0` on `attn1` (explicit softmax; SDPA does not expose weights) |

Each setup runs the **same** precomputed `image` and `control_video` tensors through the pipeline for 10 denoising steps. Captured tensors are keyed by `(block_idx, denoise_step_index)`.

---

## 3. What to Expect

### If the hypothesis holds

| Observable | FP16 | Naive W8A8 | ViDiT-Q W8A8 |
|------------|------|------------|--------------|
| Q/K/V temporal MSE vs FP16 | 0 (baseline) | Moderate | **Highest** |
| Q/K/V spatial MSE vs FP16 | 0 | Low–moderate | Low–moderate (smaller gap vs naive) |
| Temporal attention heatmaps | Sharp diagonal / localized peaks | Close to FP16 | **Washed contrast**, broader peaks |
| Mean attention entropy (temporal rows) | Baseline | Between FP16 and ViDiT | **Highest** (smudged distributions) |
| Cosine similarity (temporal) | 1.0 | Higher than ViDiT | **Lowest** |

Qualitatively, ViDiT temporal attention maps should look **flatter** than naive maps in side-by-side heatmaps. Naive W8A8 should sit **between** FP16 and ViDiT on temporal metrics but **closer to FP16 on spatial metrics**.

### If the hypothesis fails

- ViDiT and naive temporal MSE/cosine vs FP16 are **statistically indistinguishable**.
- Attention entropy differences across setups are **negligible** (< 0.05 nats).
- Spatial and temporal degradation are **uniform** (similar relative gaps), suggesting generic INT8 error rather than SmoothQuant-specific temporal dampening.
- Q/K/V spike dampening is **not visible** in temporal activation profiles.

A failed hypothesis implies the kinematic gap on `human2_motion1` arises elsewhere (FFN layers, later blocks, VAE, pose encoder) or from error compounding across the full 25-step denoising chain rather than early temporal attention flattening.

---

## 4. Evaluation Protocol

### Token layout

Video tokens (excluding text) are reshaped at runtime from the transformer config:

\[
\text{num\_video\_tokens} = L_{\text{seq}} - L_{\text{text}}, \quad T = \left\lfloor \frac{\text{num\_frames}}{\text{patch\_size\_t}} \right\rfloor, \quad S = \frac{\text{num\_video\_tokens}}{T}
\]

Activations \(Q, K, V\) are sliced to the video segment `[:, L_text:, :]` and reshaped to \((T, S, D)\).

### Decomposed attention matrices

From the video–video block of the post-softmax attention matrix \(A \in \mathbb{R}^{L_v \times L_v}\) (mean over heads and batch), reshape indices as \((t_q, s_q, t_k, s_k)\):

**Temporal aggregation** (frame-to-frame routing):

\[
A_{\text{temp}}[t_q, t_k] = \frac{1}{S^2} \sum_{s_q=1}^{S} \sum_{s_k=1}^{S} A[t_q, s_q, t_k, s_k]
\]

**Spatial aggregation** (within-frame routing):

\[
A_{\text{spat}}[s_q, s_k] = \frac{1}{T^2} \sum_{t_q=1}^{T} \sum_{t_k=1}^{T} A[t_q, s_q, t_k, s_k]
\]

### Mean Squared Error (MSE)

For variant tensor \(X\) and FP16 baseline \(B\) of the same shape:

\[
\mathrm{MSE}(X, B) = \frac{1}{N} \sum_{i=1}^{N} (X_i - B_i)^2
\]

Computed separately on:

- \(A_{\text{temp}}\) and \(A_{\text{spat}}\) (attention)
- Temporal profiles of \(Q, K, V\): `tensor.reshape(T, S, D).mean(dim=(1, 2))` → vector of length \(T\)
- Spatial profiles: `tensor.reshape(T, S, D).mean(dim=(0, 2))` → vector of length \(S\)

Report per block (1–5), per denoise step, and **macro-averaged** over all captures.

### Cosine similarity

For flattened vectors \(x, b\):

\[
\cos(x, b) = \frac{x \cdot b}{\|x\| \, \|b\|}
\]

Applied to the same tensors as MSE. Values near 1.0 indicate directional agreement with FP16; lower values indicate structural divergence.

### Shannon entropy (attention smudging)

For each temporal attention row \(p = A_{\text{temp}}[t, :]\) (already nonnegative, sums to 1):

\[
H(p) = -\sum_{k=1}^{T} p_k \log(p_k + \epsilon), \quad \epsilon = 10^{-8}
\]

Report **mean entropy** over query frames \(t\), blocks 1–5, and 10 denoise steps. Higher entropy indicates **more uniform** (smudged) attention; lower entropy indicates **sharper** peaky routing.

### Hypothesis verdict (diagnostic thresholds)

The hypothesis is considered **supported** when **all** of the following hold on macro-averaged temporal attention metrics:

| Criterion | Condition |
|-----------|-----------|
| Temporal MSE gap | \(\mathrm{MSE}_{\text{vidit}}^{\text{temp}} > \mathrm{MSE}_{\text{naive}}^{\text{temp}}\) by **> 10% relative** |
| Temporal cosine gap | \(\cos_{\text{vidit}}^{\text{temp}} < \cos_{\text{naive}}^{\text{temp}} - 0.02\) |
| Entropy smudging | \(\bar{H}_{\text{vidit}} > \bar{H}_{\text{naive}} + 0.05\) nats |
| Spatial specificity | \(\|\mathrm{MSE}_{\text{vidit}}^{\text{spat}} - \mathrm{MSE}_{\text{naive}}^{\text{spat}}\| < \frac{1}{2}\|\mathrm{MSE}_{\text{vidit}}^{\text{temp}} - \mathrm{MSE}_{\text{naive}}^{\text{temp}}\|\) |

These thresholds are **diagnostic**, not acceptance gates. Partial support (e.g., entropy + MSE but not cosine) should be reported with per-block breakdowns.

### Outputs

The notebook writes artifacts to:

```
outputs/temporal_quant_debug/human2_motion1/
├── metrics.json
├── metrics_summary.csv
└── heatmaps/
    └── block{N}_step{S}_temporal_attn.png
```
