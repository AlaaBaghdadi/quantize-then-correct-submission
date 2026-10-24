# Lightweight DynamiCtrl via Kinematics-Gated PTQ

A zero-training plan to make **DynamiCtrl-5B faster on a single 24–48 GB GPU**, using your existing pose metrics as the automatic acceptance gate. No kinematic *loss* is needed anywhere — kinematics only ever appears as a pass/fail check on generated output, which is fully automatic.

## Why this fits your constraints

| Constraint | Decision |
|---|---|
| Compute | 1× consumer GPU (24–48 GB) |
| Target | DynamiCtrl-5B, optimize for inference latency |
| Method | Post-training quantization only, **no fine-tuning** |
| Kinematics | Evaluation gate, not a training signal |
| Budget | 1–2 days, almost all of it calibration + evaluation, not training |

The only model in DynamiCtrl that dominates inference cost is the denoising DiT — a CogVideoX-5B backbone of ~42 transformer blocks (hidden ≈3072, 48 heads), run once per diffusion step. The VAE and the shared image/pose encoder run once. So all compression effort goes into the DiT linear layers and their activations.

## Core method: ViDiT-Q-style mixed-precision PTQ

Plain INT8 PTQ fails on video DiTs because activation ranges vary wildly across three axes: across **tokens**, across **channels**, and across **diffusion timesteps**. The published method that already solved this for exactly this model family (Open-Sora, Latte — both CogVideoX-like DiTs) is **ViDiT-Q**, which reaches lossless W8A8 and usable W4A8 with 2–2.5× memory saving and 1.4–1.7× end-to-end speedup. Adopt its three ideas rather than reinventing them:

1. **Token-wise dynamic activation quantization.** Compute the activation scale per token (per spatial-temporal position) at runtime instead of one scale per tensor. This is what kills the "unreadable output" failure mode of naive W8A8.

2. **Channel smoothing (SmoothQuant migration).** Migrate activation outliers into the weights with a per-channel scale `s`, so `(X / s)·(s·W)` is mathematically identical but both factors quantize cleanly. Calibrated once, folded into weights — free at inference.

3. **Timestep-aware / metric-decoupled mixed precision.** A handful of layers and a handful of early denoising timesteps are disproportionately sensitive. Keep those in FP16; push everything else to INT8 (or INT4 weights). This is the lever your kinematic gate will drive (below).

**Recommended starting point:** W8A8 everywhere except the protected set below. If the kinematic gate passes with margin, try W4A8 (4-bit weights, 8-bit activations) on the least-sensitive blocks for extra memory headroom.

### Always keep in FP16 (the protected set)
- VAE encoder/decoder (runs once, tiny share of cost, large quality impact).
- The shared image+pose latent encoder (DynamiCtrl's identity/pose fidelity lives here).
- AdaLN / modulation MLPs and time-embedding projections (small, extremely sensitive).
- First and last 1–2 DiT blocks.
- The text/joint-text cross-attention projections — DynamiCtrl's "Joint-text" control is what you don't want to corrupt.

## Where kinematics enters: the gate, not a loss

You said the correction step only works when hand-tuned per video, so it can't be a differentiable objective. Don't try. Instead use the metrics you already compute automatically (Section 4 of your report) as a **config acceptance test**.

Define a fixed calibration set — reuse the 6 DeepFashion identities (man_1…woman_3). For the FP16 model, generate each video once and record the baseline vector:
- `acceleration_variance` (temporal)
- `mean_limb_length_std`
- `total_violations` / `violation_fraction`
- `var_accel_norm` (jitter)

These come straight from the MediaPipe+OpenPose pipeline you already have — no correction tuning required, because you're measuring the *generated* skeleton's stability, not correcting it.

For any candidate quantization config Q, accept it iff, averaged over the 6 videos:

```
PSNR(Q vs FP16 output)        ≥ 30 dB        # visual fidelity vs the fp16 model
Δ acceleration_variance       ≤ +10%
Δ mean_limb_length_std        ≤ +10%
violation_fraction(Q)         ≤ violation_fraction(FP16)
Δ var_accel_norm              ≤ +15%
```

Tune the thresholds to taste — the point is they're computed, not eyeballed. This turns "is the compressed model still kinematically OK?" into a green/red number, which is the automatic role kinematics can actually play here.

### Kinematics-driven sensitivity ranking (the clever part)
Rather than guessing which blocks tolerate 4-bit, let the metrics decide:

1. Start from the all-FP16 baseline.
2. One block at a time, drop it to W8A8 (then W4A8), regenerate the 6 videos, recompute the metric vector.
3. Record each block's **kinematic degradation cost** = weighted sum of the Δs above.
4. Greedily assign the lowest bitwidth to the blocks with the smallest cost; stop lowering a block the moment it breaks the gate.

This is a few dozen forward-only evaluation passes — no gradients, no training — and it produces a mixed-precision map justified directly by motion plausibility rather than by a generic weight-norm heuristic.

## Two free add-ons (still zero training, same gate)

PTQ alone mostly buys memory bandwidth; on a 4090-class card the wall-clock win is real but modest unless you also reduce work. Both of these are training-free and pass through the identical kinematic gate, so include them if latency is the priority:

- **Fewer / better-scheduled steps.** Swap the sampler to a DPM-Solver++ style scheduler and cut steps (e.g. 50 → 20–25). This is often the single biggest latency win and costs no training. Gate it the same way.
- **Activation caching across steps (training-free "pruning in time").** DiT features change little between adjacent denoising steps; cache and reuse block outputs on a fraction of steps (the ∆-DiT / block-cache idea). This is the closest training-free analog to pruning and is gated identically. Treat as optional/stretch.

I've kept these secondary because you chose PTQ-only; flag me if you want the plan re-centered on them.

## Tooling

- **Quantization:** `torchao` (native INT8/INT4 + dynamic per-token activation quant) or `optimum-quanto`; or port the [ViDiT-Q repo](https://github.com/thu-nics/ViDiT-Q) directly since it already targets CogVideoX-family DiTs.
- **Real speedup needs INT8 GEMM kernels** — make sure the backend actually dispatches to INT8 tensor-core matmuls (torchao `int8_dynamic_activation_int8_weight`, or TensorRT). Weight-only INT8 with FP16 compute saves memory but little latency, so verify kernel dispatch with a microbenchmark before trusting the speedup.
- **Metrics harness:** wrap your existing pose-evaluation script as a function that returns the metric vector for a video directory, so the gate runs unattended in the sensitivity sweep.

## Two-day schedule

**Day 1 — baseline + plain W8A8**
- Stand up FP16 inference; generate the 6 calibration videos; record baseline metric vectors and per-video latency/VRAM. (~2–3 h)
- Collect calibration activations: 1–2 generations, capturing activations across all timesteps (store a representative subset of steps). (~1–2 h)
- Apply SmoothQuant channel scales + token-wise W8A8 with the protected set in FP16. Regenerate the 6 videos. Run the gate. (~3–4 h)
- If W8A8 passes: you already have a shippable lighter model. Stop here if satisfied.

**Day 2 — mixed precision + (optional) acceleration**
- Run the kinematics-driven sensitivity sweep; build the per-block bitwidth map; push tolerant blocks to W4A8. Re-gate. (~4–6 h)
- Optionally add step reduction and/or block caching; re-gate each. (~2–3 h)
- Final report: latency, VRAM, and the kinematic metric table (FP16 vs final config) on all 6 videos. (~1 h)

## What you get

A DynamiCtrl-5B that fits comfortably in 24 GB with ~2–2.5× lower DiT memory and a measured ~1.4–1.7× (more if you add step reduction) speedup, with a one-page table proving the compressed model's motion metrics stayed inside your tolerances — i.e. kinematics enforced automatically, no per-video tuning, no training run longer than calibration.

---

## Open questions before I turn this into runnable code

1. **Does the gate use absolute or relative thresholds?** I defaulted to "relative to FP16 output." If you'd rather hold to the *original report's* per-video numbers (Section 4.3) as fixed targets, say so — it changes pass/fail.
2. **Acceptable quality floor.** Is the 30 dB PSNR / +10% motion-variance bar right, or do you want it stricter (closer to lossless W8A8) or looser (chase max speedup)?
3. **W4 or W8 as the floor?** Comfortable risking 4-bit weights on the tolerant blocks, or keep the whole thing at INT8 for safety on the first pass?
4. **Calibration set.** Reuse the same 6 DeepFashion videos, or do you have a few held-out motion clips you'd rather calibrate/gate on so the result generalizes?
5. **Speed vs memory priority.** If forced to choose, is the goal "runs at all in 24 GB" or "lowest latency on a 48 GB card"? That decides whether INT8-GEMM kernels + step reduction are mandatory or optional.
6. **Latency target.** Is there a concrete number (e.g. seconds per video, or X% of current) that defines "done"?

## Sources
- [DynamiCtrl (arXiv 2503.21246)](https://arxiv.org/abs/2503.21246) · [GitHub](https://github.com/gulucaptain/dynamictrl)
- [ViDiT-Q (arXiv 2406.02540, ICLR'25)](https://arxiv.org/abs/2406.02540) · [GitHub](https://github.com/thu-nics/ViDiT-Q)
- [CogVideoX (Transformer3D in diffusers)](https://huggingface.co/docs/diffusers/en/api/models/cogvideox_transformer3d)
