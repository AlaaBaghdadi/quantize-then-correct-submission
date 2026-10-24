# DynamiCtrl PTQ Integration Steps

Practical guide for wiring post-training quantization (PTQ) into **DynamiCtrl** before running the plan in [`DynamiCtrl-PTQ-method.md`](DynamiCtrl-PTQ-method.md).

The high-level PTQ plan assumes you can already load the DiT, inventory layers, calibrate with pose, quantize, and benchmark. **This document covers the 1–3 days of integration work** that must happen first. The Colab notebook ([`DynamiCtrl.ipynb`](DynamiCtrl.ipynb)) proves FP16 inference works interactively; these steps turn that into **repeatable scripts** suitable for overnight calibration and sweeps.

---

## Why integration is needed

PTQ tools (ViDiT-Q, torchao, optimum-quanto) do not quantize “a 5B video model” in the abstract. They quantize a **specific program**: a class name, checkpoint layout, forward pass, and inference script.

| | Stock CogVideoX (diffusers) | DynamiCtrl |
|---|---|---|
| **Class** | `CogVideoXTransformer3DModel` | `DynamiCtrlTransformerModel` |
| **Source** | Hugging Face diffusers | `dynamictrl/models/model_dynamictrl.py` |
| **Task** | Text → video | Image + driving pose + text → animated video |
| **Extra modules** | Standard blocks | `PoseAdaptiveLN` (`control_norm1`) on all 42 blocks, `ofs` embedding, dual `patch_embed` pass for pose |

ViDiT-Q ships scripts for **Open-Sora** and **PixArt**, not DynamiCtrl. You must either **port** ViDiT-Q’s pipeline or **hook** torchao into DynamiCtrl’s actual code path.

### Verified model facts (from HF checkpoint + source)

| Component | Parameters (approx.) | Notes |
|-----------|---------------------|-------|
| DiT (`DynamiCtrlTransformerModel`) | **~5.77B** | 42 layers, 48 heads × 64 dim → 3072 hidden |
| Text encoder (T5-XXL) | **~4.76B** | Runs once per generation |
| VAE (CogVideoX) | **~431M** | Encode/decode once |
| **Full pipeline** | **~10.9B** | “DynamiCtrl-5B” refers to the DiT only |

---

## Big picture

Before the PTQ plan’s Day 1 / Day 2 work, build a small **quantization toolchain**:

```
Standalone Python scripts
        ↓
Load DiT (or full pipeline)
        ↓
Inventory Linear layers (quantize vs protect)
        ↓
Collect calibration activations (with pose, multiple timesteps)
        ↓
Apply W8A8 quant → generate video → debug
        ↓
Benchmark sec/step + verify INT8 GEMM dispatch
```

---

## Summary table

| Step | What it is | Est. time | Difficulty | Main deliverable |
|------|------------|-----------|------------|------------------|
| **1. Load DiT** | Standalone script loads `DynamiCtrlTransformerModel` outside the notebook | 2–4 h | ★★☆☆☆ | `load_dit.py` runs, prints ~5.77B params |
| **2. Layer list** | Inventory every `nn.Linear`, tag QUANTIZE vs PROTECT | 2–3 h | ★☆☆☆☆ | `quantize_layers.txt`, `protected_layers.txt` |
| **3. Calibration** | Run pipeline with pose, record activation stats per timestep | 4–8 h | ★★★★☆ | `calib_data/*.pt` from real image+motion clips |
| **4. Naive W8A8** | Apply quant, generate video, debug artifacts | 4–8 h | ★★★★☆ | First non-garbage quantized video + PSNR |
| **5. Speed verify** | Benchmark sec/step, confirm INT8 kernels | 2–4 h | ★★★☆☆ | FP16 vs W8A8 timing + VRAM table |

**Total integration:** ~1–3 working days before the 2-day PTQ schedule in the method doc realistically starts.

---

## Suggested folder layout

Create these under the cloned DynamiCtrl repo (or your project root):

```
DynamiCtrl/
  scripts/quant/
    load_dit.py           # Step 1
    list_layers.py        # Step 2
    collect_calib.py      # Step 3
    apply_w8a8.py         # Step 4
    benchmark.py          # Step 5
  configs/quant/
    quantize_layers.txt
    protected_layers.txt
  calib_data/
    clip1.pt
```

---

## Step 1 — Load DynamiCtrl DiT in a standalone quant script

**Estimated time:** 2–4 hours  
**Complexity:** Low–medium

### What it is

A normal `.py` file (not a notebook cell) that:

1. Adds the DynamiCtrl repo to `PYTHONPATH`
2. Loads the **transformer (DiT) weights** onto GPU
3. Prints parameter count, dtype, and memory — proving the quant pipeline has a target

“Standalone” matters because calibration and block sweeps need to run for hours without Colab disconnects and must be callable from the shell.

### Why you need it

Quantization tools need a live PyTorch object: `pipe.transformer` or `DynamiCtrlTransformerModel`. Nothing in ViDiT-Q or torchao loads DynamiCtrl for you.

### How to do it

#### Option A — Load full pipeline, grab DiT (simplest)

Matches what the notebook does via `dynamictrl_inference.py`:

```python
import sys
sys.path.insert(0, "/path/to/DynamiCtrl")

import torch
from dynamictrl.pipelines.dynamictrl_pipeline import DynamiCtrlAnimationPipeline

MODEL = "./checkpoints/DynamiCtrl"  # or "gulucaptain/DynamiCtrl"
DTYPE = torch.bfloat16

pipe = DynamiCtrlAnimationPipeline.from_pretrained(MODEL, torch_dtype=DTYPE)
pipe.to("cuda")

dit = pipe.transformer  # DynamiCtrlTransformerModel, ~5.77B params
print(type(dit))
print(sum(p.numel() for p in dit.parameters()) / 1e9, "B params")
```

#### Option B — Load DiT only (lighter for dry runs)

DynamiCtrl provides `from_pretrained_2d` in `model_dynamictrl.py`:

```python
from dynamictrl.models.model_dynamictrl import DynamiCtrlTransformerModel

dit = DynamiCtrlTransformerModel.from_pretrained_2d(
    "./checkpoints/DynamiCtrl",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)
dit.to("cuda")
# Prints "### All Parameters: XXXX M" automatically
```

Option B uses less RAM (no T5/VAE), but **Step 3 calibration still needs the full pipeline** because activations depend on real latents from VAE + pose.

### Example script: `scripts/quant/load_dit.py`

```python
#!/usr/bin/env python3
"""Step 1: verify DynamiCtrl DiT loads outside the notebook."""
import argparse
import sys
from pathlib import Path

import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="./checkpoints/DynamiCtrl")
    parser.add_argument("--repo-root", default=".", help="DynamiCtrl git clone root")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--dit-only", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo))

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    if args.dit_only:
        from dynamictrl.models.model_dynamictrl import DynamiCtrlTransformerModel
        dit = DynamiCtrlTransformerModel.from_pretrained_2d(
            args.model_path, subfolder="transformer", torch_dtype=dtype
        )
    else:
        from dynamictrl.pipelines.dynamictrl_pipeline import DynamiCtrlAnimationPipeline
        pipe = DynamiCtrlAnimationPipeline.from_pretrained(args.model_path, torch_dtype=dtype)
        dit = pipe.transformer

    dit.to("cuda")
    n_params = sum(p.numel() for p in dit.parameters())
    print(f"Model class: {type(dit).__name__}")
    print(f"Parameters:  {n_params / 1e9:.2f} B")
    print(f"Dtype:       {dtype}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        _ = next(dit.parameters()).device  # touch GPU
        print(f"GPU:         {torch.cuda.get_device_name()}")
    print("OK: DiT loaded successfully.")

if __name__ == "__main__":
    main()
```

Run:

```bash
cd /path/to/DynamiCtrl
export PYTHONPATH=$PWD
python scripts/quant/load_dit.py --model-path ./checkpoints/DynamiCtrl
```

### Definition of done

- [ ] Script exits 0 without missing-key errors (or you document any `strict=False` skips)
- [ ] Printed param count ≈ **5.77B**
- [ ] GPU memory for DiT weights ≈ **11–12 GB** in bf16/fp16

### Common issues

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: dynamictrl` | Set `PYTHONPATH` to DynamiCtrl repo root |
| Custom diffusers class not found | Use DynamiCtrl’s pinned `diffusers==0.32.0` |
| `huggingface-hub` version conflicts | Match `requirements.txt`; upgrade transformers if needed |
| OOM on load | Use `--dit-only` first; enable `pipe.vae.enable_slicing()` when loading full pipeline |

### Complexity breakdown

- **Easy:** Loading weights (HF layout already supported)
- **Medium:** Environment setup outside Colab, paths, CUDA
- **Hard:** None unless dependency conflicts

---

## Step 2 — List quantizable Linear layers + mark protected ones

**Estimated time:** 2–3 hours  
**Complexity:** Low (mostly mechanical)

### What it is

A **map of every matrix-multiply layer** in the DiT:

- **QUANTIZE** — large linear layers (attention projections, FFN) → target for INT8/W4
- **PROTECT** — small or sensitive layers → stay FP16
- **SKIP** — LayerNorm, SiLU, dropout, rotary embeddings (not linear layers)

PTQ is mixed precision: quantize the bulk, protect the fragile parts.

### Why you need it

Without a layer list you are guessing. The kinematic gate and ViDiT-Q-style sweeps both need named modules. DynamiCtrl adds **`control_norm1`** (pose injection) on every block — not present in stock CogVideoX protect lists.

### DynamiCtrl DiT structure (from source)

Each of **42 blocks** (`DynamiCtrlBlock`, index `0` … `41`):

| Submodule | Linear layers? | Default decision |
|-----------|----------------|------------------|
| `norm1`, `norm2` | Yes (AdaLN / `CogVideoXLayerNormZero`) | **PROTECT** — timestep modulation |
| `control_norm1` | Yes (`PoseAdaptiveLN.linear`) | **PROTECT** — pose injection (DynamiCtrl-specific) |
| `attn1` | Yes (Q, K, V, out) | **QUANTIZE** — main compute |
| `ff` | Yes (2-layer MLP) | **QUANTIZE** |
| `attn2`, `norm3` | Only if `pose_control_function == "cross_attention"` | **PROTECT** if present; notebook uses `padaln` |

Top-level `DynamiCtrlTransformerModel`:

| Module | Default decision |
|--------|------------------|
| `patch_embed` | **PROTECT** — image/pose → tokens (runs twice for pose stream) |
| `time_embedding`, `time_proj` | **PROTECT** |
| `ofs_embedding`, `ofs_proj` | **PROTECT** — DynamiCtrl-specific |
| `norm_out` (AdaLayerNorm) | **PROTECT** |
| `proj_out` | **PROTECT** initially; can try quantize later |
| `norm_final` | LayerNorm only → **SKIP** |
| `transformer_blocks.0.*` | **PROTECT** (first block) |
| `transformer_blocks.41.*` | **PROTECT** (last block) |

Expect **~300–350 Linear layers** to quantize (similar to CogVideoX-5B) plus **42** `control_norm1` modules to protect.

### Protected name patterns

```python
PROTECTED_PATTERNS = [
    "patch_embed",
    "time_embedding",
    "time_proj",
    "ofs_embedding",
    "ofs_proj",
    "norm_out",
    "control_norm1",
    "transformer_blocks.0.",
    "transformer_blocks.41.",
]
```

Adjust last-block index if `num_layers` in config changes.

### Example script: `scripts/quant/list_layers.py`

```python
#!/usr/bin/env python3
"""Step 2: inventory nn.Linear modules and classify quantize vs protect."""
import argparse
import json
import sys
from pathlib import Path

import torch.nn as nn

PROTECTED_PATTERNS = [
    "patch_embed",
    "time_embedding",
    "time_proj",
    "ofs_embedding",
    "ofs_proj",
    "norm_out",
    "control_norm1",
    "transformer_blocks.0.",
    "transformer_blocks.41.",
]

def classify(name: str) -> str:
    for pat in PROTECTED_PATTERNS:
        if pat in name:
            return "PROTECT"
    return "QUANTIZE"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--model-path", default="./checkpoints/DynamiCtrl")
    parser.add_argument("--out-dir", default="./configs/quant")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.repo_root).resolve()))
    from dynamictrl.models.model_dynamictrl import DynamiCtrlTransformerModel

    dit = DynamiCtrlTransformerModel.from_pretrained_2d(
        args.model_path, subfolder="transformer", torch_dtype=torch.bfloat16
    )

    quantize, protect, skip = [], [], []
    quant_params, protect_params = 0, 0

    for name, module in dit.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        n = module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
        tag = classify(name)
        entry = {"name": name, "in": module.in_features, "out": module.out_features, "params": n}
        if tag == "QUANTIZE":
            quantize.append(entry)
            quant_params += n
        else:
            protect.append(entry)
            protect_params += n

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "quantize_layers.json").write_text(json.dumps(quantize, indent=2))
    (out / "protected_layers.json").write_text(json.dumps(protect, indent=2))
    (out / "quantize_layers.txt").write_text("\n".join(e["name"] for e in quantize))
    (out / "protected_layers.txt").write_text("\n".join(e["name"] for e in protect))

    print(f"Quantize: {len(quantize)} Linear layers, {quant_params/1e9:.2f}B params")
    print(f"Protect:  {len(protect)} Linear layers, {protect_params/1e9:.2f}B params")
    print(f"Wrote configs to {out}/")

if __name__ == "__main__":
    main()
```

### Definition of done

- [ ] `configs/quant/quantize_layers.txt` and `protected_layers.txt` exist
- [ ] `control_norm1` appears only in **protected**
- [ ] `patch_embed`, `time_embedding`, `ofs_embedding` are **protected**
- [ ] Bulk of ~5.77B params sits in **quantize** list (attention + FFN)

### Complexity breakdown

- **Easy:** Walking `named_modules()`
- **Medium:** Policy choices (how many edge blocks to protect)
- **Hard:** None — this step is documentation for Steps 3–5

---

## Step 3 — Calibration script: pipeline with pose, activations at multiple timesteps

**Estimated time:** 4–8 hours  
**Complexity:** High (hardest integration step)

### What it is

**Calibration** = run the model on a few real examples and record **activation statistics** (min, max, absmax, percentiles) at each quantizable layer.

PTQ uses these stats to pick scales: e.g. “activations range −3…+12 → map to int8 −128…127.”

For DynamiCtrl, calibration **must** include:

- Reference image
- Driving video → DWPose → pose latents
- Text prompt
- Multiple **denoising timesteps** (activations change across the diffusion schedule)

Calibrating without pose produces scales that do not match production inference.

### Why you need it

Weight-only quantization (scales from weights alone) fails on video DiTs. ViDiT-Q uses **activation-aware** PTQ with **token-wise dynamic** activation quant and often **timestep-aware** scaling. You need real activation ranges from the DynamiCtrl forward path.

### Data flow

```
Image + motion video
        ↓
DynamiCtrlAnimationPipeline
        ↓
VAE encode + T5 encode + DWPose
        ↓
Latents + pose latents + prompt embeds
        ↓
Denoising loop (e.g. 25 steps)
        ↓
DynamiCtrlTransformerModel  ← forward hooks record stats
        ↓
calib_data/clip_name.pt
```

### Implementation outline

1. **Start from** `scripts/dynamictrl_inference.py` loading logic (same args as notebook).
2. **Load** quantize layer list from Step 2.
3. **Register forward hooks** on input activations to each quantizable `nn.Linear`.
4. **Run 1–3 calibration clips** (expand to 6 for the kinematic gate later).
5. **Subsample timesteps** — e.g. early / mid / late buckets, or every 5th step.
6. **Save statistics**, not full tensors (full tensors = terabytes).

### Hook pattern

```python
import torch.nn as nn

current_timestep = None  # set from denoising loop each iteration
activations = {}

def make_hook(name):
    def hook(module, inp, out):
        x = inp[0].detach().float()
        bucket = timestep_bucket(current_timestep)  # "early" | "mid" | "late"
        store = activations.setdefault(name, {}).setdefault(bucket, [])
        store.append({
            "absmax": x.abs().max().item(),
            "shape": tuple(x.shape),
        })
    return hook

handles = []
for name, module in dit.named_modules():
    if name in quantize_names and isinstance(module, nn.Linear):
        handles.append(module.register_forward_hook(make_hook(name)))
```

### Timestep bucketing (minimum viable)

```python
def timestep_bucket(t: int, t_max: int = 1000) -> str:
    if t > t_max * 0.66:
        return "early"
    if t > t_max * 0.33:
        return "mid"
    return "late"
```

ViDiT-Q’s full method uses finer timestep-aware channel balancing; start with 3 buckets.

### Storage strategy

| Approach | Disk size | Quality |
|----------|-----------|---------|
| absmax per (layer, timestep bucket) | MB | Good enough to start |
| Histograms / percentiles | Medium | Better for SmoothQuant |
| Full activation tensors | GB+ | Research / ViDiT-Q reproduction only |

### Practical calibration budget (first pass)

| Setting | Value |
|---------|-------|
| Clips | 1–3 image+motion pairs |
| Steps | 10–25 (or sample every 5th) |
| Resolution | Same as production (e.g. 768×1360, 37 frames) |
| CFG | Same as inference (`guidance_scale=3.0` in notebook) |
| Pose mode | `padaln` |

### Threading timestep into hooks

Forward hooks do not receive the diffusion timestep automatically. Options:

1. **Global variable** — set `current_timestep = t.item()` at the top of each denoising iteration (simplest).
2. **Pipeline callback** — use `callback_on_step_end` if you extend the pipeline call.
3. **Small fork** — copy the denoising loop from `dynamictrl_pipeline.py` into `collect_calib.py` for explicit control.

Option 1 is fine for integration; option 3 is cleaner long-term.

### Example CLI sketch: `scripts/quant/collect_calib.py`

```bash
python scripts/quant/collect_calib.py \
  --model-path ./checkpoints/DynamiCtrl \
  --quantize-layers ./configs/quant/quantize_layers.txt \
  --reference-image ./assets/human2.jpg \
  --driving-video ./assets/motion1.mp4 \
  --prompt "Input the test prompt here." \
  --num-inference-steps 25 \
  --width 768 --height 1360 --num-frames 37 \
  --pose-control-function padaln \
  --out ./calib_data/human2_motion1.pt
```

### Definition of done

- [ ] `calib_data/*.pt` files exist with dict keyed by layer name → timestep bucket → stats
- [ ] Run used **pose video** (DWPose logs visible)
- [ ] Stats differ between early vs late buckets (sanity check)
- [ ] Re-run on same clip gives similar absmax (±10%)

### Common issues

| Issue | Cause | Fix |
|-------|-------|-----|
| Empty hook dict | Wrong layer names vs Step 2 list | Re-export layer list from same model load path |
| OOM | Full pipeline + hooks at full res | `pipe.vae.enable_slicing()` + `enable_tiling()` as in inference script |
| Stats too flat | Hooks on output instead of input | Hook `inp[0]` before Linear |
| Wrong scales for pose | Calibrated text-only or no `control_hidden_states` | Use full `DynamiCtrlAnimationPipeline` call |

### Complexity breakdown

- **Easy:** Running inference once with existing script
- **Medium:** Hooks + saving stats + timestep globals
- **Hard:** Representative calibration for 6-video gate; CFG doubles batch size in activations

---

## Step 4 — First naive W8A8 attempt + debug broken output

**Estimated time:** 4–8 hours  
**Complexity:** High

### What it is

Apply the simplest post-training quantization:

- **W8A8** = 8-bit weights + 8-bit activations
- Run full inference, inspect video
- Fix artifacts by widening the protected set or fixing calibration

“Naive” means no SmoothQuant, no W4A8, no block sweep yet — only prove the plumbing works.

### Why you need it

Calibration is useless until you can quantize layers, run the denoising loop end-to-end, and get output that is not noise. The first attempt is **expected** to look bad; debugging is normal.

### How to apply (torchao path)

Install torchao (compatible with torch 2.6):

```bash
pip install torchao
```

Apply to quantize-list layers only:

```python
import torch.nn as nn
from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight

quantize_names = set(Path("configs/quant/quantize_layers.txt").read_text().splitlines())

for name, module in dit.named_modules():
    if name in quantize_names and isinstance(module, nn.Linear):
        quantize_(module, int8_dynamic_activation_int8_weight())

pipe.transformer = dit
```

### Optional: SmoothQuant-style channel smoothing (before quantize)

ViDiT-Q migrates activation outliers into weights with per-channel scale `s` so `(X/s)·(s·W)` is identical in FP but quantizes better. Fold `s` into weights once using calibration absmax per channel. Implement after naive W8A8 fails; see ViDiT-Q paper (channel balancing).

### Debug iteration loop

```
FP16 baseline video  →  save as reference
        ↓
W8A8 (quantize list only, protected list untouched)
        ↓
Generate video → visual check + PSNR vs FP16
        ↓
If bad: widen protected set OR re-calibrate OR add SmoothQuant
        ↓
Repeat until W8A8 passes coarse quality bar
```

### Symptom → fix table

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Gray mush / static | Bad activation scales | Re-run Step 3 with pose; use dynamic per-token quant |
| Pose ignored, body wrong | `control_norm1` quantized | Add to protected list |
| Text / identity drift | Early blocks or joint attention too aggressive | Protect block 0 (and 1), keep AdaLN FP16 |
| NaN / crash | dtype mismatch at quant boundaries | Keep LayerNorm in FP16; cast at Linear I/O |
| OK early steps, bad late steps | Single global scale per layer | Timestep buckets in calibration (Step 3) |

### Quality check (aligns with PTQ method doc gate)

Compare quantized vs FP16 on same seed and inputs:

```
PSNR(Q vs FP16)              ≥ 30 dB   (visual fidelity)
```

Kinematic metrics from the method doc apply after W8A8 is visually sane.

### Definition of done

- [ ] Quantized inference completes without crash
- [ ] Video is recognizable (human + roughly correct motion)
- [ ] PSNR vs FP16 measured on at least one clip
- [ ] You know which protection rule fixed the worst artifact

### Complexity breakdown

- **Easy:** Calling `quantize_` on modules
- **Medium:** SmoothQuant folding (if implemented)
- **Hard:** Subtle pose/motion bugs via MM-DiT joint attention over text + video tokens

---

## Step 5 — Verify INT8 kernels actually speed things up

**Estimated time:** 2–4 hours  
**Complexity:** Medium

### What it is

Measure whether quantization **reduces wall-clock time**, not just **VRAM**.

Many setups store weights as int8 but compute in FP16 → **memory win, no speed win**. Real latency gains need **INT8 GEMM** on tensor cores.

### Why you need it

The PTQ method doc targets ~**1.4–1.7×** end-to-end speedup. That requires fast int8 matmuls. Your notebook shows ~**5.5 s/step** on A100 at 25 steps; if W8A8 still shows ~5.5 s/step, PTQ alone is not buying latency.

### A. Microbenchmark — DiT forward only

Time one transformer forward pass with realistic tensor shapes:

```python
import time

dit.eval()
with torch.no_grad():
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        _ = dit(
            hidden_states=hidden_states,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            control_hidden_states=pose_latents,
            pose_control_function="padaln",
            ofs=ofs_emb,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )
    torch.cuda.synchronize()
print(f"{(time.perf_counter() - t0) / 10:.3f} sec/forward")
```

Use shapes captured from a real pipeline run (768×1360, 37 frames, 32-channel latents, CFG batch doubling if enabled).

### B. End-to-end benchmark — full pipeline

Time the same CLI as the notebook:

```bash
# FP16 baseline
CUDA_VISIBLE_DEVICES=0 python scripts/dynamictrl_inference.py ... 

# W8A8 (after apply_w8a8.py patches transformer)
CUDA_VISIBLE_DEVICES=0 python scripts/dynamictrl_inference.py ...
```

Record:

- Total wall time
- Denoising sec/step (from progress bar or injected timer)
- Peak VRAM: `torch.cuda.max_memory_allocated() / 1e9`

### C. Confirm kernel dispatch

After `quantize_`, inspect layer types:

```python
print(type(dit.transformer_blocks[5].attn1.to_q.weight))
# Expect torchao wrapper (e.g. AffineQuantizedTensor), not plain nn.Parameter
```

Use PyTorch profiler:

```python
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    dit(...)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

Look for int8/cutlass ops vs generic fp16 `mm`.

### Interpreting results

| Observation | Meaning |
|-------------|---------|
| VRAM ↓, time ≈ same | Weight-only or memory-bound; compute still FP16 |
| DiT forward ↓ 1.3–1.7× | INT8 GEMM likely working |
| DiT faster but total time flat | T5, VAE, DWPose dominate → add step reduction |
| Time increases | Quant/dequant overhead without fused kernels — change torchao config or backend |

### Fair comparison notes

- Step 5 e2e uses the same `--compile` flag and **1 untimed warmup** on **both** FP16 and W8A8 before timing, so e2e latency is comparable to micro DiT kernel gains (compile cost is excluded from `wall_sec`).
- Benchmark **denoising loop only** to isolate DiT (T5 load is ~1 min + ~10 GB once).
- Use identical `--seed`, steps, resolution, and inputs.
- On **24 GB** GPUs, T5-XXL may force offload; DiT microbenchmark is still valid for quant speed.

### Definition of done

- [ ] Table: FP16 vs W8A8 — peak VRAM, sec/step, total time
- [ ] Explicit conclusion: “INT8 GEMM confirmed” or “memory-only; no latency gain”
- [ ] Decision documented: pursue W4A8 vs step reduction vs T5 offload

### Complexity breakdown

- **Easy:** `perf_counter` + `cuda.synchronize`
- **Medium:** Realistic dummy tensors from one logged forward
- **Hard:** Proving kernel path on specific GPU + driver + torch 2.6 + torchao combo

---

## Recommended order of work

1. **Step 1** — unblock everything
2. **Step 2** — do immediately after Step 1
3. **Step 3** — start minimal: 1 clip, 3 timestep buckets, absmax only
4. **Step 4** — protect aggressively first (`control_norm1`, `patch_embed`, blocks 0 & 41, all AdaLN/time embeds)
5. **Step 5** — only after Step 4 produces a valid video

If Step 4 works but Step 5 shows no speedup, you still have a **memory win**; chase latency via INT8 kernel tuning or fewer denoising steps (see PTQ method doc).

---

## After integration: connect to the PTQ method plan

Once all five steps pass, proceed with [`DynamiCtrl-PTQ-method.md`](DynamiCtrl-PTQ-method.md):

| Method doc phase | Requires |
|------------------|----------|
| Day 1 — FP16 baseline on 6 videos | Step 1 + working inference |
| Day 1 — calibration activations | Step 3 at scale (6 clips) |
| Day 1 — W8A8 + kinematic gate | Steps 2 + 4 |
| Day 2 — block sensitivity sweep | Step 2 layer names + automated inference |
| Day 2 — W4A8 mixed precision | Step 4 W8A8 stable first |
| Latency / VRAM report | Step 5 |

---

## Dependencies

Match DynamiCtrl’s environment where possible:

```
torch==2.6.0
diffusers==0.32.0
transformers==4.49.0
```

Add for quantization:

```
torchao          # recommended first path for W8A8
# optimum-quanto # alternative
# ViDiT-Q repo   # port if you need exact paper reproduction
```

---

## References

- [DynamiCtrl GitHub](https://github.com/gulucaptain/DynamiCtrl)
- [DynamiCtrl weights (HF)](https://huggingface.co/gulucaptain/DynamiCtrl)
- [ViDiT-Q GitHub](https://github.com/thu-nics/ViDiT-Q)
- [ViDiT-Q paper](https://arxiv.org/abs/2406.02540)
- [torchao quantization docs](https://pytorch.org/ao/stable/index.html)
