"""Shared helpers for DynamiCtrl PTQ scripts (used by collect/apply/benchmark)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

CURRENT_TIMESTEP: int | None = None
_ORT_CUDA_ENSURED = False

# onnxruntime-gpu 1.22+ often links libcudart.so.13; Colab A100 images ship CUDA 12.x.
_ORT_GPU_PIN_CANDIDATES = tuple(
    v.strip()
    for v in os.environ.get(
        "ORT_GPU_PIN_CANDIDATES",
        "1.20.2,1.20.0,1.19.2,1.18.1",
    ).split(",")
    if v.strip()
)


def ensure_onnxruntime_cuda_compat() -> None:
    """Ensure onnxruntime-gpu loads with CUDAExecutionProvider (fixes libcudart.so.13 errors)."""
    global _ORT_CUDA_ENSURED
    if _ORT_CUDA_ENSURED:
        return

    def cuda_provider_available() -> bool:
        try:
            import onnxruntime as ort

            return "CUDAExecutionProvider" in ort.get_available_providers()
        except (OSError, ModuleNotFoundError):
            return False

    if cuda_provider_available():
        _ORT_CUDA_ENSURED = True
        return

    for pinned in _ORT_GPU_PIN_CANDIDATES:
        print(
            f"Installing CUDA-12-compatible onnxruntime-gpu=={pinned} "
            "(requirements.txt may have pulled a CUDA-13 build)..."
        )
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--force-reinstall",
                f"onnxruntime-gpu=={pinned}",
            ]
        )
        for name in list(sys.modules):
            if name == "onnxruntime" or name.startswith("onnxruntime."):
                del sys.modules[name]
        if cuda_provider_available():
            print(f"onnxruntime-gpu=={pinned} OK")
            _ORT_CUDA_ENSURED = True
            return

    raise RuntimeError(
        "onnxruntime-gpu is incompatible with this CUDA runtime (libcudart.so.13 error). "
        f"Tried pins: {', '.join(_ORT_GPU_PIN_CANDIDATES)}. "
        "In Colab, re-run setup cell C or set ORT_GPU_PIN_CANDIDATES."
    )


def get_current_timestep() -> int | None:
    """Return the active diffusion timestep for calibration hooks."""
    return CURRENT_TIMESTEP


def _extract_timestep(args: tuple, kwargs: dict) -> int | None:
    ts = kwargs.get("timestep")
    if ts is None and len(args) >= 3:
        ts = args[2]
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return int(ts)
    return int(ts.flatten()[0].item())


def timestep_bucket(t: int, t_max: int = 1000) -> str:
    if t > t_max * 0.66:
        return "early"
    if t > t_max * 0.33:
        return "mid"
    return "late"


def wrap_transformer_timestep(dit: nn.Module) -> None:
    """Track diffusion timestep for calibration forward hooks."""
    original_forward = dit.forward

    def wrapped_forward(*args, **kwargs):
        global CURRENT_TIMESTEP
        CURRENT_TIMESTEP = _extract_timestep(args, kwargs)
        return original_forward(*args, **kwargs)

    dit.forward = wrapped_forward


PROTECTED_PATTERNS = [
    "patch_embed",
    "time_embedding",
    "time_proj",
    "ofs_embedding",
    "ofs_proj",
    "norm_out",
    "proj_out",
    "control_norm1",
    ".norm1.",
    ".norm2.",
    ".norm3.",
    "attn2",
]


def build_protected_patterns(num_layers: int) -> list[str]:
    """Return protected name patterns including first/last DiT blocks."""
    last = max(0, num_layers - 1)
    return PROTECTED_PATTERNS + [
        "transformer_blocks.0.",
        f"transformer_blocks.{last}.",
    ]


def classify_layer(name: str, protected_patterns: list[str]) -> str:
    for pat in protected_patterns:
        if pat in name:
            return "PROTECT"
    return "QUANTIZE"


def resolve_repo_path(repo_root: str | Path, path: str | Path) -> Path:
    """Resolve repo-relative paths (e.g. ./configs/...) against --repo-root."""
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(repo_root).resolve() / p


def load_quantize_names(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Quantize layer list not found: {path}\n"
            "Run Step 2 (list_layers.py) first, or pass --repo-root so "
            "relative paths resolve under the DynamiCtrl clone."
        )
    names = {line.strip() for line in path.read_text().splitlines() if line.strip()}
    if not names:
        raise ValueError(f"Quantize layer list is empty: {path}")
    return names


def layers_for_block_group(
    all_quantize_names: set[str],
    block_start: int,
    block_end: int,
) -> set[str]:
    """Return transformer_blocks.X.* Linear names for blocks in [block_start, block_end]."""
    selected: set[str] = set()
    for name in all_quantize_names:
        if not name.startswith("transformer_blocks."):
            continue
        parts = name.split(".")
        if len(parts) < 3:
            continue
        try:
            block_idx = int(parts[1])
        except ValueError:
            continue
        if block_start <= block_idx <= block_end:
            selected.add(name)
    return selected


def resolve_hybrid_fp16_layers(
    quantize_names: set[str],
    block_fp16: tuple[int, int],
    block_int8: tuple[int, int],
) -> tuple[set[str], set[str]]:
    """Build fp16 skip set and INT8 target set for block-level hybrid W8A8."""
    actual_int8_layers = layers_for_block_group(
        quantize_names, block_int8[0], block_int8[1]
    )
    fp16_layers = quantize_names - actual_int8_layers

    if not actual_int8_layers:
        raise RuntimeError(
            f"No INT8 targets for blocks {block_int8}. Check quantize_layers.txt."
        )

    early_int8 = layers_for_block_group(
        quantize_names, block_fp16[0], block_fp16[1]
    )
    overlap = early_int8 & actual_int8_layers
    if overlap:
        raise RuntimeError(
            f"Blocks {block_fp16} must stay FP16; found INT8 overlap: {sorted(overlap)[:5]}"
        )
    return fp16_layers, actual_int8_layers


def load_protected_names(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Protected layer list not found: {path}")
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def apply_w8a8(
    dit: nn.Module,
    quantize_names: set[str],
    *,
    fp16_layers: set[str] | None = None,
    calib_path: Path | str | None = None,
    quantize_plan: dict | None = None,
) -> dict:
    from torchao.quantization import int8_dynamic_activation_int8_weight, quantize_

    if quantize_plan is not None:
        skip = set(quantize_plan["early_fp16_layers"])
        actual_int8_layers = set(quantize_plan["actual_int8_layers"])
        if fp16_layers is not None and set(fp16_layers) != skip:
            raise ValueError("fp16_layers disagrees with quantize_plan early_fp16_layers")
    else:
        skip = fp16_layers or set()
        actual_int8_layers = quantize_names - skip
    count = 0
    applied: set[str] = set()
    for name, module in dit.named_modules():
        if name in actual_int8_layers and isinstance(module, nn.Linear):
            quantize_(module, int8_dynamic_activation_int8_weight())
            applied.add(name)
            count += 1
    missing = sorted(actual_int8_layers - applied)
    if missing:
        raise RuntimeError(
            f"Missing Linear modules for INT8 quantize ({len(missing)}): {missing[:5]}"
        )
    if count == 0:
        raise RuntimeError("No Linear layers were quantized — check quantize_layers.txt")
    meta: dict = {
        "quantized_layers": count,
        "early_fp16_layers": sorted(skip),
        "actual_int8_layers": sorted(actual_int8_layers),
    }
    if calib_path is not None:
        meta["calib_path"] = str(calib_path)
    if quantize_plan is not None:
        meta["early_ratio_threshold"] = quantize_plan.get("early_ratio_threshold")
        meta["max_early_fp16_layers"] = quantize_plan.get("max_early_fp16_layers")
    return meta


def ensure_dynami_ctrl_on_path(*hints: str | Path | None) -> Path:
    """Ensure DynamiCtrl clone root (contains ``utils/`` and ``dynamictrl/``) is on ``sys.path``."""
    repo: Path | None = None
    for hint in hints:
        if hint is None:
            continue
        p = Path(hint).resolve()
        for candidate in (p, p.parent, p.parent.parent):
            if (candidate / "utils").is_dir() and (candidate / "dynamictrl").is_dir():
                repo = candidate
                break
        if repo is not None:
            break
    if repo is None:
        cwd = Path.cwd().resolve()
        if (cwd / "utils").is_dir() and (cwd / "dynamictrl").is_dir():
            repo = cwd
    if repo is None:
        fallback = Path("/content/DynamiCtrl")
        if (fallback / "utils").is_dir() and (fallback / "dynamictrl").is_dir():
            repo = fallback
    if repo is None:
        raise ModuleNotFoundError(
            "Cannot locate DynamiCtrl repo root (expected utils/ and dynamictrl/). "
            "Call add_repo_to_path(--repo-root) or run from the DynamiCtrl clone."
        )
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    os.environ.setdefault("PYTHONPATH", repo_str)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return repo


def prepare_fp16_pipeline(model_path: str, dtype: torch.dtype):
    ensure_onnxruntime_cuda_compat()
    ensure_dynami_ctrl_on_path(model_path)
    ensure_dynamictrl_config_files(model_path)
    from diffusers import CogVideoXDPMScheduler
    from dynamictrl.pipelines.dynamictrl_pipeline import DynamiCtrlAnimationPipeline

    pipe = DynamiCtrlAnimationPipeline.from_pretrained(model_path, torch_dtype=dtype)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return pipe


def prepare_control_video(
    reference_image_path: str,
    driving_video: str,
    width: int,
    height: int,
    num_frames: int,
):
    ensure_onnxruntime_cuda_compat()
    verify_torch_stack()
    ensure_dynami_ctrl_on_path(reference_image_path, driving_video)
    validate_clip_assets(reference_image_path, driving_video)
    ensure_dwpose_weights(repo_root=Path(reference_image_path).parent.parent)
    from diffusers.utils import load_image
    from utils.load_validation_control import load_contorl_video_from_Image
    from utils.pre_process import preprocess

    image = load_image(image=reference_image_path)
    pose_images, _ = preprocess(
        driving_video,
        reference_image_path,
        width=width,
        height=height,
        max_frame_num=num_frames - 1,
        sample_stride=1,
    )
    control = load_contorl_video_from_Image(list(pose_images), height, width).unsqueeze(0)
    return image, control


def _clone_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_clone_value(v) for v in value)
    if isinstance(value, list):
        return [_clone_value(v) for v in value]
    return value


def capture_dit_forward_kwargs(pipe, args) -> dict:
    """Run a short pipeline pass and capture real DynamiCtrlTransformerModel inputs."""
    captured: dict = {}

    def capture_hook(_module, _args, kwargs, _output):
        captured.clear()
        for key, value in kwargs.items():
            captured[key] = _clone_value(value)

    handle = pipe.transformer.register_forward_hook(capture_hook, with_kwargs=True)
    image, control = prepare_control_video(
        args.reference_image,
        args.driving_video,
        args.width,
        args.height,
        args.num_frames,
    )
    generator = make_generator(args.seed)
    with torch.no_grad():
        pipe(
            height=args.height,
            width=args.width,
            prompt=args.prompt,
            image=image,
            control_video=control,
            pose_control_function=args.pose_control_function,
            num_inference_steps=min(args.num_inference_steps, 3),
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale,
            use_dynamic_cfg=False,
            generator=generator,
        )
    handle.remove()
    if "hidden_states" not in captured:
        raise RuntimeError("Failed to capture DiT forward kwargs from pipeline run")
    return captured


def compile_model(model: nn.Module, mode: str = "max-autotune"):
    """Apply torch.compile to the model for optimized execution."""
    print(f"Compiling model with mode={mode}...")
    return torch.compile(model, mode=mode)


def sample_dit_linear_weight(dit: nn.Module) -> torch.Tensor:
    """Return a representative Linear weight (handles compile + SmoothQuant wrappers)."""
    model = getattr(dit, "_orig_mod", dit)
    layer = model.transformer_blocks[5].attn1.to_q
    if hasattr(layer, "weight"):
        return layer.weight
    inner = getattr(layer, "module", None)
    if inner is not None and hasattr(inner, "weight"):
        return inner.weight
    raise AttributeError(f"Cannot find sample weight on {type(layer)}")


def microbenchmark_dit(dit, forward_kwargs: dict, repeats: int = 10, warmup: int = 3, compile: bool = False) -> dict:
    dit.eval()
    kwargs = {k: v for k, v in forward_kwargs.items()}
    kwargs["return_dict"] = False

    if compile:
        dit = compile_model(dit)

    # Warmup runs (critical for torch.compile and CUDA graphs)
    print(f"Warming up for {warmup} iterations...")
    with torch.no_grad():
        for _ in range(warmup):
            dit(**kwargs)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    print(f"Benchmarking for {repeats} iterations...")
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            dit(**kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    sample_weight = sample_dit_linear_weight(dit)
    sample_weight_type = str(type(sample_weight))
    
    return {
        "sec_per_forward": elapsed / repeats,
        "repeats": repeats,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "sample_weight_type": sample_weight_type,
        "compiled": compile
    }


def run_inference(pipe, args, image, control_video):
    generator = make_generator(args.seed)
    with torch.no_grad():
        result = pipe(
            height=args.height,
            width=args.width,
            prompt=args.prompt,
            image=image,
            control_video=control_video,
            pose_control_function=args.pose_control_function,
            num_inference_steps=args.num_inference_steps,
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale,
            use_dynamic_cfg=False,
            generator=generator,
        )
    return result.frames[0]


def run_timed_inference(
    model_path: str,
    dtype: torch.dtype,
    args,
    quantize_names: set[str] | None,
    *,
    fp16_layers: set[str] | None = None,
    calib_path: Path | str | None = None,
    compile: bool = False,
    e2e_warmup: int = 0,
):
    """Return (wall_seconds, peak_vram_gb, frames, metadata) for FP16 or W8A8."""
    pipe = prepare_fp16_pipeline(model_path, dtype)
    w8a8_meta = None
    if quantize_names is not None:
        w8a8_meta = apply_w8a8(
            pipe.transformer,
            quantize_names,
            fp16_layers=fp16_layers,
            calib_path=calib_path,
        )
    if compile:
        pipe.transformer = compile_model(pipe.transformer)

    image, control = prepare_control_video(
        args.reference_image,
        args.driving_video,
        args.width,
        args.height,
        args.num_frames,
    )

    for i in range(e2e_warmup):
        print(f"E2E warmup {i + 1}/{e2e_warmup}...")
        run_inference(pipe, args, image, control)
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    frames = run_inference(pipe, args, image, control)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    del pipe
    torch.cuda.empty_cache()
    meta: dict = {"compiled": compile, "e2e_warmup": e2e_warmup}
    if w8a8_meta is not None:
        meta["w8a8"] = w8a8_meta
    return wall, peak, frames, meta


def run_timed_vidit_inference(
    model_path: str,
    dtype: torch.dtype,
    args,
    quantize_names: set[str],
    calib_path: Path,
    *,
    smoothquant_alpha: float = 0.5,
    early_ratio_threshold: float = 2.0,
    max_early_fp16_layers: int = 24,
    compile: bool = False,
    e2e_warmup: int = 0,
):
    """Return (wall_seconds, peak_vram_gb, frames, metadata) for ViDiT-Q W8A8."""
    from vidit_quant import apply_vidit_w8a8

    pipe = prepare_fp16_pipeline(model_path, dtype)
    vidit_meta = apply_vidit_w8a8(
        pipe.transformer,
        quantize_names,
        Path(calib_path),
        smoothquant_alpha=smoothquant_alpha,
        early_ratio_threshold=early_ratio_threshold,
        max_early_fp16_layers=max_early_fp16_layers,
    )
    compiled = False
    if compile:
        try:
            pipe.transformer = compile_model(pipe.transformer)
            compiled = True
        except Exception as exc:
            print(f"WARN: torch.compile failed on ViDiT-Q transformer ({exc}); running eager.")

    image, control = prepare_control_video(
        args.reference_image,
        args.driving_video,
        args.width,
        args.height,
        args.num_frames,
    )

    for i in range(e2e_warmup):
        print(f"E2E warmup {i + 1}/{e2e_warmup}...")
        run_inference(pipe, args, image, control)
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    frames = run_inference(pipe, args, image, control)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    del pipe
    torch.cuda.empty_cache()
    meta = {
        "compiled": compiled,
        "e2e_warmup": e2e_warmup,
        "vidit": vidit_meta,
    }
    return wall, peak, frames, meta


DYNAMICTRL_REPO_ID = "gulucaptain/DynamiCtrl"
DYNAMICTRL_GIT_URL = "https://github.com/gulucaptain/DynamiCtrl.git"

DYNAMICTRL_CONFIG_FILES = (
    "model_index.json",
    "transformer/config.json",
    "vae/config.json",
    "text_encoder/config.json",
    "tokenizer/tokenizer_config.json",
    "scheduler/scheduler_config.json",
)


def ensure_dynamictrl_config_files(model_path: str | Path) -> Path:
    """Download missing HF config sidecars before loading DynamiCtrl weights."""
    root = Path(model_path)
    missing = [
        rel
        for rel in DYNAMICTRL_CONFIG_FILES
        if not (root / rel).is_file() or (root / rel).stat().st_size == 0
    ]
    if missing:
        from huggingface_hub import hf_hub_download

        root.mkdir(parents=True, exist_ok=True)
        for rel in missing:
            hf_hub_download(
                repo_id=DYNAMICTRL_REPO_ID,
                filename=rel,
                local_dir=str(root),
            )
            print(f"Fetched missing checkpoint file: {rel}")
    return root


def is_dynami_ctrl_repo(repo: Path) -> bool:
    """True when ``repo`` looks like a full DynamiCtrl source tree."""
    return (repo / "utils").is_dir() and (repo / "dynamictrl").is_dir()


DYNAMICTRL_CORE_PINS = (
    "torch==2.6.0",
    "torchvision==0.21.0",
    "transformers==4.49.0",
    "diffusers==0.32.0",
)

# Colab Py3.12 + transformers 4.49 need numpy 2.0.x (_no_nep50_warning); block 2.1+ (kernel crashes).
NUMPY_PIN = "numpy>=2.0.0,<2.1.0"

# transformers.generation -> huggingface_hub S3 helpers import botocore; not always preinstalled on Colab.
COLAB_EXTRA_DEPS = ("botocore",)

LOCAL_DWPOSE = Path("/content/checkpoints/DWPose")
REQUIRED_DWPOSE_FILES = ("yolox_l.onnx", "dw-ll_ucoco_384.onnx")


def _pip_install(*packages: str, force: bool = False) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-q"]
    if force:
        cmd.append("--force-reinstall")
    cmd.extend(packages)
    subprocess.check_call(cmd)


def install_dynamictrl_deps(repo_root: str | Path, *, force_core: bool = True) -> None:
    """Pin torch/torchvision to DynamiCtrl-compatible versions (fixes Colab mismatches)."""
    repo = Path(repo_root)
    if force_core:
        _pip_install(*DYNAMICTRL_CORE_PINS, force=True)
        _pip_install(NUMPY_PIN)
    req = repo / "requirements.txt"
    if req.is_file():
        # Avoid requirements.txt pulling numpy 2.5+ before we re-pin (kernel crash on import).
        _pip_install("-r", str(req), "--no-deps")
    _pip_install(
        "onnxruntime-gpu==1.20.1",
        "huggingface-hub>=0.26.0,<1.0",
        "torchao",
        "scikit-image",
        "imageio",
        "matplotlib",
        "pandas",
        *COLAB_EXTRA_DEPS,
    )
    # requirements.txt / Colab extras may bump numpy to 2.5+ and crash the kernel on import.
    _pip_install(NUMPY_PIN, force=True)


def ensure_colab_deps_installed(repo_root: str | Path, deps_flag: str | Path) -> bool:
    """Install pinned deps on first Colab run, restart kernel once, then return True to continue."""
    flag = Path(deps_flag)
    if flag.exists():
        # Flag skips the heavy torch reinstall, but still ensure small Colab-only extras.
        _pip_install(*COLAB_EXTRA_DEPS)
        verify_torch_stack()
        print("Dependencies OK (skipping pip install).")
        return True

    print("Installing pinned torch/numpy stack (first run only)...")
    install_dynamictrl_deps(repo_root)
    flag.write_text("ok", encoding="utf-8")
    print(
        "\n=== Expected kernel restart (not a crash) ===\n"
        "Restarting runtime to load pinned torch/torchvision/numpy.\n"
        "Re-run this setup cell after Colab reconnects.\n"
    )
    try:
        from IPython import get_ipython

        get_ipython().kernel.do_shutdown(restart=True)
    except Exception:
        pass
    return False


def verify_torch_stack() -> None:
    """Smoke-test imports that fail when Colab's torch/torchvision versions disagree."""
    import numpy as np
    import transformers
    import torchvision
    from transformers.models.t5 import modeling_t5  # noqa: F401

    np_parts = tuple(int(p) for p in np.__version__.split(".")[:2])
    if np_parts < (2, 0):
        raise RuntimeError(
            f"numpy {np.__version__} is too old (need >=2.0, <2.1). "
            "Re-run install_dynamictrl_deps() and restart the Colab runtime."
        )
    if np_parts >= (2, 1):
        raise RuntimeError(
            f"numpy {np.__version__} is too new (need >=2.0, <2.1). "
            "Re-run install_dynamictrl_deps() and restart the Colab runtime."
        )
    if not hasattr(np, "_no_nep50_warning"):
        raise RuntimeError(
            f"numpy {np.__version__} is missing _no_nep50_warning (broken or mixed install). "
            "Delete the notebook deps flag, re-run install_dynamictrl_deps(), and restart."
        )
    if not torch.__version__.startswith("2.6"):
        raise RuntimeError(
            f"Expected torch 2.6.x, got {torch.__version__}. "
            "Re-run install_dynamictrl_deps() and restart the Colab runtime."
        )
    if not transformers.__version__.startswith("4.49"):
        raise RuntimeError(
            f"Expected transformers 4.49.x, got {transformers.__version__}. "
            "Re-run install_dynamictrl_deps() and restart the Colab runtime."
        )
    if not torchvision.__version__.startswith("0.21"):
        raise RuntimeError(
            f"Expected torchvision 0.21.x, got {torchvision.__version__}. "
            "Re-run install_dynamictrl_deps() and restart the Colab runtime."
        )
    from torchvision import transforms  # noqa: F401

    _ = transforms.Compose([transforms.Resize((64, 64))])
    print(
        f"torch={torch.__version__}, torchvision={torchvision.__version__}, "
        f"transformers={transformers.__version__}, numpy={np.__version__}"
    )


def ensure_dwpose_weights(
    drive_root: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> Path:
    """Ensure DWPose ONNX weights exist at ``/content/checkpoints/DWPose``."""
    LOCAL_DWPOSE.mkdir(parents=True, exist_ok=True)
    drive = Path(drive_root) if drive_root else None
    repo = Path(repo_root) if repo_root else None

    for name in REQUIRED_DWPOSE_FILES:
        local = LOCAL_DWPOSE / name
        if local.is_file() and local.stat().st_size > 0:
            continue
        sources: list[Path] = []
        if drive is not None:
            sources.append(drive / "checkpoints" / "DWPose" / name)
        if repo is not None:
            sources.append(repo / "checkpoints" / "DWPose" / name)
        copied = False
        for src in sources:
            if src.is_file() and src.stat().st_size > 0:
                shutil.copy2(src, local)
                print(f"Copied DWPose weight: {src} -> {local}")
                copied = True
                break
        if copied:
            continue
        from huggingface_hub import hf_hub_download

        print(f"Downloading DWPose weight: {name}")
        hf_hub_download(
            repo_id="yzd-v/DWPose",
            filename=name,
            local_dir=str(LOCAL_DWPOSE),
        )
    return LOCAL_DWPOSE


def validate_clip_assets(reference_image: str | Path, driving_video: str | Path) -> None:
    """Raise early if human2_motion1 assets are missing from the DynamiCtrl clone."""
    ref = Path(reference_image)
    vid = Path(driving_video)
    missing = [str(p) for p in (ref, vid) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing clip assets:\n  "
            + "\n  ".join(missing)
            + "\nEnsure DynamiCtrl assets/ is present (re-clone or copy from Drive)."
        )


def ensure_dynami_ctrl_clone(repo_root: str | Path = "/content/DynamiCtrl") -> Path:
    """Clone or repair DynamiCtrl so ``utils/`` and ``dynamictrl/`` exist."""
    repo = Path(repo_root).resolve()
    if is_dynami_ctrl_repo(repo):
        return repo

    if (repo / ".git").is_dir():
        print(f"Updating incomplete DynamiCtrl clone at {repo}...")
        subprocess.run(
            ["git", "-C", str(repo), "pull", "--ff-only"],
            check=False,
        )
        if is_dynami_ctrl_repo(repo):
            return repo

    if not repo.is_dir():
        repo.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning DynamiCtrl into {repo}...")
        subprocess.check_call(["git", "clone", DYNAMICTRL_GIT_URL, str(repo)])
        if is_dynami_ctrl_repo(repo):
            return repo

    # Partial stub (e.g. scripts/quant deployed before clone) — overlay source tree.
    print(
        f"DynamiCtrl at {repo} is missing utils/ or dynamictrl/; "
        "overlaying source files from GitHub..."
    )
    with tempfile.TemporaryDirectory() as tmp:
        clone_dest = Path(tmp) / "DynamiCtrl"
        subprocess.check_call(
            ["git", "clone", "--depth", "1", DYNAMICTRL_GIT_URL, str(clone_dest)]
        )
        for item in clone_dest.iterdir():
            if item.name == ".git":
                continue
            target = repo / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            elif not target.is_file():
                shutil.copy2(item, target)

    if not is_dynami_ctrl_repo(repo):
        raise FileNotFoundError(
            f"DynamiCtrl clone still incomplete at {repo} "
            f"(missing utils/ or dynamictrl/). Clone {DYNAMICTRL_GIT_URL} manually."
        )
    print(f"DynamiCtrl source tree ready at {repo}")
    return repo


def add_repo_to_path(repo_root: str | Path) -> Path:
    repo = ensure_dynami_ctrl_clone(repo_root)
    return ensure_dynami_ctrl_on_path(repo)


def parse_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bfloat16" else torch.float16


def make_generator(seed: int) -> torch.Generator:
    """Match DynamiCtrl ``scripts/dynamictrl_inference.py`` (CPU generator)."""
    return torch.Generator().manual_seed(seed)


def args_namespace(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)
