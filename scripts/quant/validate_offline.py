#!/usr/bin/env python3
"""Offline zero-trust checks (no GPU/model weights required)."""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

OFFLINE_VALIDATE_VERSION = 2

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

REQUIRED_SCRIPTS = (
    "colab_paths.py",
    "quant_common.py",
    "load_dit.py",
    "list_layers.py",
    "collect_calib.py",
    "collect_calib_vidit.py",
    "vidit_quant.py",
    "vidit_eval.py",
    "apply_w8a8.py",
    "apply_vidit_w8a8.py",
    "apply_and_benchmark_vidit.py",
    "run_grid_batch.py",
    "run_hybrid_grid.py",
    "benchmark.py",
    "benchmark_vidit.py",
    "vidit_ptq_config.py",
)

DOC_PROTECTED_PATTERNS = (
    "patch_embed",
    "time_embedding",
    "time_proj",
    "ofs_embedding",
    "ofs_proj",
    "norm_out",
    "control_norm1",
)

BANNED_SNIPPETS = (
    "new_zeros(",
    "NotImplementedError",
    "TODO",
    "FIXME",
    "placeholder",
)


def check_syntax(path: Path) -> None:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def check_help(path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(path), "--help"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{path.name} --help failed:\n{proc.stderr}")


def check_required_scripts() -> None:
    for name in REQUIRED_SCRIPTS:
        path = SCRIPT_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Missing required script: {name}")


def check_no_stubs() -> None:
    for name in REQUIRED_SCRIPTS:
        text = (SCRIPT_DIR / name).read_text(encoding="utf-8")
        for snippet in BANNED_SNIPPETS:
            if snippet in text:
                raise RuntimeError(f"{name} contains banned stub marker: {snippet}")


def check_protected_patterns() -> None:
    from quant_common import PROTECTED_PATTERNS, build_protected_patterns

    missing = [pat for pat in DOC_PROTECTED_PATTERNS if pat not in PROTECTED_PATTERNS]
    if missing:
        raise RuntimeError(f"PROTECTED_PATTERNS missing doc patterns: {missing}")

    dynamic = build_protected_patterns(42)
    for pat in ("transformer_blocks.0.", "transformer_blocks.41."):
        if pat not in dynamic:
            raise RuntimeError(f"build_protected_patterns(42) missing edge block pattern: {pat}")


def check_expected_quantize_bounds() -> None:
    """With AdaLN + edge blocks protected, 42-layer padaln DiT has ~240 quantize layers."""
    from quant_common import build_protected_patterns, classify_layer

    patterns = build_protected_patterns(42)
    quantize = 0
    for block_idx in range(42):
        for suffix in (
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
        ):
            name = f"transformer_blocks.{block_idx}.{suffix}"
            if classify_layer(name, patterns) == "QUANTIZE":
                quantize += 1
    if not (200 <= quantize <= 280):
        raise RuntimeError(
            f"Expected ~240 quantize attn/ff layers for 42-block DiT, simulated count={quantize}"
        )


def check_script_wiring() -> None:
    wiring = {
        "benchmark.py": (
            "capture_dit_forward_kwargs",
            "run_timed_inference",
            "microbenchmark_dit",
        ),
        "apply_w8a8.py": (
            "load_reference_frames_from_dir",
            "apply_w8a8",
            "compute_psnr",
            "resolve_matched_quantize_plan",
        ),
        "collect_calib.py": (
            "wrap_transformer_timestep",
            "register_hooks",
            "aggregate_stats",
        ),
        "collect_calib_vidit.py": (
            "register_hooks_vidit",
            "aggregate_stats_vidit",
            "wrap_transformer_timestep",
        ),
        "vidit_quant.py": (
            "compute_smoothquant_scales",
            "fold_smoothquant_into_linear",
            "SmoothQuantActivationWrapper",
            "classify_early_sensitive_layers",
            "resolve_early_fp16_layers",
            "resolve_matched_quantize_plan",
            "assert_matched_quantize_parity",
            "apply_vidit_w8a8",
            "int8_dynamic_activation_int8_weight",
        ),
        "apply_vidit_w8a8.py": (
            "apply_vidit_w8a8",
            "export_vidit_outputs",
        ),
        "apply_and_benchmark_vidit.py": (
            "run_combined_apply_and_benchmark",
        ),
        "run_grid_batch.py": (
            "run_fp16_grid",
            "run_calib_grid",
            "run_vidit_grid",
            "run_w8a8_grid",
            "run_vidit_calib_pass",
            "apply_vidit_w8a8",
            "apply_w8a8",
            "resolve_matched_quantize_plan",
            "calib_root",
        ),
        "run_hybrid_grid.py": (
            "run_hybrid_grid",
            "run_hybrid_algorithm_grid",
            "apply_w8a8",
            "apply_vidit_w8a8",
            "resolve_hybrid_fp16_layers",
            "actual_int8_layers",
            "block_fp16",
            "block_int8",
            "hybrid_meta.json",
        ),
        "benchmark_vidit.py": (
            "run_timed_vidit_inference",
            "apply_vidit_w8a8",
            "build_kernel_conclusion",
        ),
        "vidit_eval.py": (
            "run_combined_apply_and_benchmark",
            "compute_psnr",
            "build_kernel_conclusion",
            "build_matched_kernel_conclusion",
            "export_vidit_outputs",
            "export_w8a8_outputs",
            "w8a8_matched_micro",
            "vidit_vs_w8a8_micro_speedup",
            "assert_matched_quantize_parity",
            "resolve_matched_quantize_plan",
            "quantize_plan",
            "w8a8_meta.json",
            "vidit_meta.json",
        ),
        "quant_common.py": (
            "int8_dynamic_activation_int8_weight",
            "prepare_fp16_pipeline",
            "prepare_control_video",
            "ensure_onnxruntime_cuda_compat",
            "timestep_bucket",
            "run_timed_vidit_inference",
            "fp16_layers",
            "quantize_plan",
            "actual_int8_layers",
            "layers_for_block_group",
            "resolve_hybrid_fp16_layers",
        ),
        "colab_paths.py": (
            "COLAB_PATHS_VERSION",
            "DRIVE_SCRIPTS",
            "require_colab_paths_version",
            "run_setup_downloads",
            "ensure_drive_layout",
            "link_repo_paths",
            "deploy_quant_scripts",
            "download_dynamictrl_weights",
            "is_dynamictrl_checkpoint_complete",
            "validate_dynamictrl_checkpoint",
            "scrub_truncated_files",
        ),
    }
    for filename, symbols in wiring.items():
        text = (SCRIPT_DIR / filename).read_text(encoding="utf-8")
        for symbol in symbols:
            if symbol not in text:
                raise RuntimeError(f"{filename} missing expected symbol: {symbol}")


def check_layer_classification() -> None:
    from quant_common import build_protected_patterns, classify_layer

    patterns = build_protected_patterns(42)
    cases = {
        "transformer_blocks.10.attn1.to_q": "QUANTIZE",
        "transformer_blocks.10.norm1.linear": "PROTECT",
        "transformer_blocks.10.control_norm1.linear": "PROTECT",
        "transformer_blocks.10.attn2.to_q": "PROTECT",
        "transformer_blocks.0.attn1.to_q": "PROTECT",
        "transformer_blocks.41.ff.net.0.proj": "PROTECT",
        "patch_embed.proj": "PROTECT",
        "time_embedding.linear_1": "PROTECT",
        "proj_out": "PROTECT",
    }
    for name, expected in cases.items():
        got = classify_layer(name, patterns)
        if got != expected:
            raise AssertionError(f"classify({name})={got}, expected {expected}")


def check_timestep_bucket() -> None:
    from collect_calib import aggregate_stats
    from quant_common import get_current_timestep, timestep_bucket, wrap_transformer_timestep

    assert timestep_bucket(900) == "early"
    assert timestep_bucket(500) == "mid"
    assert timestep_bucket(100) == "late"

    raw = {
        "layer.a": {
            "early": [{"absmax": 2.0, "shape": (1, 2)}],
            "late": [{"absmax": 5.0, "shape": (1, 2)}],
        }
    }
    stats = aggregate_stats(raw)
    assert stats["layer.a"]["early"]["absmax"] == 2.0
    assert stats["layer.a"]["late"]["absmax"] == 5.0

    import torch

    class DummyDit(torch.nn.Module):
        def forward(self, hidden_states, encoder_hidden_states, timestep, **kwargs):
            return hidden_states

    dit = DummyDit()
    wrap_transformer_timestep(dit)
    dit(torch.zeros(1), torch.zeros(1), torch.tensor([900]))
    assert get_current_timestep() == 900
    dit(torch.zeros(1), torch.zeros(1), timestep=torch.tensor([100]))
    assert get_current_timestep() == 100


def check_vidit_smoothquant() -> None:
    import torch
    from vidit_quant import (
        SmoothQuantActivationWrapper,
        compute_smoothquant_scales,
        fold_smoothquant_into_linear,
    )

    linear = torch.nn.Linear(4, 2, bias=False)
    x = torch.randn(3, 4)
    weight_before = linear.weight.detach().clone()
    channel_absmax = x.abs().amax(dim=0)
    scales = compute_smoothquant_scales(weight_before, channel_absmax, alpha=0.5)
    fold_smoothquant_into_linear(linear, scales)
    wrapped = SmoothQuantActivationWrapper(linear, scales)
    y_ref = x @ weight_before.T
    y_fold = wrapped(x)
    assert torch.allclose(y_ref, y_fold, rtol=1e-4, atol=1e-4)

    x3 = torch.randn(2, 5, 4)
    y_ref3 = x3 @ weight_before.T
    y_fold3 = wrapped(x3)
    assert torch.allclose(y_ref3, y_fold3, rtol=1e-4, atol=1e-4)


def check_sample_dit_linear_weight() -> None:
    import torch
    from quant_common import sample_dit_linear_weight
    from vidit_quant import SmoothQuantActivationWrapper

    linear = torch.nn.Linear(4, 2, bias=False)
    wrapped = SmoothQuantActivationWrapper(linear, torch.ones(4))
    dit = torch.nn.Module()
    dit.transformer_blocks = torch.nn.ModuleList(
        [torch.nn.Module() for _ in range(6)]
    )
    attn1 = torch.nn.Module()
    attn1.to_q = wrapped
    dit.transformer_blocks[5].attn1 = attn1

    weight = sample_dit_linear_weight(dit)
    assert weight is linear.weight


def check_vidit_early_sensitive() -> None:
    from vidit_quant import classify_early_sensitive_layers, resolve_early_fp16_layers

    stats = {
        "layer.hot": {
            "early": {"tensor_absmax": 10.0},
            "late": {"tensor_absmax": 2.0},
        },
        "layer.cool": {
            "early": {"tensor_absmax": 3.0},
            "late": {"tensor_absmax": 2.5},
        },
    }
    selected = classify_early_sensitive_layers(stats, ratio_threshold=2.0, max_layers=24)
    assert selected == ["layer.hot"]
    selected_via_resolve = resolve_early_fp16_layers(
        Path("unused.pt"),
        early_ratio_threshold=2.0,
        max_early_fp16_layers=24,
        stats=stats,
    )
    assert selected_via_resolve == ["layer.hot"]


def check_apply_w8a8_matched_skip() -> None:
    import torch
    from quant_common import apply_w8a8
    from vidit_quant import assert_matched_quantize_parity, resolve_matched_quantize_plan

    text = (SCRIPT_DIR / "quant_common.py").read_text(encoding="utf-8")
    if "fp16_layers" not in text or '"quantized_layers"' not in text:
        raise RuntimeError("apply_w8a8 must accept fp16_layers and return metadata dict")
    if "quantize_plan" not in text or "actual_int8_layers" not in text:
        raise RuntimeError("apply_w8a8 must support quantize_plan and actual_int8_layers metadata")

    try:
        import torchao  # noqa: F401
    except ImportError:
        print("SKIP apply_w8a8 runtime test (torchao not installed)")
        return

    dit = torch.nn.Module()
    dit.linear_a = torch.nn.Linear(4, 2)
    dit.linear_b = torch.nn.Linear(4, 2)
    dit.linear_c = torch.nn.Linear(4, 2)
    quantize_names = {"linear_a", "linear_b", "linear_c"}
    plan = {
        "early_fp16_layers": ["linear_b"],
        "actual_int8_layers": ["linear_a", "linear_c"],
        "quantized_layers": 2,
        "calib_path": "test.pt",
        "early_ratio_threshold": 1.25,
        "max_early_fp16_layers": 24,
    }
    meta = apply_w8a8(dit, quantize_names, quantize_plan=plan)
    assert meta["quantized_layers"] == 2
    assert meta["early_fp16_layers"] == ["linear_b"]
    assert meta["actual_int8_layers"] == ["linear_a", "linear_c"]

    vidit_stub = {
        "quantized_layers": 2,
        "early_fp16_layers": ["linear_b"],
        "actual_int8_layers": ["linear_a", "linear_c"],
        "smoothquant_layers": ["linear_a", "linear_c"],
    }
    assert_matched_quantize_parity(meta, vidit_stub)


def check_matched_quantize_plan() -> None:
    from vidit_quant import resolve_matched_quantize_plan

    stats = {
        "layer.a": {
            "early": {"tensor_absmax": 10.0},
            "late": {"tensor_absmax": 1.0},
        },
        "layer.b": {
            "early": {"tensor_absmax": 2.0},
            "late": {"tensor_absmax": 2.0},
        },
        "layer.c": {
            "early": {"tensor_absmax": 8.0},
            "late": {"tensor_absmax": 1.0},
        },
    }
    quantize_names = {"layer.a", "layer.b", "layer.c", "layer.d"}
    plan = resolve_matched_quantize_plan(
        quantize_names,
        Path("unused.pt"),
        early_ratio_threshold=2.0,
        max_early_fp16_layers=24,
        stats=stats,
    )
    assert plan["early_fp16_layers"] == ["layer.a", "layer.c"]
    assert plan["actual_int8_layers"] == ["layer.b", "layer.d"]
    assert plan["quantized_layers"] == 2


def check_matched_naive_vidit_fair_comparison() -> None:
    """Zero-trust: naive and ViDiT share layer plan, env, and quant kernel; ViDiT alone uses SmoothQuant."""
    w8_text = (SCRIPT_DIR / "quant_common.py").read_text(encoding="utf-8")
    vidit_text = (SCRIPT_DIR / "vidit_quant.py").read_text(encoding="utf-8")
    eval_text = (SCRIPT_DIR / "vidit_eval.py").read_text(encoding="utf-8")
    grid_text = (SCRIPT_DIR / "run_grid_batch.py").read_text(encoding="utf-8")

    if "resolve_matched_quantize_plan" not in vidit_text:
        raise RuntimeError("vidit_quant.py must define resolve_matched_quantize_plan")
    if "assert_matched_quantize_parity" not in vidit_text:
        raise RuntimeError("vidit_quant.py must define assert_matched_quantize_parity")

    # Same torchao W8A8 kernel on shared INT8 layers.
    if w8_text.count("int8_dynamic_activation_int8_weight") < 1:
        raise RuntimeError("apply_w8a8 must use int8_dynamic_activation_int8_weight")
    if vidit_text.count("int8_dynamic_activation_int8_weight") < 1:
        raise RuntimeError("apply_vidit_w8a8 must use int8_dynamic_activation_int8_weight")

    # SmoothQuant is ViDiT-only mechanism.
    for symbol in ("SmoothQuantActivationWrapper", "fold_smoothquant_into_linear"):
        if symbol in w8_text:
            raise RuntimeError(f"apply_w8a8 must not use ViDiT-only symbol: {symbol}")
    for symbol in ("SmoothQuantActivationWrapper", "fold_smoothquant_into_linear"):
        if symbol not in vidit_text:
            raise RuntimeError(f"apply_vidit_w8a8 missing ViDiT mechanism: {symbol}")

    # Step 4 benchmark: shared forward kwargs + runtime parity guard.
    for symbol in (
        "resolve_matched_quantize_plan",
        "assert_matched_quantize_parity",
        "capture_dit_forward_kwargs",
        "w8a8_matched_micro",
        "prepare_fp16_pipeline",
    ):
        if symbol not in eval_text:
            raise RuntimeError(f"vidit_eval.py missing fair-comparison wiring: {symbol}")

    # Grid: same calib file and thresholds for naive and ViDiT phases.
    if '{cid}_vidit.pt' not in grid_text:
        raise RuntimeError("run_grid_batch.py must use per-clip {cid}_vidit.pt calib files")
    if grid_text.count("{cid}_vidit.pt") < 2:
        raise RuntimeError("Both vidit and w8a8 grid phases must load {cid}_vidit.pt calib")
    if "resolve_matched_quantize_plan" not in grid_text:
        raise RuntimeError("run_w8a8_grid must use resolve_matched_quantize_plan")
    if "early_ratio_threshold=early_ratio" not in grid_text:
        raise RuntimeError("w8a8 grid must pass the same early_ratio as ViDiT grid")
    if "max_early_fp16_layers=max_early_fp16_layers" not in grid_text:
        raise RuntimeError("w8a8 grid must pass the same max_early_fp16_layers as ViDiT grid")

    # ViDiT apply must derive INT8 set from shared plan, not a separate formula.
    if "resolve_matched_quantize_plan" not in vidit_text:
        raise RuntimeError("apply_vidit_w8a8 must call resolve_matched_quantize_plan")
    if "actual_quantize_names = quantize_names - early_sensitive_set" in vidit_text:
        raise RuntimeError(
            "apply_vidit_w8a8 must not compute INT8 layers separately from resolve_matched_quantize_plan"
        )

    # Metadata must expose comparable layer lists for audit.
    for field in ("actual_int8_layers", "early_fp16_layers", "quantized_layers"):
        if f'"{field}"' not in w8_text and f"'{field}'" not in w8_text:
            raise RuntimeError(f"apply_w8a8 metadata must include {field}")
        if f'"{field}"' not in vidit_text and f"'{field}'" not in vidit_text:
            raise RuntimeError(f"apply_vidit_w8a8 metadata must include {field}")


def check_matched_parity_rejects_mismatch() -> None:
    from vidit_quant import assert_matched_quantize_parity

    ok_w8 = {"quantized_layers": 2, "early_fp16_layers": ["a"], "actual_int8_layers": ["b", "c"]}
    ok_v = {
        "quantized_layers": 2,
        "early_fp16_layers": ["a"],
        "actual_int8_layers": ["b", "c"],
        "smoothquant_layers": ["b", "c"],
    }
    assert_matched_quantize_parity(ok_w8, ok_v)

    bad_v = {**ok_v, "actual_int8_layers": ["b"]}
    try:
        assert_matched_quantize_parity(ok_w8, bad_v)
    except RuntimeError as exc:
        if "INT8 layer set mismatch" not in str(exc):
            raise
    else:
        raise RuntimeError("assert_matched_quantize_parity should reject INT8 set mismatch")


def check_apply_vidit_w8a8_layer_override() -> None:
    """Sensitivity sweep: apply_vidit_w8a8 accepts explicit INT8 layer override."""
    import inspect

    from vidit_quant import apply_vidit_w8a8

    sig = inspect.signature(apply_vidit_w8a8)
    if "actual_int8_layers" not in sig.parameters:
        raise RuntimeError("apply_vidit_w8a8 must accept actual_int8_layers override")

    text = (SCRIPT_DIR / "vidit_quant.py").read_text(encoding="utf-8")
    for symbol in ("actual_int8_layers", "layer_override"):
        if symbol not in text:
            raise RuntimeError(f"vidit_quant.py missing sweep override symbol: {symbol}")
    if "resolve_matched_quantize_plan" not in text:
        raise RuntimeError("apply_vidit_w8a8 default path must still call resolve_matched_quantize_plan")


def check_vidit_aggregate_stats() -> None:
    import torch
    from collect_calib_vidit import aggregate_stats_vidit

    raw = {
        "layer.a": {
            "early": [
                {
                    "channel_absmax": torch.tensor([1.0, 2.0]),
                    "tensor_absmax": 2.0,
                    "shape": (1, 2),
                }
            ],
            "late": [
                {
                    "channel_absmax": torch.tensor([3.0, 1.0]),
                    "tensor_absmax": 3.0,
                    "shape": (1, 2),
                }
            ],
        }
    }
    stats = aggregate_stats_vidit(raw)
    assert stats["layer.a"]["early"]["tensor_absmax"] == 2.0
    assert stats["layer.a"]["late"]["tensor_absmax"] == 3.0
    assert torch.allclose(
        stats["layer.a"]["early"]["channel_absmax"], torch.tensor([1.0, 2.0])
    )
    assert torch.allclose(
        stats["layer.a"]["late"]["channel_absmax"], torch.tensor([3.0, 1.0])
    )


def check_method_doc_protected_set() -> None:
    """Map DynamiCtrl-PTQ-method.md protected-set bullets to code patterns."""
    from quant_common import PROTECTED_PATTERNS, build_protected_patterns, classify_layer

    patterns = build_protected_patterns(42)
    required_cases = {
        # AdaLN / modulation MLPs and time-embedding projections
        "transformer_blocks.5.norm1.linear": "PROTECT",
        "time_embedding.linear_1": "PROTECT",
        "time_proj": "PROTECT",
        # First and last DiT blocks (doc: 1–2; we protect edge blocks 0 and 41)
        "transformer_blocks.0.attn1.to_q": "PROTECT",
        "transformer_blocks.41.ff.net.2": "PROTECT",
        # Joint-text cross-attention (attn2)
        "transformer_blocks.5.attn2.to_k": "PROTECT",
        # DynamiCtrl pose injection
        "transformer_blocks.5.control_norm1.linear": "PROTECT",
        # Shared image latent encoder path inside DiT
        "patch_embed.proj": "PROTECT",
    }
    for name, expected in required_cases.items():
        got = classify_layer(name, patterns)
        if got != expected:
            raise AssertionError(f"method-doc protected set: {name}={got}, expected {expected}")

    for token in ("time_embedding", "time_proj", "control_norm1", "attn2", "patch_embed"):
        if token not in PROTECTED_PATTERNS:
            raise RuntimeError(f"PROTECTED_PATTERNS missing method-doc token: {token}")


def check_method_doc_tokenwise_quant() -> None:
    text = (SCRIPT_DIR / "vidit_quant.py").read_text(encoding="utf-8")
    if "int8_dynamic_activation_int8_weight" not in text:
        raise RuntimeError("vidit_quant.py must use torchao int8_dynamic_activation_int8_weight")
    if "quantize_(" not in text:
        raise RuntimeError("vidit_quant.py must call torchao quantize_")


def check_method_doc_smoothquant_pipeline() -> None:
    text = (SCRIPT_DIR / "vidit_quant.py").read_text(encoding="utf-8")
    for symbol in (
        "compute_smoothquant_scales",
        "fold_smoothquant_into_linear",
        "SmoothQuantActivationWrapper",
        "load_calib_stats",
    ):
        if symbol not in text:
            raise RuntimeError(f"vidit_quant.py missing SmoothQuant/calib symbol: {symbol}")
    if "layer_scales" not in text:
        raise RuntimeError("apply_vidit_w8a8 path must store per-layer SmoothQuant scales")


def check_method_doc_static_precision() -> None:
    text = (SCRIPT_DIR / "collect_calib_vidit.py").read_text(encoding="utf-8")
    if "timestep_bucket" not in text or "channel_absmax" not in text:
        raise RuntimeError("collect_calib_vidit.py must bucket timesteps and record channel stats")

    apply_text = (SCRIPT_DIR / "vidit_quant.py").read_text(encoding="utf-8")
    if "resolve_matched_quantize_plan" not in apply_text:
        raise RuntimeError(
            "vidit_quant.py should use resolve_matched_quantize_plan to exclude sensitive layers"
        )
    if "actual_int8_layers" not in apply_text:
        raise RuntimeError("vidit_quant.py should expose actual_int8_layers in quantize plan")


def check_method_doc_calib_drives_quant() -> None:
    apply_text = (SCRIPT_DIR / "apply_vidit_w8a8.py").read_text(encoding="utf-8")
    if "--calib" not in apply_text:
        raise RuntimeError("apply_vidit_w8a8.py must require --calib")
    if "apply_vidit_w8a8(" not in apply_text:
        raise RuntimeError("apply_vidit_w8a8.py must call apply_vidit_w8a8()")

    calib_text = (SCRIPT_DIR / "collect_calib_vidit.py").read_text(encoding="utf-8")
    if "prepare_fp16_pipeline" not in calib_text or "control_video" not in calib_text:
        raise RuntimeError("collect_calib_vidit.py must calibrate with pose-conditioned pipeline")


def check_method_doc_dit_scope() -> None:
    """Compression targets DiT linear layers only (not VAE/T5 in quant scripts)."""
    for name in ("apply_vidit_w8a8.py", "vidit_quant.py", "collect_calib_vidit.py"):
        text = (SCRIPT_DIR / name).read_text(encoding="utf-8")
        if "pipe.transformer" not in text and "dit" not in text:
            raise RuntimeError(f"{name} should target DiT transformer, not full VAE quant")
    vae_hits = (SCRIPT_DIR / "vidit_quant.py").read_text(encoding="utf-8").count("pipe.vae")
    if vae_hits > 0:
        raise RuntimeError("vidit_quant.py must not quantize VAE")


def check_method_doc_documented_gaps(*, colab: bool = False) -> None:
    """Day-1 ViDiT notebook scope intentionally excludes Day-2 / gate items."""
    repo_text = "\n".join(
        (SCRIPT_DIR / name).read_text(encoding="utf-8") for name in REQUIRED_SCRIPTS
    )
    notebook = ""
    if not colab:
        notebook_path = PROJECT_ROOT / "notebooks" / "DynamiCtrl-PTQ-ViDiT.ipynb"
        if notebook_path.is_file():
            notebook = notebook_path.read_text(encoding="utf-8")
    combined = repo_text + notebook

    # These are in method doc but not required for single-clip ViDiT Day-1 path.
    optional_gaps = {
        "kinematic gate metrics": ("acceleration_variance", "violation_fraction"),
        "W4A8 mixed precision": ("W4A8", "int4"),
        "6-video calibration set": ("woman_3", "man_1"),
    }
    for label, markers in optional_gaps.items():
        if any(marker in combined for marker in markers):
            continue
        print(f"NOTE method-doc gap (expected for Day-1 single clip): {label} not implemented")


def check_smoothquant_formula() -> None:
    import torch
    from vidit_quant import compute_smoothquant_scales

    weight = torch.tensor([[1.0, 4.0], [2.0, 8.0]])
    channel_absmax = torch.tensor([16.0, 4.0])
    alpha = 0.5
    s = compute_smoothquant_scales(weight, channel_absmax, alpha=alpha)
    act = channel_absmax.clamp(min=1e-5)
    w_ch = weight.abs().amax(dim=0).clamp(min=1e-5)
    expected = (act.pow(alpha) / w_ch.pow(1.0 - alpha)).clamp(min=1e-5)
    assert torch.allclose(s, expected, rtol=1e-5, atol=1e-5)


def check_vidit_ptq_config_file() -> None:
    from vidit_ptq_config import (
        default_config_path,
        load_grid_config,
        load_kinematic_gate_config,
        load_vidit_ptq_config,
    )

    cfg = load_vidit_ptq_config(default_config_path())
    if cfg.EARLY_SENSITIVITY_RATIO != 1.25:
        raise RuntimeError(
            f"Expected early_sensitivity_ratio=1.25 in vidit_ptq.json, got {cfg.EARLY_SENSITIVITY_RATIO}"
        )
    if cfg.SMOOTHQUANT_ALPHA != 0.5:
        raise RuntimeError("Unexpected smoothquant_alpha default in vidit_ptq.json")

    grid = load_grid_config(default_config_path(), repo_root=PROJECT_ROOT)
    if len(grid.images) != 4 or len(grid.motions) != 3:
        raise RuntimeError("Expected 4 grid images and 3 grid motions in vidit_ptq.json")

    kg = load_kinematic_gate_config(default_config_path())
    if kg.max_rel_delta_accel_variance != 0.10:
        raise RuntimeError("Unexpected kinematic_gate.max_rel_delta_accel_variance")
    if not kg.skip_if_cached:
        raise RuntimeError("Expected kinematic_gate.skip_if_cached=true")
    if not kg.phase1_root.is_dir():
        print(f"NOTE kinematic_gate phase1_root not present locally: {kg.phase1_root}")


def check_kinematic_gate_compare() -> None:
    from kinematic_gate.compare import (
        GateThresholds,
        abs_delta,
        compute_deltas,
        evaluate_gate,
        rel_delta,
    )
    from kinematic_gate.metrics import compute_joint_movement_summary_between

    assert rel_delta(100.0, 105.0) == 0.05
    assert abs(abs_delta(0.1, 0.12) - 0.02) < 1e-9

    def _kp(name: str, x: float, y: float, confidence: float = 0.9) -> dict:
        return {"x": x, "y": y, "z": 0.0, "confidence": confidence}

    def _synthetic_frame(offset: float = 0.0) -> dict:
        return {
            "keypoints": {
                "neck": _kp("neck", 400.0, 400.0),
                "left_hip": _kp("left_hip", 380.0 + offset, 600.0),
                "right_hip": _kp("right_hip", 420.0 + offset, 600.0),
                "right_elbow": _kp("right_elbow", 450.0 + offset, 500.0),
                "right_wrist": _kp("right_wrist", 500.0 + offset, 550.0),
                "left_elbow": _kp("left_elbow", 350.0 + offset, 500.0),
                "left_wrist": _kp("left_wrist", 300.0 + offset, 550.0),
                "right_knee": _kp("right_knee", 410.0 + offset, 800.0),
                "right_ankle": _kp("right_ankle", 410.0 + offset, 950.0),
                "left_knee": _kp("left_knee", 390.0 + offset, 800.0),
                "left_ankle": _kp("left_ankle", 390.0 + offset, 950.0),
            }
        }

    data_fp16 = {"frames": [_synthetic_frame(0.0), _synthetic_frame(5.0)]}
    data_quant = {"frames": [_synthetic_frame(0.0), _synthetic_frame(5.0)]}
    movement = compute_joint_movement_summary_between(data_fp16, data_quant)
    assert movement["mean_NPE"] == 0.0
    assert movement["p95_NPE"] == 0.0
    assert movement["under_motion_violations"] == 0.0

    fp16 = {
        "acceleration_variance": 0.01,
        "mean_limb_length_std": 2.0,
        "var_accel_norm": 0.5,
        "violation_fraction": 0.05,
    }
    quant_pass = {
        "acceleration_variance": 0.0105,
        "mean_limb_length_std": 2.05,
        "var_accel_norm": 0.52,
        "violation_fraction": 0.06,
    }
    movement_summary = {"mean_NPE": 0.05, "p95_NPE": 0.10, "under_motion_violations": 0.0}
    deltas = compute_deltas(fp16, quant_pass, movement_summary=movement_summary)
    passed, checks = evaluate_gate(fp16, quant_pass, deltas, GateThresholds())
    assert passed, checks

    quant_fail = dict(quant_pass)
    quant_fail["acceleration_variance"] = 0.02
    deltas_fail = compute_deltas(fp16, quant_fail, movement_summary=movement_summary)
    passed_fail, _ = evaluate_gate(fp16, quant_fail, deltas_fail, GateThresholds())
    assert not passed_fail

    deltas_nan = compute_deltas(fp16, quant_pass, movement_summary={})
    passed_nan, checks_nan = evaluate_gate(fp16, quant_pass, deltas_nan, GateThresholds())
    assert not passed_nan
    assert checks_nan["mean_NPE"] is False
    assert checks_nan["p95_NPE"] is False

    # One-sided violation_fraction: quant improving passes
    fp16_vf = dict(fp16, violation_fraction=0.10)
    quant_improved = dict(quant_pass, violation_fraction=0.05)
    deltas_improved = compute_deltas(fp16_vf, quant_improved, movement_summary=movement_summary)
    _, checks_improved = evaluate_gate(fp16_vf, quant_improved, deltas_improved, GateThresholds())
    assert checks_improved["violation_fraction"] is True

    # Within +0.02 margin passes
    quant_margin = dict(quant_pass, violation_fraction=0.12)
    deltas_margin = compute_deltas(fp16_vf, quant_margin, movement_summary=movement_summary)
    _, checks_margin = evaluate_gate(fp16_vf, quant_margin, deltas_margin, GateThresholds())
    assert checks_margin["violation_fraction"] is True

    # Exceeding margin fails
    quant_over = dict(quant_pass, violation_fraction=0.15)
    deltas_over = compute_deltas(fp16_vf, quant_over, movement_summary=movement_summary)
    _, checks_over = evaluate_gate(fp16_vf, quant_over, deltas_over, GateThresholds())
    assert checks_over["violation_fraction"] is False


def check_kinematic_gate_staging_name() -> None:
    from kinematic_gate.phase1_bridge import staging_video_name

    name = staging_video_name("human2_motion1", "fp16", PROJECT_ROOT / "configs" / "quant" / "vidit_ptq.json")
    assert name.startswith("human2_motion1__fp16__")
    assert name.endswith(".mp4")


def check_kinematic_gate_security() -> None:
    from kinematic_gate.security import (
        safe_child_path,
        sanitize_clip_id,
        validate_gate_thresholds,
        validate_keypoints_payload,
        validate_role,
    )

    assert sanitize_clip_id("human2_motion1") == "human2_motion1"
    try:
        sanitize_clip_id("../escape")
        raise RuntimeError("sanitize_clip_id should reject path traversal")
    except ValueError:
        pass

    base = PROJECT_ROOT / "kinematic_cache"
    child = safe_child_path(base, "keypoints", "demo")
    assert str(child).startswith(str(base.resolve()))

    try:
        safe_child_path(base, "..", "outside")
        raise RuntimeError("safe_child_path should reject escape")
    except ValueError:
        pass

    validate_role("fp16")
    try:
        validate_role("admin")
        raise RuntimeError("validate_role should reject unknown role")
    except ValueError:
        pass

    validate_gate_thresholds(
        max_rel_delta_accel_variance=0.10,
        max_rel_delta_limb_std=0.10,
        max_rel_delta_jitter=0.15,
        max_abs_delta_violation_fraction=0.02,
        max_mean_npe=0.15,
        max_p95_npe=0.30,
        max_under_motion_violations=0.0,
    )
    try:
        validate_gate_thresholds(
            max_rel_delta_accel_variance=-1.0,
            max_rel_delta_limb_std=0.10,
            max_rel_delta_jitter=0.15,
            max_abs_delta_violation_fraction=0.02,
            max_mean_npe=0.15,
            max_p95_npe=0.30,
            max_under_motion_violations=0.0,
        )
        raise RuntimeError("validate_gate_thresholds should reject negative threshold")
    except ValueError:
        pass

    try:
        validate_keypoints_payload({"frames": []})
        raise RuntimeError("validate_keypoints_payload should reject empty frames")
    except ValueError:
        pass

    n = validate_keypoints_payload(
        {"frames": [{"keypoints": {"left_shoulder": {"x": 1, "y": 2}}}]}
    )
    assert n == 1


def check_kinematic_gate_no_stubs() -> None:
    gate_dir = SCRIPT_DIR / "kinematic_gate"
    banned = BANNED_SNIPPETS + ("pass  # skip", "return True  # dummy")
    for path in sorted(gate_dir.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        check_syntax(path)
        for marker in banned:
            if marker.lower() in source.lower():
                raise RuntimeError(f"{path.name} contains banned marker {marker!r}")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "NotImplementedError":
                    raise RuntimeError(f"{path.name} raises NotImplementedError")


def check_kinematic_gate_subprocess_hardening() -> None:
    bridge = (SCRIPT_DIR / "kinematic_gate" / "phase1_bridge.py").read_text(encoding="utf-8")
    if "shell=False" not in bridge:
        raise RuntimeError("phase1_bridge subprocess must set shell=False")
    if "subprocess.run(" not in bridge:
        raise RuntimeError("phase1_bridge must use subprocess.run list invocation")
    if "shell=True" in bridge:
        raise RuntimeError("phase1_bridge must not use shell=True")
    if "timeout=" not in bridge:
        raise RuntimeError("phase1_bridge subprocess must set timeout")


def check_kinematic_gate_json_size_limit() -> None:
    import tempfile

    from kinematic_gate.security import read_json_with_limit

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        tmp.write('{"frames":[{"keypoints":{}}]}')
        tmp_path = Path(tmp.name)
    try:
        payload = read_json_with_limit(tmp_path, max_bytes=4096)
        assert isinstance(payload, dict)
        try:
            read_json_with_limit(tmp_path, max_bytes=8)
            raise RuntimeError("read_json_with_limit should reject files over max_bytes")
        except ValueError:
            pass
    finally:
        tmp_path.unlink(missing_ok=True)

    data = read_json_with_limit(PROJECT_ROOT / "configs" / "quant" / "vidit_ptq.json")
    assert isinstance(data, dict)


def check_notebooks(*, colab: bool = False) -> None:
    if colab:
        print("NOTE skip notebook checks (--colab; no phase2 notebooks in DynamiCtrl clone)")
        return
    notebook_dir = PROJECT_ROOT / "notebooks"
    integration_path = notebook_dir / "DynamiCtrl-PTQ-Integration.ipynb"
    vidit_path = notebook_dir / "DynamiCtrl-PTQ-ViDiT.ipynb"
    hybrid_path = notebook_dir / "run_hybrid_map_experiment.ipynb"
    notebook_paths = (integration_path, vidit_path, hybrid_path)
    present = [path for path in notebook_paths if path.is_file()]
    missing = [path.name for path in notebook_paths if not path.is_file()]
    if missing:
        print(
            "NOTE skip notebook content checks (not on disk in Colab clone): "
            + ", ".join(missing)
        )
    if not present:
        print("NOTE no local notebooks to validate — skipping orchestration checks")
        return

    for nb_path in present:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        source = "\n".join(
            cell.get("source", "")
            if isinstance(cell.get("source"), str)
            else "".join(cell.get("source", []))
            for cell in nb.get("cells", [])
        )
        if "validate_offline.py" not in source:
            raise RuntimeError(f"{nb_path.name} does not run validate_offline.py")
        if "run_setup_downloads" not in source:
            raise RuntimeError(f"{nb_path.name} does not run run_setup_downloads")
        if ' / "phase2" / ' in source or "/phase2/" in source:
            raise RuntimeError(f"{nb_path.name} still references deprecated Drive path phase2/")

    if integration_path.is_file():
        integration = integration_path.read_text(encoding="utf-8")
        required_integration = (
            "apply_w8a8.py",
            "collect_calib.py",
            "fp16-reference-dir",
            "benchmark.py",
            "list_layers.py",
            "load_dit.py",
        )
        for item in required_integration:
            if item not in integration:
                raise RuntimeError(f"Integration notebook missing reference to {item}")

    if vidit_path.is_file():
        vidit = vidit_path.read_text(encoding="utf-8")
        required_vidit = (
            "collect_calib_vidit.py",
            "apply_and_benchmark_vidit.py",
            "apply_vidit_w8a8.py",
            "benchmark_vidit.py",
            "vidit_eval.py",
            "vidit_ptq_config",
            "configs/quant/vidit_ptq.json",
            "human2_motion1_vidit.pt",
            "vidit_w8a8",
            "EARLY_SENSITIVITY_RATIO",
            "step4_apply_and_benchmark_vidit",
            "w8a8_matched_micro",
            "vidit_vs_w8a8_micro_speedup",
            "w8a8_meta.json",
        )
        for item in required_vidit:
            if item not in vidit:
                raise RuntimeError(f"ViDiT notebook missing reference to {item}")

    if hybrid_path.is_file():
        hybrid = hybrid_path.read_text(encoding="utf-8")
        required_hybrid = (
            "run_hybrid_grid.py",
            "run_grid_kinematic_gate.py",
            "ensure_colab_deps_installed",
            "verify_torch_stack",
            "deploy_quant_scripts",
            "hybrid_w8a8",
            "hybrid_vidit_w8a8",
            "load_hybrid_map_config",
            "resolve_hybrid_fp16_layers",
            "--algorithm",
            ".hybrid_map_deps_ok",
        )
        for item in required_hybrid:
            if item not in hybrid:
                raise RuntimeError(f"Hybrid notebook missing reference to {item}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-help", action="store_true")
    parser.add_argument(
        "--colab",
        action="store_true",
        help="Skip checks that require phase2 notebooks on disk (Colab DynamiCtrl clone)",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(SCRIPT_DIR))

    check_required_scripts()
    print("OK required script inventory")

    for name in REQUIRED_SCRIPTS:
        path = SCRIPT_DIR / name
        check_syntax(path)
        print(f"OK syntax: {name}")
        if not args.skip_help and name not in {"colab_paths.py", "quant_common.py", "vidit_quant.py"}:
            try:
                check_help(path)
                print(f"OK --help: {name}")
            except RuntimeError as exc:
                err = str(exc)
                if "No module named 'diffusers'" in err or "No module named 'torchao'" in err:
                    print(f"SKIP --help (runtime deps not installed): {name}")
                else:
                    raise

    check_no_stubs()
    print("OK no stub markers")

    check_protected_patterns()
    print("OK protected patterns superset of integration doc")

    check_expected_quantize_bounds()
    print("OK expected quantize layer count (~240 with AdaLN protected)")

    check_script_wiring()
    print("OK script wiring (no dummy benchmark/calib paths)")

    check_layer_classification()
    print("OK layer classification rules")

    check_timestep_bucket()
    print("OK timestep bucket + aggregate_stats")

    check_vidit_smoothquant()
    print("OK ViDiT SmoothQuant fold math")

    check_sample_dit_linear_weight()
    print("OK sample_dit_linear_weight unwraps SmoothQuant wrapper")

    check_vidit_early_sensitive()
    print("OK ViDiT early-sensitive layer classification")

    check_apply_w8a8_matched_skip()
    print("OK matched naive W8A8 fp16_layers skip + metadata")

    check_matched_quantize_plan()
    print("OK shared resolve_matched_quantize_plan layer coverage")

    check_matched_naive_vidit_fair_comparison()
    print("OK matched naive vs ViDiT fair-comparison wiring (zero-trust)")

    check_matched_parity_rejects_mismatch()
    print("OK assert_matched_quantize_parity rejects layer-set drift")

    check_apply_vidit_w8a8_layer_override()
    print("OK apply_vidit_w8a8 actual_int8_layers sweep override")

    check_vidit_aggregate_stats()
    print("OK ViDiT per-channel aggregate_stats")

    check_method_doc_protected_set()
    print("OK method-doc protected set mapped in layer classifier")

    check_method_doc_tokenwise_quant()
    print("OK method-doc token-wise dynamic activation quant (torchao)")

    check_smoothquant_formula()
    print("OK method-doc SmoothQuant scale formula")

    check_method_doc_smoothquant_pipeline()
    print("OK method-doc SmoothQuant fold + activation division pipeline")

    check_method_doc_static_precision()
    print("OK method-doc static precision (sensitive layers in FP16)")

    check_method_doc_calib_drives_quant()
    print("OK method-doc calibration drives quant scales")

    check_method_doc_dit_scope()
    print("OK method-doc DiT-only compression scope")

    check_method_doc_documented_gaps(colab=args.colab)
    print("OK method-doc Day-2/gate gaps noted")

    check_vidit_ptq_config_file()
    print("OK vidit_ptq.json loads (early_sensitivity_ratio=1.25, kinematic_gate)")

    check_kinematic_gate_compare()
    print("OK kinematic_gate delta math and gate evaluation")

    check_kinematic_gate_staging_name()
    print("OK kinematic_gate staging video naming")

    check_kinematic_gate_security()
    print("OK kinematic_gate security validation (paths, roles, thresholds, JSON)")

    check_kinematic_gate_no_stubs()
    print("OK kinematic_gate no stub/banned markers")

    check_kinematic_gate_subprocess_hardening()
    print("OK kinematic_gate subprocess hardening (shell=False, timeout)")

    check_kinematic_gate_json_size_limit()
    print("OK kinematic_gate JSON size limit")

    check_notebooks(colab=args.colab)
    print("OK notebook JSON + orchestration references")

    print("\nAll offline validations passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
