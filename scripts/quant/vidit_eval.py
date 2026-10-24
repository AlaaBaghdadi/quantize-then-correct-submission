"""Shared ViDiT-Q apply, PSNR, and benchmark orchestration."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

try:
    from skimage.metrics import peak_signal_noise_ratio
except ImportError:
    peak_signal_noise_ratio = None

from quant_common import (
    apply_w8a8,
    capture_dit_forward_kwargs,
    compile_model,
    load_quantize_names,
    microbenchmark_dit,
    parse_dtype,
    prepare_control_video,
    prepare_fp16_pipeline,
    resolve_repo_path,
    run_inference,
    run_timed_inference,
)
from vidit_quant import apply_vidit_w8a8, assert_matched_quantize_parity, resolve_early_fp16_layers, resolve_matched_quantize_plan


def frames_to_arrays(frames) -> list[np.ndarray]:
    return [np.array(frame.convert("RGB"), dtype=np.float32) for frame in frames]


def load_reference_frames_from_dir(ref_dir: Path, num_frames: int) -> list[np.ndarray]:
    import imageio.v3 as iio

    mp4s = sorted(ref_dir.glob("*.mp4"))
    if not mp4s:
        raise FileNotFoundError(f"No MP4 files in FP16 reference dir: {ref_dir}")
    frames = []
    for frame in iio.imiter(str(mp4s[0])):
        frames.append(np.array(frame, dtype=np.float32))
        if len(frames) >= num_frames:
            break
    if not frames:
        raise RuntimeError(f"Could not decode frames from {mp4s[0]}")
    return frames


def compute_psnr(ref_frames, quant_frames) -> float:
    if peak_signal_noise_ratio is None:
        raise RuntimeError("scikit-image is required for PSNR (pip install scikit-image)")
    n = min(len(ref_frames), len(quant_frames))
    psnrs = [
        peak_signal_noise_ratio(ref_frames[i], quant_frames[i], data_range=255.0)
        for i in range(n)
    ]
    return float(np.mean(psnrs))


def build_matched_kernel_conclusion(w8a8_micro: dict, vidit_micro: dict) -> tuple[float, str]:
    """Compare ViDiT vs matched naive W8A8 at equal INT8 layer count (>1 means ViDiT is faster)."""
    speedup = w8a8_micro["sec_per_forward"] / max(vidit_micro["sec_per_forward"], 1e-9)
    if speedup > 1.02:
        conclusion = "ViDiT faster than matched naive W8A8 at equal INT8 layer count"
    elif speedup < 0.98:
        conclusion = "ViDiT slower than matched naive W8A8 (SmoothQuant wrapper overhead)"
    else:
        conclusion = "ViDiT and matched naive W8A8 have similar DiT forward latency"
    return speedup, conclusion


def build_kernel_conclusion(fp16_micro: dict, quant_micro: dict) -> tuple[float, str]:
    speedup = fp16_micro["sec_per_forward"] / max(quant_micro["sec_per_forward"], 1e-9)
    if speedup >= 1.15:
        kernel_conclusion = "INT8 GEMM likely working (DiT forward speedup >= 1.15x)"
    elif quant_micro["peak_vram_gb"] < fp16_micro["peak_vram_gb"] * 0.95:
        kernel_conclusion = "Memory-only win; no meaningful latency gain detected"
    else:
        kernel_conclusion = "Inconclusive; inspect profiler or weight types"
    return speedup, kernel_conclusion


def export_vidit_outputs(
    out_dir: Path,
    quant_frames,
    ref_arrays: list[np.ndarray],
    vidit_meta: dict,
    args,
    *,
    compiled: bool,
) -> tuple[Path, float]:
    from diffusers.utils import export_to_video

    out_dir.mkdir(parents=True, exist_ok=True)
    image_name = Path(args.reference_image).stem
    video_name = Path(args.driving_video).stem
    out_mp4 = out_dir / f"vidit_w8a8_{image_name}_{video_name}_seed{args.seed}.mp4"
    export_to_video(quant_frames[: args.num_frames], str(out_mp4), fps=args.fps)
    print(f"ViDiT-Q W8A8 output: {out_mp4}")

    psnr = compute_psnr(ref_arrays, frames_to_arrays(quant_frames[: args.num_frames]))
    print(f"Mean PSNR vs FP16: {psnr:.2f} dB")
    (out_dir / "psnr.txt").write_text(f"mean_psnr={psnr:.4f}\n")

    meta_out = {
        **vidit_meta,
        "mean_psnr": psnr,
        "compiled": compiled,
        "reference_image": args.reference_image,
        "driving_video": args.driving_video,
    }
    (out_dir / "vidit_meta.json").write_text(json.dumps(meta_out, indent=2))
    return out_mp4, psnr


def export_w8a8_outputs(
    out_dir: Path,
    quant_frames,
    ref_arrays: list[np.ndarray],
    args,
    *,
    clip_id: str | None = None,
    w8a8_meta: dict | None = None,
) -> tuple[Path, float]:
    from diffusers.utils import export_to_video

    out_dir.mkdir(parents=True, exist_ok=True)
    if clip_id is not None:
        out_mp4 = out_dir / f"w8a8_{clip_id}_seed{args.seed}.mp4"
    else:
        image_name = Path(args.reference_image).stem
        video_name = Path(args.driving_video).stem
        out_mp4 = out_dir / f"w8a8_{image_name}_{video_name}_seed{args.seed}.mp4"
    export_to_video(quant_frames[: args.num_frames], str(out_mp4), fps=args.fps)
    print(f"W8A8 output: {out_mp4}")

    psnr = compute_psnr(ref_arrays, frames_to_arrays(quant_frames[: args.num_frames]))
    print(f"Mean PSNR vs FP16: {psnr:.2f} dB")
    (out_dir / "psnr.txt").write_text(f"mean_psnr={psnr:.4f}\n")
    if w8a8_meta is not None:
        meta_out = {
            **w8a8_meta,
            "mean_psnr": psnr,
            "reference_image": args.reference_image,
            "driving_video": args.driving_video,
        }
        (out_dir / "w8a8_meta.json").write_text(json.dumps(meta_out, indent=2))
    return out_mp4, psnr


def run_timed_vidit_e2e_on_pipe(
    pipe,
    args,
    image,
    control,
    *,
    e2e_warmup: int,
    compiled: bool,
    vidit_meta: dict,
) -> tuple[float, float, list, dict]:
    for i in range(e2e_warmup):
        print(f"E2E warmup {i + 1}/{e2e_warmup}...")
        run_inference(pipe, args, image, control)
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    quant_frames = run_inference(pipe, args, image, control)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    meta = {
        "compiled": compiled,
        "e2e_warmup": e2e_warmup,
        "vidit": vidit_meta,
    }
    return wall, peak, quant_frames, meta


def run_combined_apply_and_benchmark(args) -> dict:
    """Apply ViDiT-Q W8A8, export quality artifacts, and benchmark FP16 vs matched naive vs ViDiT."""
    repo = resolve_repo_path(args.repo_root, ".")
    dtype = parse_dtype(args.dtype)
    quantize_names = load_quantize_names(
        resolve_repo_path(repo, args.quantize_layers)
    )
    calib_path = resolve_repo_path(repo, args.calib)
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    if not args.fp16_reference_dir:
        raise ValueError("--fp16-reference-dir is required for combined apply+benchmark")

    model_path = str(resolve_repo_path(repo, args.model_path))
    args.reference_image = str(resolve_repo_path(repo, args.reference_image))
    args.driving_video = str(resolve_repo_path(repo, args.driving_video))
    out_dir = resolve_repo_path(repo, args.output_path)
    log_path = Path(args.log_json)
    if not log_path.is_absolute():
        log_path = resolve_repo_path(repo, args.log_json)

    ref_arrays = load_reference_frames_from_dir(
        Path(args.fp16_reference_dir), args.num_frames
    )

    early_fp16 = resolve_early_fp16_layers(
        calib_path,
        early_ratio_threshold=args.early_ratio,
        max_early_fp16_layers=args.max_early_fp16_layers,
    )
    fp16_layers = set(early_fp16)
    quantize_plan = resolve_matched_quantize_plan(
        quantize_names,
        calib_path,
        early_ratio_threshold=args.early_ratio,
        max_early_fp16_layers=args.max_early_fp16_layers,
    )
    print(
        f"Calib-selected FP16 layers (shared by matched naive and ViDiT): {len(fp16_layers)}"
    )
    print(f"Shared INT8 target layers: {quantize_plan['quantized_layers']}")

    e2e_warmup = args.e2e_warmup if args.e2e_warmup is not None else (1 if args.compile else 0)
    if args.compile and e2e_warmup > 0:
        print(
            f"\nE2E: each variant runs {e2e_warmup} untimed warmup(s) before timing "
            "(compile cost is excluded from wall_sec)."
        )

    image, control = prepare_control_video(
        args.reference_image,
        args.driving_video,
        args.width,
        args.height,
        args.num_frames,
    )

    print("Benchmarking FP16 baseline (DiT micro)...")
    fp16_pipe = prepare_fp16_pipeline(model_path, dtype)
    forward_kwargs = capture_dit_forward_kwargs(fp16_pipe, args)
    fp16_micro = microbenchmark_dit(
        fp16_pipe.transformer,
        forward_kwargs,
        repeats=args.micro_repeats,
        warmup=args.warmup_repeats,
        compile=args.compile,
    )
    del fp16_pipe
    torch.cuda.empty_cache()

    print("\nBenchmarking matched naive W8A8 (DiT micro)...")
    w8a8_pipe = prepare_fp16_pipeline(model_path, dtype)
    w8a8_matched_meta = apply_w8a8(
        w8a8_pipe.transformer,
        quantize_names,
        quantize_plan=quantize_plan,
        calib_path=calib_path,
    )
    print(
        f"Matched naive W8A8 applied to {w8a8_matched_meta['quantized_layers']} layers "
        f"({len(fp16_layers)} FP16 layers skipped)"
    )
    w8a8_compiled = False
    if args.compile:
        try:
            w8a8_pipe.transformer = compile_model(w8a8_pipe.transformer)
            w8a8_compiled = True
        except Exception as exc:
            print(f"WARN: torch.compile failed on matched W8A8 ({exc}); running eager.")
    w8a8_matched_micro = microbenchmark_dit(
        w8a8_pipe.transformer,
        forward_kwargs,
        repeats=args.micro_repeats,
        warmup=args.warmup_repeats,
        compile=False,
    )
    w8a8_matched_micro["compiled"] = w8a8_compiled
    del w8a8_pipe
    torch.cuda.empty_cache()

    print("\nApplying ViDiT-Q W8A8 (fresh pipeline)...")
    vidit_pipe = prepare_fp16_pipeline(model_path, dtype)
    vidit_meta = apply_vidit_w8a8(
        vidit_pipe.transformer,
        quantize_names,
        calib_path,
        smoothquant_alpha=args.smoothquant_alpha,
        early_ratio_threshold=args.early_ratio,
        max_early_fp16_layers=args.max_early_fp16_layers,
    )
    print(f"ViDiT-Q W8A8 applied to {vidit_meta['quantized_layers']} layers")
    print(f"SmoothQuant folded: {len(vidit_meta['smoothquant_layers'])} layers")
    print(
        f"Static FP16 layers (all timesteps): {len(vidit_meta['early_fp16_layers'])} layers"
    )
    assert_matched_quantize_parity(w8a8_matched_meta, vidit_meta)
    print("OK: matched naive and ViDiT quantize the same layer sets")

    vidit_compiled = False
    if args.compile:
        try:
            vidit_pipe.transformer = compile_model(vidit_pipe.transformer)
            vidit_compiled = True
        except Exception as exc:
            print(f"WARN: torch.compile failed on ViDiT ({exc}); running eager.")

    print("\nBenchmarking ViDiT-Q W8A8 (DiT micro)...")
    vidit_micro = microbenchmark_dit(
        vidit_pipe.transformer,
        forward_kwargs,
        repeats=args.micro_repeats,
        warmup=args.warmup_repeats,
        compile=False,
    )
    vidit_micro["compiled"] = vidit_compiled

    print("Running timed ViDiT-Q W8A8 inference (quality + e2e timing)...")
    vidit_wall, vidit_peak, quant_frames, vidit_e2e_meta = run_timed_vidit_e2e_on_pipe(
        vidit_pipe,
        args,
        image,
        control,
        e2e_warmup=e2e_warmup,
        compiled=vidit_compiled,
        vidit_meta=vidit_meta,
    )
    del vidit_pipe
    torch.cuda.empty_cache()

    out_mp4, psnr = export_vidit_outputs(
        out_dir,
        quant_frames,
        ref_arrays,
        vidit_meta,
        args,
        compiled=vidit_compiled,
    )

    print("\nRunning timed matched naive W8A8 inference (e2e timing)...")
    w8a8_wall, w8a8_peak, _, w8a8_e2e_meta = run_timed_inference(
        model_path,
        dtype,
        args,
        quantize_names=quantize_names,
        fp16_layers=fp16_layers,
        calib_path=calib_path,
        compile=args.compile,
        e2e_warmup=e2e_warmup,
    )

    vidit_fp16_speedup, kernel_conclusion = build_kernel_conclusion(fp16_micro, vidit_micro)
    w8a8_fp16_speedup, _ = build_kernel_conclusion(fp16_micro, w8a8_matched_micro)
    vidit_vs_w8a8_micro, matched_conclusion = build_matched_kernel_conclusion(
        w8a8_matched_micro, vidit_micro
    )
    vidit_vs_w8a8_e2e = w8a8_wall / max(vidit_wall, 1e-9)

    report = {
        "fp16_micro": fp16_micro,
        "w8a8_matched_micro": w8a8_matched_micro,
        "vidit_w8a8_micro": vidit_micro,
        "w8a8_matched_meta": w8a8_matched_meta,
        "quantize_plan": quantize_plan,
        "dit_forward_speedup_vs_fp16": {
            "w8a8_matched": w8a8_fp16_speedup,
            "vidit_w8a8": vidit_fp16_speedup,
        },
        "dit_forward_speedup": vidit_fp16_speedup,
        "vidit_vs_w8a8_micro_speedup": vidit_vs_w8a8_micro,
        "vidit_vs_w8a8_e2e_speedup": vidit_vs_w8a8_e2e,
        "kernel_conclusion": kernel_conclusion,
        "matched_conclusion": matched_conclusion,
        "vidit_meta": vidit_meta,
        "calib_path": str(calib_path),
        "mean_psnr": psnr,
        "output_mp4": str(out_mp4),
        "output_dir": str(out_dir),
        "w8a8_matched_e2e": {
            "wall_sec": w8a8_wall,
            "peak_vram_gb": w8a8_peak,
            "sec_per_step": w8a8_wall / args.num_inference_steps,
            **w8a8_e2e_meta,
        },
        "vidit_w8a8_e2e": {
            "wall_sec": vidit_wall,
            "peak_vram_gb": vidit_peak,
            "sec_per_step": vidit_wall / args.num_inference_steps,
            **vidit_e2e_meta,
        },
    }

    if not args.skip_fp16_e2e:
        fp16_wall, fp16_peak, _, fp16_meta = run_timed_inference(
            model_path,
            dtype,
            args,
            quantize_names=None,
            compile=args.compile,
            e2e_warmup=e2e_warmup,
        )
        report["fp16_e2e"] = {
            "wall_sec": fp16_wall,
            "peak_vram_gb": fp16_peak,
            "sec_per_step": fp16_wall / args.num_inference_steps,
            **fp16_meta,
        }
        report["e2e_speedup"] = fp16_wall / max(vidit_wall, 1e-9)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {log_path}")
    print(f"Conclusion (FP16 vs ViDiT): {kernel_conclusion}")
    print(f"Conclusion (matched naive vs ViDiT): {matched_conclusion}")
    return report
