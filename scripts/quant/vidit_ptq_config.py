"""Load and validate DynamiCtrl ViDiT-Q PTQ settings from configs/quant/vidit_ptq.json."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

DEFAULT_CONFIG_REL = Path("configs/quant/vidit_ptq.json")


def _repo_root_from_script() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def default_config_path(repo_root: Path | None = None) -> Path:
    root = repo_root or _repo_root_from_script()
    return root / DEFAULT_CONFIG_REL


def _require_section(data: dict, key: str) -> dict:
    section = data.get(key)
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{key}' must be an object")
    return section


def _require_float(section: dict, key: str) -> float:
    if key not in section:
        raise KeyError(f"Missing required config key: {key}")
    return float(section[key])


def _require_int(section: dict, key: str) -> int:
    if key not in section:
        raise KeyError(f"Missing required config key: {key}")
    return int(section[key])


def _require_bool(section: dict, key: str) -> bool:
    if key not in section:
        raise KeyError(f"Missing required config key: {key}")
    value = section[key]
    if not isinstance(value, bool):
        raise TypeError(f"Config key '{key}' must be a boolean")
    return value


def _require_str(section: dict, key: str) -> str:
    if key not in section:
        raise KeyError(f"Missing required config key: {key}")
    value = section[key]
    if not isinstance(value, str):
        raise TypeError(f"Config key '{key}' must be a string")
    return value


def load_vidit_ptq_config(path: Path | str | None = None) -> SimpleNamespace:
    """Load vidit_ptq.json and expose flat attributes for notebooks and scripts."""
    config_path = Path(path) if path is not None else default_config_path()
    if not config_path.is_file():
        raise FileNotFoundError(
            f"ViDiT PTQ config not found: {config_path}\n"
            "Upload configs/quant/vidit_ptq.json to Drive under diffusion_quantization/configs/quant/."
        )

    data = json.loads(config_path.read_text(encoding="utf-8"))
    paths = _require_section(data, "paths")
    setup = _require_section(data, "setup")
    clip = _require_section(data, "clip")
    inference = _require_section(data, "inference")
    vidit = _require_section(data, "vidit_quantization")
    benchmark = _require_section(data, "benchmark")
    quality = _require_section(data, "quality_gate")

    early_ratio = _require_float(vidit, "early_sensitivity_ratio")
    if early_ratio <= 0:
        raise ValueError("early_sensitivity_ratio must be > 0")

    smooth_alpha = _require_float(vidit, "smoothquant_alpha")
    if not 0.0 <= smooth_alpha <= 1.0:
        raise ValueError("smoothquant_alpha must be in [0, 1]")

    return SimpleNamespace(
        config_path=str(config_path.resolve()),
        description=data.get("description", ""),
        DRIVE_ROOT=_require_str(paths, "drive_root"),
        REPO=_require_str(paths, "repo"),
        MODEL_PATH=_require_str(paths, "model_path"),
        CALIB_FILE=_require_str(paths, "calib_file"),
        QUANTIZE_LAYERS=_require_str(paths, "quantize_layers"),
        FP16_REFERENCE_DIR=_require_str(paths, "fp16_reference_dir"),
        VIDIT_OUTPUT_DIR=_require_str(paths, "vidit_output_dir"),
        BENCHMARK_LOG_JSON=_require_str(paths, "benchmark_log_json"),
        RESUME=_require_bool(setup, "resume_downloads"),
        REFERENCE_IMAGE=_require_str(clip, "reference_image"),
        DRIVING_VIDEO=_require_str(clip, "driving_video"),
        PROMPT=_require_str(inference, "prompt"),
        NUM_INFERENCE_STEPS=_require_int(inference, "num_inference_steps"),
        WIDTH=_require_int(inference, "width"),
        HEIGHT=_require_int(inference, "height"),
        NUM_FRAMES=_require_int(inference, "num_frames"),
        POSE_CONTROL_FUNCTION=_require_str(inference, "pose_control_function"),
        GUIDANCE_SCALE=_require_float(inference, "guidance_scale"),
        SEED=_require_int(inference, "seed"),
        FPS=_require_int(inference, "fps"),
        DTYPE=_require_str(inference, "dtype"),
        SMOOTHQUANT_ALPHA=smooth_alpha,
        EARLY_SENSITIVITY_RATIO=early_ratio,
        MAX_EARLY_FP16_LAYERS=_require_int(vidit, "max_early_fp16_layers"),
        USE_COMPILE=_require_bool(vidit, "use_compile"),
        WARMUP_REPEATS=_require_int(benchmark, "warmup_repeats"),
        MIN_PSNR_DB=_require_float(quality, "min_psnr_db"),
        TARGET_PSNR_DB=_require_float(quality, "target_psnr_db"),
    )


DEFAULT_GRID_IMAGES = [
    "./assets/human1.jpg",
    "./assets/human2.jpg",
    "./assets/human3.jpg",
    "./assets/human4.jpg",
]
DEFAULT_GRID_MOTIONS = [
    "./assets/motion1.mp4",
    "./assets/motion2.mp4",
    "./assets/motion3.mp4",
]


def _optional_str_list(section: dict, key: str, default: list[str]) -> list[str]:
    if key not in section:
        return list(default)
    value = section[key]
    if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
        raise TypeError(f"Config key '{key}' must be a non-empty list of strings")
    return value


def load_grid_config(
    path: Path | str | None = None,
    *,
    repo_root: Path | None = None,
) -> SimpleNamespace:
    """Load optional grid section for 4×3 batch runs."""
    config_path = Path(path) if path is not None else default_config_path(repo_root)
    if not config_path.is_file():
        raise FileNotFoundError(f"ViDiT PTQ config not found: {config_path}")

    data = json.loads(config_path.read_text(encoding="utf-8"))
    section = data.get("grid")
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise ValueError("Config section 'grid' must be an object")

    return SimpleNamespace(
        images=_optional_str_list(section, "images", DEFAULT_GRID_IMAGES),
        motions=_optional_str_list(section, "motions", DEFAULT_GRID_MOTIONS),
        fp16_output_dir=section.get("fp16_output_dir", "./outputs/fp16_grid"),
        vidit_output_dir=section.get("vidit_output_dir", "./outputs/vidit_w8a8_grid"),
        w8a8_output_dir=section.get("w8a8_output_dir", "./outputs/w8a8_grid"),
        calib_dir=section.get("calib_dir", "./calib_data"),
    )


def _optional_float(section: dict, key: str, default: float) -> float:
    if key not in section:
        return default
    return float(section[key])


def _optional_bool(section: dict, key: str, default: bool) -> bool:
    if key not in section:
        return default
    value = section[key]
    if not isinstance(value, bool):
        raise TypeError(f"Config key '{key}' must be a boolean")
    return value


def _optional_str_or_none(section: dict, key: str) -> str | None:
    if key not in section:
        return None
    value = section[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"Config key '{key}' must be a string or null")
    return value


DEFAULT_HYBRID_BLOCK_FP16 = (1, 13)
DEFAULT_HYBRID_BLOCK_INT8 = (14, 40)
DEFAULT_HYBRID_SWEEP_IMAGE = "./assets/human2.jpg"
DEFAULT_HYBRID_SWEEP_MOTIONS = [
    "./assets/motion1.mp4",
    "./assets/motion2.mp4",
    "./assets/motion3.mp4",
]


def _optional_int_pair(section: dict, key: str, default: tuple[int, int]) -> tuple[int, int]:
    if key not in section:
        return default
    value = section[key]
    if not isinstance(value, list) or len(value) != 2:
        raise TypeError(f"Config key '{key}' must be a two-element integer list")
    start, end = int(value[0]), int(value[1])
    if start > end:
        raise ValueError(f"Config key '{key}' must have start <= end")
    return start, end


def load_hybrid_map_config(
    path: Path | str | None = None,
    *,
    repo_root: Path | None = None,
) -> SimpleNamespace:
    """Load optional hybrid_map section for block-level mixed-precision W8A8."""
    config_path = Path(path) if path is not None else default_config_path(repo_root)
    root = repo_root or _repo_root_from_script()
    if not config_path.is_file():
        raise FileNotFoundError(f"ViDiT PTQ config not found: {config_path}")

    data = json.loads(config_path.read_text(encoding="utf-8"))
    section = data.get("hybrid_map")
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise ValueError("Config section 'hybrid_map' must be an object")

    block_fp16 = _optional_int_pair(section, "block_fp16", DEFAULT_HYBRID_BLOCK_FP16)
    block_int8 = _optional_int_pair(section, "block_int8", DEFAULT_HYBRID_BLOCK_INT8)
    sweep_image = section.get("sweep_image", DEFAULT_HYBRID_SWEEP_IMAGE)
    sweep_motions = _optional_str_list(
        section, "sweep_motions", DEFAULT_HYBRID_SWEEP_MOTIONS
    )
    naive_output_dir = section.get(
        "naive_output_dir", section.get("output_dir", "./outputs/hybrid_w8a8_grid")
    )
    vidit_output_dir = section.get(
        "vidit_output_dir", "./outputs/hybrid_vidit_w8a8_grid"
    )
    calib_dir = section.get("calib_dir", "./calib_data")
    naive_quant_prefix = section.get("naive_quant_prefix", "hybrid_w8a8")
    vidit_quant_prefix = section.get("vidit_quant_prefix", "hybrid_vidit_w8a8")
    gate_report_dir = section.get("gate_report_dir", "./logs/kinematic_gate_hybrid_grid")
    gate_summary_json = section.get(
        "gate_summary_json", "./logs/kinematic_gate_hybrid_grid.json"
    )
    for key, value in (
        ("sweep_image", sweep_image),
        ("naive_output_dir", naive_output_dir),
        ("vidit_output_dir", vidit_output_dir),
        ("calib_dir", calib_dir),
        ("naive_quant_prefix", naive_quant_prefix),
        ("vidit_quant_prefix", vidit_quant_prefix),
        ("gate_report_dir", gate_report_dir),
        ("gate_summary_json", gate_summary_json),
    ):
        if not isinstance(value, str):
            raise TypeError(f"hybrid_map.{key} must be a string")

    def _resolve_path(rel: str) -> str:
        p = Path(rel)
        return str(
            (root / rel).resolve() if not p.is_absolute() else p.resolve()
        )

    return SimpleNamespace(
        config_path=str(config_path.resolve()),
        block_fp16=block_fp16,
        block_int8=block_int8,
        sweep_image=sweep_image,
        sweep_motions=sweep_motions,
        naive_output_dir=_resolve_path(naive_output_dir),
        vidit_output_dir=_resolve_path(vidit_output_dir),
        calib_dir=_resolve_path(calib_dir),
        naive_quant_prefix=naive_quant_prefix,
        vidit_quant_prefix=vidit_quant_prefix,
        output_dir=_resolve_path(naive_output_dir),
        quant_prefix=naive_quant_prefix,
        gate_report_dir=_resolve_path(gate_report_dir),
        gate_summary_json=_resolve_path(gate_summary_json),
    )


def load_kinematic_gate_config(
    path: Path | str | None = None,
    *,
    repo_root: Path | None = None,
) -> SimpleNamespace:
    """
    Load kinematic_gate section from vidit_ptq.json with backward-compatible defaults.
    """
    config_path = Path(path) if path is not None else default_config_path(repo_root)
    root = repo_root or _repo_root_from_script()
    if not config_path.is_file():
        raise FileNotFoundError(f"ViDiT PTQ config not found: {config_path}")

    data = json.loads(config_path.read_text(encoding="utf-8"))
    section = data.get("kinematic_gate")
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise ValueError("Config section 'kinematic_gate' must be an object")

    phase1_rel = section.get("phase1_root", "../phase1")
    if not isinstance(phase1_rel, str):
        raise TypeError("kinematic_gate.phase1_root must be a string")

    openpose_raw = _optional_str_or_none(section, "openpose_root")
    cache_rel = section.get("cache_dir", "./kinematic_cache")
    report_rel = section.get("report_json", "./logs/kinematic_gate.json")
    if not isinstance(cache_rel, str) or not isinstance(report_rel, str):
        raise TypeError("kinematic_gate cache_dir and report_json must be strings")

    phase1_root = (root / phase1_rel).resolve() if not Path(phase1_rel).is_absolute() else Path(phase1_rel).resolve()
    openpose_root = None
    if openpose_raw:
        openpose_root = (
            (root / openpose_raw).resolve()
            if not Path(openpose_raw).is_absolute()
            else Path(openpose_raw).resolve()
        )

    max_rel_delta_accel_variance = _optional_float(section, "max_rel_delta_accel_variance", 0.10)
    max_rel_delta_limb_std = _optional_float(section, "max_rel_delta_limb_std", 0.10)
    max_rel_delta_jitter = _optional_float(section, "max_rel_delta_jitter", 0.15)
    max_abs_delta_violation_fraction = _optional_float(section, "max_abs_delta_violation_fraction", 0.02)
    max_mean_npe = _optional_float(section, "max_mean_npe", 0.15)
    max_p95_npe = _optional_float(section, "max_p95_npe", 0.30)
    max_under_motion_violations = _optional_float(section, "max_under_motion_violations", 0.0)

    from kinematic_gate.security import validate_gate_thresholds

    validate_gate_thresholds(
        max_rel_delta_accel_variance=max_rel_delta_accel_variance,
        max_rel_delta_limb_std=max_rel_delta_limb_std,
        max_rel_delta_jitter=max_rel_delta_jitter,
        max_abs_delta_violation_fraction=max_abs_delta_violation_fraction,
        max_mean_npe=max_mean_npe,
        max_p95_npe=max_p95_npe,
        max_under_motion_violations=max_under_motion_violations,
    )

    return SimpleNamespace(
        config_path=str(config_path.resolve()),
        phase1_root=phase1_root,
        openpose_root=openpose_root,
        cache_dir=str((root / cache_rel).resolve() if not Path(cache_rel).is_absolute() else Path(cache_rel).resolve()),
        report_json=str((root / report_rel).resolve() if not Path(report_rel).is_absolute() else Path(report_rel).resolve()),
        max_rel_delta_accel_variance=max_rel_delta_accel_variance,
        max_rel_delta_limb_std=max_rel_delta_limb_std,
        max_rel_delta_jitter=max_rel_delta_jitter,
        max_abs_delta_violation_fraction=max_abs_delta_violation_fraction,
        max_mean_npe=max_mean_npe,
        max_p95_npe=max_p95_npe,
        max_under_motion_violations=max_under_motion_violations,
        skip_if_cached=_optional_bool(section, "skip_if_cached", True),
    )


def print_vidit_ptq_config(cfg: SimpleNamespace) -> None:
    """Human-readable summary for notebook cells."""
    print(f"Config: {cfg.config_path}")
    if cfg.description:
        print(cfg.description)
    print(f"  clip: {cfg.REFERENCE_IMAGE} + {cfg.DRIVING_VIDEO}")
    print(f"  smoothquant_alpha={cfg.SMOOTHQUANT_ALPHA}")
    print(f"  early_sensitivity_ratio={cfg.EARLY_SENSITIVITY_RATIO}")
    print(f"  max_early_fp16_layers={cfg.MAX_EARLY_FP16_LAYERS}")
    print(f"  use_compile={cfg.USE_COMPILE}")
    print(f"  min_psnr_db={cfg.MIN_PSNR_DB} (target {cfg.TARGET_PSNR_DB})")
