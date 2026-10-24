"""Run pose_corrector in-process with variant kwargs for benchmarks and ablations."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from experiments.benchmark_comparison.config import KAPT_ALPHA, PROJECT_ROOT
from pose_handler.common_io import read_json, write_json


def _load_pose_corrector_module():
    """Import pose_corrector with module registered (avoids dataclass import issues)."""
    path = PROJECT_ROOT / "pose-corrector" / "pose_corrector.py"
    if not path.is_file():
        raise FileNotFoundError(f"pose_corrector not found: {path}")
    name = "pose_corrector_kapt_runner"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pose_corrector from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_kapt_correction(
    in_json: Path,
    out_json: Path,
    *,
    alpha: float = KAPT_ALPHA,
    hard_clamp: bool = False,
    max_drift: float = 0.05,
    force: bool = False,
    variant_id: Optional[str] = None,
) -> None:
    """
    Apply PoseCorrector to a keypoints JSON and write corrected output.

    Skips if out_json exists unless force=True.
    """
    if out_json.is_file() and not force:
        return

    in_json = in_json.resolve()
    if not in_json.is_file():
        raise FileNotFoundError(f"Input keypoints not found: {in_json}")

    out_json.parent.mkdir(parents=True, exist_ok=True)

    mod = _load_pose_corrector_module()
    process_video_keypoints = mod.process_video_keypoints
    PITCH_MIN = mod.PITCH_MIN
    PITCH_MAX = mod.PITCH_MAX
    LEFT_YAW_MIN = mod.LEFT_YAW_MIN
    LEFT_YAW_MAX = mod.LEFT_YAW_MAX
    RIGHT_YAW_MIN = mod.RIGHT_YAW_MIN
    RIGHT_YAW_MAX = mod.RIGHT_YAW_MAX

    data = read_json(in_json)
    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"No frames in {in_json}")

    corrected_frames, stats = process_video_keypoints(
        frames,
        alpha=alpha,
        max_drift=max_drift,
        use_soft_cone=not hard_clamp,
    )

    output = copy.deepcopy(data)
    output["frames"] = corrected_frames
    output["format"] = "pose_corrector/constrained_keypoints_v1"
    output["correction_params"] = {
        "variant_id": variant_id,
        "alpha": alpha,
        "hard_clamp": hard_clamp,
        "max_drift": max_drift,
        "use_soft_cone": not hard_clamp,
        "pitch_range": [PITCH_MIN, PITCH_MAX],
        "left_yaw_range": [LEFT_YAW_MIN, LEFT_YAW_MAX],
        "right_yaw_range": [RIGHT_YAW_MIN, RIGHT_YAW_MAX],
        "stats": stats,
    }
    write_json(out_json, output)
