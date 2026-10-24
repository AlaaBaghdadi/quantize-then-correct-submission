"""Hyperparameter sensitivity sweep definitions (Experiment C, extended)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Literal

from experiments.benchmark_comparison.config import KAPT_ALPHA

SensitivityParam = Literal["alpha", "max_drift", "process_noise"]

DEFAULT_MAX_DRIFT = 0.05
DEFAULT_PROCESS_NOISE = 1e-2
DEFAULT_MEASUREMENT_NOISE = 1e-1

# One-at-a-time grids (default value included in each list).
ALPHA_SWEEP: List[float] = [0.3, 0.5, 0.7, 0.9]
MAX_DRIFT_SWEEP: List[float] = [0.02, 0.05, 0.10, 0.20]
PROCESS_NOISE_SWEEP: List[float] = [1e-3, 1e-2, 1e-1]

DEFAULTS = {
    "alpha": KAPT_ALPHA,
    "max_drift": DEFAULT_MAX_DRIFT,
    "process_noise": DEFAULT_PROCESS_NOISE,
}

PRIMARY_METRICS = ["accel_var", "angle_viol_frac", "limb_std", "jitter"]


@dataclass(frozen=True)
class SensitivitySweep:
    param: SensitivityParam
    display_name: str
    values: List[float]
    default: float
    primary_metric: str  # metric emphasized in figure caption


SENSITIVITY_SWEEPS: List[SensitivitySweep] = [
    SensitivitySweep(
        param="alpha",
        display_name="EMA alpha",
        values=ALPHA_SWEEP,
        default=KAPT_ALPHA,
        primary_metric="accel_var",
    ),
    SensitivitySweep(
        param="max_drift",
        display_name="Length max_drift",
        values=MAX_DRIFT_SWEEP,
        default=DEFAULT_MAX_DRIFT,
        primary_metric="limb_std",
    ),
    SensitivitySweep(
        param="process_noise",
        display_name="Kalman process noise (Q)",
        values=PROCESS_NOISE_SWEEP,
        default=DEFAULT_PROCESS_NOISE,
        primary_metric="jitter",
    ),
]


def format_sweep_value(param: SensitivityParam, value: float) -> str:
    """Filesystem-safe tag for a sweep value (e.g. 0.7, 1e-2)."""
    if param == "process_noise":
        if value <= 0:
            return f"{value:g}"
        exp = round(math.log10(value))
        if abs(value - 10.0 ** exp) < 1e-15 * max(1.0, abs(value)):
            return f"1e{exp}"
    return f"{value:g}"


def condition_dir_name(param: SensitivityParam, value: float) -> str:
    tag = format_sweep_value(param, value)
    return f"sensitivity_{param}_{tag}"
