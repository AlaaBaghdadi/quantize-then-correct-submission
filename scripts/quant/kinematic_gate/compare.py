"""Delta math and gate evaluation for FP16 vs quantized kinematic metrics."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

_EPS = 1e-9


def rel_delta(baseline: float, candidate: float, *, eps: float = _EPS) -> float:
    """Relative change: |candidate - baseline| / max(|baseline|, eps)."""
    if not (math.isfinite(baseline) and math.isfinite(candidate)):
        return float("nan")
    return abs(candidate - baseline) / max(abs(baseline), eps)


def abs_delta(baseline: float, candidate: float) -> float:
    """Absolute difference."""
    if not (math.isfinite(baseline) and math.isfinite(candidate)):
        return float("nan")
    return abs(candidate - baseline)


@dataclass(frozen=True)
class GateThresholds:
    max_rel_delta_accel_variance: float = 0.10
    max_rel_delta_limb_std: float = 0.10
    max_rel_delta_jitter: float = 0.15
    max_abs_delta_violation_fraction: float = 0.02
    max_mean_npe: float = 0.15
    max_p95_npe: float = 0.30
    max_under_motion_violations: float = 0.0

    def __post_init__(self) -> None:
        from .security import validate_gate_thresholds

        validate_gate_thresholds(
            max_rel_delta_accel_variance=self.max_rel_delta_accel_variance,
            max_rel_delta_limb_std=self.max_rel_delta_limb_std,
            max_rel_delta_jitter=self.max_rel_delta_jitter,
            max_abs_delta_violation_fraction=self.max_abs_delta_violation_fraction,
            max_mean_npe=self.max_mean_npe,
            max_p95_npe=self.max_p95_npe,
            max_under_motion_violations=self.max_under_motion_violations,
        )


def compute_deltas(
    metrics_fp16: Mapping[str, float],
    metrics_quant: Mapping[str, float],
    *,
    movement_summary: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Compute all delta values between FP16 baseline and quantized candidate."""
    deltas: Dict[str, float] = {
        "rel_delta_acceleration_variance": rel_delta(
            float(metrics_fp16.get("acceleration_variance", float("nan"))),
            float(metrics_quant.get("acceleration_variance", float("nan"))),
        ),
        "rel_delta_mean_limb_length_std": rel_delta(
            float(metrics_fp16.get("mean_limb_length_std", float("nan"))),
            float(metrics_quant.get("mean_limb_length_std", float("nan"))),
        ),
        "rel_delta_var_accel_norm": rel_delta(
            float(metrics_fp16.get("var_accel_norm", float("nan"))),
            float(metrics_quant.get("var_accel_norm", float("nan"))),
        ),
        "abs_delta_violation_fraction": abs_delta(
            float(metrics_fp16.get("violation_fraction", float("nan"))),
            float(metrics_quant.get("violation_fraction", float("nan"))),
        ),
        "abs_delta_total_violations": abs_delta(
            float(metrics_fp16.get("total_violations", float("nan"))),
            float(metrics_quant.get("total_violations", float("nan"))),
        ),
    }
    if movement_summary:
        deltas["mean_NPE"] = float(movement_summary.get("mean_NPE", float("nan")))
        deltas["p95_NPE"] = float(movement_summary.get("p95_NPE", float("nan")))
        deltas["under_motion_violations"] = float(
            movement_summary.get("under_motion_violations", float("nan"))
        )
    return deltas


def evaluate_gate(
    metrics_fp16: Mapping[str, float],
    metrics_quant: Mapping[str, float],
    deltas: Mapping[str, float],
    thresholds: GateThresholds,
) -> Tuple[bool, Dict[str, bool]]:
    """
    Evaluate pass/fail per metric. Returns (overall_passed, per_check).
    """
    checks: Dict[str, bool] = {}

    def _check(name: str, delta: float, limit: float) -> None:
        if not math.isfinite(delta):
            checks[name] = False
            return
        checks[name] = delta <= limit

    _check(
        "acceleration_variance",
        float(deltas.get("rel_delta_acceleration_variance", float("nan"))),
        thresholds.max_rel_delta_accel_variance,
    )
    _check(
        "mean_limb_length_std",
        float(deltas.get("rel_delta_mean_limb_length_std", float("nan"))),
        thresholds.max_rel_delta_limb_std,
    )
    _check(
        "var_accel_norm",
        float(deltas.get("rel_delta_var_accel_norm", float("nan"))),
        thresholds.max_rel_delta_jitter,
    )

    vf_fp16 = float(metrics_fp16.get("violation_fraction", float("nan")))
    vf_quant = float(metrics_quant.get("violation_fraction", float("nan")))
    if not (math.isfinite(vf_fp16) and math.isfinite(vf_quant)):
        checks["violation_fraction"] = False
    else:
        checks["violation_fraction"] = (
            vf_quant <= vf_fp16 + thresholds.max_abs_delta_violation_fraction
        )

    mean_npe = float(deltas.get("mean_NPE", float("nan")))
    p95_npe = float(deltas.get("p95_NPE", float("nan")))
    under_motion = float(deltas.get("under_motion_violations", float("nan")))

    checks["mean_NPE"] = math.isfinite(mean_npe) and mean_npe <= thresholds.max_mean_npe
    checks["p95_NPE"] = math.isfinite(p95_npe) and p95_npe <= thresholds.max_p95_npe
    checks["under_motion_violations"] = (
        math.isfinite(under_motion)
        and under_motion <= thresholds.max_under_motion_violations
    )

    for key in ("acceleration_variance", "mean_limb_length_std", "var_accel_norm", "violation_fraction"):
        v0 = float(metrics_fp16.get(key, float("nan")))
        v1 = float(metrics_quant.get(key, float("nan")))
        if not (math.isfinite(v0) and math.isfinite(v1)):
            checks[f"finite_{key}"] = False
        else:
            checks[f"finite_{key}"] = True

    passed = all(checks.values())
    return passed, checks
