"""Kinematic delta gate: compare FP16 vs quantized video pose metrics via phase1."""

from .compare import evaluate_gate, compute_deltas
from .metrics import (
    compute_joint_movement_summary_between,
    compute_metric_vector,
    load_keypoints_json,
)

__all__ = [
    "compute_deltas",
    "evaluate_gate",
    "compute_joint_movement_summary_between",
    "compute_metric_vector",
    "load_keypoints_json",
]
