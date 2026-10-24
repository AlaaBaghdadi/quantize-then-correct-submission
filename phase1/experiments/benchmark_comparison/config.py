"""Defaults for KAPT vs SmoothNet comparison."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_KALMAN_DIR = PROJECT_ROOT / "kalman_out"
DEFAULT_WORK_DIR = PROJECT_ROOT / "experiments" / "reviewer_r5_r6"

# Primary benchmark clips (exclude *_comparison* re-runs)
DEFAULT_VIDEO_STEMS = [
    "man1",
    "man2",
    "man3",
    "woman1",
    "woman2",
    "woman3",
]

KALMAN_JSON_SUFFIX = "_smoothed_keypoints.json"
KAPT_ALPHA = 0.7

METRIC_COLUMNS = [
    "accel_var",
    "viol_frac",
    "limb_std",
    "jitter",
]
