"""Defaults and SmoothNet model hyperparameters for the baseline runner."""

from __future__ import annotations

from pathlib import Path

# Project root: experiments/smoothnet_baseline/config.py -> phase1/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_WORK_DIR = PROJECT_ROOT / "experiments" / "reviewer_r5_r6"
DEFAULT_SMOOTHNET_ROOT = PROJECT_ROOT / "third_party" / "SmoothNet"
DEFAULT_CHECKPOINT = (
    DEFAULT_SMOOTHNET_ROOT / "data" / "checkpoints" / "h36m_fcn_3D" / "checkpoint_32.pth.tar"
)

# Verified against checkpoint_32.pth.tar load test
WINDOW_SIZE = 32
WINDOW_STEP = 1
HIDDEN_SIZE = 512
RES_HIDDEN_SIZE = 128
NUM_BLOCKS = 5
DROPOUT = 0.25
INFERENCE_BATCH_SIZE = 64

H36M_NUM_JOINTS = 17
H36M_FLAT_DIM = H36M_NUM_JOINTS * 2  # 34

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
KALMAN_JSON_SUFFIX = "_smoothed_keypoints.json"
