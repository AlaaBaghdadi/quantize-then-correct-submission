"""Default paths and settings for the video Kalman pipeline."""

import os
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_default_openpose_root() -> Path:
    env_root = os.environ.get("OPENPOSE_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return get_project_root() / "openpose-1.7.0-binaries-win64-cpu-python3.7-flir-3d" / "openpose"


def get_default_out_dir() -> Path:
    return get_project_root() / "kalman_out"
