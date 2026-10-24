"""
Shared pose I/O helpers.

These functions re-export existing implementations from `kalman.video_pose_io`
under a neutral namespace so that both video and image pipelines can share
them without duplicating code.
"""

from .openpose_converter import openpose_json_to_named_keypoints  # noqa: F401
from .mediapipe_scaler import (  # noqa: F401
    scale_mediapipe_entries,
    calc_scale_factor,
)
from .keypoint_averager import average_frame_keypoints  # noqa: F401

__all__ = [
    "openpose_json_to_named_keypoints",
    "scale_mediapipe_entries",
    "calc_scale_factor",
    "average_frame_keypoints",
]

