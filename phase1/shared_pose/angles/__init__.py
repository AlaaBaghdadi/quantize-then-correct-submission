"""
Angle utilities shared across pipelines.

This subpackage re-exports the angle computation helpers from
`pose_handler.calc_angles` via `shared_pose.angles.calc_angles`, so callers
can import from `shared_pose.angles` without depending directly on the
`pose_handler` package layout.
"""

from .calc_angles import *  # noqa: F401,F403

__all__ = [
    "TINY",
    "EPS",
    "_norm",
    "body_frame_orthonormal",
    "calc_angles_from_landmarks",
    "calc_angles_flat",
    "keypoints_to_arrays",
    "angles_from_keypoints",
    "_angles_from_arrays",
    "JointVisualizer",
]

