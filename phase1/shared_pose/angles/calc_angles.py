"""
Thin re-export layer for angle utilities.

All logic lives in `pose_handler.calc_angles`; this module simply re-exports
those symbols under the `shared_pose.angles.calc_angles` namespace so that
other parts of the project can depend on a shared package instead of on
`pose_handler` directly.
"""

from typing import Dict, Optional, Union  # noqa: F401

import numpy as np  # noqa: F401

from pose_handler.calc_angles import (  # type: ignore[attr-defined]
    TINY,
    EPS,
    _norm,
    body_frame_orthonormal,
    calc_angles_from_landmarks,
    calc_angles_flat,
    keypoints_to_arrays,
    angles_from_keypoints,
    _angles_from_arrays,
    JointVisualizer,
)

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

