"""
MediaPipe + OpenPose keypoint averaging (shared wrapper).

This module re-exports the frame-level averaging helper from
`kalman.video_pose_io` so that multiple pipelines can share the same
implementation.
"""

from typing import Dict, Optional

from kalman.video_pose_io import (  # type: ignore[attr-defined]
    average_frame_keypoints as _impl_average_frame_keypoints,
)


def average_frame_keypoints(
    mp_scaled_entry: Dict,
    op_named_keypoints: Dict[str, Optional[Dict[str, float]]],
    image_size: Dict[str, int],
    min_conf: float = 0.3,
) -> Dict[str, Optional[Dict[str, float]]]:
    """
    Average MediaPipe (scaled) and OpenPose keypoints for one frame.

    Delegates to `kalman.video_pose_io.average_frame_keypoints`. The
    parameters and return value intentionally mirror that function.
    """
    return _impl_average_frame_keypoints(
        mp_scaled_entry,
        op_named_keypoints,
        image_size,
        min_conf=min_conf,
    )


__all__ = ["average_frame_keypoints"]

