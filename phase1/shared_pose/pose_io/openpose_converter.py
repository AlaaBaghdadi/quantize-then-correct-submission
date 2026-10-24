"""
OpenPose JSON → named keypoints converter (shared wrapper).

This module does not introduce a new implementation. It simply exposes the
existing `kalman.video_pose_io.openpose_json_to_named_keypoints` function
under a shared namespace so both video and image pipelines can rely on the
same core logic.
"""

from typing import Dict, Optional

from kalman.video_pose_io import (  # type: ignore[attr-defined]
    openpose_json_to_named_keypoints as _impl_openpose_json_to_named_keypoints,
)


def openpose_json_to_named_keypoints(
    data: Dict,
    person_index: int = 0,
    min_conf: float = 0.0,
) -> Dict[str, Optional[Dict[str, float]]]:
    """
    Convert an OpenPose JSON object to a dict of named keypoints.

    This is a thin wrapper around
    `kalman.video_pose_io.openpose_json_to_named_keypoints`. See that
    function for full details of the format and behavior.
    """
    return _impl_openpose_json_to_named_keypoints(
        data,
        person_index=person_index,
        min_conf=min_conf,
    )


__all__ = ["openpose_json_to_named_keypoints"]

