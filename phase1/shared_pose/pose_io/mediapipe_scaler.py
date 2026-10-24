"""
MediaPipe Z scaling helpers (shared wrapper).

This module re-exports the z-scaling logic from `kalman.video_pose_io` so
that other code can depend on a single shared implementation without
duplicating algorithms.
"""

from typing import Dict, List

from kalman.video_pose_io import (  # type: ignore[attr-defined]
    scale_mediapipe_entries as _impl_scale_mediapipe_entries,
    _calc_scale_factor as _impl_calc_scale_factor,
)


def scale_mediapipe_entries(entries: List[Dict]) -> List[Dict]:
    """
    Scale z (world meters → normalized units) for each MediaPipe entry.

    Delegates directly to `kalman.video_pose_io.scale_mediapipe_entries`.
    """
    return _impl_scale_mediapipe_entries(entries)


def calc_scale_factor(landmarks: Dict) -> float:
    """
    Estimate scale factor between normalized and world coordinates for a
    single set of landmarks.

    Delegates to `kalman.video_pose_io._calc_scale_factor`.
    """
    return float(_impl_calc_scale_factor(landmarks))


__all__ = ["scale_mediapipe_entries", "calc_scale_factor"]

