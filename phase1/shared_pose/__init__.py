"""
Shared pose-related utilities used across `kalman`, `pose_handler`,
and `pose-corrector`.

This package intentionally does not introduce new algorithms. Instead, it
provides a stable import location that re-exports existing, battle-tested
implementations from those packages so they can be shared without code
duplication.
"""

from . import angles  # noqa: F401
from . import pose_io  # noqa: F401

__all__ = ["angles", "pose_io"]

