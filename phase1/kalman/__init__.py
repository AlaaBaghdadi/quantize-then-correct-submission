"""Video pose pipeline with Kalman smoothing: OpenPose + MediaPipe, average, smooth, angles."""

from kalman.keypoint_smoother import KeypointSmoother

__all__ = ["KeypointSmoother"]
