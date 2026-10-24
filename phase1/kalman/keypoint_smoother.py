"""
Kalman filter smoother for 2D keypoints (x, y).
One independent cv2.KalmanFilter per keypoint; constant velocity model.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np


class KeypointSmoother:
    """
    Manages Kalman filters for pose keypoints. One filter per keypoint name.
    State: [x, y, vx, vy]. Measurement: [x, y]. Constant velocity model.
    """

    def __init__(
        self,
        keypoint_names: List[str],
        dt: float = 1.0,
        process_noise: float = 1e-2,
        measurement_noise: float = 1e-1,
        initial_error_cov: float = 1.0,
        visibility_threshold: float = 0.3,
    ):
        self.keypoint_names = list(keypoint_names)
        self.dt = dt
        self.visibility_threshold = visibility_threshold
        self._filters: Dict[str, cv2.KalmanFilter] = {}
        self._initialized: Dict[str, bool] = {name: False for name in self.keypoint_names}

        # State 4D, measurement 2D
        state_dim = 4
        measure_dim = 2

        for name in self.keypoint_names:
            kf = cv2.KalmanFilter(state_dim, measure_dim, 0)

            # Transition F: [x, y, vx, vy] -> [x+dt*vx, y+dt*vy, vx, vy]
            kf.transitionMatrix = np.array(
                [
                    [1, 0, dt, 0],
                    [0, 1, 0, dt],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                dtype=np.float32,
            )

            # Measurement H: we observe only (x, y)
            kf.measurementMatrix = np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                ],
                dtype=np.float32,
            )

            # Process noise Q
            kf.processNoiseCov = np.eye(state_dim, dtype=np.float32) * process_noise

            # Measurement noise R
            kf.measurementNoiseCov = np.eye(measure_dim, dtype=np.float32) * measurement_noise

            # Initial error covariance P (high to allow quick convergence)
            kf.errorCovPost = np.eye(state_dim, dtype=np.float32) * initial_error_cov

            self._filters[name] = kf

    def update(
        self,
        keypoints: Dict[str, Optional[Dict]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Process one frame of keypoints. Returns smoothed (x, y) per keypoint;
        z and confidence are not smoothed (caller should pass through from input).
        """
        out: Dict[str, Dict[str, float]] = {}
        for name in self.keypoint_names:
            kf = self._filters[name]
            kp = keypoints.get(name) if keypoints else None

            if kp is None or float(kp.get("confidence", 0.0)) < self.visibility_threshold:
                # Missing or occluded: predict only
                state = kf.predict()
                x, y = self._state_xy(state)
                out[name] = {"x": x, "y": y}
                continue

            x_meas = float(kp.get("x", 0.0))
            y_meas = float(kp.get("y", 0.0))
            measurement = np.array([[x_meas], [y_meas]], dtype=np.float32)

            if not self._initialized[name]:
                # First valid detection: set state directly
                kf.statePost = np.array(
                    [[x_meas], [y_meas], [0.0], [0.0]],
                    dtype=np.float32,
                )
                self._initialized[name] = True
                out[name] = {"x": x_meas, "y": y_meas}
                continue

            # Predict then correct
            kf.predict()
            kf.correct(measurement)
            state = kf.statePost
            x, y = self._state_xy(state)
            out[name] = {"x": x, "y": y}

        return out

    @staticmethod
    def _state_xy(state: np.ndarray) -> tuple[float, float]:
        """Extract (x, y) from Kalman statePost/predict output (shape 4x1 or 4,)."""
        flat = np.asarray(state, dtype=np.float32).reshape(-1)
        return float(flat[0]), float(flat[1])

    def reset(self, keypoint_name: Optional[str] = None) -> None:
        """Reset filter(s). If keypoint_name is None, reset all."""
        if keypoint_name is not None:
            self._initialized[keypoint_name] = False
        else:
            for name in self.keypoint_names:
                self._initialized[name] = False
