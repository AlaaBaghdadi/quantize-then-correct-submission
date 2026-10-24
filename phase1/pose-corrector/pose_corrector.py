"""
Constrained Kinematic Correction for Human Pose Estimation.

This module implements biomechanically-constrained correction of shoulder angles
derived from MediaPipe landmarks. It clamps impossible angles to valid ranges,
applies temporal smoothing, and reconstructs 3D landmark positions.

Algorithm:
    A. Deconstruct: Compute upper-arm vectors and Pitch/Yaw from landmarks.
    B. Soft Conical Constraints: Replace hard pitch/yaw clamping with a
       feasible cone per shoulder (axis ~30° forward, 45° down; max deviation
       ~55°). In a buffer zone near the limit, smoothstep resistance is
       applied; beyond the limit, the direction is clamped to the cone edge.
    C. Smooth: Apply EMA on the resulting angles; yaw uses wrap-aware interpolation.
    D. Optional: Velocity limit (max angular change per frame).
    E. Kinematic Chain Reconstruction: Rotate the forearm by the same rotation
       that aligns the old upper-arm direction to the new one (Rodrigues);
       New Elbow = Shoulder + U_new, New Wrist = New Elbow + R @ F_old.

Coordinate System (from calc_angles.py):
    - Body frame: Xb (lateral right), Yb (forward), Zb (up from hip)
    - Pitch: Angle from -Zb (rest/down). Range [0°, 180°].
    - Yaw: Angle in horizontal plane. 0° = forward (Yb).
           Positive = toward Xb (lateral right).
           Negative = toward -Xb (lateral left).

Biomechanical Constraints:
    - Pitch: [0°, 180°] - rest to overhead.
    - Left Shoulder Yaw: [-45°, +135°] - across chest to out/back.
    - Right Shoulder Yaw: [-135°, +45°] - out/back to across chest.

Motion-aware correction (why not "nearest boundary" only):
    - Nearest-boundary clamping ignores motion intent: a single frame may be
      noisy or out-of-range in the direction of movement; snapping to the
      "closest" boundary in linear angle space can reverse or contradict the
      trajectory (e.g. arm moving forward then snapping backward).
    - Yaw is periodic (360°). The shortest path from e.g. -180° to the valid
      range may go through the "high" boundary (+135°) rather than the "low"
      (-45°). Wrap-aware distance chooses the boundary that is actually
      nearest on the circle, avoiding large visual jumps.
    - When both boundaries are nearly equidistant (tie), we choose the boundary
      in the direction of motion (using previous frame's smoothed angle as
      velocity proxy), so the corrected pose continues the ongoing motion
      instead of snapping the other way.
    - After clamping, we optionally cap the per-frame angular change so that
      large corrections are spread over several frames (velocity limiting),
      producing smooth approach to the limit instead of one-frame teleport.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Import angle calculation utilities from calc_angles.py
from pose_handler.calc_angles import (
    TINY,
    EPS,
    _norm,
    body_frame_orthonormal,
    keypoints_to_arrays,
)
from pose_handler.common_io import read_json, write_json, ensure_dir


# ---------------------------------------------------------------------------
# Biomechanical Constraint Constants
# ---------------------------------------------------------------------------

# Pitch limits (degrees): 0° = arm at rest (down), 180° = arm overhead
PITCH_MIN = 0.0
PITCH_MAX = 180.0

# Yaw limits (degrees): 0° = forward, positive = lateral out (toward Xb)
# Left shoulder: can reach back (+135°) but limited crossing chest (-45°)
LEFT_YAW_MIN = -45.0   # Across chest limit
LEFT_YAW_MAX = 135.0   # Out/back limit

# Right shoulder: mirror of left (negative = out, positive = across)
RIGHT_YAW_MIN = -135.0  # Out/back limit
RIGHT_YAW_MAX = 45.0    # Across chest limit

# Default EMA smoothing factor (higher = more responsive, lower = smoother)
DEFAULT_ALPHA = 0.7

# Yaw is periodic (full circle); pitch is not
YAW_PERIOD = 360.0

# When distances to min/max boundary differ by less than this (degrees), use
# direction of motion to choose boundary (tie-break)
YAW_TIE_THRESHOLD_DEG = 10.0

# Max angular change per frame (degrees) for velocity limiting; None = disabled
DEFAULT_MAX_PITCH_STEP_DEG = 25.0
DEFAULT_MAX_YAW_STEP_DEG = 25.0

# Soft conical constraint: feasible cone per shoulder (replaces hard pitch/yaw clamp)
# Cone axis: approx. CONE_FORWARD_DEG forward (+Yb) and CONE_DOWN_DEG down (-Zb)
CONE_FORWARD_DEG = 30.0   # Left: +Yb; Right: mirrored
CONE_DOWN_DEG = 45.0      # Down from horizontal (-Zb)
CONE_MAX_DEVIATION_DEG = 55.0   # Max angle (degrees) from cone axis
CONE_BUFFER_DEG = 10.0    # Buffer zone (last N degrees) for soft resistance


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ShoulderAngles:
    """Container for shoulder angle data."""
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0  # Pass-through, not corrected


@dataclass
class CorrectionState:
    """State maintained across frames for temporal smoothing."""
    left_shoulder: ShoulderAngles = field(default_factory=ShoulderAngles)
    right_shoulder: ShoulderAngles = field(default_factory=ShoulderAngles)
    initialized: bool = False


@dataclass
class CorrectionResult:
    """Result of a single frame correction."""
    keypoints: Dict[str, Any]  # Corrected keypoints
    original_angles: Dict[str, ShoulderAngles]  # Before correction
    clamped_angles: Dict[str, ShoulderAngles]   # After clamping
    smoothed_angles: Dict[str, ShoulderAngles]  # After EMA
    corrections_applied: Dict[str, bool]  # Whether each shoulder was corrected


def angles_file_dict_to_shoulder_angles(
    angles: Dict[str, float],
) -> Dict[str, ShoulderAngles]:
    """
    Convert a flat angles dict (e.g. from Kalman angles JSON) to left/right ShoulderAngles.
    Keys expected: left_shoulder_pitch, left_shoulder_yaw, left_shoulder_roll,
    right_shoulder_pitch, right_shoulder_yaw, right_shoulder_roll.
    Missing keys default to 0.0.
    """
    def get(name: str) -> float:
        return float(angles.get(name, 0.0))

    return {
        "left": ShoulderAngles(
            pitch=get("left_shoulder_pitch"),
            yaw=get("left_shoulder_yaw"),
            roll=get("left_shoulder_roll"),
        ),
        "right": ShoulderAngles(
            pitch=get("right_shoulder_pitch"),
            yaw=get("right_shoulder_yaw"),
            roll=get("right_shoulder_roll"),
        ),
    }


# ---------------------------------------------------------------------------
# Angle Calculation (extracted from calc_angles.py for local use)
# ---------------------------------------------------------------------------

def _calculate_shoulder_angles(
    shoulder: np.ndarray,
    elbow: np.ndarray,
    wrist: np.ndarray,
    Xb: np.ndarray,
    Yb: np.ndarray,
    Zb: np.ndarray,
) -> ShoulderAngles:
    """
    Calculate pitch, yaw, roll for a single shoulder using atan2.
    
    Pitch: Angle between upper arm and -Zb (down), range [0°, 180°].
    Yaw: Angle in horizontal plane from Yb (forward), using atan2.
    Roll: Rotation of arm plane around the upper arm axis.
    """
    # Upper arm vector
    upper_arm = elbow - shoulder
    upper_arm_norm = np.linalg.norm(upper_arm)
    if upper_arm_norm < TINY:
        return ShoulderAngles()
    
    U = upper_arm / upper_arm_norm
    
    # Forearm vector (for roll calculation)
    forearm = wrist - elbow
    forearm_norm = np.linalg.norm(forearm)
    F = forearm / forearm_norm if forearm_norm > TINY else np.zeros(3)
    
    # --- Pitch: angle with -Zb ---
    neg_Zb = -Zb
    cross_norm = np.linalg.norm(np.cross(U, neg_Zb))
    dot_val = np.dot(U, neg_Zb)
    pitch = float(np.degrees(np.arctan2(cross_norm, dot_val)))
    
    # --- Yaw: angle in horizontal plane from Yb ---
    # Project U onto plane perpendicular to Zb
    U_proj = U - np.dot(U, Zb) * Zb
    u_proj_norm = np.linalg.norm(U_proj)
    
    if u_proj_norm > EPS:
        U_proj = U_proj / u_proj_norm
        yaw = float(np.degrees(np.arctan2(
            np.dot(np.cross(Yb, U_proj), Zb),
            np.dot(Yb, U_proj),
        )))
    else:
        yaw = 0.0
    
    # --- Roll: rotation around upper arm axis ---
    N = np.cross(U, F)
    n_norm = np.linalg.norm(N)
    if n_norm > TINY:
        N = N / n_norm
    else:
        N = Xb.copy()
    
    Nref = np.cross(neg_Zb, U)
    nref_norm = np.linalg.norm(Nref)
    if nref_norm < EPS:
        Nref = Xb.copy()
    else:
        Nref = Nref / nref_norm
    
    roll = float(np.degrees(np.arctan2(
        np.dot(np.cross(Nref, N), U),
        np.dot(Nref, N),
    )))
    
    return ShoulderAngles(pitch=pitch, yaw=yaw, roll=roll)


def _angles_to_unit_vector(
    pitch: float,
    yaw: float,
    Xb: np.ndarray,
    Yb: np.ndarray,
    Zb: np.ndarray,
) -> np.ndarray:
    """
    Convert pitch and yaw angles back to a unit vector in world coordinates.
    
    Spherical coordinate conversion in body frame:
        - pitch is angle from -Zb (down)
        - yaw is angle in horizontal plane from Yb (forward)
    
    Unit vector formula:
        u = sin(pitch)*sin(yaw)*Xb + sin(pitch)*cos(yaw)*Yb - cos(pitch)*Zb
    
    Verification:
        - pitch=0°: u = -Zb (arm down) ✓
        - pitch=90°, yaw=0°: u = Yb (arm forward) ✓
        - pitch=90°, yaw=90°: u = Xb (arm lateral right) ✓
        - pitch=180°: u = +Zb (arm overhead) ✓
    """
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)
    
    sin_pitch = np.sin(pitch_rad)
    cos_pitch = np.cos(pitch_rad)
    sin_yaw = np.sin(yaw_rad)
    cos_yaw = np.cos(yaw_rad)
    
    unit_vector = (
        sin_pitch * sin_yaw * Xb +
        sin_pitch * cos_yaw * Yb -
        cos_pitch * Zb
    )
    
    # Ensure unit length (handle numerical errors)
    norm = np.linalg.norm(unit_vector)
    if norm > TINY:
        unit_vector = unit_vector / norm
    
    return unit_vector


# ---------------------------------------------------------------------------
# Angular distance and motion-aware clamping helpers
# ---------------------------------------------------------------------------

def _shortest_angular_distance(
    from_angle: float,
    to_angle: float,
    period: float = 360.0,
) -> float:
    """
    Signed shortest angular distance from `from_angle` to `to_angle` on a circle.

    Returns a value in (-period/2, period/2]. Positive means to_angle is
    counterclockwise from from_angle (in standard math convention).

    Used for wrap-aware yaw: e.g. from 170° to -170° gives -20° (short way).
    """
    delta = (to_angle - from_angle) % period
    if delta > period / 2:
        delta -= period
    if delta <= -period / 2:
        delta += period
    return delta


def _clamp_angle(value: float, min_val: float, max_val: float) -> Tuple[float, bool]:
    """
    Clamp angle to range with minimal displacement (for non-periodic angles).
    Returns (clamped_value, was_clamped). Use for pitch [0, 180].
    """
    if value < min_val:
        return min_val, True
    if value > max_val:
        return max_val, True
    return value, False


def _clamp_yaw_motion_aware(
    value: float,
    min_val: float,
    max_val: float,
    prev_value: Optional[float],
    period: float = YAW_PERIOD,
    tie_threshold: float = YAW_TIE_THRESHOLD_DEG,
) -> Tuple[float, bool]:
    """
    Clamp yaw to [min_val, max_val] using wrap-aware distance and motion.

    - If in range: return value unchanged.
    - If out of range: choose the boundary that is nearest on the circle
      (shortest angular distance). If both boundaries are nearly equidistant
      (within tie_threshold), choose the one in the direction of motion
      (using prev_value as reference) so the pose continues the trajectory.

    Returns (clamped_value, was_clamped).
    """
    # In range: no change
    if min_val <= value <= max_val:
        return value, False

    # Distances to each boundary (unsigned, shortest path on circle)
    dist_to_min = abs(_shortest_angular_distance(value, min_val, period))
    dist_to_max = abs(_shortest_angular_distance(value, max_val, period))

    # Clear winner: choose nearer boundary
    if dist_to_min < dist_to_max - tie_threshold:
        return min_val, True
    if dist_to_max < dist_to_min - tie_threshold:
        return max_val, True

    # Tie or near-tie: use direction of motion from previous frame
    if prev_value is not None:
        velocity = _shortest_angular_distance(prev_value, value, period)
        # Prefer the boundary we're moving toward
        if velocity > 0:
            return max_val, True
        if velocity < 0:
            return min_val, True

    # No previous (first frame): use strictly nearer boundary
    return (min_val, True) if dist_to_min <= dist_to_max else (max_val, True)


def _clamp_shoulder_angles(
    angles: ShoulderAngles,
    is_left: bool,
) -> Tuple[ShoulderAngles, bool]:
    """
    Apply biomechanical constraints to shoulder angles.
    
    Args:
        angles: Current shoulder angles.
        is_left: True for left shoulder, False for right.
    
    Returns:
        (clamped_angles, any_correction_applied)
    """
    # Clamp pitch (same for both shoulders)
    clamped_pitch, pitch_corrected = _clamp_angle(
        angles.pitch, PITCH_MIN, PITCH_MAX
    )
    
    # Clamp yaw (different limits per shoulder)
    if is_left:
        yaw_min, yaw_max = LEFT_YAW_MIN, LEFT_YAW_MAX
    else:
        yaw_min, yaw_max = RIGHT_YAW_MIN, RIGHT_YAW_MAX
    
    clamped_yaw, yaw_corrected = _clamp_angle(angles.yaw, yaw_min, yaw_max)
    
    return (
        ShoulderAngles(
            pitch=clamped_pitch,
            yaw=clamped_yaw,
            roll=angles.roll,  # Pass-through unchanged
        ),
        pitch_corrected or yaw_corrected,
    )


# ---------------------------------------------------------------------------
# Exponential Moving Average (wrap-aware for yaw)
# ---------------------------------------------------------------------------

def _ema_angles(
    current: ShoulderAngles,
    previous: ShoulderAngles,
    alpha: float,
    yaw_period: float = YAW_PERIOD,
) -> ShoulderAngles:
    """
    Apply Exponential Moving Average between current and previous angles.

    Pitch and roll: linear interpolation.
    Yaw: interpolated on the circle (shortest path) so that e.g. 170° and -170°
    smooth toward 180° instead of 0°.

    Formula:
        pitch/roll: smoothed = alpha * current + (1 - alpha) * previous
        yaw: delta = shortest_angular_distance(previous.yaw, current.yaw);
             smoothed.yaw = previous.yaw + alpha * delta (normalized to same cycle)

    Args:
        current: Current frame's (clamped) angles.
        previous: Previous frame's smoothed angles.
        alpha: Smoothing factor in [0, 1]. Higher = more responsive.
        yaw_period: Period for yaw wrap (default 360).

    Returns:
        Smoothed angles.
    """
    smooth_pitch = alpha * current.pitch + (1 - alpha) * previous.pitch
    smooth_roll = alpha * current.roll + (1 - alpha) * previous.roll

    yaw_delta = _shortest_angular_distance(previous.yaw, current.yaw, yaw_period)
    smooth_yaw = previous.yaw + alpha * yaw_delta
    # Keep yaw in a consistent range for downstream (e.g. -180..180 or 0..360)
    smooth_yaw = (smooth_yaw + yaw_period / 2) % yaw_period - yaw_period / 2

    return ShoulderAngles(pitch=smooth_pitch, yaw=smooth_yaw, roll=smooth_roll)


def _apply_velocity_limit(
    previous: ShoulderAngles,
    smoothed: ShoulderAngles,
    max_pitch_step: Optional[float],
    max_yaw_step: Optional[float],
    yaw_period: float = YAW_PERIOD,
) -> ShoulderAngles:
    """
    Cap the per-frame change from previous to smoothed (velocity limiting).

    Prevents one-frame "teleports" when a large correction is applied; the
    correction is spread over several frames for more natural motion.
    If max_*_step is None, that channel is not limited.
    """
    pitch = smoothed.pitch
    if max_pitch_step is not None:
        delta_p = smoothed.pitch - previous.pitch
        delta_p = max(-max_pitch_step, min(max_pitch_step, delta_p))
        pitch = previous.pitch + delta_p

    yaw = smoothed.yaw
    if max_yaw_step is not None:
        delta_y = _shortest_angular_distance(previous.yaw, smoothed.yaw, yaw_period)
        delta_y = max(-max_yaw_step, min(max_yaw_step, delta_y))
        yaw = previous.yaw + delta_y
        yaw = (yaw + yaw_period / 2) % yaw_period - yaw_period / 2

    roll = smoothed.roll  # optional: could add max_roll_step
    return ShoulderAngles(pitch=pitch, yaw=yaw, roll=roll)


# ---------------------------------------------------------------------------
# Length Stabilization (soft-clipping for bone lengths)
# ---------------------------------------------------------------------------

# Default length stabilizer: EMA factor, safe linear zone (% of ref), tanh strength
DEFAULT_LENGTH_ALPHA = 0.4
DEFAULT_MAX_DRIFT = 0.05   # 5% of ref_length is "safe" linear zone
DEFAULT_SOFT_SCALE = 0.02  # strength of tanh compression (in same units as length)


class LengthStabilizer:
    """
    Soft-clipping length stabilizer for temporal consistency of bone lengths.
    Avoids hard thresholds that cause popping; uses EMA of stabilized output
    so the reference does not drift under sustained diffusion noise.

    Per frame:
      - Zone A (linear): if |raw - ref| <= ref * max_drift, trust input.
      - Zone B (compression): otherwise compress excess via tanh.
      - Reference is updated with EMA of the stabilized length (not raw).
    """

    def __init__(
        self,
        alpha: float = DEFAULT_LENGTH_ALPHA,
        max_drift: float = DEFAULT_MAX_DRIFT,
        soft_scale: float = DEFAULT_SOFT_SCALE,
    ):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.max_drift = float(max(1e-6, max_drift))
        self.soft_scale = float(max(1e-10, soft_scale))
        self._ref_length: Optional[float] = None

    def reset(self) -> None:
        """Reset reference so next frame re-initializes from observed length."""
        self._ref_length = None

    def step(self, raw_length: float) -> float:
        """
        Process one frame: return stabilized length and update internal reference.

        First frame: ref_length = raw_length, return raw_length.
        Later: apply zone A/B logic, then ref = EMA(ref, stabilized).
        """
        raw_length = float(raw_length)
        if self._ref_length is None:
            self._ref_length = raw_length
            return raw_length

        ref = self._ref_length
        delta = raw_length - ref
        safe_threshold = ref * self.max_drift

        if abs(delta) <= safe_threshold:
            stabilized = raw_length
        else:
            sign = 1.0 if delta >= 0 else -1.0
            excess = abs(delta) - safe_threshold
            scale = max(self.soft_scale, 1e-10)
            compressed = scale * np.tanh(excess / scale)
            stabilized = ref + sign * (safe_threshold + compressed)

        self._ref_length = (1.0 - self.alpha) * ref + self.alpha * stabilized
        return stabilized


# ---------------------------------------------------------------------------
# Main Pose Corrector Class
# ---------------------------------------------------------------------------

class PoseCorrector:
    """
    Constrained Kinematic Correction for human pose estimation.
    
    Maintains state across video frames for temporal smoothing.
    Applies biomechanical constraints and reconstructs landmark positions.
    
    Usage:
        corrector = PoseCorrector(alpha=0.7)
        for frame in video_frames:
            result = corrector.correct_frame(frame['keypoints'])
            corrected_keypoints = result.keypoints
    
    Args:
        alpha: EMA smoothing factor. Default 0.7.
               Higher = more responsive to new data.
               Lower = smoother but more latency.
        pitch_range: Optional tuple (min, max) to override pitch limits.
        left_yaw_range: Optional tuple (min, max) for left shoulder yaw.
        right_yaw_range: Optional tuple (min, max) for right shoulder yaw.
        max_pitch_step: Max degrees change per frame for pitch (velocity limit).
                        None = disabled. Default 25.
        max_yaw_step: Max degrees change per frame for yaw (velocity limit).
                      None = disabled. Default 25.
        length_alpha: EMA factor for length stabilizer reference. Default 0.4.
        max_drift: Safe linear zone as fraction of ref length (e.g. 0.05 = 5%). Default 0.05.
        soft_scale: Tanh compression strength for length excess. Default 0.02.
    """

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        pitch_range: Optional[Tuple[float, float]] = None,
        left_yaw_range: Optional[Tuple[float, float]] = None,
        right_yaw_range: Optional[Tuple[float, float]] = None,
        max_pitch_step: Optional[float] = DEFAULT_MAX_PITCH_STEP_DEG,
        max_yaw_step: Optional[float] = DEFAULT_MAX_YAW_STEP_DEG,
        length_alpha: float = DEFAULT_LENGTH_ALPHA,
        max_drift: float = DEFAULT_MAX_DRIFT,
        soft_scale: float = DEFAULT_SOFT_SCALE,
        use_soft_cone: bool = True,
    ):
        self.alpha = np.clip(alpha, 0.0, 1.0)
        self.use_soft_cone = use_soft_cone

        # Biomechanical limits (allow override for testing/tuning)
        self.pitch_min = pitch_range[0] if pitch_range else PITCH_MIN
        self.pitch_max = pitch_range[1] if pitch_range else PITCH_MAX
        self.left_yaw_min = left_yaw_range[0] if left_yaw_range else LEFT_YAW_MIN
        self.left_yaw_max = left_yaw_range[1] if left_yaw_range else LEFT_YAW_MAX
        self.right_yaw_min = right_yaw_range[0] if right_yaw_range else RIGHT_YAW_MIN
        self.right_yaw_max = right_yaw_range[1] if right_yaw_range else RIGHT_YAW_MAX

        # Velocity limiting (spread large corrections over frames)
        # Hard-clamp ablation disables velocity limiting per EXPERIMENTS.md
        if use_soft_cone:
            self._max_pitch_step = max_pitch_step
            self._max_yaw_step = max_yaw_step
        else:
            self._max_pitch_step = None
            self._max_yaw_step = None

        # Length stabilizers (one per bone per arm): soft-clip limb lengths
        self._len_stab_left_upper = LengthStabilizer(alpha=length_alpha, max_drift=max_drift, soft_scale=soft_scale)
        self._len_stab_left_forearm = LengthStabilizer(alpha=length_alpha, max_drift=max_drift, soft_scale=soft_scale)
        self._len_stab_right_upper = LengthStabilizer(alpha=length_alpha, max_drift=max_drift, soft_scale=soft_scale)
        self._len_stab_right_forearm = LengthStabilizer(alpha=length_alpha, max_drift=max_drift, soft_scale=soft_scale)

        # Temporal state
        self._state = CorrectionState()
        
        # Statistics
        self._frame_count = 0
        self._corrections_count = {"left": 0, "right": 0}
    
    def reset(self) -> None:
        """Reset temporal state. Call between videos or after tracking loss."""
        self._state = CorrectionState()
        self._len_stab_left_upper.reset()
        self._len_stab_left_forearm.reset()
        self._len_stab_right_upper.reset()
        self._len_stab_right_forearm.reset()
        self._frame_count = 0
        self._corrections_count = {"left": 0, "right": 0}
    
    @property
    def frame_count(self) -> int:
        """Number of frames processed since last reset."""
        return self._frame_count
    
    @property
    def correction_stats(self) -> Dict[str, int]:
        """Count of frames where corrections were applied per shoulder."""
        return self._corrections_count.copy()
    
    def _extract_landmarks(
        self,
        keypoints: Dict[str, Any],
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract required landmarks as numpy arrays.
        Returns None if any required landmark is missing.
        """
        return keypoints_to_arrays(keypoints)
    
    def _build_body_frame(
        self,
        landmarks: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build orthonormal body frame from landmarks."""
        return body_frame_orthonormal(
            landmarks["LS"],
            landmarks["RS"],
            landmarks["LH"],
        )
    
    def _clamp_with_limits(
        self,
        angles: ShoulderAngles,
        is_left: bool,
        prev_smoothed: Optional[ShoulderAngles] = None,
    ) -> Tuple[ShoulderAngles, bool]:
        """Clamp angles using instance limits; yaw uses motion-aware wrap-aware logic."""
        # Pitch: non-periodic, simple clamp
        clamped_pitch, pitch_corr = _clamp_angle(
            angles.pitch, self.pitch_min, self.pitch_max
        )

        # Yaw: wrap-aware and motion-aware (use previous frame for direction when tied)
        yaw_min = self.left_yaw_min if is_left else self.right_yaw_min
        yaw_max = self.left_yaw_max if is_left else self.right_yaw_max
        prev_yaw = prev_smoothed.yaw if prev_smoothed else None
        clamped_yaw, yaw_corr = _clamp_yaw_motion_aware(
            angles.yaw, yaw_min, yaw_max, prev_value=prev_yaw
        )

        return (
            ShoulderAngles(pitch=clamped_pitch, yaw=clamped_yaw, roll=angles.roll),
            pitch_corr or yaw_corr,
        )

    def _get_cone_axis(
        self,
        is_left: bool,
        Xb: np.ndarray,
        Yb: np.ndarray,
        Zb: np.ndarray,
    ) -> np.ndarray:
        """Feasible cone axis for the shoulder in body frame (unit vector).
        Left: ~30° forward (+Yb), 45° down (-Zb). Right: mirrored."""
        down_rad = np.radians(CONE_DOWN_DEG)
        # Axis: CONE_DOWN_DEG below horizontal; forward +Yb (left) / -Yb (right)
        sign = 1.0 if is_left else -1.0
        axis = sign * np.cos(down_rad) * Yb - np.sin(down_rad) * Zb
        n = np.linalg.norm(axis)
        if n < TINY:
            return -Zb.copy()
        return axis / n

    def _soft_cone_constraint(
        self,
        U_old_dir: np.ndarray,
        cone_axis: np.ndarray,
        max_deg: float = CONE_MAX_DEVIATION_DEG,
        buffer_deg: float = CONE_BUFFER_DEG,
    ) -> np.ndarray:
        """Apply soft conical limit: return corrected unit vector U_new_dir.
        If angle θ from axis is within (max - buffer), return U_old_dir.
        In buffer zone, apply smoothstep resistance; beyond max, clamp to cone edge."""
        dot_val = np.dot(U_old_dir, cone_axis)
        dot_val = np.clip(dot_val, -1.0, 1.0)
        theta_deg = float(np.degrees(np.arccos(dot_val)))
        if theta_deg <= max_deg - buffer_deg:
            return U_old_dir.copy()
        # Plane: U_old_dir and cone_axis. Perp = component of U_old_dir perpendicular to axis
        along = np.dot(U_old_dir, cone_axis) * cone_axis
        perp = U_old_dir - along
        perp_norm = np.linalg.norm(perp)
        if perp_norm < TINY:
            return cone_axis.copy()
        perp = perp / perp_norm
        max_rad = np.radians(max_deg)
        if theta_deg >= max_deg:
            # Clamp to cone boundary
            U_new_dir = np.cos(max_rad) * cone_axis + np.sin(max_rad) * perp
        else:
            # Buffer zone: smoothstep pull toward axis
            t = (theta_deg - (max_deg - buffer_deg)) / buffer_deg
            t = np.clip(t, 0.0, 1.0)
            smooth = t * t * (3.0 - 2.0 * t)
            theta_new_deg = theta_deg - smooth * (theta_deg - (max_deg - buffer_deg))
            theta_new_rad = np.radians(theta_new_deg)
            U_new_dir = np.cos(theta_new_rad) * cone_axis + np.sin(theta_new_rad) * perp
        n = np.linalg.norm(U_new_dir)
        if n < TINY:
            return cone_axis.copy()
        return U_new_dir / n

    def _rodrigues_rotate(self, v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
        """Rotate vector v by angle_rad around unit axis (Rodrigues)."""
        k = np.asarray(axis, dtype=np.float64).ravel()
        nk = np.linalg.norm(k)
        if nk < TINY:
            return v.copy()
        k = k / nk
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        return v * cos_a + np.cross(k, v) * sin_a + k * np.dot(k, v) * (1.0 - cos_a)

    def _angles_from_unit_vector(
        self,
        U: np.ndarray,
        Xb: np.ndarray,
        Yb: np.ndarray,
        Zb: np.ndarray,
    ) -> ShoulderAngles:
        """Compute pitch and yaw (degrees) from a unit upper-arm direction in body frame. Roll set to 0."""
        if np.linalg.norm(U) < TINY:
            return ShoulderAngles()
        U = U / np.linalg.norm(U)
        neg_Zb = -Zb
        cross_norm = np.linalg.norm(np.cross(U, neg_Zb))
        dot_val = np.dot(U, neg_Zb)
        pitch = float(np.degrees(np.arctan2(cross_norm, dot_val)))
        U_proj = U - np.dot(U, Zb) * Zb
        u_proj_norm = np.linalg.norm(U_proj)
        if u_proj_norm > EPS:
            U_proj = U_proj / u_proj_norm
            yaw = float(np.degrees(np.arctan2(
                np.dot(np.cross(Yb, U_proj), Zb),
                np.dot(Yb, U_proj),
            )))
        else:
            yaw = 0.0
        return ShoulderAngles(pitch=pitch, yaw=yaw, roll=0.0)

    def _reconstruct_arm_kinematic(
        self,
        shoulder: np.ndarray,
        old_elbow: np.ndarray,
        old_wrist: np.ndarray,
        U_new_dir: np.ndarray,
        len_u: Optional[float] = None,
        len_f: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct elbow and wrist using kinematic chain rotation.
        U_new_dir is the corrected unit upper-arm direction (angle logic unchanged).
        Optional len_u/len_f are stabilized lengths (caller steps LengthStabilizer each frame).
        Corrected direction + len_u define new elbow; forearm direction is rotated then
        scaled by len_f."""
        U_old = old_elbow - shoulder
        F_old = old_wrist - old_elbow
        raw_len_u = float(np.linalg.norm(U_old))
        raw_len_f = float(np.linalg.norm(F_old))

        if raw_len_u < TINY:
            return old_elbow.copy(), old_wrist.copy()

        use_len_u = raw_len_u if len_u is None else len_u
        use_len_f = raw_len_f if len_f is None else len_f

        U_old_dir = U_old / raw_len_u
        U_new = U_new_dir * use_len_u
        new_elbow = shoulder + U_new

        dot_uu = np.dot(U_old_dir, U_new_dir)
        dot_uu = np.clip(dot_uu, -1.0, 1.0)
        angle_rad = np.arccos(dot_uu)
        cross_uu = np.cross(U_old_dir, U_new_dir)
        cross_norm = np.linalg.norm(cross_uu)
        if cross_norm < TINY:
            F_rotated_dir = (F_old / raw_len_f) if raw_len_f > TINY else np.zeros_like(F_old)
        else:
            axis = cross_uu / cross_norm
            F_old_dir = F_old / raw_len_f if raw_len_f > TINY else np.zeros_like(F_old)
            F_rotated = self._rodrigues_rotate(F_old_dir, axis, angle_rad)
            fn = np.linalg.norm(F_rotated)
            F_rotated_dir = F_rotated / fn if fn > TINY else F_old_dir
        F_new = F_rotated_dir * use_len_f
        new_wrist = new_elbow + F_new
        return new_elbow, new_wrist

    def _reconstruct_arm(
        self,
        shoulder: np.ndarray,
        old_elbow: np.ndarray,
        old_wrist: np.ndarray,
        corrected_angles: ShoulderAngles,
        Xb: np.ndarray,
        Yb: np.ndarray,
        Zb: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct elbow and wrist positions from corrected angles.
        
        Algorithm:
            1. Compute upper arm length.
            2. Convert corrected pitch/yaw to unit vector.
            3. New elbow = shoulder + unit_vector * upper_arm_length.
            4. Compute displacement (new_elbow - old_elbow).
            5. New wrist = old_wrist + displacement (rigid forearm).
        
        Returns:
            (new_elbow, new_wrist)
        """
        # Upper arm length (preserve original)
        upper_arm_vec = old_elbow - shoulder
        upper_arm_length = np.linalg.norm(upper_arm_vec)
        
        if upper_arm_length < TINY:
            # Degenerate case: elbow at shoulder
            return old_elbow.copy(), old_wrist.copy()
        
        # Convert corrected angles to unit vector
        unit_vec = _angles_to_unit_vector(
            corrected_angles.pitch,
            corrected_angles.yaw,
            Xb, Yb, Zb,
        )
        
        # New elbow position
        new_elbow = shoulder + unit_vec * upper_arm_length
        
        # Displacement vector
        displacement = new_elbow - old_elbow
        
        # Apply same displacement to wrist (rigid forearm)
        new_wrist = old_wrist + displacement
        
        return new_elbow, new_wrist
    
    def correct_frame(
        self,
        keypoints: Dict[str, Any],
        apply_smoothing: bool = True,
        angles_from_file: Optional[Dict[str, float]] = None,
        length_only: bool = False,
    ) -> CorrectionResult:
        """
        Apply constrained kinematic correction to a single frame.

        Args:
            keypoints: Dict of keypoints with x, y, z, z_scaled, confidence.
            apply_smoothing: Whether to apply EMA smoothing (ignored when length_only=True).
            angles_from_file: Optional pre-calculated angles (e.g. from Kalman angles JSON).
                Keys: left_shoulder_pitch, left_shoulder_yaw, left_shoulder_roll,
                right_shoulder_pitch, right_shoulder_yaw, right_shoulder_roll.
                When provided, these are used as the current frame angles instead of
                recomputing from keypoints, so pipeline and corrector stay consistent.
            length_only: When True, only stabilize limb lengths; do not change angles/directions.
                Use this when the frame has length violation but no angle violation, so
                shoulder direction is preserved and only bone lengths are corrected.

        Returns:
            CorrectionResult with corrected keypoints and diagnostic info.
            Returns original keypoints if landmarks are missing.
        """
        # Deep copy to avoid modifying input
        corrected_keypoints = copy.deepcopy(keypoints)
        
        # Extract landmarks
        landmarks = self._extract_landmarks(keypoints)
        
        if landmarks is None:
            # Missing landmarks - return original, don't update state
            return CorrectionResult(
                keypoints=corrected_keypoints,
                original_angles={
                    "left": ShoulderAngles(),
                    "right": ShoulderAngles(),
                },
                clamped_angles={
                    "left": ShoulderAngles(),
                    "right": ShoulderAngles(),
                },
                smoothed_angles={
                    "left": ShoulderAngles(),
                    "right": ShoulderAngles(),
                },
                corrections_applied={"left": False, "right": False},
            )
        
        # Build body frame (always from keypoints; needed for cone axes and reconstruction)
        Xb, Yb, Zb = self._build_body_frame(landmarks)
        
        # Step A: Current angles — from file when provided, else compute from keypoints
        if angles_from_file is not None and isinstance(angles_from_file, dict):
            original_angles = angles_file_dict_to_shoulder_angles(angles_from_file)
            left_angles = original_angles["left"]
            right_angles = original_angles["right"]
        else:
            left_angles = _calculate_shoulder_angles(
                landmarks["LS"], landmarks["LE"], landmarks["LW"],
                Xb, Yb, Zb,
            )
            right_angles = _calculate_shoulder_angles(
                landmarks["RS"], landmarks["RE"], landmarks["RW"],
                Xb, Yb, Zb,
            )
            original_angles = {
                "left": ShoulderAngles(
                    pitch=left_angles.pitch,
                    yaw=left_angles.yaw,
                    roll=left_angles.roll,
                ),
                "right": ShoulderAngles(
                    pitch=right_angles.pitch,
                    yaw=right_angles.yaw,
                    roll=right_angles.roll,
                ),
            }
        
        if length_only:
            # Only stabilize lengths; keep angles/directions unchanged (no cone, no EMA).
            clamped_angles = dict(original_angles)
            smoothed_left = original_angles["left"]
            smoothed_right = original_angles["right"]
            smoothed_angles = {"left": smoothed_left, "right": smoothed_right}
            left_corrected = False
            right_corrected = False
            # Do not update angle state so next frame's full correction is unaffected
        else:
            # Step B: Angle constraints (soft cone or hard clamp)
            if self.use_soft_cone:
                cone_axis_left = self._get_cone_axis(True, Xb, Yb, Zb)
                cone_axis_right = self._get_cone_axis(False, Xb, Yb, Zb)

                def _apply_cone_and_angles(
                    shoulder: np.ndarray,
                    elbow: np.ndarray,
                    cone_axis: np.ndarray,
                    original: ShoulderAngles,
                ) -> Tuple[ShoulderAngles, bool]:
                    U_old = elbow - shoulder
                    len_u = np.linalg.norm(U_old)
                    if len_u < TINY:
                        return original, False
                    U_old_dir = U_old / len_u
                    U_new_dir = self._soft_cone_constraint(U_old_dir, cone_axis)
                    clamped = self._angles_from_unit_vector(U_new_dir, Xb, Yb, Zb)
                    clamped = ShoulderAngles(pitch=clamped.pitch, yaw=clamped.yaw, roll=original.roll)
                    corrected = (
                        abs(clamped.pitch - original.pitch) > 0.01
                        or abs(clamped.yaw - original.yaw) > 0.01
                    )
                    return clamped, corrected

                clamped_left, left_corrected = _apply_cone_and_angles(
                    landmarks["LS"], landmarks["LE"], cone_axis_left, left_angles
                )
                clamped_right, right_corrected = _apply_cone_and_angles(
                    landmarks["RS"], landmarks["RE"], cone_axis_right, right_angles
                )
            else:
                prev_left = self._state.left_shoulder if self._state.initialized else None
                prev_right = self._state.right_shoulder if self._state.initialized else None
                clamped_left, left_corrected = self._clamp_with_limits(
                    left_angles, is_left=True, prev_smoothed=prev_left
                )
                clamped_right, right_corrected = self._clamp_with_limits(
                    right_angles, is_left=False, prev_smoothed=prev_right
                )

            clamped_angles = {"left": clamped_left, "right": clamped_right}

            if left_corrected:
                self._corrections_count["left"] += 1
            if right_corrected:
                self._corrections_count["right"] += 1
            
            # Step C: Apply EMA smoothing (wrap-aware for yaw)
            if apply_smoothing and self._state.initialized:
                smoothed_left = _ema_angles(clamped_left, self._state.left_shoulder, self.alpha)
                smoothed_right = _ema_angles(clamped_right, self._state.right_shoulder, self.alpha)
            else:
                smoothed_left = clamped_left
                smoothed_right = clamped_right

            # Step D: Velocity limiting (cap per-frame change for natural motion)
            if self._state.initialized and (self._max_pitch_step is not None or self._max_yaw_step is not None):
                smoothed_left = _apply_velocity_limit(
                    self._state.left_shoulder,
                    smoothed_left,
                    self._max_pitch_step,
                    self._max_yaw_step,
                )
                smoothed_right = _apply_velocity_limit(
                    self._state.right_shoulder,
                    smoothed_right,
                    self._max_pitch_step,
                    self._max_yaw_step,
                )

            smoothed_angles = {"left": smoothed_left, "right": smoothed_right}

            # Update state for next frame
            self._state.left_shoulder = smoothed_left
            self._state.right_shoulder = smoothed_right
            self._state.initialized = True
            self._frame_count += 1

        # Step E: Length stabilization (step every frame so ref tracks)
        raw_left_upper = float(np.linalg.norm(landmarks["LE"] - landmarks["LS"]))
        raw_left_forearm = float(np.linalg.norm(landmarks["LW"] - landmarks["LE"]))
        raw_right_upper = float(np.linalg.norm(landmarks["RE"] - landmarks["RS"]))
        raw_right_forearm = float(np.linalg.norm(landmarks["RW"] - landmarks["RE"]))
        len_left_upper = self._len_stab_left_upper.step(raw_left_upper)
        len_left_forearm = self._len_stab_left_forearm.step(raw_left_forearm)
        len_right_upper = self._len_stab_right_upper.step(raw_right_upper)
        len_right_forearm = self._len_stab_right_forearm.step(raw_right_forearm)

        # Step F: Decide whether we must change directions (angles)
        # Even when angles are already valid, we still want to stabilize lengths.
        # In that case, keep the *current frame's* directions and only rescale the
        # bone vectors to the stabilized lengths (no angle change).
        need_left_reconstruct = (
            abs(smoothed_left.pitch - left_angles.pitch) > 0.01 or
            abs(smoothed_left.yaw - left_angles.yaw) > 0.01
        )
        need_right_reconstruct = (
            abs(smoothed_right.pitch - right_angles.pitch) > 0.01 or
            abs(smoothed_right.yaw - right_angles.yaw) > 0.01
        )

        # Step G: Reconstruct every frame using stabilized lengths
        # Left arm
        if need_left_reconstruct:
            U_left_dir = _angles_to_unit_vector(smoothed_left.pitch, smoothed_left.yaw, Xb, Yb, Zb)
        else:
            U_left = landmarks["LE"] - landmarks["LS"]
            U_left_dir = U_left / np.linalg.norm(U_left) if np.linalg.norm(U_left) > TINY else np.zeros_like(U_left)

        new_le, new_lw = self._reconstruct_arm_kinematic(
            landmarks["LS"], landmarks["LE"], landmarks["LW"],
            U_left_dir,
            len_u=len_left_upper,
            len_f=len_left_forearm,
        )
        self._update_keypoint(corrected_keypoints, "left_elbow", new_le)
        self._update_keypoint(corrected_keypoints, "left_wrist", new_lw)

        # Right arm
        if need_right_reconstruct:
            U_right_dir = _angles_to_unit_vector(smoothed_right.pitch, smoothed_right.yaw, Xb, Yb, Zb)
        else:
            U_right = landmarks["RE"] - landmarks["RS"]
            U_right_dir = U_right / np.linalg.norm(U_right) if np.linalg.norm(U_right) > TINY else np.zeros_like(U_right)

        new_re, new_rw = self._reconstruct_arm_kinematic(
            landmarks["RS"], landmarks["RE"], landmarks["RW"],
            U_right_dir,
            len_u=len_right_upper,
            len_f=len_right_forearm,
        )
        self._update_keypoint(corrected_keypoints, "right_elbow", new_re)
        self._update_keypoint(corrected_keypoints, "right_wrist", new_rw)
        
        return CorrectionResult(
            keypoints=corrected_keypoints,
            original_angles=original_angles,
            clamped_angles=clamped_angles,
            smoothed_angles=smoothed_angles,
            corrections_applied={
                "left": left_corrected,
                "right": right_corrected,
            },
        )
    
    def _update_keypoint(
        self,
        keypoints: Dict[str, Any],
        name: str,
        new_pos: np.ndarray,
    ) -> None:
        """Update a keypoint's position in the keypoints dict."""
        if name not in keypoints:
            return
        
        kp = keypoints[name]
        
        # Check if this keypoint uses scaled coordinates
        # (x, y are in pixels; z_scaled is depth)
        if "z_scaled" in kp:
            kp["x"] = float(new_pos[0])
            kp["y"] = float(new_pos[1])
            kp["z_scaled"] = float(new_pos[2])
            # Also update raw z if present (scaled back approximately)
            if "z" in kp:
                # z_scaled = z * scale_factor, but we don't know scale_factor
                # Keep z_scaled as the authoritative depth
                pass
        else:
            # Simple x, y, z format
            kp["x"] = float(new_pos[0])
            kp["y"] = float(new_pos[1])
            kp["z"] = float(new_pos[2])


# ---------------------------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------------------------

def process_video_keypoints(
    frames: List[Dict[str, Any]],
    alpha: float = DEFAULT_ALPHA,
    **corrector_kwargs,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Process an entire video's keypoints with biomechanical correction.
    
    Args:
        frames: List of frame dicts, each with "keypoints" key.
        alpha: EMA smoothing factor.
        **corrector_kwargs: Additional args for PoseCorrector.
    
    Returns:
        (corrected_frames, stats)
        - corrected_frames: List of frames with corrected keypoints.
        - stats: Processing statistics.
    """
    corrector = PoseCorrector(alpha=alpha, **corrector_kwargs)
    corrected_frames = []
    
    angle_history = {
        "left_pitch": [],
        "left_yaw": [],
        "right_pitch": [],
        "right_yaw": [],
    }
    
    for frame in frames:
        keypoints = frame.get("keypoints", {})
        result = corrector.correct_frame(keypoints)
        
        # Build corrected frame
        corrected_frame = copy.deepcopy(frame)
        corrected_frame["keypoints"] = result.keypoints
        
        # Optionally store correction metadata
        corrected_frame["_correction_info"] = {
            "original": {
                "left_pitch": result.original_angles["left"].pitch,
                "left_yaw": result.original_angles["left"].yaw,
                "right_pitch": result.original_angles["right"].pitch,
                "right_yaw": result.original_angles["right"].yaw,
            },
            "smoothed": {
                "left_pitch": result.smoothed_angles["left"].pitch,
                "left_yaw": result.smoothed_angles["left"].yaw,
                "right_pitch": result.smoothed_angles["right"].pitch,
                "right_yaw": result.smoothed_angles["right"].yaw,
            },
            "corrected": result.corrections_applied,
        }
        
        corrected_frames.append(corrected_frame)
        
        # Track angle history
        angle_history["left_pitch"].append(result.smoothed_angles["left"].pitch)
        angle_history["left_yaw"].append(result.smoothed_angles["left"].yaw)
        angle_history["right_pitch"].append(result.smoothed_angles["right"].pitch)
        angle_history["right_yaw"].append(result.smoothed_angles["right"].yaw)
    
    stats = {
        "total_frames": corrector.frame_count,
        "corrections_applied": corrector.correction_stats,
        "alpha": alpha,
        "angle_ranges": {
            "left_pitch": (
                min(angle_history["left_pitch"]) if angle_history["left_pitch"] else 0,
                max(angle_history["left_pitch"]) if angle_history["left_pitch"] else 0,
            ),
            "left_yaw": (
                min(angle_history["left_yaw"]) if angle_history["left_yaw"] else 0,
                max(angle_history["left_yaw"]) if angle_history["left_yaw"] else 0,
            ),
            "right_pitch": (
                min(angle_history["right_pitch"]) if angle_history["right_pitch"] else 0,
                max(angle_history["right_pitch"]) if angle_history["right_pitch"] else 0,
            ),
            "right_yaw": (
                min(angle_history["right_yaw"]) if angle_history["right_yaw"] else 0,
                max(angle_history["right_yaw"]) if angle_history["right_yaw"] else 0,
            ),
        },
    }
    
    return corrected_frames, stats


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> int:
    """Command-line interface for pose correction."""
    ap = argparse.ArgumentParser(
        description="Apply biomechanical constraints to pose keypoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Biomechanical Constraints Applied:
  - Shoulder Pitch: 0° (rest) to 180° (overhead)
  - Left Shoulder Yaw: -45° (across chest) to +135° (out/back)
  - Right Shoulder Yaw: -135° (out/back) to +45° (across chest)

Example:
  python -m pose-corrector.pose_corrector --in_json kalman_out/video1_smoothed_keypoints.json
        """,
    )
    ap.add_argument(
        "--in_json", required=True,
        help="Input keypoints JSON (Kalman-smoothed or raw MediaPipe).",
    )
    ap.add_argument(
        "--out_dir", default=None,
        help="Output directory. Default: same as input with '_corrected' suffix.",
    )
    ap.add_argument(
        "--out_path", default=None,
        help="Explicit output path (overrides --out_dir).",
    )
    ap.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA,
        help=f"EMA smoothing factor [0, 1]. Default: {DEFAULT_ALPHA}",
    )
    ap.add_argument(
        "--no_smooth", action="store_true",
        help="Disable temporal smoothing (only apply clamping).",
    )
    ap.add_argument(
        "--hard_clamp", action="store_true",
        help="Use hard pitch/yaw clamping instead of soft conical constraints (ablation A3).",
    )
    ap.add_argument(
        "--max_drift", type=float, default=DEFAULT_MAX_DRIFT,
        help=f"Length stabilizer safe zone as fraction of ref length. Default: {DEFAULT_MAX_DRIFT}. "
             "Use 1.0 to disable length stabilization (ablation A4).",
    )
    ap.add_argument(
        "--stats", action="store_true",
        help="Print processing statistics.",
    )
    args = ap.parse_args()
    
    # Load input
    in_path = Path(args.in_json)
    data = read_json(in_path)
    
    # Determine output path
    if args.out_path:
        out_path = Path(args.out_path)
    elif args.out_dir:
        out_root = ensure_dir(args.out_dir)
        out_path = out_root / f"{in_path.stem}_corrected.json"
    else:
        out_path = in_path.parent / f"{in_path.stem}_corrected.json"
    
    # Process based on format
    if "frames" in data:
        # Video format
        frames = data["frames"]
        
        if args.no_smooth:
            # Process without smoothing
            corrector = PoseCorrector(
                alpha=1.0,
                max_drift=args.max_drift,
                use_soft_cone=not args.hard_clamp,
            )
            corrected_frames = []
            for frame in frames:
                result = corrector.correct_frame(frame.get("keypoints", {}), apply_smoothing=False)
                corrected_frame = copy.deepcopy(frame)
                corrected_frame["keypoints"] = result.keypoints
                corrected_frames.append(corrected_frame)
            stats = {"total_frames": len(frames), "smoothing": "disabled"}
        else:
            corrected_frames, stats = process_video_keypoints(
                frames,
                alpha=args.alpha,
                max_drift=args.max_drift,
                use_soft_cone=not args.hard_clamp,
            )
        
        # Build output
        output = copy.deepcopy(data)
        output["frames"] = corrected_frames
        output["format"] = "pose_corrector/constrained_keypoints_v1"
        output["correction_params"] = {
            "alpha": args.alpha if not args.no_smooth else None,
            "hard_clamp": args.hard_clamp,
            "max_drift": args.max_drift,
            "use_soft_cone": not args.hard_clamp,
            "pitch_range": [PITCH_MIN, PITCH_MAX],
            "left_yaw_range": [LEFT_YAW_MIN, LEFT_YAW_MAX],
            "right_yaw_range": [RIGHT_YAW_MIN, RIGHT_YAW_MAX],
        }
        
        if args.stats:
            print(f"Processed {stats['total_frames']} frames")
            print(f"Left shoulder corrections: {stats.get('corrections_applied', {}).get('left', 'N/A')}")
            print(f"Right shoulder corrections: {stats.get('corrections_applied', {}).get('right', 'N/A')}")
            if "angle_ranges" in stats:
                ar = stats["angle_ranges"]
                print(f"Left pitch range: {ar['left_pitch'][0]:.1f}° to {ar['left_pitch'][1]:.1f}°")
                print(f"Left yaw range: {ar['left_yaw'][0]:.1f}° to {ar['left_yaw'][1]:.1f}°")
                print(f"Right pitch range: {ar['right_pitch'][0]:.1f}° to {ar['right_pitch'][1]:.1f}°")
                print(f"Right yaw range: {ar['right_yaw'][0]:.1f}° to {ar['right_yaw'][1]:.1f}°")
    
    elif "entries" in data:
        # Image batch format (no temporal smoothing)
        corrector = PoseCorrector(alpha=1.0)
        corrected_entries = []
        
        for entry in data["entries"]:
            keypoints = entry.get("keypoints", {})
            result = corrector.correct_frame(keypoints, apply_smoothing=False)
            corrected_entry = copy.deepcopy(entry)
            corrected_entry["keypoints"] = result.keypoints
            corrected_entries.append(corrected_entry)
        
        output = copy.deepcopy(data)
        output["entries"] = corrected_entries
        output["format"] = "pose_corrector/constrained_keypoints_v1"
        
        if args.stats:
            print(f"Processed {len(corrected_entries)} entries (no temporal smoothing)")
    
    else:
        # Single image format
        corrector = PoseCorrector(alpha=1.0)
        keypoints = data.get("keypoints", {})
        result = corrector.correct_frame(keypoints, apply_smoothing=False)
        
        output = copy.deepcopy(data)
        output["keypoints"] = result.keypoints
        output["format"] = "pose_corrector/constrained_keypoints_v1"
        
        if args.stats:
            print("Processed single image (no temporal smoothing)")
    
    # Write output
    write_json(out_path, output)
    print(f"Wrote: {out_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

