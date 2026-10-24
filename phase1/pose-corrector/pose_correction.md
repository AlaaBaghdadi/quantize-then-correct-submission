## Pose Correction: Constrained Kinematic Loop

This document explains the logic behind the **Constrained Kinematic Correction**
implemented in `pose_corrector.py` inside the `pose-corrector` package.

The goal is to take **MediaPipe 3D landmarks** (optionally already Kalman-smoothed),
detect **biomechanically impossible shoulder configurations**, and correct them
*in a physically consistent way* by:

- **Clamping** shoulder Pitch/Yaw to realistic human limits.
- **Smoothing** angles over time for temporal coherence.
- **Reconstructing** the 3D elbow and wrist positions to match the corrected angles.

The output is a new set of landmarks that respect human joint limits while
remaining as close as possible to the original signal.

---

### 1. Inputs and Coordinate Frames

- **Input landmarks**: MediaPipe-style dict with keys:
  - `left_shoulder`, `right_shoulder`
  - `left_elbow`, `right_elbow`
  - `left_wrist`, `right_wrist`
  - `left_hip`
- These are converted to NumPy arrays (`LS, RS, LE, RE, LW, RW, LH`) using
  `pose_handler.calc_angles.keypoints_to_arrays`.

- **Body frame** is constructed via `body_frame_orthonormal(LS, RS, LH)`:
  - **Xb**: Lateral right (approx. rightward from the body).
  - **Yb**: Forward (approx. facing direction).
  - **Zb**: Up from hip toward left shoulder.

This body frame is **orthonormal and right-handed**, which makes angular
calculations and reconstruction stable and interpretable.

---

### 2. Shoulder Angles (Deconstruct Step A)

For each shoulder we derive:

- **Pitch**: Elevation angle relative to `-Zb` (arm down).
  - \( 0^\circ \): arm down / rest (aligned with \(-Z_b\)).
  - \( 90^\circ \): arm horizontal.
  - \( 180^\circ \): arm overhead (aligned with \(+Z_b\)).

- **Yaw**: Rotation in the horizontal plane (perpendicular to `Zb`),
  measured using an `atan2` formulation:
  - \( 0^\circ \): arm pointing forward along `Yb`.
  - \( +90^\circ \): arm lateral toward `Xb` (right).
  - \( -90^\circ \): arm lateral toward `-Xb` (left).

- **Roll**: Rotation around the upper-arm axis (used for diagnostics and
  pass-through; landmarks are not modified based on roll).

Implementation details:

- Upper arm unit vector:
  - `U = (Elbow - Shoulder) / ||Elbow - Shoulder||`
- Pitch:
  - Angle between `U` and `-Zb` via robust `atan2(norm(cross), dot)` logic.
- Yaw:
  - Project `U` onto plane ⟂ `Zb`, then measure angle from `Yb` using `atan2`.
- Roll:
  - Defined by orientation of the arm–forearm plane relative to a reference plane.

This logic mirrors `calc_angles.py` so the **sign conventions and angle
definitions are consistent** throughout the pipeline.

---

### 3. Biomechanical Constraints (Clamp Step B)

We define a **feasible cone** for shoulder motion based on simple but realistic
biomechanical bounds:

- **Pitch range** (both shoulders):
  - \( \text{Pitch} \in [0^\circ, 180^\circ] \)
  - 0°: down, 180°: overhead.

- **Yaw range** (asymmetric due to chest collision):
  - **Left shoulder**:
    - \( \text{Yaw}_L \in [-45^\circ, +135^\circ] \)
    - \(-45^\circ\): arm crossing chest.
    - \(+135^\circ\): arm rotated out/back behind the body.
  - **Right shoulder**:
    - \( \text{Yaw}_R \in [-135^\circ, +45^\circ] \)
    - \(-135^\circ\): arm rotated out/back behind the body.
    - \(+45^\circ\): arm crossing chest.

These ranges follow the **sign logic** from `calc_angles.py`:

- Yaw sign is determined in the `Xb–Yb` plane.
- Left vs right constraints are **mirrored** to prevent the arm from entering
  the chest region more than ±45°, while allowing large out/back motion.

Clamping is done in a **motion-aware** and **wrap-aware** way:

- If an angle is inside the range: **leave it unchanged**.
- **Pitch** (non-periodic): clamp to the nearest boundary (min or max).
- **Yaw** (periodic 360°):
  - Compute the **shortest angular distance** on the circle to each boundary.
  - Choose the **nearest boundary** (so e.g. -180° snaps to +135° if that is 45° away, rather than -45° at 135° away).
  - When both boundaries are nearly equidistant (tie), use the **direction of motion** from the previous frame’s smoothed angle: choose the boundary we are moving toward, so the pose continues the trajectory instead of snapping the other way.
- Optional **velocity limiting**: after EMA, the per-frame change in pitch/yaw can be capped (e.g. 25° per frame), so large corrections are spread over several frames and the arm approaches the limit smoothly instead of “teleporting” in one frame.

---

### 4. Temporal Smoothing (Smooth Step C)

To avoid frame-to-frame jitter or abrupt corrections, the class
`PoseCorrector` maintains **state across frames**:

- `prev_valid_angles` is represented by `CorrectionState`:
  - `left_shoulder: ShoulderAngles`
  - `right_shoulder: ShoulderAngles`
  - `initialized: bool`

For each frame, after clamping:

- We compute **Exponential Moving Average (EMA)**:
  - **Pitch and roll**: \( \theta_{\text{smooth}} = \alpha \theta_{\text{current}} + (1-\alpha) \theta_{\text{prev}} \).
  - **Yaw**: interpolated on the circle (shortest path): \( \Delta = \text{shortest\_angle}(\theta_{\text{prev}}, \theta_{\text{current}}) \), then \( \theta_{\text{smooth}} = \theta_{\text{prev}} + \alpha \Delta \). This avoids averaging 170° and -170° to 0° and instead smooths toward 180°.
  - `alpha` is typically `0.7`:
    - High enough to respond to real motion.
    - Low enough to suppress short-lived spikes.

- Optional **velocity limiting** caps the change from the previous frame (e.g. 25° per frame), so large corrections ease in over multiple frames.

- On the **first valid frame**, or when smoothing is disabled, the angles are
  taken directly from the clamped values (no averaging).

This EMA creates **temporally coherent** corrections, critical for video-based
pose where sudden angle snaps look physically wrong.

---

### 5. Reconstruction of 3D Landmarks (Reconstruct Step D)

Correcting angles alone is insufficient; we must move the landmarks so that
the geometry matches the constrained kinematics.

For each shoulder:

1. **Compute original upper-arm length**:
   - \( L_{\text{upper}} = \|\text{Elbow} - \text{Shoulder}\| \)
   - This length is **preserved** to avoid unnatural bone scaling.

2. **Convert corrected pitch/yaw back to a 3D unit vector** in body frame:

   Given pitch \(\theta\) and yaw \(\phi\):

   \[
   \mathbf{u} =
       \sin(\theta)\sin(\phi)\,\hat{X}_b \;+\;
       \sin(\theta)\cos(\phi)\,\hat{Y}_b \;-\;
       \cos(\theta)\,\hat{Z}_b
   \]

   - \(\theta = 0^\circ \Rightarrow \mathbf{u} = -\hat{Z}_b\) (arm down).
   - \(\theta = 90^\circ, \phi = 0^\circ \Rightarrow \mathbf{u} = \hat{Y}_b\) (forward).
   - \(\theta = 90^\circ, \phi = 90^\circ \Rightarrow \mathbf{u} = \hat{X}_b\) (lateral).
   - \(\theta = 180^\circ \Rightarrow \mathbf{u} = +\hat{Z}_b\) (overhead).

3. **Reconstruct new elbow position**:

   \[
   \text{Elbow}_{\text{new}} = \text{Shoulder} + \mathbf{u} \cdot L_{\text{upper}}
   \]

4. **Compute displacement** imparted by correction:

   \[
   \Delta = \text{Elbow}_{\text{new}} - \text{Elbow}_{\text{old}}
   \]

5. **Translate the wrist** by the same displacement to keep the forearm rigid:

   \[
   \text{Wrist}_{\text{new}} = \text{Wrist}_{\text{old}} + \Delta
   \]

This approach:

- Preserves both **upper-arm length** and **forearm length**.
- Treats the forearm as a **rigid body** under a pure translation, which is
  a good approximation for shoulder-only corrections.
- Ensures the corrected skeleton is **kinematically plausible** and spatially
  close to the input.

Reconstruction is only applied when **corrected angles differ meaningfully**
from the original (to avoid unnecessary numerical noise).

---

### 6. Robustness and Failure Modes

The correction loop is designed to fail gracefully:

- **Missing landmarks**:
  - If any required keypoint (`LS, RS, LE, RE, LW, RW, LH`) is missing,
    the frame is **returned unchanged**, and internal state is not updated.

- **Degenerate geometry**:
  - If the upper-arm length is near zero (elbow at shoulder), the code skips
    reconstruction and passes through original elbow/wrist.
  - Norm checks use small thresholds (`TINY`, `EPS`) to avoid division by zero.

- **Angle computation** is fully `atan2`-based (norm–dot with cross), which is
  numerically robust even when vectors are nearly parallel.

---

### 7. Temporal API and Usage Patterns

- **Class API** (per-frame streaming):
  - Instantiate once per sequence:
    - `corrector = PoseCorrector(alpha=0.7)`
  - For each video frame:
    - `result = corrector.correct_frame(frame["keypoints"])`
    - Use `result.keypoints` as corrected landmarks.
  - `reset()` between independent sequences or when tracking is lost.

- **Batch API**:
  - `process_video_keypoints(frames, alpha=0.7)` takes a list of frames and
    returns:
    - Corrected frames with `_correction_info` metadata.
    - Global statistics about corrections and angle ranges.

---

### 8. Design Rationale

- **Minimal change principle**:
  - Only clamp when angles leave the biomechanical envelope.
  - Clamp to the **nearest feasible boundary**.
  - Preserve original bone lengths and frame-to-frame continuity.

- **Biomechanics-aware**:
  - The yaw limits explicitly model **chest collision** and
    **out/back reach**, instead of allowing symmetric 360° yaw.

- **Decoupled from estimation noise**:
  - Upstream smoothing (e.g., Kalman) handles measurement noise.
  - This module focuses on **kinematic feasibility** and **temporal coherence**.

Together, these decisions produce a corrected pose that is **numerically
stable**, **biomechanically constrained**, and **compatible with existing
angle logic** defined in `pose_handler/calc_angles.py`.

