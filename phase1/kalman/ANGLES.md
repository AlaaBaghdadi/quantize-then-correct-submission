# Upper-Body Joint Angle Definitions (Hip-Based Body Frame)

This document describes how shoulder (Pitch, Yaw, Roll) and elbow angles are computed in the Kalman video pipeline. The logic lives in `pose_handler/calc_angles.py`. It uses a **hip-based body reference frame** and 3D MediaPipe-style keypoints (`x`, `y`, `z` / `z_scaled`).

---

## 1. Keypoint Abbreviations

| Abbreviation | Keypoint        | Description        |
|-------------|------------------|--------------------|
| LS          | left_shoulder    | Left shoulder      |
| RS          | right_shoulder   | Right shoulder     |
| LE          | left_elbow       | Left elbow         |
| RE          | right_elbow      | Right elbow        |
| LW          | left_wrist       | Left wrist         |
| RW          | right_wrist      | Right wrist        |
| LH          | left_hip         | Left hip           |
| RH          | right_hip        | Right hip (optional) |

---

## 2. Body Reference Frame (Hip-Based, Orthorectified)

The body frame is built from the **left hip** and shoulders, then orthorectified so \(X_b\), \(Y_b\), \(Z_b\) are perfectly orthogonal and unit length.

### 2.1 Axis Definitions

- **Zb (blue, “up” from hip)**  
  \[
  Z_b = \frac{LS - LH}{\|LS - LH\|}
  \]  
  From left hip to left shoulder, normalized.

- **Yb (green, forward)**  
  \[
  Y_b = \frac{Z_b \times (RS - LS)}{\|Z_b \times (RS - LS)\|}
  \]  
  Cross of Zb with shoulder line; normalized.

- **Xb (red, lateral)**  
  \[
  X_b = \frac{Y_b \times Z_b}{\|Y_b \times Z_b\|}
  \]  
  Completes right-handed frame; normalized.

- **Orthorectification**  
  Axes are re-built from cross products so \(X_b, Y_b, Z_b\) form a perfect orthonormal right-handed basis (no drift from floating-point).

---

## 3. Shoulder Pitch (Green Arc)

Angle between the **upper arm** (shoulder → elbow) and **−Zb** (downward from shoulder in body frame). Computed with an **atan2-based** formula (no `acos`) for robustness and to avoid NaN when the dot product slightly exceeds 1.0.

### What the person must do (joint reference)

| Angle | Pose / action |
|-------|----------------|
| **0°**  | Arm down at side (upper arm aligned with −Zb). |
| **90°** | Arm forward, horizontal (perpendicular to −Zb in the forward plane). |
| **180°**| Arm straight up (upper arm aligned with +Zb). |

---

## 4. Shoulder Yaw (Red Arc)

Angle of the **upper arm projected onto the plane perpendicular to Zb**, measured from the **forward** direction **Yb** (primary axis). Uses `atan2` for full range and sign; **Xb** (lateral) is the secondary axis.  
- Upper arm projected onto plane ⊥ Zb:  
  \(U_{proj} = U - (U \cdot Z_b) Z_b\), then normalized.  
- Yaw = `atan2( cross(Yb, U_proj)·Zb, Yb·U_proj )` in degrees (Yb = 0° reference, Xb = ±90°).

### What the person must do (joint reference)

| Angle | Pose / action |
|-------|----------------|
| **0°**  | Arm **forward** (horizontal); arm’s projection aligns with Yb (forward). |
| **90°** | Arm **lateral** (out to the side); arm’s projection aligns with Xb (left arm → +90°, right arm → −90°). |

---

## 5. Shoulder Roll (Cyan Arc)

Rotation of the **arm plane** (upper arm × forearm) relative to a reference normal in the plane perpendicular to the upper arm.  
- Arm plane normal: \(N = \mathrm{normalize}(U \times F)\) (upper arm × forearm).  
- Reference normal: \(N_{ref} = \mathrm{normalize}((-Z_b) \times U)\); if degenerate, use **Xb**.  
- Roll = `atan2( cross(Nref, N)·U, Nref·N )` in degrees.

### What the person must do (joint reference)

| Angle | Pose / action |
|-------|----------------|
| **0°**  | Neutral twist (e.g. palm roughly facing leg or forward, no internal/external rotation). |
| **±90°**| Maximum internal or external rotation of the humerus (twist so elbow/wrist move in/out). |

---

## 6. Elbow Angle (Magenta Arc)

Angle **at the elbow** between the two segments: shoulder→elbow and elbow→wrist. Computed with **atan2** between the two unit vectors (no `acos`) so it is robust to floating-point error.  
- Vectors from elbow: shoulder−elbow and wrist−elbow; angle between them in **[0°, 180°]**.

### What the person must do (joint reference)

| Angle | Pose / action |
|-------|----------------|
| **180°** | Straight arm (shoulder, elbow, wrist in a line). |
| **90°**  | L-shape (forearm perpendicular to upper arm). |
| **0°**   | Fully closed (forearm folded against upper arm). |

---

## 7. Summary: Joint Reference (0°, 90°, 180°)

| Joint / angle   | 0° | 90° | 180° |
|----------------|----|-----|------|
| **Shoulder Pitch** | Arm down | Arm forward (horizontal) | Arm straight up |
| **Shoulder Yaw**   | Arm forward | Arm lateral | — |
| **Shoulder Roll**  | Neutral twist | Max internal/external rotation | — |
| **Elbow**          | Fully closed (folded) | L-shape | Straight arm |

---

## 8. Output Structure

### Structured (nested)

```json
{
  "left_shoulder":  { "pitch": 12.5, "yaw": -5.2, "roll": 3.1 },
  "right_shoulder": { "pitch": 10.1, "yaw": 4.8,  "roll": -2.0 },
  "left_elbow":     { "angle": 165.2 },
  "right_elbow":    { "angle": 92.4 }
}
```

### Flat (e.g. for JSON logs / pipeline)

```json
{
  "left_shoulder_pitch": 12.5,
  "left_shoulder_yaw": -5.2,
  "left_shoulder_roll": 3.1,
  "left_elbow_angle": 165.2,
  "right_shoulder_pitch": 10.1,
  "right_shoulder_yaw": 4.8,
  "right_shoulder_roll": -2.0,
  "right_elbow_angle": 92.4
}
```

All angles are in **degrees**, rounded to 2 decimal places.

---

## 9. Implementation Notes (`pose_handler/calc_angles.py`)

- **Entry points**
  - `calc_angles_from_landmarks(landmarks)` → structured dict (nested).
  - `calc_angles_flat(landmarks)` → flat dict.
  - `angles_from_keypoints(kps, image_size=..., flat=...)` → from MediaPipe-style keypoints.
- **Body frame**: `body_frame_orthonormal(LS, RS, LH)` returns orthorectified \(X_b, Y_b, Z_b\).
- **Robustness**
  - Shoulder pitch and elbow angle use **atan2(norm(cross(a,b)), dot(a,b))** instead of `acos(dot(...))` to avoid NaN when dot is slightly &gt; 1 and to keep numeric stability.
  - Yaw and roll already use `atan2` in the MATLAB-derived formulation.
- **Compatibility**
  - `JointVisualizer(LS, RS, LE, RE, LW, RW, LH, RH=None)` and `.calculate_angles()` return the **flat** angle dict for existing callers (e.g. visualizers).

---

## 10. Naming vs Previous Systems

| This implementation | Old / other names |
|--------------------|--------------------|
| Shoulder **Pitch** | Shoulder flexion (sagittal); 0° down, 90° forward, 180° up |
| Shoulder **Yaw**   | 0° = forward (Yb), 90° = lateral (Xb) |
| Shoulder **Roll**  | Internal/external rotation |
| Elbow **angle**    | 180° = straight, 90° = L, 0° = fully closed (kinematic convention) |

The current pipeline uses the **hip-based** body frame and the **atan2-based** formulas above.
