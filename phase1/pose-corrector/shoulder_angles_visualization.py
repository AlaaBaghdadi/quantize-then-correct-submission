import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


MIN_VIS = 0.5  # visibility threshold for using a landmark


def landmark_to_np(landmark) -> np.ndarray:
    """Convert a MediaPipe landmark to a numpy 3D point (x, y, z)."""
    return np.array([landmark.x, landmark.y, landmark.z], dtype=np.float32)


def get_point_if_visible(landmarks, index: int, min_vis: float = MIN_VIS) -> Optional[np.ndarray]:
    lm = landmarks[index]
    if lm.visibility < min_vis:
        return None
    return landmark_to_np(lm)


def calculate_midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


def build_shoulder_frame(
    shoulder: np.ndarray,
    mid_shoulders: np.ndarray,
    hip: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build local coordinate frame at the shoulder:
    - x_axis: lateral, from shoulder to mid-shoulders (across chest)
    - y_axis: vertical, from shoulder to hip (down the torso)
    - z_axis: depth, perpendicular to chest (x × y)
    """
    # Y axis (down torso)
    y_axis = hip - shoulder
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-6:
        raise ValueError("Hip and shoulder are too close to define vertical axis.")
    y_axis /= y_norm

    # X axis (across chest)
    x_axis = mid_shoulders - shoulder
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-6:
        raise ValueError("Mid-shoulders and shoulder are too close to define lateral axis.")
    x_axis /= x_norm

    # Z axis (depth: x × y)
    z_axis = np.cross(x_axis, y_axis)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-6:
        raise ValueError("Failed to build a valid depth axis (body might be degenerate in view).")
    z_axis /= z_norm

    return x_axis, y_axis, z_axis


def calculate_shoulder_angles(
    shoulder: np.ndarray,
    elbow: np.ndarray,
    hip: np.ndarray,
    mid_shoulders: np.ndarray,
) -> Tuple[float, float]:
    """
    Calculate flexion/extension and abduction/adduction for one shoulder.

    Returns:
        (flexion_deg, abduction_deg)
    """
    x_axis, y_axis, z_axis = build_shoulder_frame(shoulder, mid_shoulders, hip)

    # Upper arm vector
    arm_vec = elbow - shoulder
    arm_norm = np.linalg.norm(arm_vec)
    if arm_norm < 1e-6:
        raise ValueError("Elbow and shoulder are too close to define arm vector.")
    arm_vec /= arm_norm

    # Projections on local axes
    v_x = float(np.dot(arm_vec, x_axis))
    v_y = float(np.dot(arm_vec, y_axis))
    v_z = float(np.dot(arm_vec, z_axis))

    # Angles in degrees
    flexion = np.degrees(np.arctan2(v_z, v_y))      # sagittal plane (YZ)
    abduction = np.degrees(np.arctan2(v_x, v_y))    # frontal plane (XY)

    return float(flexion), float(abduction)


def draw_text_block(
    frame,
    lines,
    origin=(10, 30),
    line_height=24,
    color=(0, 255, 0),
    thickness=1,
    font_scale=0.6,
):
    x, y = origin
    for line in lines:
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y += line_height


def process_video(
    source: str,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    flip_horizontal: bool = True,
):
    """
    Run MediaPipe Pose on a video/webcam source and visualize shoulder angles.

    `source` can be:
      - a path to a video file
      - "0", "1", ... to refer to a webcam index
    """
    # Determine if source is webcam index
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    with mp_pose.Pose(
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as pose:
        while True:
            success, frame = cap.read()
            if not success:
                break

            if flip_horizontal:
                frame = cv2.flip(frame, 1)

            # Convert to RGB as MediaPipe expects
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)
            image_rgb.flags.writeable = True

            left_angles = None
            right_angles = None

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Get 3D points if visible
                left_shoulder = get_point_if_visible(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
                right_shoulder = get_point_if_visible(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
                left_elbow = get_point_if_visible(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW.value)
                right_elbow = get_point_if_visible(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW.value)
                left_hip = get_point_if_visible(landmarks, mp_pose.PoseLandmark.LEFT_HIP.value)
                right_hip = get_point_if_visible(landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value)

                # Mid-shoulder (neck) in normalized coordinates
                if left_shoulder is not None and right_shoulder is not None:
                    mid_shoulders = calculate_midpoint(left_shoulder, right_shoulder)
                else:
                    mid_shoulders = None

                # Compute angles if all required points are available and frame is well-posed
                try:
                    if (
                        left_shoulder is not None
                        and left_elbow is not None
                        and left_hip is not None
                        and mid_shoulders is not None
                    ):
                        left_angles = calculate_shoulder_angles(
                            left_shoulder, left_elbow, left_hip, mid_shoulders
                        )
                except ValueError:
                    left_angles = None

                try:
                    if (
                        right_shoulder is not None
                        and right_elbow is not None
                        and right_hip is not None
                        and mid_shoulders is not None
                    ):
                        right_angles = calculate_shoulder_angles(
                            right_shoulder, right_elbow, right_hip, mid_shoulders
                        )
                except ValueError:
                    right_angles = None

                # Draw pose skeleton on frame (2D projection)
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

            # Prepare text lines
            lines = []
            if left_angles is not None:
                lf, la = left_angles
                lines.append(f"Left  Flex: {lf:6.1f} deg")
                lines.append(f"Left  Abd : {la:6.1f} deg")
            else:
                lines.append("Left  shoulder: N/A")

            if right_angles is not None:
                rf, ra = right_angles
                lines.append(f"Right Flex: {rf:6.1f} deg")
                lines.append(f"Right Abd : {ra:6.1f} deg")
            else:
                lines.append("Right shoulder: N/A")

            draw_text_block(frame, lines, origin=(10, 30))

            cv2.imshow("Shoulder Angles (MediaPipe Pose)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):  # ESC or q to quit
                break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract and visualize shoulder flexion/abduction angles using MediaPipe Pose."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help=(
            "Video source: path to a video file or a webcam index (as string), "
            "e.g. '0' for default webcam."
        ),
    )
    parser.add_argument(
        "--min_detection_confidence",
        type=float,
        default=0.5,
        help="MediaPipe minimum detection confidence.",
    )
    parser.add_argument(
        "--min_tracking_confidence",
        type=float,
        default=0.5,
        help="MediaPipe minimum tracking confidence.",
    )
    parser.add_argument(
        "--no_flip",
        action="store_true",
        help="Do not horizontally flip the input (by default webcam is mirrored).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Expand relative video path for convenience
    source = args.source
    if not source.isdigit():
        source_path = Path(source)
        source = str(source_path)

    process_video(
        source=source,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        flip_horizontal=not args.no_flip,
    )


if __name__ == "__main__":
    main()

