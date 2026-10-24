import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from estimator.config import POSE_CONNECTIONS


class PoseEstimator:
    def __init__(self, smoothing_window=5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
        self.smoothing_window = smoothing_window
        self.landmark_buffer = {}  # {joint_name: deque}
        self.joint_names = [lm.name for lm in self.mp_pose.PoseLandmark]
        
        # print("all landmarsk:", [lm.name for lm in self.mp_pose.PoseLandmark])

    def extract_landmarks(self, frame):
        """Process one frame, extract 3D landmarks."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        if not results.pose_landmarks:
            return None  # No pose detected

        landmarks = results.pose_landmarks.landmark
        joint_positions = {}
        for idx, joint_name in enumerate(self.joint_names):
            lm = landmarks[idx]
            joint_positions[joint_name] = (lm.x, lm.y, lm.z)
        return joint_positions

    def update_buffer(self, joint_positions):
        """Update buffers for moving average smoothing."""
        for name, pos in joint_positions.items():
            if name not in self.landmark_buffer:
                self.landmark_buffer[name] = deque(maxlen=self.smoothing_window)
            self.landmark_buffer[name].append(np.array(pos))

    def smooth_landmarks(self):
        """Return smoothed positions using simple moving average."""
        smoothed = {}
        for name, buffer in self.landmark_buffer.items():
            if len(buffer) > 1:
                avg = np.mean(buffer, axis=0)
                smoothed[name] = tuple(avg)
            else:
                smoothed[name] = tuple(buffer[-1])  # Not enough for window, return latest
        return smoothed

    def process_video(self, video_path):
        """Main loop for video processing."""
        cap = cv2.VideoCapture(video_path)
        
        self._frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = cap.get(cv2.CAP_PROP_FPS)
        all_frames_landmarks = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            joint_positions = self.extract_landmarks(frame)
            if joint_positions is None:
                all_frames_landmarks.append(None)
                continue
            
            self.update_buffer(joint_positions)
            smoothed_positions = self.smooth_landmarks()

            all_frames_landmarks.append(smoothed_positions)

        cap.release()
        self.pose.close()
        return all_frames_landmarks
    
    
    def _draw_skeleton(self, joints, image_size=(480, 640)):
        """Draw pose skeleton on a white image. joints: dict of {name: (x, y, z)}"""
        h, w = image_size
        blank = np.ones((h, w, 3), dtype=np.uint8) * 255

        # Draw bones
        for joint1, joint2 in POSE_CONNECTIONS:
            if joint1 in joints and joint2 in joints:
                x1, y1, _ = joints[joint1]
                x2, y2, _ = joints[joint2]
                pt1 = (int(x1 * w), int(y1 * h))
                pt2 = (int(x2 * w), int(y2 * h))
                cv2.line(blank, pt1, pt2, (0, 200, 0), 3)

        # Draw joints
        for name, (x, y, z) in joints.items():
            center = (int(x * w), int(y * h))
            cv2.circle(blank, center, 6, (0, 0, 255), -1)
        return blank

    def create_pose_video(self, skeleton_sequence, output_path="pose_video.mp4"):
        
        print("Details:",self._fps, (self._frame_width, self._frame_height))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self._fps, (self._frame_width, self._frame_height))
        for joints in skeleton_sequence:
            # 👈 Pass image_size as (height, width), note the order!
            frame = self._draw_skeleton(
                joints if joints is not None else {},
                image_size=(self._frame_height, self._frame_width)  
            )
            if frame.shape != (self._frame_height, self._frame_width, 3):
                print(f"[Warning] Frame shape: {frame.shape}")
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            out.write(frame)
        out.release()
        print(f"Pose video written to {output_path}")


