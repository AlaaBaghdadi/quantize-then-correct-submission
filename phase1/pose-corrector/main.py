# import cv2 
# import mediapipe as mp
# import numpy as np
# import logging
# logging.getLogger().setLevel(logging.ERROR)
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# cap = cv2.VideoCapture("p1.mp4")
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)
# out = cv2.VideoWriter("posed.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# print("FPS: ", fps)
# while cap.isOpened():
#     ret, image = cap.read()
#     if not ret:
#         break
#     image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
#     image.flags.writeable = False
#     results = pose.process(image)
#     image.flags.writeable = True

#     # Create a white background
#     white_bg = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

#     # Draw pose on white background
#     if results.pose_landmarks is not None:
#         mp_drawing.draw_landmarks(
#             white_bg, 
#             results.pose_landmarks, 
#             mp_pose.POSE_CONNECTIONS,
#             mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=3), # landmark circles
#             mp_drawing.DrawingSpec(color=(0,255,0), thickness=2) # connections
#         )
#     out.write(white_bg)

# pose.close()
# cap.release()
# out.release()

from estimator.pose_estimator import PoseEstimator

from estimator.skeleton_model import SkeletonKinematicModel

from estimator.config import POSE_CONNECTIONS

video_path = 'p1.mp4'
estimator = PoseEstimator(smoothing_window=3)
skeleton_sequence = estimator.process_video(video_path)
# print("sek",skeleton_sequence)
# estimator.create_pose_video(skeleton_sequence=skeleton_sequence)

kinematic_model = SkeletonKinematicModel()
# kinematic_model.compute_average_bone_lengths(skeleton_sequence, N_init=3)

angles_sequence = kinematic_model.compute_joint_angles_sequence(skeleton_sequence)

for fi, angles in enumerate(angles_sequence):
    print(f"Frame {fi}: {angles}")
    
    
import cv2

# Open video for reading
output_path = 'output_with_angles.mp4'

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: cannot open video!")
    exit()

# Get video properties
fps    = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Prepare video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Or use 'XVID', 'avc1', etc
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret or frame_idx >= len(angles_sequence):
        break

    # Get the angles for the current frame
    angles = angles_sequence[frame_idx]

    # Overlay the angles as text
    y0 = 30
    dy = 30
    for j, (joint, angle) in enumerate(angles.items()):
        text = f"{joint}: {angle:.2f}"
        y = y0 + j * dy
        # text, origin, font, font_scale, color(BGR), thickness, lineType
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Write the frame to output video
    out.write(frame)

    frame_idx += 1

# Release everything
cap.release()
out.release()
print(f"Saved with overlays: {output_path}")