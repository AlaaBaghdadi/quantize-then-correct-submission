import cv2 
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Skeleton connections to draw
POSE_BODY_CONNECTIONS = [
    (11, 12),  # left shoulder - right shoulder
    (23, 24),  # left hip - right hip
    (11, 13),  # left shoulder - left elbow
    (13, 15),  # left elbow - left wrist
    (12, 14),  # right shoulder - right elbow
    (14, 16),  # right elbow - right wrist
    (23, 25),  # left hip - left knee
    (25, 27),  # left knee - left ankle
    (27, 29),  # left ankle - left heel
    (29, 31),  # left heel - left foot index
    (24, 26),  # right hip - right knee
    (26, 28),  # right knee - right ankle
    (28, 30),  # right ankle - right heel
    (30, 32),  # right heel - right foot index
]

MIN_VIS = 0.3

# Open video file
cap = cv2.VideoCapture("p2.mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter("test_skeleton_output.mp4", fourcc, fps, (frame_width, frame_height))

# Initialize MediaPipe Pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    frame_num = 0
    
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break
        
        # Flip image horizontally for mirror effect (optional)
        frame = cv2.flip(image, 1)
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Process the frame
        results = pose.process(image_rgb)
        
        image_rgb.flags.writeable = True
        
        # Draw skeleton if pose detected
        if results.pose_landmarks is not None:
            landmarks = results.pose_landmarks.landmark
            
            # Draw skeleton connections
            for (a, b) in POSE_BODY_CONNECTIONS:
                # Check visibility of both landmarks
                if (landmarks[a].visibility > MIN_VIS) and (landmarks[b].visibility > MIN_VIS):
                    pt1 = (int(landmarks[a].x * frame_width), int(landmarks[a].y * frame_height))
                    pt2 = (int(landmarks[b].x * frame_width), int(landmarks[b].y * frame_height))
                    
                    # Draw line (BGR format: green color)
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 3)
            
            # Draw landmarks as circles
            for idx, landmark in enumerate(landmarks):
                if landmark.visibility > MIN_VIS:
                    pt = (int(landmark.x * frame_width), int(landmark.y * frame_height))
                    # Draw circle for each landmark (BGR format: blue color)
                    cv2.circle(frame, pt, 5, (255, 0, 0), -1)
        
        # Write frame to output video
        out_writer.write(frame)
        frame_num += 1
        
        # Optional: Display frame (uncomment to see live preview)
        # cv2.imshow('Skeleton Detection', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

# Release resources
cap.release()
out_writer.release()
# cv2.destroyAllWindows()  # Uncomment if using imshow

print(f"Done. Processed {frame_num} frames. Output saved to test_skeleton_output.mp4")
