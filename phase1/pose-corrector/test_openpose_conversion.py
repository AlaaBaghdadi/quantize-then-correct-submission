import cv2 
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

# OpenPose COCO format (18 keypoints) mapping from MediaPipe
# OpenPose index: (MediaPipe index, description)
OPENPOSE_MAPPING = {
    0: (0, "Nose"),                    # Nose
    1: None,                           # Neck - calculated from shoulders
    2: (12, "Right Shoulder"),         # Right Shoulder
    3: (14, "Right Elbow"),            # Right Elbow
    4: (16, "Right Wrist"),            # Right Wrist
    5: (11, "Left Shoulder"),          # Left Shoulder
    6: (13, "Left Elbow"),             # Left Elbow
    7: (15, "Left Wrist"),             # Left Wrist
    8: (24, "Right Hip"),              # Right Hip
    9: (26, "Right Knee"),             # Right Knee
    10: (28, "Right Ankle"),           # Right Ankle
    11: (23, "Left Hip"),              # Left Hip
    12: (25, "Left Knee"),             # Left Knee
    13: (27, "Left Ankle"),            # Left Ankle
    14: (5, "Right Eye"),              # Right Eye
    15: (2, "Left Eye"),               # Left Eye
    16: (8, "Right Ear"),              # Right Ear
    17: (7, "Left Ear"),               # Left Ear
}

# OpenPose skeleton connections (COCO format)
OPENPOSE_CONNECTIONS = [
    (1, 2),   # Neck - Right Shoulder
    (1, 5),   # Neck - Left Shoulder
    (1, 8),   # Neck - Right Hip
    (1, 11),  # Neck - Left Hip
    (2, 3),   # Right Shoulder - Right Elbow
    (3, 4),   # Right Elbow - Right Wrist
    (5, 6),   # Left Shoulder - Left Elbow
    (6, 7),   # Left Elbow - Left Wrist
    (8, 9),   # Right Hip - Right Knee
    (9, 10),  # Right Knee - Right Ankle
    (11, 12), # Left Hip - Left Knee
    (12, 13), # Left Knee - Left Ankle
    (0, 1),   # Nose - Neck
    (0, 14),  # Nose - Right Eye
    (0, 15),  # Nose - Left Eye
    (14, 16), # Right Eye - Right Ear
    (15, 17), # Left Eye - Left Ear
]

MIN_VIS = 0.3

def calculate_neck_point(left_shoulder, right_shoulder):
    """Calculate neck point as midpoint between left and right shoulders."""
    if left_shoulder is None or right_shoulder is None:
        return None
    
    # Calculate midpoint
    neck_x = (left_shoulder.x + right_shoulder.x) / 2
    neck_y = (left_shoulder.y + right_shoulder.y) / 2
    neck_z = (left_shoulder.z + right_shoulder.z) / 2
    
    # Average visibility
    neck_visibility = (left_shoulder.visibility + right_shoulder.visibility) / 2
    
    # Create a simple object with the required attributes
    class NeckPoint:
        def __init__(self, x, y, z, visibility):
            self.x = x
            self.y = y
            self.z = z
            self.visibility = visibility
    
    return NeckPoint(neck_x, neck_y, neck_z, neck_visibility)

def convert_to_openpose(landmarks):
    """Convert MediaPipe landmarks to OpenPose COCO format (18 keypoints)."""
    openpose_keypoints = [None] * 18
    
    # Map MediaPipe landmarks to OpenPose format
    for openpose_idx, mapping in OPENPOSE_MAPPING.items():
        if mapping is None:
            # Neck point (index 1) - calculate from shoulders
            left_shoulder = landmarks[11]  # MediaPipe left shoulder
            right_shoulder = landmarks[12]  # MediaPipe right shoulder
            openpose_keypoints[1] = calculate_neck_point(left_shoulder, right_shoulder)
        else:
            mediapipe_idx, _ = mapping
            openpose_keypoints[openpose_idx] = landmarks[mediapipe_idx]
    
    return openpose_keypoints

# Open video file
cap = cv2.VideoCapture("p2.mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter("test_openpose_output.mp4", fourcc, fps, (frame_width, frame_height))

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
            
            # Convert to OpenPose format
            openpose_keypoints = convert_to_openpose(landmarks)
            
            # Draw skeleton connections using OpenPose format
            for (a, b) in OPENPOSE_CONNECTIONS:
                # Check if both keypoints exist and are visible
                if (openpose_keypoints[a] is not None and 
                    openpose_keypoints[b] is not None and
                    openpose_keypoints[a].visibility > MIN_VIS and 
                    openpose_keypoints[b].visibility > MIN_VIS):
                    
                    pt1 = (int(openpose_keypoints[a].x * frame_width), 
                           int(openpose_keypoints[a].y * frame_height))
                    pt2 = (int(openpose_keypoints[b].x * frame_width), 
                           int(openpose_keypoints[b].y * frame_height))
                    
                    # Draw line (BGR format: green color)
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 3)
            
            # Draw landmarks as circles
            for idx, keypoint in enumerate(openpose_keypoints):
                if keypoint is not None and keypoint.visibility > MIN_VIS:
                    pt = (int(keypoint.x * frame_width), int(keypoint.y * frame_height))
                    # Draw circle for each landmark (BGR format: blue color)
                    cv2.circle(frame, pt, 5, (255, 0, 0), -1)
        
        # Write frame to output video
        out_writer.write(frame)
        frame_num += 1
        
        # Optional: Display frame (uncomment to see live preview)
        # cv2.imshow('OpenPose Skeleton Detection', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

# Release resources
cap.release()
out_writer.release()
# cv2.destroyAllWindows()  # Uncomment if using imshow

print(f"Done. Processed {frame_num} frames. Output saved to test_openpose_output.mp4")
