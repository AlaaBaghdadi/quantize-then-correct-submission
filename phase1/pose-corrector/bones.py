import cv2 
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

# ONLY the bones you want to monitor
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


bone_names = [
    "LShoulder-RShoulder", "LHip-RHip", "LShoulder-LElbow", "LElbow-LWrist",
    "RShoulder-RElbow", "RElbow-RWrist", "LHip-LKnee", "LKnee-LAnkle",
    "LAnkle-LHeel", "LHeel-LFootIndex", "RHip-RKnee", "RKnee-RAnkle", "RAnkle-RHeel", "RHeel-RFootIndex"
]
def euclidean3d(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def angle3d(a, b, c):
    # a, b, c: np.array or tuple, in 3D
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_ang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    # Clamp for numerical safety
    cos_ang = np.clip(cos_ang, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_ang))
    return angle_deg

MIN_VIS = 0.3
cap = cv2.VideoCapture("p2.mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
white_bg_writer = cv2.VideoWriter("posed_white.mp4", fourcc, fps, (frame_width, frame_height))
orig_writer = cv2.VideoWriter("posed_orig.mp4", fourcc, fps, (frame_width, frame_height))
font = cv2.FONT_HERSHEY_SIMPLEX
threshold = 0.3

LEFT_SHOULDER = 11
LEFT_ELBOW    = 13
LEFT_WRIST    = 15

# Baselines: For each bone, keep mean of first N_BASE valid len
N_BASE = 3
baseline_bone_lengths = [None for _ in POSE_BODY_CONNECTIONS]
baseline_bone_lists = [[] for _ in POSE_BODY_CONNECTIONS]

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    frame_num = 0
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break
        orig_frame = cv2.flip(image, 1)
        white_bg = np.ones_like(orig_frame) * 255
        image_rgb = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        image_rgb.flags.writeable = True

        if results.pose_landmarks is not None:
            landmarks = results.pose_landmarks.landmark
            points_xyz = [(lm.x, lm.y, lm.z) for lm in landmarks]
            curr_lengths = []
            valid_mask = []
            for i, (a, b) in enumerate(POSE_BODY_CONNECTIONS):
                if (landmarks[a].visibility > MIN_VIS) and (landmarks[b].visibility > MIN_VIS):
                    l = euclidean3d(points_xyz[a], points_xyz[b])
                    curr_lengths.append(l)
                    valid_mask.append(True)
                    # Build baseline for this bone
                    if baseline_bone_lengths[i] is None:
                        if len(baseline_bone_lists[i]) < N_BASE:
                            baseline_bone_lists[i].append(l)
                        if len(baseline_bone_lists[i]) == N_BASE:
                            baseline_bone_lengths[i] = np.mean(baseline_bone_lists[i])
                            print(f"Set baseline for bone {bone_names[i]} as {baseline_bone_lengths[i]:.5f}")
                    
                    if (landmarks[LEFT_SHOULDER].visibility > MIN_VIS and
                        landmarks[LEFT_ELBOW].visibility > MIN_VIS and
                        landmarks[LEFT_WRIST].visibility > MIN_VIS):
                        ls = points_xyz[LEFT_SHOULDER]
                        le = points_xyz[LEFT_ELBOW]
                        lw = points_xyz[LEFT_WRIST]
                        left_elbow_ang = angle3d(ls, le, lw)
                        # Elbow normal: 0 (straight) to 160~ (flexed), negative is past straight, rarely possible
                        elbow_good = 0 < left_elbow_ang < 160
                        elbow_text = f"L-elbow: {left_elbow_ang:.1f} deg"
                        pt2d = (int(landmarks[LEFT_ELBOW].x * frame_width), int(landmarks[LEFT_ELBOW].y * frame_height))
                        color = (0,0,0) if elbow_good else (0,0,255)
                        cv2.putText(white_bg, elbow_text, (pt2d[0]+10, pt2d[1]-20), font, 0.7, color, 2, cv2.LINE_AA)
                        cv2.putText(orig_frame, elbow_text, (pt2d[0]+10, pt2d[1]-20), font, 0.7, color, 2, cv2.LINE_AA)

                        # Optional: Visual alarm
                        if not elbow_good:
                            cv2.putText(white_bg, "UNREALISTIC ELBOW!", (pt2d[0], pt2d[1]-50), font, 1.0, (0,0,255), 3, cv2.LINE_AA)
                            cv2.putText(orig_frame, "UNREALISTIC ELBOW!", (pt2d[0], pt2d[1]-50), font, 1.0, (0,0,255), 3, cv2.LINE_AA)
                else:
                    curr_lengths.append(np.nan)
                    valid_mask.append(False)

            # --- DRAW section ---
            for idx, ((a, b), name) in enumerate(zip(POSE_BODY_CONNECTIONS, bone_names)):
                curr_l = curr_lengths[idx]
                base_l = baseline_bone_lengths[idx]
                # Only plot if both current and baseline available
                if np.isnan(curr_l) or (base_l is None):
                    continue
                print(f"Frame {frame_num}, {name}: Length {curr_l:.5f}")
                change = abs(curr_l - base_l) / base_l
                pt1 = (int(landmarks[a].x * frame_width), int(landmarks[a].y * frame_height))
                pt2 = (int(landmarks[b].x * frame_width), int(landmarks[b].y * frame_height))
                color = (0, 200, 0) if change <= threshold else (0, 0, 255)
                cv2.line(white_bg, pt1, pt2, color, 3)
                cv2.line(orig_frame, pt1, pt2, color, 3)
                midpt = ((pt1[0]+pt2[0])//2, (pt1[1]+pt2[1])//2)
                # txt = f"{change*100:.1f}%"
                # cv2.putText(white_bg, txt, midpt, font, 0.5, (0,0,0), 1, cv2.LINE_AA)
                # cv2.putText(orig_frame, txt, midpt, font, 0.5, (0,0,0), 1, cv2.LINE_AA)

            # ALARM logic: Only for bones with baseline set, not NaN, and current value available
            if any(
                (baseline_bone_lengths[i] is not None) and (not np.isnan(curr_lengths[i])) 
                and (abs(curr_lengths[i] - baseline_bone_lengths[i]) / baseline_bone_lengths[i] > threshold)
                for i in range(len(baseline_bone_lengths))
            ):
                cv2.putText(white_bg, "ALARM: Bone abnormal length!", (50,80), font, 1.0, (0,0,255), 3, cv2.LINE_AA)
                cv2.putText(orig_frame, "ALARM: Bone abnormal length!", (50,80), font, 1.0, (0,0,255), 3, cv2.LINE_AA)

            # 3D coords for each used point
            printed_points = set([idx for pair in POSE_BODY_CONNECTIONS for idx in pair])
            for idx in printed_points:
                lm = landmarks[idx]
                pt2d = (int(lm.x * frame_width), int(lm.y * frame_height))
                text = f"{idx}:({lm.x:.2f},{lm.y:.2f},{lm.z:.2f})"
                cv2.putText(white_bg, text, (pt2d[0]+5, pt2d[1]-5), font, 0.4, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(orig_frame, text, (pt2d[0]+5, pt2d[1]-5), font, 0.4, (0,0,0), 1, cv2.LINE_AA)

        white_bg_writer.write(white_bg)
        orig_writer.write(orig_frame)
        frame_num += 1

cap.release()
white_bg_writer.release()
orig_writer.release()
print("Done.")