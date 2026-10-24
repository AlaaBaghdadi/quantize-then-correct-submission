from __future__ import annotations

from typing import Dict, List, Optional, Tuple


# OpenPose COCO-18 names (correct order matching OpenPose documentation)
OPENPOSE18_KEYPOINT_NAMES: List[str] = [
    "Nose",           # 0
    "Neck",           # 1
    "Right Shoulder", # 2
    "Right Elbow",    # 3
    "Right Wrist",    # 4
    "Left Shoulder",  # 5
    "Left Elbow",     # 6
    "Left Wrist",     # 7
    "Right Hip",      # 8
    "Right Knee",     # 9
    "Right Ankle",    # 10
    "Left Hip",       # 11
    "Left Knee",      # 12
    "Left Ankle",     # 13
    "Right Eye",      # 14
    "Left Eye",       # 15
    "Right Ear",      # 16
    "Left Ear",       # 17
]


# OpenPose COCO-18 skeleton connections (pairs of indices)
OPENPOSE18_CONNECTIONS: List[Tuple[int, int]] = [
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


# OpenPose index -> MediaPipe Pose landmark index (None means derived)
# Mirrors the mapping used in `pose-corrector/mediapipe_to_pise_csv.py`.
OPENPOSE18_FROM_MEDIAPIPE: Dict[int, Optional[int]] = {
    0: 0,    # Nose
    1: None, # Neck (midpoint of shoulders)
    2: 12,   # Right Shoulder
    3: 14,   # Right Elbow
    4: 16,   # Right Wrist
    5: 11,   # Left Shoulder
    6: 13,   # Left Elbow
    7: 15,   # Left Wrist
    8: 24,   # Right Hip
    9: 26,   # Right Knee
    10: 28,  # Right Ankle
    11: 23,  # Left Hip
    12: 25,  # Left Knee
    13: 27,  # Left Ankle
    14: 5,   # Right Eye
    15: 2,   # Left Eye
    16: 8,   # Right Ear
    17: 7,   # Left Ear
}


# PISE-18 ordering (reorders OpenPose COCO-18 by swapping eyes/ears positions)
# Matches `pose-corrector/mediapipe_to_pise_csv.py`.
PISE18_ORDER: Tuple[int, ...] = (
    0,   # Nose
    1,   # Neck
    2,   # RShoulder
    3,   # RElbow
    4,   # RWrist
    5,   # LShoulder
    6,   # LElbow
    7,   # LWrist
    8,   # RHip
    10,  # RAnkle
    9,   # RKnee
    11,  # LHip
    13,  # LAnkle
    12,  # LKnee
    15,  # LEye
    14,  # REye
    17,  # LEar
    16,  # REar
)

