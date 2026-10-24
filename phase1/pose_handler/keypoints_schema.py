from __future__ import annotations

from typing import Dict, List, Literal, TypedDict


FORMAT_MEDIAPIPE_POSE_V1 = "pose_handler/mediapipe_pose_v1"
FORMAT_MEDIAPIPE_POSE_SCALED_V1 = "pose_handler/mediapipe_pose_scaled_v1"
FORMAT_OPENPOSE18_V1 = "pose_handler/openpose18_v1"
FORMAT_UNIFIED_KEYPOINTS_V1 = "pose_handler/unified_keypoints_v1"  # Unified format for all keypoint files


# MediaPipe Pose landmark indices we care about (for OpenPose-18 + scale/angles)
# Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
MEDIAPIPE_POSE_INDEX_BY_NAME: Dict[str, int] = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

MEDIAPIPE_POSE_NAMES: List[str] = list(MEDIAPIPE_POSE_INDEX_BY_NAME.keys())


class ImageSize(TypedDict):
    width: int
    height: int


class MpPoint(TypedDict):
    x: float
    y: float
    z: float


class MpLandmark(TypedDict):
    visibility: float
    normalized: MpPoint
    world_3d: MpPoint
    pixel: Dict[str, float]


class MediaPipePoseEntry(TypedDict):
    image_name: str
    image_path: str
    image_size: ImageSize
    detected: bool
    landmarks: Dict[str, MpLandmark]


class MediaPipePoseFile(TypedDict):
    format: Literal["pose_handler/mediapipe_pose_v1"]
    entries: List[MediaPipePoseEntry]


class MpScaledPoint(TypedDict):
    x: float
    y: float
    z_world: float
    z_scaled: float
    visibility: float


class MediaPipePoseScaledEntry(TypedDict):
    image_name: str
    image_path: str
    image_size: ImageSize
    detected: bool
    scale_factor: float
    keypoints: Dict[str, MpScaledPoint]


class MediaPipePoseScaledFile(TypedDict):
    format: Literal["pose_handler/mediapipe_pose_scaled_v1"]
    entries: List[MediaPipePoseScaledEntry]


class OpenPoseKp(TypedDict):
    x: float
    y: float
    confidence: float


class OpenPose18Entry(TypedDict):
    image_name: str
    image_path: str
    image_size: ImageSize
    keypoints: List[OpenPoseKp | None]  # length=18


class OpenPose18File(TypedDict):
    format: Literal["pose_handler/openpose18_v1"]
    keypoint_names: List[str]  # length=18
    connections: List[List[int]]  # pairs
    entries: List[OpenPose18Entry]

