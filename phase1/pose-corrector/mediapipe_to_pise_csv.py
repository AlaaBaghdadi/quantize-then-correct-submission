import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import mediapipe as mp

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

# PISE-18 order mapping (from make_pise_fashion_sample.py)
# This reorders OpenPose COCO-18 to PISE-18 format (swaps eyes/ears)
PISE18_ORDER = (
    0,  # Nose
    1,  # Neck
    2,  # RShoulder
    3,  # RElbow
    4,  # RWrist
    5,  # LShoulder
    6,  # LElbow
    7,  # LWrist
    8,  # RHip
    9,  # RKnee
    10,  # RAnkle
    11,  # LHip
    12,  # LKnee
    13,  # LAnkle
    15,  # LEye  (swap: OpenPose 15 -> PISE 14)
    14,  # REye  (swap: OpenPose 14 -> PISE 15)
    17,  # LEar  (swap: OpenPose 17 -> PISE 16)
    16,  # REar  (swap: OpenPose 16 -> PISE 17)
)

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


def convert_mediapipe_to_openpose18(landmarks, img_width: int, img_height: int) -> List[Optional[Tuple[float, float, float]]]:
    """
    Convert MediaPipe landmarks to OpenPose COCO-18 format.
    Returns list of (x, y, confidence) tuples in pixel coordinates.
    """
    openpose_keypoints = [None] * 18
    
    # Map MediaPipe landmarks to OpenPose format
    for openpose_idx, mapping in OPENPOSE_MAPPING.items():
        if mapping is None:
            # Neck point (index 1) - calculate from shoulders
            left_shoulder = landmarks[11]  # MediaPipe left shoulder
            right_shoulder = landmarks[12]  # MediaPipe right shoulder
            neck_point = calculate_neck_point(left_shoulder, right_shoulder)
            if neck_point:
                openpose_keypoints[1] = (
                    neck_point.x * img_width,
                    neck_point.y * img_height,
                    neck_point.visibility
                )
        else:
            mediapipe_idx, _ = mapping
            landmark = landmarks[mediapipe_idx]
            openpose_keypoints[openpose_idx] = (
                landmark.x * img_width,
                landmark.y * img_height,
                landmark.visibility
            )
    
    return openpose_keypoints


def convert_openpose18_to_pise18(openpose18: List[Optional[Tuple[float, float, float]]]) -> List[Optional[Tuple[float, float, float]]]:
    """
    Reorder OpenPose COCO-18 keypoints to PISE-18 order.
    Swaps eyes and ears positions.
    """
    pise18 = [openpose18[i] if openpose18[i] is not None else None for i in PISE18_ORDER]
    return pise18


def extract_xy_lists_from_mediapipe(
    openpose18: List[Optional[Tuple[float, float, float]]], 
    score_thresh: float = 0.15
) -> Tuple[List[int], List[int]]:
    """
    Convert OpenPose COCO-18 keypoints to (y_list, x_list) in PISE-18 order.
    Missing keypoints are encoded as -1.
    """
    # Reorder to PISE-18 format
    pise18 = convert_openpose18_to_pise18(openpose18)
    
    xs: List[int] = []
    ys: List[int] = []
    
    for kp in pise18:
        if kp is None:
            xs.append(-1)
            ys.append(-1)
            continue
        x, y, s = kp
        if not isinstance(s, (int, float)) or s < score_thresh:
            xs.append(-1)
            ys.append(-1)
            continue
        xs.append(int(round(float(x))))
        ys.append(int(round(float(y))))
    
    return ys, xs


def process_image_to_csv(
    image_path: str,
    output_csv_path: str = None,
    score_thresh: float = 0.15,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5
) -> Tuple[List[int], List[int]]:
    """
    Process an image with MediaPipe, convert to OpenPose format, and return PISE-18 format keypoints.
    
    Args:
        image_path: Path to input image
        output_csv_path: Optional path to write CSV file
        score_thresh: Confidence threshold for keypoints
        min_detection_confidence: MediaPipe detection confidence
        min_tracking_confidence: MediaPipe tracking confidence
    
    Returns:
        Tuple of (y_list, x_list) in PISE-18 order
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    img_height, img_width = img.shape[:2]
    image_name = Path(image_path).name
    
    print(f"Processing image: {image_path} ({img_width}x{img_height})")
    
    # Extract MediaPipe keypoints
    with mp_pose.Pose(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    ) as pose:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        
        if results.pose_landmarks is None:
            raise ValueError(f"No pose detected in image: {image_path}")
        
        landmarks = results.pose_landmarks.landmark
        
        # Convert to OpenPose COCO-18 format
        openpose18 = convert_mediapipe_to_openpose18(landmarks, img_width, img_height)
        
        # Count valid keypoints
        valid_count = sum(1 for kp in openpose18 if kp is not None and kp[2] >= score_thresh)
        print(f"Detected {valid_count}/18 valid keypoints")
        
        # Convert to PISE-18 format (y_list, x_list)
        ys, xs = extract_xy_lists_from_mediapipe(openpose18, score_thresh)
        
        # Write CSV if output path provided
        if output_csv_path:
            ann_csv = Path(output_csv_path)
            lines = [
                "name:keypoints_y:keypoints_x",
                f"{image_name}:{json.dumps(ys)}:{json.dumps(xs)}",
                "",
            ]
            ann_csv.write_text("\n".join(lines), encoding="utf-8")
            print(f"Wrote CSV to: {output_csv_path}")
        
        return ys, xs


def main():
    ap = argparse.ArgumentParser(
        description="Extract MediaPipe keypoints, convert to OpenPose format, and save as PISE CSV format."
    )
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--output_csv", default=None, help="Output CSV path (default: <image_name>_keypoints.csv)")
    ap.add_argument("--score_thresh", type=float, default=0.15, help="Keypoints below this confidence are set to -1")
    ap.add_argument(
        "--min_detection_confidence", 
        type=float, 
        default=0.5, 
        help="MediaPipe minimum detection confidence"
    )
    ap.add_argument(
        "--min_tracking_confidence", 
        type=float, 
        default=0.5, 
        help="MediaPipe minimum tracking confidence"
    )
    
    args = ap.parse_args()
    
    # Default output CSV name
    if args.output_csv is None:
        image_path = Path(args.image)
        args.output_csv = str(image_path.parent / f"{image_path.stem}_keypoints.csv")
    
    try:
        ys, xs = process_image_to_csv(
            args.image,
            args.output_csv,
            args.score_thresh,
            args.min_detection_confidence,
            args.min_tracking_confidence
        )
        
        print("\n" + "="*60)
        print("KEYPOINTS EXTRACTED (PISE-18 order)")
        print("="*60)
        print(f"Y coordinates: {ys}")
        print(f"X coordinates: {xs}")
        print(f"Valid keypoints: {sum(1 for y in ys if y != -1)}/18")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
