import argparse
import cv2
import mediapipe as mp
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os

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

# OpenPose keypoint names for reference
OPENPOSE_KEYPOINT_NAMES = [
    "Nose", "Neck", "Right Shoulder", "Right Elbow", "Right Wrist",
    "Left Shoulder", "Left Elbow", "Left Wrist", "Right Hip", "Right Knee",
    "Right Ankle", "Left Hip", "Left Knee", "Left Ankle",
    "Right Eye", "Left Eye", "Right Ear", "Left Ear"
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

def convert_to_openpose(landmarks, img_width, img_height):
    """Convert MediaPipe landmarks to OpenPose COCO format (18 keypoints).
    Returns list of (x, y, confidence) tuples in pixel coordinates."""
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

def load_openpose_keypoints_from_csv(csv_path, image_name=None):
    """Load OpenPose keypoints from CSV file (PISE format).
    CSV format: name:keypoints_y:keypoints_x
    Keypoints are in PISE-18 order, need to convert back to OpenPose COCO-18 order.
    Returns list of (x, y, confidence) tuples for 18 keypoints in OpenPose COCO-18 order."""
    
    # Reverse mapping: PISE-18 index -> OpenPose COCO-18 index
    # PISE swaps eyes/ears compared to OpenPose COCO-18
    pise18_to_openpose18 = (
        0,   # 0: Nose -> OpenPose 0
        1,   # 1: Neck -> OpenPose 1
        2,   # 2: RShoulder -> OpenPose 2
        3,   # 3: RElbow -> OpenPose 3
        4,   # 4: RWrist -> OpenPose 4
        5,   # 5: LShoulder -> OpenPose 5
        6,   # 6: LElbow -> OpenPose 6
        7,   # 7: LWrist -> OpenPose 7
        8,   # 8: RHip -> OpenPose 8
        9,   # 9: RKnee -> OpenPose 9
        10,  # 10: RAnkle -> OpenPose 10
        11,  # 11: LHip -> OpenPose 11
        12,  # 12: LKnee -> OpenPose 12
        13,  # 13: LAnkle -> OpenPose 13
        15,  # 14: LEye (PISE) -> OpenPose 15 (LEye)
        14,  # 15: REye (PISE) -> OpenPose 14 (REye)
        17,  # 16: LEar (PISE) -> OpenPose 17 (LEar)
        16,  # 17: REar (PISE) -> OpenPose 16 (REar)
    )
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Skip header
    if len(lines) < 2:
        raise ValueError(f"CSV file {csv_path} has no data rows")
    
    # Parse data rows
    for line in lines[1:]:
        if not line.strip():
            continue
        
        parts = line.strip().split(':')
        if len(parts) != 3:
            continue
        
        name = parts[0]
        y_str = parts[1]
        x_str = parts[2]
        
        if image_name is None or name == image_name:
            y_cords = json.loads(y_str)
            x_cords = json.loads(x_str)
            
            # Convert from PISE-18 order to OpenPose COCO-18 order
            # Initialize with None
            openpose_keypoints = [None] * 18
            
            for pise_idx, (y, x) in enumerate(zip(y_cords, x_cords)):
                if y != -1 and x != -1:
                    # Map PISE index to OpenPose index
                    openpose_idx = pise18_to_openpose18[pise_idx]
                    # Use a default confidence of 1.0 (CSV doesn't store confidence)
                    # You might want to adjust this based on your needs
                    openpose_keypoints[openpose_idx] = (float(x), float(y), 1.0)
            
            return openpose_keypoints
    
    raise ValueError(f"Image {image_name} not found in CSV" if image_name else "No valid data rows found in CSV")

def load_openpose_keypoints(json_path):
    """Load OpenPose keypoints from JSON file.
    Returns list of (x, y, confidence) tuples for 18 keypoints."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if 'people' not in data or len(data['people']) == 0:
        return None
    
    # Get first person's pose keypoints
    keypoints_flat = data['people'][0]['pose_keypoints_2d']
    
    # Convert flat array to list of (x, y, confidence) tuples
    keypoints = []
    for i in range(0, len(keypoints_flat), 3):
        x = keypoints_flat[i]
        y = keypoints_flat[i + 1]
        conf = keypoints_flat[i + 2]
        keypoints.append((x, y, conf))
    
    return keypoints

def calculate_distances(mediapipe_kp, openpose_kp):
    """Calculate Euclidean distances between corresponding keypoints.
    Returns list of distances and statistics."""
    distances = []
    valid_pairs = []
    
    for i in range(18):
        if mediapipe_kp[i] is not None and openpose_kp[i] is not None:
            mp_x, mp_y, mp_conf = mediapipe_kp[i]
            op_x, op_y, op_conf = openpose_kp[i]
            
            # Only compare if both have reasonable confidence
            if mp_conf > MIN_VIS and op_conf > MIN_VIS:
                dist = np.sqrt((mp_x - op_x)**2 + (mp_y - op_y)**2)
                distances.append(dist)
                valid_pairs.append((i, dist, mp_conf, op_conf))
    
    stats = {
        'mean': np.mean(distances) if distances else 0,
        'median': np.median(distances) if distances else 0,
        'std': np.std(distances) if distances else 0,
        'min': np.min(distances) if distances else 0,
        'max': np.max(distances) if distances else 0,
        'count': len(distances)
    }
    
    return distances, valid_pairs, stats

def visualize_comparison(img, mediapipe_kp, openpose_kp, distances_dict, output_path='pose_comparison.png'):
    """Visualize both MediaPipe and OpenPose keypoints side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Left plot: MediaPipe
    axes[0].imshow(img_rgb)
    axes[0].set_title('MediaPipe (Converted to OpenPose Format)', fontsize=14, fontweight='bold')
    
    # Draw MediaPipe skeleton
    for (a, b) in OPENPOSE_CONNECTIONS:
        if (mediapipe_kp[a] is not None and mediapipe_kp[b] is not None):
            mp_a_x, mp_a_y, mp_a_conf = mediapipe_kp[a]
            mp_b_x, mp_b_y, mp_b_conf = mediapipe_kp[b]
            if mp_a_conf > MIN_VIS and mp_b_conf > MIN_VIS:
                axes[0].plot([mp_a_x, mp_b_x], [mp_a_y, mp_b_y], 'g-', linewidth=2, alpha=0.7)
    
    # Draw MediaPipe keypoints
    for i, kp in enumerate(mediapipe_kp):
        if kp is not None:
            x, y, conf = kp
            if conf > MIN_VIS:
                axes[0].plot(x, y, 'bo', markersize=8, alpha=0.8)
                axes[0].text(x + 5, y + 5, f'{i}', fontsize=8, color='blue')
    
    axes[0].axis('off')
    
    # Right plot: OpenPose
    axes[1].imshow(img_rgb)
    axes[1].set_title('OpenPose (Original)', fontsize=14, fontweight='bold')
    
    # Draw OpenPose skeleton
    for (a, b) in OPENPOSE_CONNECTIONS:
        if (openpose_kp[a] is not None and openpose_kp[b] is not None):
            op_a_x, op_a_y, op_a_conf = openpose_kp[a]
            op_b_x, op_b_y, op_b_conf = openpose_kp[b]
            if op_a_conf > MIN_VIS and op_b_conf > MIN_VIS:
                axes[1].plot([op_a_x, op_b_x], [op_a_y, op_b_y], 'r-', linewidth=2, alpha=0.7)
    
    # Draw OpenPose keypoints
    for i, kp in enumerate(openpose_kp):
        if kp is not None:
            x, y, conf = kp
            if conf > MIN_VIS:
                axes[1].plot(x, y, 'ro', markersize=8, alpha=0.8)
                axes[1].text(x + 5, y + 5, f'{i}', fontsize=8, color='red')
    
    axes[1].axis('off')
    
    # Add statistics text
    stats_text = f"""Comparison Statistics:
    Mean Distance: {distances_dict['mean']:.2f} pixels
    Median Distance: {distances_dict['median']:.2f} pixels
    Std Deviation: {distances_dict['std']:.2f} pixels
    Min Distance: {distances_dict['min']:.2f} pixels
    Max Distance: {distances_dict['max']:.2f} pixels
    Valid Keypoints: {distances_dict['count']}/18"""
    
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Comparison visualization saved to {output_path}")
    plt.close()

def create_overlay_visualization(img, mediapipe_kp, openpose_kp, output_path='pose_overlay.png'):
    """Create overlay visualization showing both methods on same image."""
    img_rgb = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(img_rgb)
    ax.set_title('MediaPipe (Blue) vs OpenPose (Red) Overlay', fontsize=14, fontweight='bold')
    
    # Draw MediaPipe skeleton (blue)
    for (a, b) in OPENPOSE_CONNECTIONS:
        if (mediapipe_kp[a] is not None and mediapipe_kp[b] is not None):
            mp_a_x, mp_a_y, mp_a_conf = mediapipe_kp[a]
            mp_b_x, mp_b_y, mp_b_conf = mediapipe_kp[b]
            if mp_a_conf > MIN_VIS and mp_b_conf > MIN_VIS:
                ax.plot([mp_a_x, mp_b_x], [mp_a_y, mp_b_y], 'b-', linewidth=2, alpha=0.6, label='MediaPipe' if a == 1 else '')
    
    # Draw OpenPose skeleton (red)
    for (a, b) in OPENPOSE_CONNECTIONS:
        if (openpose_kp[a] is not None and openpose_kp[b] is not None):
            op_a_x, op_a_y, op_a_conf = openpose_kp[a]
            op_b_x, op_b_y, op_b_conf = openpose_kp[b]
            if op_a_conf > MIN_VIS and op_b_conf > MIN_VIS:
                ax.plot([op_a_x, op_b_x], [op_a_y, op_b_y], 'r-', linewidth=2, alpha=0.6, label='OpenPose' if a == 1 else '')
    
    # Draw MediaPipe keypoints (blue)
    for i, kp in enumerate(mediapipe_kp):
        if kp is not None:
            x, y, conf = kp
            if conf > MIN_VIS:
                ax.plot(x, y, 'bo', markersize=6, alpha=0.8)
    
    # Draw OpenPose keypoints (red)
    for i, kp in enumerate(openpose_kp):
        if kp is not None:
            x, y, conf = kp
            if conf > MIN_VIS:
                ax.plot(x, y, 'ro', markersize=6, alpha=0.8)
    
    # Draw lines connecting corresponding keypoints
    for i in range(18):
        if (mediapipe_kp[i] is not None and openpose_kp[i] is not None):
            mp_x, mp_y, mp_conf = mediapipe_kp[i]
            op_x, op_y, op_conf = openpose_kp[i]
            if mp_conf > MIN_VIS and op_conf > MIN_VIS:
                ax.plot([mp_x, op_x], [mp_y, op_y], 'y--', linewidth=1, alpha=0.4)
    
    ax.axis('off')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Overlay visualization saved to {output_path}")
    plt.close()

def print_detailed_comparison(mediapipe_kp, openpose_kp, valid_pairs):
    """Print detailed comparison of each keypoint."""
    print("\n" + "="*80)
    print("DETAILED KEYPOINT COMPARISON")
    print("="*80)
    print(f"{'ID':<4} {'Keypoint Name':<20} {'MediaPipe (x,y)':<25} {'OpenPose (x,y)':<25} {'Distance':<10} {'MP Conf':<10} {'OP Conf':<10}")
    print("-"*80)
    
    for i in range(18):
        mp_str = "N/A"
        op_str = "N/A"
        dist_str = "N/A"
        mp_conf_str = "N/A"
        op_conf_str = "N/A"
        
        if mediapipe_kp[i] is not None:
            mp_x, mp_y, mp_conf = mediapipe_kp[i]
            mp_str = f"({mp_x:.1f}, {mp_y:.1f})"
            mp_conf_str = f"{mp_conf:.3f}"
        
        if openpose_kp[i] is not None:
            op_x, op_y, op_conf = openpose_kp[i]
            op_str = f"({op_x:.1f}, {op_y:.1f})"
            op_conf_str = f"{op_conf:.3f}"
        
        # Find distance if available
        for pair_id, dist, mp_c, op_c in valid_pairs:
            if pair_id == i:
                dist_str = f"{dist:.2f}"
                break
        
        print(f"{i:<4} {OPENPOSE_KEYPOINT_NAMES[i]:<20} {mp_str:<25} {op_str:<25} {dist_str:<10} {mp_conf_str:<10} {op_conf_str:<10}")

def main():
    ap = argparse.ArgumentParser(description="Compare MediaPipe and OpenPose keypoints")
    ap.add_argument("--image", type=str, default='test_1.png', help="Path to input image")
    ap.add_argument("--openpose_csv", type=str, default=None, help="Path to OpenPose CSV file (format: name:keypoints_y:keypoints_x)")
    ap.add_argument("--openpose_json", type=str, default=None, help="Path to OpenPose JSON file (alternative to CSV)")
    ap.add_argument("--image_name", type=str, default=None, help="Image name in CSV (if None, uses first row)")
    ap.add_argument("--min_detection_confidence", type=float, default=0.5, help="MediaPipe detection confidence")
    ap.add_argument("--min_tracking_confidence", type=float, default=0.5, help="MediaPipe tracking confidence")
    
    args = ap.parse_args()
    
    # Load image
    img_path = args.image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image {img_path}")
        return
    
    img_height, img_width = img.shape[:2]
    print(f"Loaded image: {img_path} ({img_width}x{img_height})")
    
    # Extract MediaPipe keypoints
    print("\nExtracting MediaPipe keypoints...")
    with mp_pose.Pose(
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence
    ) as pose:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        
        if results.pose_landmarks is None:
            print("Error: No pose detected in image")
            return
        
        landmarks = results.pose_landmarks.landmark
        mediapipe_kp = convert_to_openpose(landmarks, img_width, img_height)
        print(f"MediaPipe: Detected {sum(1 for kp in mediapipe_kp if kp is not None)}/18 keypoints")
    
    # Load OpenPose keypoints
    print("\nLoading OpenPose keypoints...")
    if args.openpose_csv:
        openpose_kp = load_openpose_keypoints_from_csv(args.openpose_csv, args.image_name)
        print(f"OpenPose: Loaded from CSV {args.openpose_csv}")
    elif args.openpose_json:
        openpose_kp = load_openpose_keypoints(args.openpose_json)
        if openpose_kp is None:
            print("Error: Could not load OpenPose keypoints from JSON")
            return
        print(f"OpenPose: Loaded from JSON {args.openpose_json}")
    else:
        # Default: try CSV first, then JSON
        # Try common CSV names
        csv_paths = ['test_1_mediapipe_keypoints.csv', 'test_1_keypoints.csv']
        json_path = 'test_1_key.json'  # Default JSON name
        
        csv_path = None
        for path in csv_paths:
            if os.path.exists(path):
                csv_path = path
                break
        if csv_path:
            openpose_kp = load_openpose_keypoints_from_csv(csv_path, args.image_name)
            print(f"OpenPose: Loaded from CSV {csv_path} (default)")
        elif os.path.exists(json_path):
            openpose_kp = load_openpose_keypoints(json_path)
            if openpose_kp is None:
                print("Error: Could not load OpenPose keypoints")
                return
            print(f"OpenPose: Loaded from JSON {json_path} (default)")
        else:
            print("Error: No OpenPose file found. Please specify --openpose_csv or --openpose_json")
            return
    
    print(f"OpenPose: Loaded {sum(1 for kp in openpose_kp if kp is not None)}/18 keypoints")
    
    # Calculate distances and statistics
    print("\nCalculating distances...")
    distances, valid_pairs, stats = calculate_distances(mediapipe_kp, openpose_kp)
    
    # Print statistics
    print("\n" + "="*80)
    print("COMPARISON STATISTICS")
    print("="*80)
    print(f"Mean Distance: {stats['mean']:.2f} pixels")
    print(f"Median Distance: {stats['median']:.2f} pixels")
    print(f"Standard Deviation: {stats['std']:.2f} pixels")
    print(f"Min Distance: {stats['min']:.2f} pixels")
    print(f"Max Distance: {stats['max']:.2f} pixels")
    print(f"Valid Keypoint Pairs: {stats['count']}/18")
    
    # Print detailed comparison
    print_detailed_comparison(mediapipe_kp, openpose_kp, valid_pairs)
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualize_comparison(img, mediapipe_kp, openpose_kp, stats, 'pose_comparison.png')
    create_overlay_visualization(img, mediapipe_kp, openpose_kp, 'pose_overlay.png')
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print("Output files:")
    print("  - pose_comparison.png: Side-by-side comparison")
    print("  - pose_overlay.png: Overlay visualization")

if __name__ == '__main__':
    main()
