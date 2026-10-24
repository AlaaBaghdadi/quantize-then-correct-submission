"""
Visualize keypoints from MediaPipe CSV on images.

This script reads the PISE fashion dataset format CSV and draws keypoints
on top of the images to verify the annotations are correct.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw

# PISE 18-joint skeleton connections
LIMB_SEQ = [
    [1, 2],   # Neck - RShoulder
    [1, 5],   # Neck - LShoulder
    [2, 3],   # RShoulder - RElbow
    [3, 4],   # RElbow - RWrist
    [5, 6],   # LShoulder - LElbow
    [6, 7],   # LElbow - LWrist
    [1, 8],   # Neck - RHip
    [8, 9],   # RHip - RKnee
    [9, 10],  # RKnee - RAnkle
    [1, 11],  # Neck - LHip
    [11, 12], # LHip - LKnee
    [12, 13], # LKnee - LAnkle
    [1, 0],   # Neck - Nose
    [0, 14],  # Nose - LEye
    [14, 16], # LEye - LEar
    [0, 15],  # Nose - REye
    [15, 17], # REye - REar
]

# Colors for each joint (RGB)
JOINT_COLORS = [
    (255, 0, 0),      # 0: nose
    (255, 85, 0),     # 1: neck
    (255, 170, 0),    # 2: RShoulder
    (255, 255, 0),    # 3: RElbow
    (170, 255, 0),    # 4: RWrist
    (85, 255, 0),     # 5: LShoulder
    (0, 255, 0),      # 6: LElbow
    (0, 255, 85),     # 7: LWrist
    (0, 255, 170),    # 8: RHip
    (0, 255, 255),    # 9: RKnee
    (0, 170, 255),    # 10: RAnkle
    (0, 85, 255),     # 11: LHip
    (0, 0, 255),      # 12: LKnee
    (85, 0, 255),     # 13: LAnkle
    (170, 0, 255),    # 14: LEye
    (255, 0, 255),    # 15: REye
    (255, 0, 170),    # 16: LEar
    (255, 0, 85),     # 17: REar
]

JOINT_LABELS = [
    'nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri',
    'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear'
]

MISSING_VALUE = -1


def load_keypoints_from_csv(csv_path: Path, image_name: str = None) -> Tuple[str, np.ndarray]:
    """
    Load keypoints from CSV file.
    
    Args:
        csv_path: Path to annotation CSV file
        image_name: Optional image name to filter (if None, uses first row)
    
    Returns:
        Tuple of (image_name, keypoints_array) where keypoints_array is shape (18, 2) with [y, x] coordinates
    """
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Skip header
    if len(lines) < 2:
        raise ValueError(f"CSV file {csv_path} has no data rows")
    
    # Parse first data row (or find matching image_name if provided)
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
            
            # Convert to numpy array: shape (18, 2) with [y, x] format
            pose_cords = np.array([[y, x] for y, x in zip(y_cords, x_cords)])
            return name, pose_cords
    
    raise ValueError(f"Image {image_name} not found in CSV" if image_name else "No valid data rows found in CSV")


def draw_circle(draw: ImageDraw, center: Tuple[int, int], radius: int, color: Tuple[int, int, int], fill: bool = True):
    """Draw a filled circle using PIL ImageDraw."""
    x, y = center
    bbox = [x - radius, y - radius, x + radius, y + radius]
    if fill:
        draw.ellipse(bbox, fill=color, outline=color)
    else:
        draw.ellipse(bbox, outline=color, width=2)


def visualize_keypoints_on_image(
    image_path: Path,
    keypoints: np.ndarray,
    output_path: Path,
    joint_radius: int = 4,
    line_width: int = 2,
) -> None:
    """
    Draw keypoints on an image and save the result.
    
    Args:
        image_path: Path to input image
        keypoints: Array of shape (18, 2) with [y, x] coordinates
        output_path: Where to save the visualization
        joint_radius: Radius of joint circles
        line_width: Width of skeleton lines
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # Draw skeleton connections (limbs)
    for f, t in LIMB_SEQ:
        from_y, from_x = keypoints[f, 0], keypoints[f, 1]
        to_y, to_x = keypoints[t, 0], keypoints[t, 1]
        
        # Skip if either endpoint is missing
        if (from_y == MISSING_VALUE or from_x == MISSING_VALUE or 
            to_y == MISSING_VALUE or to_x == MISSING_VALUE):
            continue
        
        # Draw line between joints (use average color of the two joints)
        color1 = JOINT_COLORS[f]
        color2 = JOINT_COLORS[t]
        avg_color = tuple((c1 + c2) // 2 for c1, c2 in zip(color1, color2))
        draw.line([(from_x, from_y), (to_x, to_y)], fill=avg_color, width=line_width)
    
    # Draw joints (circles)
    for i, (y, x) in enumerate(keypoints):
        if y == MISSING_VALUE or x == MISSING_VALUE:
            continue
        
        # Draw filled circle for joint
        draw_circle(draw, (int(x), int(y)), joint_radius, JOINT_COLORS[i], fill=True)
        # Draw outline for better visibility
        draw_circle(draw, (int(x), int(y)), joint_radius, (255, 255, 255), fill=False)
    
    # Save result
    img.save(output_path)
    print(f"Saved visualization to: {output_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Visualize keypoints from MediaPipe CSV on images"
    )
    ap.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to annotation CSV file (format: name:keypoints_y:keypoints_x)",
    )
    ap.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for visualization (default: <image_name>_keypoints.png)",
    )
    ap.add_argument(
        "--image_name",
        type=str,
        default=None,
        help="Image name in CSV to use (if None, uses first row)",
    )
    ap.add_argument(
        "--joint_radius",
        type=int,
        default=4,
        help="Radius of joint circles in pixels",
    )
    ap.add_argument(
        "--line_width",
        type=int,
        default=2,
        help="Width of skeleton lines",
    )
    args = ap.parse_args()
    
    csv_path = Path(args.csv)
    image_path = Path(args.image)
    
    # Default output path
    if args.output is None:
        output_path = image_path.parent / f"{image_path.stem}_keypoints.png"
    else:
        output_path = Path(args.output)
    
    # Verify files exist
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    print(f"Loading keypoints from: {csv_path}")
    print(f"Loading image from: {image_path}")
    
    # Load keypoints
    image_name, keypoints = load_keypoints_from_csv(csv_path, args.image_name)
    
    # Count valid keypoints
    valid_count = np.sum((keypoints[:, 0] != MISSING_VALUE) & 
                         (keypoints[:, 1] != MISSING_VALUE))
    print(f"Image name in CSV: {image_name}")
    print(f"Valid keypoints: {valid_count}/18")
    
    # Print keypoint details
    print("\nKeypoint details:")
    print("-" * 60)
    for i, (y, x) in enumerate(keypoints):
        if y != MISSING_VALUE and x != MISSING_VALUE:
            print(f"{i:2d} {JOINT_LABELS[i]:8s}: ({int(x):4d}, {int(y):4d})")
        else:
            print(f"{i:2d} {JOINT_LABELS[i]:8s}: (missing)")
    print("-" * 60)
    
    # Visualize
    visualize_keypoints_on_image(
        image_path,
        keypoints,
        output_path,
        joint_radius=args.joint_radius,
        line_width=args.line_width,
    )
    
    print(f"\nVisualization complete! Saved to: {output_path}")


if __name__ == "__main__":
    main()
