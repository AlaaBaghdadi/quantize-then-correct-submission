# pose_handler

`pose_handler` is a small set of scripts to handle pose tasks in a clear way (one file per task).

## Unified JSON Format

All keypoint JSON files can be read and processed in a **unified format** (`pose_handler/unified_keypoints_v1`):

```json
{
  "format": "pose_handler/unified_keypoints_v1",
  "image_name": "img1.jpeg",
  "image_path": "path/to/image.jpg",
  "image_size": {"width": 1920, "height": 1080},
  "keypoints": {
    "nose": {"x": 950.0, "y": 127.0, "confidence": 0.95},
    "left_shoulder": {"x": 1042.0, "y": 239.0, "confidence": 0.77},
    ...
  }
}
```

### How it works:
1. **Each script outputs two files**: 
   - Legacy format (e.g., `mediapipe_pose.json`, `mediapipe_pose_scaled.json`, `openpose18.json`)
   - Unified format (e.g., `img1_unified.json`, `openpose_named.json`, `averaged_keypoints.json`)

2. **All scripts can read both formats**:
   - Processing scripts (`scale_mediapipe_z.py`, `convert_mediapipe_to_openpose18.py`) accept both legacy and unified formats as input
   - Visualization scripts automatically detect and convert any format

3. **The unified format**:
   - Uses **pixel coordinates** (x, y) for easy visualization
   - Uses **confidence** (0-1) for all keypoints
   - Optional **z_scaled** for 3D visualization (from scaled MediaPipe data)
   - Consistent keypoint names: `nose`, `left_shoulder`, `right_elbow`, etc.

## Output folder

All scripts write into `pose_handler_out/` by default (configurable with `--out_dir`).

Outputs:
- `pose_handler_out/mediapipe_pose.json` (legacy format, also writes `*_unified.json`)
- `pose_handler_out/mediapipe_pose_scaled.json` (legacy format, also writes `*_unified.json`)
- `pose_handler_out/openpose18.json` (legacy format, also writes `*_unified.json`)
- `pose_handler_out/openpose_named.json` (unified format)
- `pose_handler_out/averaged_keypoints.json` (unified format)
- `pose_handler_out/pise18_keypoints.csv`
- `pose_handler_out/angles.json`
- `pose_handler_out/vis_2d/` (2D overlays)
- `pose_handler_out/vis_3d/` (3D plots)

## 1) Extract MediaPipe pose from a folder

File: `pose_handler/extract_mediapipe_from_folder.py`  
What it does: Runs MediaPipe Pose on every image in a folder and saves a clean JSON with normalized + world_3d + pixel coordinates.

Command:

```bash
python -m pose_handler.extract_mediapipe_from_folder --input_dir path/to/images --out_dir pose_handler_out
```

## 2) Scale Z (world meters → normalized units)

File: `pose_handler/scale_mediapipe_z.py`  
What it does: Reads `mediapipe_pose.json`, estimates a scale factor, and writes `mediapipe_pose_scaled.json` with `z_scaled`.

Command:

```bash
python -m pose_handler.scale_mediapipe_z --in_json pose_handler_out/mediapipe_pose.json --out_dir pose_handler_out
```

## 3) Convert MediaPipe → OpenPose COCO-18

File: `pose_handler/convert_mediapipe_to_openpose18.py`  
What it does: Converts the extracted MediaPipe landmarks into OpenPose COCO-18 keypoints (pixel coords). Neck is computed as the midpoint of shoulders.

Command:

```bash
python -m pose_handler.convert_mediapipe_to_openpose18 --in_json pose_handler_out/mediapipe_pose.json --out_dir pose_handler_out
```

## 4) Export PISE CSV (PISE-18)

File: `pose_handler/export_pise18_csv.py`  
What it does: Converts `openpose18.json` to the PISE CSV format: `name:keypoints_y:keypoints_x` with `-1` for missing/low-confidence joints.

Command:

```bash
python -m pose_handler.export_pise18_csv --in_json pose_handler_out/openpose18.json --out_dir pose_handler_out --score_thresh 0.15
```

## 5) Visualize 2D overlay

File: `pose_handler/visualize_2d_overlay.py`  
What it does: Draws the pose skeleton on each image and saves PNGs in `pose_handler_out/vis_2d/`.

Command (recommended: OpenPose18 input):

```bash
python -m pose_handler.visualize_2d_overlay --input_dir path/to/images --keypoints_json pose_handler_out/openpose18.json --out_dir pose_handler_out
```

Command (also supported: MediaPipe input):

```bash
python -m pose_handler.visualize_2d_overlay --input_dir path/to/images --keypoints_json pose_handler_out/mediapipe_pose.json --out_dir pose_handler_out
```

## 6) Visualize 3D keypoints

File: `pose_handler/visualize_3d_keypoints.py`  
What it does: Saves a 3D matplotlib plot per image using `z_scaled`, in `pose_handler_out/vis_3d/`.

Command:

```bash
python -m pose_handler.visualize_3d_keypoints --in_json pose_handler_out/mediapipe_pose_scaled.json --out_dir pose_handler_out
```

## 7) Compute shoulder + elbow angles

File: `pose_handler/calc_angles.py`  
What it does: Computes shoulder (horizontal/vertical) + elbow angles from scaled keypoints and writes `pose_handler_out/angles.json`.

Command:

```bash
python -m pose_handler.calc_angles --in_json pose_handler_out/mediapipe_pose_scaled.json --out_dir pose_handler_out
```

## 8) Convert OpenPose JSON → named keypoints JSON (MediaPipe-style names)

File: `pose_handler/openpose_json_to_named.py`  
What it does: Reads an OpenPose `pose_keypoints_2d` JSON and writes a clean JSON with keys like `left_shoulder`, `right_elbow`, etc.

Command:

```bash
python -m pose_handler.openpose_json_to_named --in_json openpose.json --out_dir pose_handler_out
```

## 9) Average MediaPipe and OpenPose keypoints

File: `pose_handler/average_mediapipe_openpose.py`  
What it does: Takes `mediapipe_pose.json` (or `mediapipe_pose_scaled.json`) and `openpose_named.json`, calculates the midpoint/average for each matching keypoint, and writes `pose_handler_out/averaged_keypoints.json`.

Command:

```bash
python -m pose_handler.average_mediapipe_openpose --mediapipe_json pose_handler_out/mediapipe_pose_scaled.json --openpose_json pose_handler_out/openpose_named.json --out_dir pose_handler_out
```

Optional: specify which image to process if MediaPipe JSON has multiple entries:

```bash
python -m pose_handler.average_mediapipe_openpose --mediapipe_json pose_handler_out/mediapipe_pose_scaled.json --openpose_json pose_handler_out/openpose_named.json --image_name img1.jpeg --out_dir pose_handler_out
```

## 10) Visualize `openpose_named.json` on the original image (2D overlay)

File: `pose_handler/visualize_openpose_named_on_image.py`  
What it does: Overlays the skeleton from `openpose_named.json` on a single image and saves a PNG.

Command:

```bash
python -m pose_handler.visualize_openpose_named_on_image --image path/to/image.png --named_json pose_handler_out/openpose_named.json --out_dir pose_handler_out
```

