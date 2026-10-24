
# Activate diffusion PS 
cript:"diffusion\Scripts\activate"

#1 Extract skeleton using averaged (mediapipe,openpose), smoothed using kalman filter. 

- COMMAND: python -m kalman.run_video_pipeline --video VIDEO_PATH
- Important Note: Before Running the diffusion environment should be activated using this comman `diffusion\Scripts\activate`
- note: replace vidoo pat with the real video path
- note2: The output will ba saved on kalman_output folder
- note3: This process will output: 1- skeleton points, 2- angels, 3- video of the skeleton



#2 Vidoe Correction + Frame generation
-COMMAND: python run_video_correction_pipeline.py --video media/VIDOE_NAME.mp4  --keypoints_json kalman_out/VIDOE_NAME_smoothed_keypoints.json --angles_json kalman_out/VIDOE_NAME_angles.json --length_drift_threshold 0.01

- note1: This command run the pipeline, calculate the angels, correct them, and regenerate the frames using PIES.


#2 Video Builder
-COMMAND: python build_comparison_videos.py --video media/VIDE_NAME.mp4 --comparison_dir kalman_out\VIDEO_NAME\comparison_frames

- note1: This command regenerate the videos, after regeneration of the frames. 
- note2: Two videos are built, one with the correct skeleton, one with original skeleton.
- note3: The videos are saved in kalman_out\VIDEO_NAME\comparison_frames

# Pose Evaluator
VIDEONAME_original_generated_smoothed_keypoints.json
VIDEONAME_corrected_generated_smoothed_keypoints.json
-COMMAND: python pose_evaluator.py --keypoints_dir kalman_out




# Note: After frame generation and run the video builder, we re run the kalman on the new 2 videos, SO we extract the new generated keypoints. Finally run the pos evaluator on these keypoints extracted from the new generated video