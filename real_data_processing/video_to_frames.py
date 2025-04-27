import cv2 
import os 
import numpy as np


def split_video_to_frames(video_path, output_folder, get_timestamps=False):
    video_filename = os.path.basename(video_path).replace(".mp4","")

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None 

    # Get video details
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames

    print(f"Total frames: {total_frames}")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    timestamps = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Generate filename for each frame
        frame_filename = os.path.join(output_folder, f"{video_filename}_frame_{frame_count:05d}.png")
        
        # Save the frame as PNG
        cv2.imwrite(frame_filename, frame)
        
        timestamp = frame_count / fps
        timestamps.append(timestamp)
        frame_count += 1
        # print(f"Processing frame {frame_count}/{total_frames}")
    
    # Release the video capture object
    cap.release()

    if get_timestamps:
        # Return the list of timestamps if requested
        return np.array(timestamps) 
    
video_path = "C:/Users/NegiA/Desktop/abhay_ws/marker_detection_failure_recovery/real_data_processing/raw_data/calibration_26042025_2200.mp4"
output_folder = "C:/Users/NegiA/Desktop/abhay_ws/marker_detection_failure_recovery/real_data_processing/raw_data/realsense415_charuco_calibration_frames_v2"  # Folder to save the PNGs
split_video_to_frames(video_path, output_folder, get_timestamps=False)
