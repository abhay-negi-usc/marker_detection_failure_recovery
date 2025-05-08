import cv2
import os
from pathlib import Path

def pngs_to_mp4(input_dir, output_path, fps=30):
    input_dir = Path(input_dir)
    images = sorted([img for img in input_dir.glob("*.png")])

    if not images:
        raise ValueError(f"No PNG files found in {input_dir}")

    # Read the first image to get dimensions
    first_img = cv2.imread(str(images[0]))
    height, width, _ = first_img.shape

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' or 'avc1' for broader codec support
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Warning: Could not read {img_path}")
            continue
        video_writer.write(frame)

    video_writer.release()
    print(f"Saved video to {output_path}")

if __name__ == "__main__":
    pngs_to_mp4(
        input_dir="./real_data_processing/raw_data/realsense/realsense_6_frames_LBCV/keypoints_overlay",  # 🔁 your input folder here
        output_path="./real_data_processing/raw_data/realsense/realsense_6_frames_LBCV/keypoints_overlay/keypoints_video.mp4",
        fps=30  # 🔁 change FPS as needed
    )
