import os
from pathlib import Path
import cv2
import numpy as np

def concatenate_images_across_folders(folder_paths, output_folder):
    """
    Args:
        folder_paths (list of str or Path): Paths to M folders each containing N images.
        output_folder (str or Path): Path to save concatenated images.
    """
    folder_paths = [Path(p) for p in folder_paths]
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Get sorted list of images in each folder
    image_lists = []
    for folder in folder_paths:
        images = sorted([f for f in folder.glob("*") if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])
        image_lists.append(images)

    # Check that all folders have the same number of images
    num_images_per_folder = [len(images) for images in image_lists]
    if len(set(num_images_per_folder)) != 1:
        raise ValueError(f"Folders have different number of images: {num_images_per_folder}")

    N = num_images_per_folder[0]
    M = len(folder_paths)

    # Loop over all images
    for i in range(N):
        images_to_concat = []
        for m in range(M):
            img_path = image_lists[m][i]
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"Could not read image: {img_path}")
            images_to_concat.append(img)

        # Optionally resize images to match heights if needed
        min_height = min(img.shape[0] for img in images_to_concat)
        resized_images = [cv2.resize(img, (int(img.shape[1] * min_height / img.shape[0]), min_height)) 
                          if img.shape[0] != min_height else img 
                          for img in images_to_concat]

        concatenated = cv2.hconcat(resized_images)

        # Save output
        output_path = output_folder / f"concat_{i:04d}.png"
        cv2.imwrite(str(output_path), concatenated)
        print(f"Saved: {output_path}")

    print("Done!")

# -------------------
# Example usage

if __name__ == "__main__":
    # ablations = ["distance", "skew", "truncation", "underexposure", "glare", "glint", "shadow"]
    ablations = ["shadow"]
    for ablation in ablations: 
        folder_paths = [
            Path(f"./ablations/data/real_experiments/{ablation}/CCV_results"),
            Path(f"./ablations/data/real_experiments/{ablation}/LBCV_segmentation_results"),
            Path(f"./ablations/data/real_experiments/{ablation}/LBCV_keypoints_results"),
        ]
        output_folder = f"./ablations/data/real_experiments/{ablation}/results/combined_images"
        concatenate_images_across_folders(folder_paths, output_folder)
