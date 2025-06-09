# [Cleaned + Parallelized Main Script for Dataset Processing and Saving]
import os
import time
import json
import multiprocessing as mp
from PIL import Image
from tqdm import tqdm
import numpy as np
from functools import partial
from random import shuffle
import matplotlib.pyplot as plt
from synthetic_data_generation.data_preprocessing_utils import DataProcessor  # Assuming the DataProcessor class is stored separately
from keypoints_model.utils import overlay_points_on_image  # Assuming this is defined externally

# -------------------- Parallel Filter Helper (Must be top-level) --------------------
def filter_datapoint_worker(dp, min_tag_area=1000, min_tag_pix_mean=70, max_tag_pix_mean=250):
    try:
        dp.compute_diffusion_reflectance()
        seg_img = dp.preprocess_seg_img()
        seg_img = np.array(seg_img)

        dp.tag_pix_area = np.sum(seg_img == 255)

        marker_pixels = np.argwhere(seg_img == 255)
        rgb_img = np.array(Image.open(dp.rgb_filepath))
        marker_rgb_values = rgb_img[marker_pixels[:, 0], marker_pixels[:, 1]]
        marker_grey_values = np.mean(marker_rgb_values, axis=1)

        if marker_grey_values.size == 0:
            return dp, False

        dp.tag_pix_contrast = marker_grey_values.max() - marker_grey_values.min()
        dp.tag_pix_mean = marker_grey_values.mean()

        is_valid = (
            dp.tag_pix_area > min_tag_area and
            min_tag_pix_mean < dp.tag_pix_mean < max_tag_pix_mean
        )

        return dp, is_valid
    except Exception as e:
        print(f"[ERROR] Filtering failed for {dp.rgb_filepath}: {e}")
        return dp, False

# -------------------- Parallel Split Helper (Must be top-level) --------------------
def split_worker(item_train_idx):
    item, train_indices = item_train_idx
    idx, dp = item
    return dp, idx in train_indices

# -------------------- Define Helper Function for Saving --------------------
def save_single_datapoint(dp, idx, dataset_type, out_dir, save_rgb, save_seg, save_keypoints, save_metadata, save_summary_image, save_roi, num_augmentations, keypoints_tag_frame, camera_matrix):
    dataset_dir = os.path.join(out_dir, dataset_type)
    os.makedirs(dataset_dir, exist_ok=True)

    if save_rgb:
        os.makedirs(os.path.join(dataset_dir, "rgb"), exist_ok=True)
        img = Image.open(dp.rgb_filepath)
        img.save(os.path.join(dataset_dir, "rgb", f"img_{idx}.png"))

    if save_seg:
        os.makedirs(os.path.join(dataset_dir, "seg"), exist_ok=True)
        seg = dp.preprocess_seg_img()
        seg.save(os.path.join(dataset_dir, "seg", f"seg_{idx}.png"))

    if save_keypoints:
        os.makedirs(os.path.join(dataset_dir, "keypoints"), exist_ok=True)
        keypoints = dp.compute_keypoints(keypoints_tag_frame, camera_matrix)
        keypoints_json = {f"keypoints_{i}": kp.tolist() for i, kp in enumerate(keypoints)}
        with open(os.path.join(dataset_dir, "keypoints", f"keypoints_{idx}.json"), 'w') as f:
            json.dump(keypoints_json, f)

    if save_metadata:
        os.makedirs(os.path.join(dataset_dir, "metadata"), exist_ok=True)
        with open(os.path.join(dataset_dir, "metadata", f"metadata_{idx}.json"), 'w') as f:
            json.dump(dp.metadata, f)

    if save_summary_image:
        os.makedirs(os.path.join(dataset_dir, "summary_images"), exist_ok=True)

        try:
            rgb_img = Image.open(dp.rgb_filepath).convert("RGB")
            seg_img = np.array(dp.preprocess_seg_img())
            augmented_img = np.array(rgb_img)  # no augmentation applied here
            keypoints = dp.compute_keypoints(keypoints_tag_frame, camera_matrix)
            roi_image = np.array(rgb_img)
            roi_keypoints = keypoints

            plt.figure(figsize=(12, 8))
            plt.subplot(2, 3, 1)
            plt.imshow(np.array(rgb_img))
            plt.axis('off')
            plt.title(f'Original Image {idx}')

            plt.subplot(2, 3, 2)
            plt.imshow(augmented_img)
            plt.axis('off')
            plt.title(f'Augmented Image {idx}')

            plt.subplot(2, 3, 3)
            plt.imshow(seg_img, cmap='viridis')
            plt.axis('off')
            plt.title(f'Segmentation Image {idx}')

            keypoints_image = overlay_points_on_image(image=np.array(augmented_img), pixel_points=keypoints, radius=1)
            plt.subplot(2, 3, 4)
            plt.imshow(keypoints_image)
            plt.axis('off')
            plt.title(f'Keypoints Image {idx}')

            plt.subplot(2, 3, 5)
            plt.imshow(roi_image)
            plt.axis('off')
            plt.title(f'ROI Image {idx}')

            roi_keypoints_image = overlay_points_on_image(image=roi_image, pixel_points=roi_keypoints, radius=1)
            plt.subplot(2, 3, 6)
            plt.imshow(roi_keypoints_image)
            plt.axis('off')
            plt.title(f'ROI Keypoints Image {idx}')

            metadata_str = dp.__repr__()
            plt.text(1.05, 0.5, metadata_str, fontsize=12, ha='left', va='center', transform=plt.gca().transAxes,
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=1'))

            plt.tight_layout()
            plt.subplots_adjust(right=0.8)

            save_path = os.path.join(dataset_dir, "summary_images", f"summary_image_{idx}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()

        except Exception as e:
            print(f"[ERROR] Failed to generate summary image for {idx}: {e}")

# -------------------- Parallel Save Helper --------------------
def save_datapoint_wrapper(args):
    dp, idx, dataset_type, config = args
    try:
        out_dir = config["out_dir"]
        save_single_datapoint(
            dp, idx, dataset_type, out_dir=out_dir,
            **{k: v for k, v in config.items() if k not in ["out_dir"]}
        )
    except Exception as e:
        print(f"[ERROR] Failed to save datapoint {idx} ({dataset_type}): {e}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # -------------------- Configuration --------------------
    data_folders = [
        "/home/anegi/abhay_ws/marker_detection_failure_recovery/output/sdg_markers_20250423-191220/",
        "/home/anegi/abhay_ws/marker_detection_failure_recovery/output/sdg_markers_20250423-191357/",
        "/home/anegi/abhay_ws/marker_detection_failure_recovery/output/sdg_markers_20250423-191716/",
        "/home/anegi/abhay_ws/marker_detection_failure_recovery/output/sdg_markers_20250423-191924/",
    ]

    OUT_DIR = f"/home/anegi/abhay_ws/marker_detection_failure_recovery/segmentation_model/data/data_{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[INFO] Output directory created: {OUT_DIR}")

    print("[INFO] Initializing data processor...")
    processor = DataProcessor(data_folders, OUT_DIR)
    processor.set_marker(
        image_path="./synthetic_data_generation/assets/tags/4x4_1000-31.png",
        num_squares=8,
        side_length=0.100
    )

    print("[INFO] Processing data folders...")
    processor.process_folders()
    print(f"[INFO] Total datapoints loaded: {len(processor.datapoints)}")

    # # Optional truncation for fast testing
    # MAX_DATAPOINTS = 100
    # processor.datapoints = processor.datapoints[:MAX_DATAPOINTS]
    # print(f"[INFO] Truncated to {len(processor.datapoints)} datapoints for debugging.")

    print("[INFO] Filtering datapoints based on tag visibility and lighting...")
    with mp.Pool(mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(partial(filter_datapoint_worker), processor.datapoints), total=len(processor.datapoints)))

    processor.datapoints_filtered = [dp for dp, valid in results if valid]
    processor.datapoints_filtered_out = [dp for dp, valid in results if not valid]
    print(f"[INFO] Datapoints after filtering: {len(processor.datapoints_filtered)}")

    def parallel_split(datapoints, frac_train=0.95):
        indices = list(range(len(datapoints)))
        shuffle(indices)
        split_idx = int(len(datapoints) * frac_train)
        train_indices = set(indices[:split_idx])

        datapoints_with_index = list(enumerate(datapoints))
        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.map(split_worker, [(item, train_indices) for item in datapoints_with_index])

        train_set = [dp for dp, is_train in results if is_train]
        val_set = [dp for dp, is_train in results if not is_train]
        return train_set, val_set

    print("[INFO] Splitting filtered datapoints into train/val...")
    processor.datapoints_train, processor.datapoints_val = parallel_split(processor.datapoints_filtered, frac_train=0.95)
    print(f"[INFO] Training set size: {len(processor.datapoints_train)}")
    print(f"[INFO] Validation set size: {len(processor.datapoints_val)}")

    print("[INFO] Saving preprocessed train/val data in parallel...")
    config = {
        "save_rgb": True,
        "save_seg": True,
        "save_keypoints": True,
        "save_metadata": True,
        "save_summary_image": False,
        "save_roi": False,
        "num_augmentations": 0,
        "out_dir": OUT_DIR,
        "keypoints_tag_frame": processor.keypoints_tag_frame,
        "camera_matrix": processor.camera_matrix
    }

    args_list = []
    for dataset_type in ['train', 'val']:
        datapoints = processor.datapoints_train if dataset_type == 'train' else processor.datapoints_val
        for i, dp in enumerate(datapoints):
            args_list.append((dp, i, dataset_type, config))

    with mp.Pool(processes=mp.cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(save_datapoint_wrapper, args_list), total=len(args_list)))

    print("[INFO] Dataset processing and saving complete.")
