import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pose_estimation_model.utils import *
from scipy.spatial.transform import Rotation as R

def get_rgb_and_seg_images(dir_rgb, dir_seg, idx=None):
    files_rgb = sorted([f for f in os.listdir(dir_rgb) if f.endswith('.png') or f.endswith('.jpg')])
    files_seg = [f.replace(dir_rgb, dir_seg) for f in files_rgb]
    
    if idx is None:
        idx = np.random.randint(0, len(files_rgb))
    
    if idx < 0 or idx >= len(files_rgb):
        raise IndexError("Index out of range for image files.")
    
    
    img_rgb = cv2.imread(os.path.join(dir_rgb, files_rgb[idx]))
    img_seg = cv2.imread(os.path.join(dir_seg, files_seg[idx]))
    img_rgb = cv2.resize(img_rgb, (640, 480))
    img_seg = cv2.resize(img_seg, (640, 480))
    return img_rgb, img_seg

def main():
    # dir_rgb = "/home/rp/abhay_ws/marker_detection_failure_recovery/real_data_processing/raw_data/controlled_tests/dark_test_3_frames/"
    # dir_seg = "/home/rp/abhay_ws/marker_detection_failure_recovery/real_data_processing/raw_data/controlled_tests/dark_test_3_frames/results/LBCV_segmentation/"
    dir_rgb = "/home/rp/abhay_ws/marker_detection_failure_recovery/real_data_processing/raw_data/controlled_tests/bright_test_4_frames/"
    dir_seg = "/home/rp/abhay_ws/marker_detection_failure_recovery/real_data_processing/raw_data/controlled_tests/bright_test_4_frames/results/LBCV_segmentation/"
    img_marker_path = "./synthetic_data_generation/assets/tags/tag36h11_0.png"
    fx, fy, cx, cy = 1363.85, 1365.40, 958.58, 552.25
    sx, sy = 640 / 1920, 480 / 1080
    camera_matrix = np.array([[fx * sx, 0, cx * sx],
                              [0, fy * sy, cy * sy],
                              [0, 0, 1]], dtype=np.float32)

    img_rgb, img_seg = get_rgb_and_seg_images(dir_rgb, dir_seg, idx=None)
    img_rgb_masked = crop_rgb_using_seg(img_rgb, img_seg)
    corners = find_segmentation_four_corners(img_seg)
    if corners is None:
        print("No corners found in the segmentation image.")
        # show image with no corners found
        plt.imshow(cv2.cvtColor(img_rgb_masked, cv2.COLOR_BGR2RGB))
        plt.title("No Corners Found")
        plt.axis('off')
        plt.show()
        return
    tf_candidates = compute_tf_candidates_from_corners(corners, marker_size=(0.1, 0.1), camera_matrix=camera_matrix)

    keypoints_rgb_image_space = find_keypoints(img_rgb, img_seg)
    img_marker = cv2.imread(img_marker_path)
    keypoints_marker_image_space = find_keypoints(img_marker)
    keypoints_marker_cartesian_space = convert_marker_keypoints_to_cartesian(
        keypoints_marker_image_space, image_size=(img_marker.shape[0], img_marker.shape[1]), marker_size=(0.1, 0.1)
    )

    refined_tf_candidates = []
    residuals = []

    for i, tf in enumerate(tf_candidates):
        refined_tf, residual = refine_pose_icp_3d2d_auto_match(
            keypoints_marker_cartesian_space, keypoints_rgb_image_space, camera_matrix,
            tf, max_iterations=100, show_iteration_images=False
        )
        refined_tf_candidates.append(refined_tf)
        residuals.append(residual)

        eul_init = R.from_matrix(tf[:3, :3]).as_euler('xyz', degrees=True)
        eul_refined = R.from_matrix(refined_tf[:3, :3]).as_euler('xyz', degrees=True)
        print(f"Initial Euler: {np.round(eul_init, 1)}, Refined Euler: {np.round(eul_refined, 1)}")

    min_idx = np.argmin(residuals)
    print(f"Min Residual: {residuals[min_idx]:.4f} at index {min_idx}")
    tf_final = refined_tf_candidates[min_idx]

    # --- Prepare detected keypoints overlay on masked image ---
    keypoints_rgb_image_space = np.asarray(keypoints_rgb_image_space, dtype=np.float32).reshape(-1, 2)
    img_masked_with_keypoints = img_rgb_masked.copy()
    for pt in keypoints_rgb_image_space:
        cv2.circle(img_masked_with_keypoints, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)

    # --- Prepare projected keypoints overlay on original RGB image ---
    img_projected_overlay = overlay_3D_points_on_image(
        img_rgb.copy(), keypoints_marker_cartesian_space, camera_matrix, tf_final,
        color=(255, 0, 0), radius=3
    )

    # --- Create side-by-side plot ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].imshow(cv2.cvtColor(img_masked_with_keypoints, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Masked RGB with Detected Keypoints")
    axs[0].axis('off')

    axs[1].imshow(cv2.cvtColor(img_projected_overlay, cv2.COLOR_BGR2RGB))
    axs[1].set_title(f"Full RGB with Projected Keypoints\nResidual: {residuals[min_idx]:.4f}")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
