import numpy as np
import os
import cv2
import torch
from scipy.spatial.transform import Rotation as R
from torch.optim import Adam
import torch.nn.functional as F
import math 
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real

from pose_estimation_model.optimal_overlap_torch import Datapoint, Processor, custom_overlap_loss, marker_reprojection_differentiable, estimate_initial_pose_from_segmentation

# Keep other functions (e.g. xyzabc_to_tf, euler_to_rot_matrix, warp, project_points, etc.) unchanged
# --- Assume all functions and classes like Processor, custom_overlap_loss, etc. remain the same ---

def main():
    p = Processor(
        dir_seg="output/sdg_markers_20250323-031913/seg/",
        dir_rgb="output/sdg_markers_20250323-031913/rgb/",
        marker_path="synthetic_data_generation/assets/tags/4x4_1000-31.png"
    )

    width, height = 640, 480
    focal_length = 24.0
    horiz_aperture = 20.955
    vert_aperture = height/width * horiz_aperture
    fx = width * focal_length / horiz_aperture
    fy = height * focal_length / vert_aperture
    cx, cy = width / 2, height / 2

    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    marker_corners_2d = np.array([
        [0, 0],
        [0, p.marker.shape[0]],
        [p.marker.shape[1], p.marker.shape[0]],
        [p.marker.shape[1], 0]
    ], dtype=np.float32)

    marker_length = 0.10
    marker_corners_3d = np.array([
        [0, 0, 0],
        [marker_length, 0, 0],
        [marker_length, marker_length, 0],
        [0, marker_length, 0]
    ], dtype=np.float32)

    dp = p.datapoints[0]
    seg_gt = dp.get_seg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Estimate initial pose using heuristic or PnP method
    pose_initial = estimate_initial_pose_from_segmentation(
        seg_gt,
        camera_matrix,
        marker_length,
        marker_corners_2d,
        marker_corners_3d,
        p.marker,
        image_size=(height, width),
        device=device
    )

    print("Starting Bayesian Optimization...")
    optimized_pose, loss_history = p.optimize_pose_bayesian(
        p.marker, marker_corners_2d, marker_corners_3d, seg_gt, camera_matrix, device, n_calls=250, pose_initial=pose_initial)

    print("Optimized pose:", optimized_pose.cpu().numpy())
    p.plot_loss_curve(loss_history)
    p.save_overlay(optimized_pose, marker_corners_2d, marker_corners_3d, camera_matrix, device)

if __name__ == "__main__":
    main()
