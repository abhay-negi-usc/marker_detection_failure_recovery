import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from optimal_overlap_torch import Processor, custom_overlap_loss, marker_reprojection_differentiable
import cv2 
from scipy.spatial.transform import Rotation as R

def estimate_initial_pose_from_segmentation(seg_mask: np.ndarray,
                                            camera_matrix: np.ndarray,
                                            marker_length: float,
                                            marker_corners_2d: np.ndarray,
                                            marker_corners_3d: np.ndarray,
                                            marker_image: np.ndarray,
                                            image_size=(480, 640),
                                            device="cpu") -> torch.Tensor:
    import torch.nn.functional as F

    contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in segmentation mask.")

    largest_contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

    if len(approx) != 4:
        raise ValueError("Could not find 4 corners in segmentation mask.")

    approx = approx.reshape(4, 2).astype(np.float32)

    best_loss = float('inf')
    best_pose = None
    seg_tensor = torch.from_numpy(seg_mask).float().to(device).unsqueeze(0) / 255.0
    rgb_img = torch.from_numpy(cv2.cvtColor(marker_image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    for i in range(4):
        perm = np.roll(approx, -i, axis=0).astype(np.float32)
        success, rvec, tvec = cv2.solvePnP(marker_corners_3d, perm, camera_matrix, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            continue

        rot = cv2.Rodrigues(rvec)[0]
        rot_euler = R.from_matrix(rot).as_euler("xyz", degrees=True)
        pose = np.concatenate([tvec.flatten(), rot_euler])
        pose_tensor = torch.tensor(pose, dtype=torch.float32, device=device)

        marker_seg = marker_image.copy()
        marker_seg[:, :, :] = 1

        rendered = marker_reprojection_differentiable(marker_image, marker_corners_2d, marker_corners_3d,
                                                       pose_tensor, camera_matrix, image_size=image_size)
        rendered_seg = marker_reprojection_differentiable(marker_seg, marker_corners_2d, marker_corners_3d, pose_tensor, camera_matrix)
        loss = custom_overlap_loss(rgb_img, seg_tensor, rendered, rendered_seg)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_pose = pose_tensor.clone()

    if best_pose is None:
        raise RuntimeError("Failed to find a valid pose from any corner ordering.")

    best_pose.requires_grad_()
    return best_pose

def plot_loss_vs_distance(pose_init, img_tensor, seg_tensor, marker_img_tensor, marker_seg_tensor, marker_corners_2d, marker_corners_3d, camera_matrix, device, std_devs=None, n_samples=200):
    if std_devs is None:
        std_devs = torch.tensor([0.02, 0.02, 0.02, 2.0, 2.0, 2.0], device=device)

    losses = []
    distances = []

    for _ in range(n_samples):
        noise = torch.randn_like(pose_init) * std_devs
        sample_pose = pose_init + noise

        rendered = marker_reprojection_differentiable(marker_img_tensor, marker_corners_2d, marker_corners_3d, sample_pose, camera_matrix)
        rendered_seg = marker_reprojection_differentiable(marker_seg_tensor, marker_corners_2d, marker_corners_3d, sample_pose, camera_matrix)

        loss = custom_overlap_loss(img_tensor, seg_tensor, rendered, rendered_seg)
        distance = torch.norm(noise / std_devs).item()

        losses.append(loss.item())
        distances.append(distance)

    plt.figure(figsize=(7, 5))
    plt.scatter(distances, losses, alpha=0.7, edgecolors='k')
    y_min, y_max = np.percentile(losses, [5, 95])
    plt.ylim(y_min, y_max)
    plt.xlabel("Distance from initial pose")
    plt.ylabel("Custom Overlap Loss")
    plt.title("Loss vs. Distance from Initial Pose")
    plt.grid(True)
    os.makedirs("visualization", exist_ok=True)
    plt.savefig("visualization/loss_vs_distance_scatter.png")
    plt.close()
    print("Saved scatter plot to visualization/loss_vs_distance_scatter.png")

def visualize_loss_landscape(pose_center, p: Processor, marker_corners_2d, marker_corners_3d, seg_gt, camera_matrix, 
                             img_tensor, marker_img_tensor, marker_seg_tensor, device, idx_pair=(0, 1),
                             perturb_range=(-0.05, 0.05), resolution=50):
    x_idx, y_idx = idx_pair
    x_vals = np.linspace(-perturb_range[0], +perturb_range[0], resolution)
    y_vals = np.linspace(-perturb_range[1], +perturb_range[1], resolution)
    
    loss_grid = np.zeros((resolution, resolution))

    for i, dx in enumerate(x_vals):
        for j, dy in enumerate(y_vals):
            pose_perturbed = pose_center.clone()
            pose_perturbed[x_idx] += dx
            pose_perturbed[y_idx] += dy

            rendered = marker_reprojection_differentiable(marker_img_tensor, marker_corners_2d, marker_corners_3d, pose_perturbed, camera_matrix)
            rendered_seg = marker_reprojection_differentiable(marker_seg_tensor, marker_corners_2d, marker_corners_3d, pose_perturbed, camera_matrix)

            loss = custom_overlap_loss(img_tensor, seg_gt, rendered, rendered_seg)
            loss_grid[j, i] = loss.item()

    plt.figure(figsize=(6, 5))
    plt.contourf(x_vals, y_vals, loss_grid, levels=50, cmap="viridis")
    plt.colorbar(label="Loss")
    plt.xlabel(f"Perturbation in dim {x_idx}")
    plt.ylabel(f"Perturbation in dim {y_idx}")
    plt.title("Loss Landscape")
    plt.tight_layout()
    os.makedirs("visualization", exist_ok=True)
    plt.savefig(f"visualization/loss_landscape_dim{x_idx}_dim{y_idx}.png")
    print(f"Saved loss landscape to visualization/loss_landscape_dim{x_idx}_dim{y_idx}.png")
    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p = Processor(
        dir_seg="output/sdg_markers_20250323-031913/seg/",
        dir_rgb="output/sdg_markers_20250323-031913/rgb/",
        marker_path="synthetic_data_generation/assets/tags/4x4_1000-31.png"
    )

    dp = p.datapoints[0]
    seg_np = dp.get_seg()
    seg_tensor = torch.from_numpy(seg_np).float().to(device).unsqueeze(0) / 255.0
    img_tensor = torch.from_numpy(cv2.imread(dp.rgb_path)).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    marker_img_tensor = p.marker
    marker_seg_tensor = np.ones_like(p.marker)

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

    pose_init = estimate_initial_pose_from_segmentation(
        seg_np,
        camera_matrix,
        marker_length,
        marker_corners_2d,
        marker_corners_3d,
        p.marker,
        image_size=(height, width),
        device=device
    )
    print("Initial pose estimated from segmentation:", pose_init)

    # plot_loss_vs_distance(pose_init, img_tensor, seg_tensor, marker_img_tensor, marker_seg_tensor, marker_corners_2d, marker_corners_3d, camera_matrix, device)

    perturb_ranges = np.array([0.100, 0.100, 1.0, 10.0, 10.0, 10.0])
    for i in range(6):
        for j in range(i + 1, 6):
            perturb_range = (perturb_ranges[i], perturb_ranges[j]) 
            resolution = 50
            visualize_loss_landscape(pose_init, p, marker_corners_2d, marker_corners_3d, seg_tensor, camera_matrix, 
                                     img_tensor, marker_img_tensor, marker_seg_tensor, device, idx_pair=(i, j), perturb_range=perturb_range, resolution=resolution)
