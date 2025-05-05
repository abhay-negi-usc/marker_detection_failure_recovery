import numpy as np
import os
import cv2
import torch
from scipy.spatial.transform import Rotation as R
from torch.optim import Adam
import torch.nn.functional as F
import math 
import matplotlib.pyplot as plt

def xyzabc_to_tf(xyzabc):
    t = xyzabc[:3]
    rot = R.from_euler("xyz", xyzabc[3:], degrees=True).as_matrix()
    tf = np.eye(4)
    tf[:3, :3] = rot
    tf[:3, 3] = t
    return tf

def euler_to_rot_matrix(euler_deg: torch.Tensor) -> torch.Tensor:
    """ Convert xyz Euler angles in degrees to a rotation matrix (3x3) in a differentiable way. """
    euler_rad = torch.deg2rad(euler_deg)
    x, y, z = euler_rad

    Rx = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(x), -torch.sin(x)],
        [0, torch.sin(x), torch.cos(x)]
    ], device=euler_deg.device)

    Ry = torch.tensor([
        [torch.cos(y), 0, torch.sin(y)],
        [0, 1, 0],
        [-torch.sin(y), 0, torch.cos(y)]
    ], device=euler_deg.device)

    Rz = torch.tensor([
        [torch.cos(z), -torch.sin(z), 0],
        [torch.sin(z), torch.cos(z), 0],
        [0, 0, 1]
    ], device=euler_deg.device)

    return Rz @ Ry @ Rx  # XYZ order

def compute_homography_dlt_batched(src_pts: torch.Tensor, dst_pts: torch.Tensor) -> torch.Tensor:
    B = src_pts.shape[0]
    assert src_pts.shape == (B, 4, 2)
    assert dst_pts.shape == (B, 4, 2)

    A_list = []
    for i in range(4):
        x = src_pts[:, i, 0]
        y = src_pts[:, i, 1]
        u = dst_pts[:, i, 0]
        v = dst_pts[:, i, 1]

        zeros = torch.zeros_like(x)

        row1 = torch.stack([-x, -y, -torch.ones_like(x), zeros, zeros, zeros, u * x, u * y, u], dim=1)
        row2 = torch.stack([zeros, zeros, zeros, -x, -y, -torch.ones_like(x), v * x, v * y, v], dim=1)

        A_list.extend([row1, row2])

    A = torch.stack(A_list, dim=1)  # (B, 8, 9)
    _, _, V = torch.linalg.svd(A)
    h = V[:, -1, :]  # (B, 9)
    H = h.view(B, 3, 3)
    H = H / H[:, 2:3, 2:3]
    return H

def warp_perspective_torch(image: torch.Tensor, H: torch.Tensor, out_size: tuple) -> torch.Tensor:
    B, C, H_img, W_img = image.shape
    H_out, W_out = out_size

    y, x = torch.meshgrid(
        torch.linspace(0, H_out - 1, H_out, device=image.device),
        torch.linspace(0, W_out - 1, W_out, device=image.device),
        indexing="ij"
    )
    grid = torch.stack([x, y, torch.ones_like(x)], dim=-1)
    grid = grid.view(-1, 3).permute(1, 0).unsqueeze(0)

    H_inv = torch.inverse(H)
    warped = H_inv @ grid
    warped = warped[:, :2, :] / warped[:, 2:3, :]

    norm_x = 2 * warped[:, 0, :] / (W_img - 1) - 1
    norm_y = 2 * warped[:, 1, :] / (H_img - 1) - 1
    grid = torch.stack([norm_x, norm_y], dim=-1).view(1, H_out, W_out, 2)

    out = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return out

def project_points_torch(points_3d, rvec, tvec, K):
    """
    Differentiable implementation of cv2.projectPoints.

    Args:
        points_3d: (N, 3) torch tensor
        rvec: (3,) torch tensor, axis-angle
        tvec: (3,) torch tensor
        K: (3, 3) torch tensor

    Returns:
        points_2d: (N, 2) torch tensor in pixel space
    """
    # Convert rvec (axis-angle) to rotation matrix using Rodrigues formula
    theta = torch.norm(rvec)
    if theta < 1e-6:
        R = torch.eye(3, device=rvec.device, dtype=rvec.dtype)
    else:
        k = rvec / theta
        K_cross = torch.tensor([
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0]
        ], device=rvec.device, dtype=rvec.dtype)
        R = torch.eye(3, device=rvec.device) + torch.sin(theta) * K_cross + (1 - torch.cos(theta)) * (K_cross @ K_cross)

    points_cam = (R @ points_3d.T).T + tvec  # (N, 3)
    points_proj = (K @ points_cam.T).T  # (N, 3)
    points_2d = points_proj[:, :2] / points_proj[:, 2:3]  # (N, 2)

    return points_2d

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def custom_overlap_loss(
    img: torch.Tensor,
    seg: torch.Tensor,
    reproj_img: torch.Tensor,
    reproj_seg: torch.Tensor,
    sharpening: float = 1.0,
    overlap_reward_weight: float = 10.0
) -> torch.Tensor:
    """
    Custom loss:
        Encourages image similarity in overlapping segmented regions AND promotes larger overlap.

        loss = sum[(sI * sP) * abs(bI - bP)] - alpha * sum[sI * sP]
    Where:
        sI: segmentation of input image (seg)
        sP: segmentation of projection (reproj_seg)
        bI: binarized grayscale of input image
        bP: binarized grayscale of projected image
        alpha: overlap_reward_weight
    """
    _, _, H_r, W_r = reproj_img.shape
    img_resized = F.interpolate(img, size=(H_r, W_r), mode='bilinear', align_corners=True)

    if seg.ndim == 3:
        seg = seg.unsqueeze(1)
    seg_resized = F.interpolate(seg, size=(H_r, W_r), mode='nearest')

    # Convert to grayscale
    gray_img = 0.2989 * img_resized[:, 0, :, :] + 0.5870 * img_resized[:, 1, :, :] + 0.1140 * img_resized[:, 2, :, :]
    gray_reproj = 0.2989 * reproj_img[:, 0, :, :] + 0.5870 * reproj_img[:, 1, :, :] + 0.1140 * reproj_img[:, 2, :, :]

    # Soft binarization
    bin_img = torch.sigmoid((gray_img - 0.5) * sharpening)
    bin_reproj = torch.sigmoid((gray_reproj - 0.5) * sharpening)
    bin_seg = torch.sigmoid((seg_resized.squeeze(1) - 0.5) * sharpening)
    bin_reproj_seg = torch.sigmoid((reproj_seg.squeeze(1) - 0.5) * sharpening)

    # Overlap mask
    overlap_mask = bin_seg * bin_reproj_seg

    # Loss term
    diff = torch.abs(bin_img - bin_reproj)
    masked_diff = overlap_mask * diff
    image_diff_loss = masked_diff.sum()

    # Overlap reward term
    overlap_reward = overlap_mask.sum()

    # Final loss
    total_loss = image_diff_loss - overlap_reward_weight * overlap_reward

    return total_loss


def marker_reprojection_differentiable(marker_image_np, marker_corners_2d, marker_corners_3d, xyzabc, camera_matrix, image_size=(480, 640)):
    if marker_image_np.ndim == 2:
        marker_tensor = torch.from_numpy(marker_image_np).float() / 255.0
        marker_tensor = marker_tensor.unsqueeze(0).unsqueeze(0)
    elif marker_image_np.ndim == 3:
        marker_tensor = torch.from_numpy(marker_image_np).permute(2, 0, 1).float() / 255.0
        marker_tensor = marker_tensor.unsqueeze(0)
    else:
        raise ValueError("Unexpected marker image shape")

    device = xyzabc.device
    marker_tensor = marker_tensor.to(device)

    t = xyzabc[:3]
    rot = R.from_euler("xyz", xyzabc[3:].detach().cpu().numpy(), degrees=True)
    R_mat = euler_to_rot_matrix(xyzabc[3:])
    rvec, _ = cv2.Rodrigues(R_mat.cpu().numpy())
    rvec = torch.from_numpy(rvec).squeeze().float().to(device)

    marker_corners_3d = torch.from_numpy(marker_corners_3d).float().to(device)
    K = torch.from_numpy(camera_matrix).float().to(device)
    tvec = t.float().unsqueeze(0)

    marker_corners_2d_proj = project_points_torch(marker_corners_3d, rvec, tvec.squeeze(0), K)  # (4, 2)

    src_pts = torch.from_numpy(marker_corners_2d).unsqueeze(0).float().to(device)
    dst_pts = marker_corners_2d_proj.unsqueeze(0)
    H = compute_homography_dlt_batched(src_pts, dst_pts)

    warped = warp_perspective_torch(marker_tensor, H, image_size)
    return warped


class Datapoint():
    def __init__(self, rgb_path, seg_path):
        self.rgb_path = rgb_path
        self.seg_path = seg_path

    def get_rgb(self):
        return cv2.imread(self.rgb_path)

    def get_seg(self):
        image = cv2.imread(self.seg_path)
        key_color = (25, 255, 140)
        return cv2.inRange(image, np.array(key_color), np.array(key_color))

class Processor():
    def __init__(self, dir_rgb, dir_seg, marker_path):
        self.dir_rgb = dir_rgb
        self.dir_seg = dir_seg
        self.marker_path = marker_path
        self.datapoints = self.get_datapoints()
        self.marker = self.get_marker()
        self.loss_history = []  # ← add this

    def get_datapoints(self):
        rgb_files = sorted(os.listdir(self.dir_rgb))
        return [Datapoint(os.path.join(self.dir_rgb, f), os.path.join(self.dir_seg, f.replace("rgb", "seg"))) for f in rgb_files]

    def get_marker(self):
        marker_image = cv2.imread(self.marker_path)
        if marker_image is None:
            raise FileNotFoundError(f"Could not read marker image at {self.marker_path}")
        return marker_image

    
    def _optimize_pose_custom(self, marker_image, marker_corners_2d, marker_corners_3d, seg_gt, camera_matrix, pose_initial, device, lr=1e-1, steps=1000):
        pose = pose_initial.clone().detach().requires_grad_().to(device)
        optimizer = Adam([pose], lr=lr)
        self.loss_history = []

        seg_tensor = torch.from_numpy(seg_gt).float().to(device).unsqueeze(0) / 255.0
        rgb_img = torch.from_numpy(cv2.imread(self.datapoints[0].rgb_path)).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

        marker_seg = marker_image.copy() 
        marker_seg[:, :, :] = 1 

        for i in range(steps):
            optimizer.zero_grad()
            rendered = marker_reprojection_differentiable(marker_image, marker_corners_2d, marker_corners_3d, pose, camera_matrix)
            rendered_seg = marker_reprojection_differentiable(marker_seg, marker_corners_2d, marker_corners_3d, pose, camera_matrix)
            loss = custom_overlap_loss(rgb_img, seg_tensor, rendered, rendered_seg)
            self.loss_history.append(loss.item())
            loss.backward()

            if i % 100 == 0:
                print(f"Iteration {i}: Loss: {loss.item():.6f}, Gradients: {pose.grad.detach().cpu().numpy()}")

            # Add gradient noise
            noise = torch.randn_like(pose) * 0.01
            pose.grad += noise

            optimizer.step()


        return pose.detach().cpu(), self.loss_history


    def plot_loss_curve(self, loss_history):
        plt.figure(figsize=(8, 4))
        plt.plot(loss_history, label="Loss")
        plt.xlabel("Iteration")
        plt.ylabel("MSE Loss")
        plt.title("Pose Optimization Loss Curve")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("loss_curve.png")
        plt.close()
        print("Saved loss curve to loss_curve.png")

def estimate_initial_pose_from_segmentation(seg_mask: np.ndarray,
                                            camera_matrix: np.ndarray,
                                            marker_length: float,
                                            marker_corners_2d: np.ndarray,
                                            marker_corners_3d: np.ndarray,
                                            marker_image: np.ndarray,
                                            image_size=(480, 640),
                                            device="cpu") -> torch.Tensor:
    """
    Estimate initial pose using detected corners and PnP.
    Tries all 4 permutations of corner order and returns the one with the lowest loss.
    """
    import torch.nn.functional as F

    # Find contours
    contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in segmentation mask.")

    # Assume largest contour is the marker
    largest_contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

    if len(approx) != 4:
        raise ValueError("Could not find 4 corners in segmentation mask.")

    approx = approx.reshape(4, 2).astype(np.float32)

    # Try all 4 permutations of the square corner ordering
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

        rendered = marker_reprojection_differentiable(marker_image, marker_corners_2d, marker_corners_3d,
                                                       pose_tensor, camera_matrix, image_size=image_size)
        loss = custom_overlap_loss(rgb_img, seg_tensor, rendered, rendered)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_pose = pose_tensor.clone()

    if best_pose is None:
        raise RuntimeError("Failed to find a valid pose from any corner ordering.")

    best_pose.requires_grad_()
    return best_pose


import numpy as np
import os
import cv2
import torch
from scipy.spatial.transform import Rotation as R
from torch.optim import Adam
import torch.nn.functional as F
import math 
import matplotlib.pyplot as plt

def main():
    p = Processor(
        dir_seg="output/sdg_markers_20250323-031913/seg/",
        dir_rgb="output/sdg_markers_20250323-031913/rgb/",
        marker_path="synthetic_data_generation/assets/tags/4x4_1000-31.png"
    )

    width = 640 
    height = 480 
    focal_length = 24.0 
    horiz_aperture = 20.955
    vert_aperture = height/width * horiz_aperture
    fx = width * focal_length / horiz_aperture
    fy = height * focal_length / vert_aperture
    cx = width / 2
    cy = height / 2

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


    marker_length = 0.10  # meters
    marker_corners_3d = np.array([
        [0, 0, 0],
        [marker_length, 0, 0],
        [marker_length, marker_length, 0],
        [0, marker_length, 0]
    ], dtype=np.float32)

    dp = p.datapoints[0]
    seg_gt = dp.get_seg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pose_initial = estimate_initial_pose_from_segmentation(seg_gt, camera_matrix, marker_length, marker_corners_2d, marker_corners_3d, p.marker, image_size=(height, width), device=device)
    print("Initial pose:", pose_initial.detach().cpu().numpy())

    pose_initial = (pose_initial.clone().detach() +
                torch.tensor([0.0, -0.0, 0.0, 0.0, 0, 10], device=pose_initial.device)).requires_grad_()

    optimized_pose, loss_history = p._optimize_pose_custom(p.marker, marker_corners_2d, marker_corners_3d, seg_gt, camera_matrix, pose_initial, device, lr=1e-4, steps=1_000)

    print("Final pose:", optimized_pose.numpy())
    p.plot_loss_curve(loss_history)

        # --- Overlay visualization ---
    with torch.no_grad():
        initial_rendered = marker_reprojection_differentiable(p.marker, marker_corners_2d, marker_corners_3d, pose_initial.to(device), camera_matrix)
        final_rendered = marker_reprojection_differentiable(p.marker, marker_corners_2d, marker_corners_3d, optimized_pose.to(device), camera_matrix)

    rgb = cv2.resize(dp.get_rgb(), (final_rendered.shape[-1], final_rendered.shape[-2]))
    rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    initial_np = (initial_rendered.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    final_np = (final_rendered.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    blended_initial = cv2.addWeighted(rgb, 0.4, initial_np, 0.6, 0)
    blended_final = cv2.addWeighted(rgb, 0.4, final_np, 0.6, 0)

    comparison = np.hstack([blended_initial, blended_final])
    cv2.imwrite("comparison_overlay.png", comparison)
    print("Saved initial and final overlay comparison to comparison_overlay.png")



if __name__ == "__main__":
    main()
