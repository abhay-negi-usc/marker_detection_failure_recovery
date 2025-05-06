import numpy as np 
import torch

def se3_loss(pred, target):
    """Mean squared error loss on se(3) 6D vector."""
    return torch.mean((pred - target) ** 2)

def lie_to_matrix(lie_vec):
    """
    Convert 6D vector to SE(3) matrix.
    """
    from scipy.spatial.transform import Rotation as R
    rot = R.from_rotvec(lie_vec[:3]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = lie_vec[3:]
    return T

def keypoint_loss(pred, target):
    return torch.mean((pred - target) ** 2)  # MSE over (x, y) pairs

def generate_heatmap(keypoints, img_size, heatmap_size, sigma=2):
    """keypoints: (K, 2), returns (K, H, W)"""
    K = keypoints.shape[0]
    heatmaps = np.zeros((K, heatmap_size, heatmap_size), dtype=np.float32)
    for i in range(K):
        x, y = keypoints[i]
        x = int(x / img_size[0] * heatmap_size)
        y = int(y / img_size[1] * heatmap_size)
        if x < 0 or y < 0 or x >= heatmap_size or y >= heatmap_size:
            continue
        xx, yy = np.meshgrid(np.arange(heatmap_size), np.arange(heatmap_size))
        heatmaps[i] = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
    return heatmaps

def heatmap_loss(pred, target):
    return torch.mean((pred - target) ** 2)
