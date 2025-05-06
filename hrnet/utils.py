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
