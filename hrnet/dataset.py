import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

class TagPoseDataset(Dataset):
    def __init__(self, image_dir, pose_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.pose_dir = Path(pose_dir)
        self.image_paths = sorted(self.image_dir.glob("rgb_*.png"))
        self.pose_paths = sorted(self.pose_dir.glob("pose_*.json"))

        assert len(self.image_paths) == len(self.pose_paths), "Mismatch between images and poses"

        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        with open(self.pose_paths[idx], 'r') as f:
            pose_json = json.load(f)
        
        tag_T = np.array(pose_json['tag'], dtype=np.float32)  # 4x4 matrix

        # Extract rotation matrix and translation
        R_mat = tag_T[:3, :3]
        t_vec = tag_T[:3, 3]

        # Convert rotation to so(3) (Lie algebra) using scipy
        rvec = R.from_matrix(R_mat).as_rotvec()  # shape (3,)
        pose_6d = np.concatenate([rvec, t_vec], axis=0).astype(np.float32)  # (6,)

        return self.transform(image), torch.tensor(pose_6d)
