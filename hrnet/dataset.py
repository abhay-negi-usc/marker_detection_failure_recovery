import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import os 

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


class MarkersDataset(Dataset):
    def __init__(self, image_dir, keypoints_dir, transform=None, heatmap_size=64):
        self.image_dir = image_dir
        self.keypoints_dir = keypoints_dir
        self.transform = transform or transforms.ToTensor()
        self.images = sorted(os.listdir(image_dir))  # 👈 make sure this is deterministic
        self.heatmap_size = heatmap_size

    def __len__(self):
        return len(self.images)  # ✅ this is the missing method


    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        img_filename = os.path.basename(img_path)
        # keypoints_filename = img_filename.replace("_", "_keypoints_").replace(".png", ".json")
        keypoints_filename = img_filename.replace("img", "keypoints").replace("_0.png", ".json")
        keypoints_path = os.path.join(self.keypoints_dir, keypoints_filename)

        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        image_tensor = self.transform(image)  # This resizes to e.g. 256×256

        # load raw keypoints
        with open(keypoints_path, 'r') as f:
            keypoints_data = json.load(f)
        keypoints = np.stack([np.array(v) for v in keypoints_data.values()])  # shape (K, 2)

        # ✅ Normalize to [0, 1] by original image size
        keypoints[:, 0] /= w
        keypoints[:, 1] /= h

        # flatten to [2K]
        keypoints = keypoints.flatten()
        keypoints = torch.tensor(keypoints, dtype=torch.float32)

        return image_tensor, keypoints

