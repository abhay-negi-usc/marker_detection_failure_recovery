import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import json 
from PIL import Image
from PIL import Image
import numpy as np
import json
import os
import torch
from torch.utils.data import Dataset
import torch.nn as nn

class KeypointsDataset(Dataset):
    def __init__(self, image_dir, keypoints_dir, transform=None):
        self.image_dir = image_dir
        self.keypoints_dir = keypoints_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_filename = self.image_files[index]
        img_path = os.path.join(self.image_dir, img_filename)

        keypoints_filename = img_filename.replace("roi", "roi_keypoints").replace(".png", ".json")
        keypoints_path = os.path.join(self.keypoints_dir, keypoints_filename)

        # Load image
        image = np.array(Image.open(img_path).convert("RGB")).astype(np.float32)

        # Load keypoints from JSON
        with open(keypoints_path, 'r') as f:
            keypoints_data = json.load(f)

        # Sort keys to ensure correct order
        keypoints_list = [keypoints_data[key] for key in sorted(keypoints_data.keys())]
        keypoints = np.array(keypoints_list, dtype=np.float32)

        # Prepare keypoints for Albumentations
        keypoints_tuples = [tuple(pt) for pt in keypoints]

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, keypoints=keypoints_tuples)
            image = augmented["image"]
            keypoints = np.array(augmented["keypoints"], dtype=np.float32)

        # Flatten keypoints for regression output
        keypoints_flat = keypoints.flatten()

        return image, torch.tensor(keypoints_flat, dtype=torch.float32)



from torch.utils.data import DataLoader

def get_vit_loaders(
    train_img_dir,
    train_keypoints_dir,
    val_img_dir,
    val_keypoints_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = KeypointsDataset(
        image_dir=train_img_dir,
        keypoints_dir=train_keypoints_dir,
        transform=train_transform,
    )

    val_ds = KeypointsDataset(
        image_dir=val_img_dir,
        keypoints_dir=val_keypoints_dir,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader

def evaluate_l1_loss(loader, model, device):
    model.eval()
    total_l1_loss = 0.0
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)

            # Do NOT permute again!
            data = data.to(torch.float32)

            outputs = model(data)
            l1_loss = nn.L1Loss()(outputs, targets)
            total_l1_loss += l1_loss.item()
    avg_l1_loss = total_l1_loss / len(loader)
    return avg_l1_loss
