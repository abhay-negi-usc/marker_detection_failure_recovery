import os
import torch
import numpy as np
from torchvision.utils import draw_keypoints
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from hrnet.model import HRNetKeypoint
from keypoints_model.dataset import MarkersDataset

@torch.no_grad()
def evaluate_model(
    image_dir,
    keypoints_dir,
    model_path,
    save_dir,
    num_keypoints,
    device='cuda'
):
    os.makedirs(save_dir, exist_ok=True)

    # Load model
    model = HRNetKeypoint(num_keypoints=num_keypoints).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load dataset
    dataset = MarkersDataset(image_dir, keypoints_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for idx, (img_tensor, _) in enumerate(tqdm(dataloader)):
        img_tensor = img_tensor.permute(0, 3, 1, 2)  # [B, H, W, C] → [B, C, H, W]
        img_tensor = img_tensor.float() / 255.0  # Convert to float32 normalized
        img_tensor = img_tensor.to(device).float()  # Ensure float32
        pred_flat = model(img_tensor)[0].cpu()  # shape: [2K]
        keypoints = pred_flat.view(num_keypoints, 2)  # shape: [K, 2] with normalized coords in [0,1]

        # Get image dimensions
        img_vis = (img_tensor[0].cpu() * 255).clamp(0, 255).byte()
        # img_vis = (img_tensor[0].cpu() * 255).byte()
        C, H_img, W_img = img_vis.shape

        # ✅ Rescale from [0,1] to image coordinates
        keypoints[:, 0] *= W_img  # x
        keypoints[:, 1] *= H_img  # y
        keypoints = keypoints.int()

        # Draw and save
        vis_img = draw_keypoints(img_vis, keypoints.unsqueeze(0), colors="red", radius=4)
        vis_img = to_pil_image(vis_img)
        vis_img.save(Path(save_dir) / f"vis_{idx:04d}.png")



if __name__ == "__main__":
    eval_config = {
        # "image_dir": "./segmentation_model/data/sdg_markers_20250330-013534_val/val/rgb",
        # "keypoints_dir": "./segmentation_model/data/sdg_markers_20250330-013534_val/val/keypoints",
        "image_dir": "./segmentation_model/data/data_20250603-201339/val/rgb",
        "keypoints_dir": "./segmentation_model/data/data_20250603-201339/val/keypoints",
        "model_path": "./hrnet/checkpoints/hrnet_keypoint_epoch000014.pth",
        "save_dir": "./eval_outputs/keypoints_vis",
        # "num_keypoints": (10 + 1)**2,
        "num_keypoints": (6 + 1)**2,
    }

    evaluate_model(**eval_config)
