import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
import wandb

from keypoints_model.model import RegressorMobileNetV3_from_scratch
from keypoints_model.utils import (
    load_checkpoint, save_checkpoint, get_loaders,
    evaluate_l1_loss, overlay_points_on_image
)

import matplotlib
matplotlib.use('Agg')

# -------------------- Utility Functions --------------------

def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
    return ((img_tensor * std + mean) * 255).clamp(0, 255).byte()

def save_predictions_as_images(loader, model, folder, device="cuda", max_batches=1):
    os.makedirs(folder, exist_ok=True)
    model.eval()
    to_pil = ToPILImage()

    # Define normalization transform (same as training)
    normalize = A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=255.0
    )

    with torch.no_grad():
        for idx, (x_np, _) in enumerate(loader):  # x_np is [B, H, W, C], still numpy-based

            batch_size = x_np.shape[0]
            for j in range(batch_size):
                raw_img = x_np[j].cpu().numpy()  # shape: [H, W, C], dtype: float32 or uint8

                # Store original image before normalization
                original_img = raw_img.astype(np.uint8).copy()

                # Normalize using Albumentations
                norm_result = normalize(image=raw_img)
                norm_img = norm_result["image"]  # shape: [H, W, C]
                norm_img_tensor = torch.tensor(norm_img).permute(2, 0, 1).unsqueeze(0).float().to(device)

                # Model prediction
                pred = model(norm_img_tensor)[0].detach().cpu().numpy().reshape(-1, 2)

                # Overlay on original image (not normalized)
                keypoints_img = overlay_points_on_image(image=original_img, pixel_points=pred, radius=1)

                save_path = os.path.join(folder, f"pred_{idx * batch_size + j}.png")
                plt.imshow(keypoints_img)
                plt.axis('off')
                plt.savefig(save_path)
                plt.close()

                wandb.log({f"predictions/image_{idx * batch_size + j}": wandb.Image(save_path)})

            if idx + 1 >= max_batches:
                break

def train_one_epoch(loader, model, optimizer, loss_fn, scaler, device):
    model.train()
    loop = tqdm(loader, desc="Training")
    total_loss = 0
    total_samples = 0

    for data, targets in loop:
        data = data.to(device).float().permute(0, 3, 1, 2)
        targets = targets.float().to(device)

        with torch.amp.autocast(device_type=device):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * data.size(0)
        total_samples += data.size(0)

        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / total_samples
    return avg_loss


def evaluate_loss(loader, model, loss_fn, device="cuda"):
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device).float().permute(0, 3, 1, 2)
            targets = targets.float().to(device)
            predictions = model(data)
            batch_loss = loss_fn(predictions, targets)
            total_loss += batch_loss.item() * data.size(0)
            total_samples += data.size(0)

    return total_loss / total_samples


# -------------------- Main Training Function --------------------

def main():
    # ---------- Config Dictionary ----------
    config = {
        "learning_rate": 1e-3,
        "batch_size": 128,
        "num_epochs": 10000,
        "num_workers": 32,
        "image_height": 480,
        "image_width": 640,
        "test_frequency": 10,
        "pin_memory": True,
        "load_model": True,
        "num_epoch_dont_save": 0,
        "data_dir": "./segmentation_model/data/data_20250607-214821/",
        "checkpoint_path": "./keypoints_model/checkpoints/my_checkpoint.pth.tar",
        "load_checkpoint_path": "./keypoints_model/checkpoints/my_checkpoint.pth.tar_epoch_147.pth.tar",
        "save_dir": "./keypoints_model/saved_images/",
        "wandb_project": "keypoint-regression",
        "wandb_run_name": "mobilenetv3-keypoints",
        "weight_decay": 1e-4,
        "dropout_p": 0.2,
    }

    # ---------- WandB Init ----------
    wandb.init(
        project=config["wandb_project"],
        name=config["wandb_run_name"],
        config=config
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_transform = A.Compose([
        A.Resize(height=config["image_height"], width=config["image_width"]),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(height=config["image_height"], width=config["image_width"]),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0),
        ToTensorV2(),
    ])

    train_loader, val_loader = get_loaders(
        os.path.join(config["data_dir"], "train", "rgb"),
        os.path.join(config["data_dir"], "train", "keypoints"),
        os.path.join(config["data_dir"], "val", "rgb"),
        os.path.join(config["data_dir"], "val", "keypoints"),
        config["batch_size"],
        train_transform,
        val_transform,
        config["num_workers"],
        config["pin_memory"]
    )

    # model = RegressorMobileNetV3_with_dropouts(dropout_p=config["dropout_p"]).to(device)
    model = RegressorMobileNetV3_from_scratch().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler()

    if config["load_model"]:
        load_checkpoint(torch.load(config["load_checkpoint_path"]), model)

    best_loss = float("inf")

    for epoch in range(config["num_epochs"]):
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, device)
        wandb.log({"train/loss": train_loss, "epoch": epoch})

        val_loss = evaluate_loss(val_loader, model, loss_fn, device=device)
        wandb.log({"val/loss": val_loss, "epoch": epoch})
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_loss and epoch > config["num_epoch_dont_save"]:
            best_loss = val_loss
            save_checkpoint({
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, config["checkpoint_path"])

        if epoch % 5 == 0:
            # save checkpoint with epoch number
            save_checkpoint({
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, f"{config['checkpoint_path']}_epoch_{epoch}.pth.tar")

        if epoch % config["test_frequency"] == 0:
            save_predictions_as_images(val_loader, model, folder=config["save_dir"], device=device)

if __name__ == "__main__":
    main()