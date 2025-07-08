import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from vit_keypoints_model.model import ViTKeypointRegressor
import os
import wandb
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToPILImage
import matplotlib

matplotlib.use('Agg')
from keypoints_model.utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    # evaluate_l1_loss,
    overlay_points_on_image,
)
from vit_keypoints_model.utils import get_vit_loaders, evaluate_l1_loss

# === Hyperparameters ===
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
NUM_EPOCHS = 100_000 
NUM_WORKERS = 8
IMAGE_HEIGHT = 224  # ViT requires 224x224 input size
IMAGE_WIDTH = 224
PIN_MEMORY = True
LOAD_MODEL = True
LOAD_PATH = "./vit_keypoints_model/checkpoints/vit_keypoints_model.pth.tar"
MAIN_DIR = "./segmentation_model/data/data_20250330-013534_reaugmented/"
TRAIN_IMG_DIR = os.path.join(MAIN_DIR, "train", "roi_rgb_reaugmented")
TRAIN_KEYPOINTS_DIR = os.path.join(MAIN_DIR, "train", "roi_keypoints")
VAL_IMG_DIR = os.path.join(MAIN_DIR, "val", "roi_rgb_reaugmented")
VAL_KEYPOINTS_DIR = os.path.join(MAIN_DIR, "val", "roi_keypoints")
SAVE_DIR = "./vit_keypoints_model/checkpoints/vit_keypoints_model.pth.tar" 

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    running_loss = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE).to(torch.float32)
        targets = targets.float().to(DEVICE)

        with torch.amp.autocast(device_type=DEVICE):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(loader)
    return avg_loss

def main():
    wandb.init(project="vit-keypoints", name="vit-keypoints", config={
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "image_size": (IMAGE_HEIGHT, IMAGE_WIDTH),
        "train_data": TRAIN_IMG_DIR,
        "val_data": VAL_IMG_DIR,
        "loss": "L1Loss",
    })

    train_transform = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    val_transform = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    model = ViTKeypointRegressor(num_keypoints=11**2).to(DEVICE)
    loss_fn = nn.L1Loss()  # Using L1 loss for keypoint regression
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_vit_loaders(
        train_img_dir=TRAIN_IMG_DIR,
        train_keypoints_dir=TRAIN_KEYPOINTS_DIR,
        val_img_dir=VAL_IMG_DIR,
        val_keypoints_dir=VAL_KEYPOINTS_DIR,
        batch_size=BATCH_SIZE,
        train_transform=train_transform,
        val_transform=val_transform,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


    if LOAD_MODEL:
        checkpoint = torch.load(LOAD_PATH)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 0)
        print(f"[INFO] Resuming from epoch {start_epoch}")
    else:
        start_epoch = 0

    scaler = torch.amp.GradScaler()

    for epoch in range(start_epoch, NUM_EPOCHS):
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        wandb.log({"train/epoch_loss": train_loss}, step=epoch)

        val_loss = evaluate_l1_loss(val_loader, model, device=DEVICE)
        wandb.log({"val/mae_loss": val_loss}, step=epoch)
        print(f"EPOCH: {epoch}. Validation MAE: {val_loss:.4f}")

        save_checkpoint({
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
        }, SAVE_DIR)

    wandb.finish()

if __name__ == "__main__":
    main()