import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
from pathlib import Path
import os
import wandb

# from hrnet.model import HRNetModel, HRNetKeypoint, HRNetCorners  # updated HRNet backbone
from hrnet.keypoint_hrnet import HRNetKeypoint, HRNetCorners
from hrnet.utils import heatmap_loss
from hrnet.dataset import MarkersDataset  # your keypoint dataset

def train(
    train_image_dir: str,
    train_pose_dir: str,
    val_image_dir: str,
    val_pose_dir: str,
    batch_size: int = 16,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    save_dir: str = "checkpoints",
    load_model_path: str = None  
):
    wandb.init(
        project="keypoint-hrnet",
        name=f"hrnet_keypoints_bs{batch_size}_lr{learning_rate}" if not load_model_path else f"resume_{Path(load_model_path).stem}",
        config={
            "batch_size": batch_size,
            "epochs": num_epochs,
            "learning_rate": learning_rate,
            "train_image_dir": train_image_dir,
            "train_pose_dir": train_pose_dir,
            "val_image_dir": val_image_dir,
            "val_pose_dir": val_pose_dir,
            "load_model_path": load_model_path
        }
    )

    # -------- DATASETS --------
    train_dataset = MarkersDataset(train_image_dir, train_pose_dir, heatmap_size=None, indices=[0,10,110,120])
    val_dataset = MarkersDataset(val_image_dir, val_pose_dir, heatmap_size=None, indices=[0,10,110,120])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # -------- MODEL + LOSS --------
    NUM_KEYPOINTS = train_dataset[0][1].shape[0] // 2
    # model = HRNetKeypoint(num_keypoints=NUM_KEYPOINTS).cuda()
    model = HRNetCorners().cuda()
    # loss_fn = heatmap_loss
    loss_fn = nn.MSELoss()  # Mean Squared Error for keypoint regression 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # -------- LOAD MODEL --------
    if load_model_path is not None:
        print(f"[INFO] Loading model from {load_model_path}")
        checkpoint = torch.load(load_model_path, map_location='cuda')
        model.load_state_dict(checkpoint)

    os.makedirs(save_dir, exist_ok=True)

    # -------- TRAINING LOOP --------
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for imgs, keypoints in train_loader:
            imgs, keypoints = imgs.cuda(), keypoints.cuda()  # keypoints: [B, 2K]
            preds = model(imgs)  # output: [B, 2K]

            loss = loss_fn(preds, keypoints)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # -------- VALIDATION LOOP --------
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for imgs, keypoints in val_loader:
                imgs, keypoints = imgs.cuda(), keypoints.cuda()
                preds = model(imgs)
                loss = loss_fn(preds, keypoints)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"[Epoch {epoch+1:02d}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })

        # Optionally save only every N epochs
        if (epoch + 1) % 1 == 0 or (epoch + 1) == num_epochs:
            # torch.save(model.state_dict(), Path(save_dir) / f"hrnet_keypoint_epoch{epoch+1:06d}.pth")
            torch.save(model.state_dict(), Path(save_dir) / f"hrnet_corners_epoch{epoch+1:06d}.pth")

    wandb.finish()

if __name__ == "__main__":
    main_dir = "./segmentation_model/data/data_20250330-013534/"

    train(
        train_image_dir=f"{main_dir}/train/rgb",
        train_pose_dir=f"{main_dir}/train/keypoints",
        val_image_dir=f"{main_dir}/val/rgb",
        val_pose_dir=f"{main_dir}/val/keypoints",
        batch_size=32,
        num_epochs=1_000_000,
        learning_rate=1e-3,
        save_dir="./hrnet/checkpoints",
        load_model_path=None,
    )
