import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
from torchvision import transforms
from .model import HRNetSE3
from .dataset import TagPoseDataset
from .utils import se3_loss
from pathlib import Path
import os
import wandb

def train(
    image_dir: str,
    pose_dir: str,
    batch_size: int = 16,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    save_dir: str = "checkpoints",
    val_split: float = 0.2,
    load_model_path: str = None  
):
    # Init WandB
    wandb.init(
        project="se3-pose-estimation",
        name=f"hrnet_run_bs{batch_size}_lr{learning_rate}" if not load_model_path else f"resume_{Path(load_model_path).stem}",
        config={
            "batch_size": batch_size,
            "epochs": num_epochs,
            "learning_rate": learning_rate,
            "image_dir": image_dir,
            "pose_dir": pose_dir,
            "load_model_path": load_model_path
        }
    )

    # Dataset and DataLoader
    full_dataset = TagPoseDataset(image_dir, pose_dir)
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model, loss, optimizer
    model = HRNetSE3().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ✅ Load pretrained weights if provided
    if load_model_path is not None:
        print(f"[INFO] Loading model from {load_model_path}")
        checkpoint = torch.load(load_model_path, map_location='cuda')
        model.load_state_dict(checkpoint)

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for imgs, poses in train_loader:
            imgs, poses = imgs.cuda(), poses.cuda()
            preds = model(imgs)

            loss = se3_loss(preds, poses)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation step
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for imgs, poses in val_loader:
                imgs, poses = imgs.cuda(), poses.cuda()
                preds = model(imgs)
                loss = se3_loss(preds, poses)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"[Epoch {epoch+1:02d}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # ✅ Log to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })

        # Save checkpoint
        torch.save(model.state_dict(), Path(save_dir) / f"hrnet_pose_epoch{epoch+1:02d}.pth")

    wandb.finish()

if __name__ == "__main__":
    main_dir = "/media/rp/Elements1/abhay_ws/marker_detection_failure_recovery/synthetic_data_generation/output/sdg_markers_20250422-205424/"
    train(
        image_dir=f"{main_dir}/rgb",
        pose_dir=f"{main_dir}/pose",
        batch_size=256,
        num_epochs=1_000_000,
        learning_rate=1e-7,
        save_dir="./hrnet/checkpoints",
        load_model_path= None, # ✅ or None
    )
0