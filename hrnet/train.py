import torch
from torch.utils.data import DataLoader
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
    save_dir: str = "checkpoints"
):
    # Init WandB
    wandb.init(
        project="se3-pose-estimation",
        name=f"hrnet_run_bs{batch_size}_lr{learning_rate}",
        config={
            "batch_size": batch_size,
            "epochs": num_epochs,
            "learning_rate": learning_rate,
            "image_dir": image_dir,
            "pose_dir": pose_dir
        }
    )

    # Dataset and DataLoader
    dataset = TagPoseDataset(image_dir, pose_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Model, loss, optimizer
    model = HRNetSE3().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for imgs, poses in dataloader:
            imgs, poses = imgs.cuda(), poses.cuda()
            preds = model(imgs)

            loss = se3_loss(preds, poses)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1:02d}] Loss: {avg_loss:.6f}")

        # ✅ Log to WandB
        wandb.log({"epoch": epoch + 1, "train_loss": avg_loss})

        # Save checkpoint
        torch.save(model.state_dict(), Path(save_dir) / f"hrnet_pose_epoch{epoch+1:02d}.pth")

    wandb.finish()


if __name__ == "__main__":
    train(
        image_dir="./output/sdg_markers_20250323-153257/rgb",
        pose_dir="./output/sdg_markers_20250323-153257/pose/",
        batch_size=64,
        num_epochs=50,
        learning_rate=1e-5,
        save_dir="./checkpoints"
    )
