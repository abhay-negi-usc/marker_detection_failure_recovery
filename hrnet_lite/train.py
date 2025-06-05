import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torchvision import transforms
from pathlib import Path
import os
import wandb
from PIL import Image
import json
import numpy as np
from hrnet_lite.model import LiteHRNet

# ------------ MODEL ------------
class HRNetLiteKeypoint(nn.Module):
    def __init__(self, num_keypoints, input_width, input_height):
        super().__init__()
        self.backbone = LiteHRNet(num_keypoints=num_keypoints)
        self.input_width = input_width
        self.input_height = input_height

    def forward(self, x):
        return self.backbone(x)

# ------------ DATASET ------------
class MarkersDataset(Dataset):
    def __init__(self, image_dir, keypoint_dir, input_width, input_height, transform=None, indices=None):
        self.image_paths = []
        self.keypoint_paths = []
        self.input_width = input_width
        self.input_height = input_height
        self.indices = indices

        for img_path in sorted(Path(image_dir).glob("*.png")):
            keypoints_filename = img_path.name.replace("img", "keypoints").replace("_0.png", ".json")
            keypoint_path = Path(keypoint_dir) / keypoints_filename
            if keypoint_path.exists():
                self.image_paths.append(img_path)
                self.keypoint_paths.append(keypoint_path)

        self.transform = transform or transforms.Compose([
            transforms.Resize((input_height, input_width)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        w, h = image.size
        image_tensor = self.transform(image)

        with open(self.keypoint_paths[idx], 'r') as f:
            keypoints_data = json.load(f)
        keypoints = np.stack([np.array(v) for v in keypoints_data.values()])
        if self.indices is not None:
            keypoints = keypoints[self.indices]

        keypoints[:, 0] /= w
        keypoints[:, 1] /= h

        keypoints = keypoints.flatten()
        keypoints = torch.tensor(keypoints, dtype=torch.float32)

        return image_tensor, keypoints

# ------------ TRAINING SCRIPT ------------
def train(
    train_image_dir, train_keypoint_dir,
    val_image_dir, val_keypoint_dir,
    batch_size=32, num_epochs=100,
    learning_rate=1e-4, save_dir="checkpoints",
    load_model_path=None,
    input_width=256,
    input_height=192
):
    wandb.init(project="lite-hrnet-keypoint")

    train_dataset = MarkersDataset(train_image_dir, train_keypoint_dir, input_width, input_height)
    val_dataset = MarkersDataset(val_image_dir, val_keypoint_dir, input_width, input_height)

    if len(train_dataset) == 0:
        raise RuntimeError(f"No training data found in {train_image_dir} and {train_keypoint_dir}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    num_keypoints = train_dataset[0][1].shape[0] // 2
    model = HRNetLiteKeypoint(num_keypoints, input_width, input_height).cuda()

    if load_model_path:
        model.load_state_dict(torch.load(load_model_path))

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for imgs, keypoints in train_loader:
            imgs, keypoints = imgs.cuda(), keypoints.cuda()
            preds = model(imgs)
            loss = loss_fn(preds * input_width, keypoints * input_width)
            loss_y = loss_fn(preds * input_height, keypoints * input_height)
            loss = (loss + loss_y) / 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, keypoints in val_loader:
                imgs, keypoints = imgs.cuda(), keypoints.cuda()
                preds = model(imgs)
                loss = loss_fn(preds * input_width, keypoints * input_width)
                loss_y = loss_fn(preds * input_height, keypoints * input_height)
                loss = (loss + loss_y) / 2
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} px, Val Loss = {avg_val_loss:.4f} px")
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), Path(save_dir) / f"lite_hrnet_epoch{epoch+1:03d}.pth")

    wandb.finish()

# Example call
if __name__ == "__main__":
    data_dir = "/home/anegi/abhay_ws/marker_detection_failure_recovery/segmentation_model/data/data_20250603-201339/"
    train(
        train_image_dir=f"{data_dir}/train/rgb",
        train_keypoint_dir=f"{data_dir}/train/keypoints",
        val_image_dir=f"{data_dir}/val/rgb",
        val_keypoint_dir=f"{data_dir}/val/keypoints",
        batch_size=64,
        num_epochs=10000,
        learning_rate=1e-3,
        save_dir="./checkpoints",
        load_model_path=None,
        input_width=640,
        input_height=480
    )
