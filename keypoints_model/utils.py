import torch
import torchvision
from dataset import MarkersDataset
from torch.utils.data import DataLoader
import torch.nn as nn 

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_keypointsdir,
    val_dir,
    val_keypointsdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = MarkersDataset(
        image_dir=train_dir,
        keypoints_dir=train_keypointsdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = MarkersDataset(
        image_dir=val_dir,
        keypoints_dir=val_keypointsdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def evaluate_mse_loss(loader, model, device): 
    model.eval() 
    total_mse_loss = 0.0 
    with torch.no_grad(): 
        for data, targets in loader: 
            data, targets = data.to(device), targets.to(device) 
            data = data.to(torch.float32).permute(0,3,1,2) 
            outputs = model(data) 

            # compute MSE loss 
            mse_loss = nn.MSELoss()(outputs, targets) 
            total_mse_loss += mse_loss.item() 
        avg_mse_loss = total_mse_loss / len(loader) 
        return avg_mse_loss

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(x, f"{folder}/rgb_{idx}.png", normalize=True)
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/seg_{idx}.png", normalize=True)

    model.train()