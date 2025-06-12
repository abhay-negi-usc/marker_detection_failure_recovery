import torch 
import albumentations as A 
from albumentations.pytorch import ToTensorV2 
from tqdm import tqdm 
import torch.nn as nn 
import torch.optim as optim 
from keypoints_model.model import RegressorMobileNetV3 
from keypoints_model.utils import (
    load_checkpoint, 
    save_checkpoint, 
    get_loaders, 
    evaluate_mse_loss,
    evaluate_l1_loss, 
    overlay_points_on_image,
)
import os 
import wandb
from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np 
from torchvision.transforms import ToPILImage
import matplotlib
matplotlib.use('Agg')


# === Hyperparameters ===
LEARNING_RATE = 1e-4 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
BATCH_SIZE = 128          
NUM_EPOCHS = 1000000 
num_epoch_dont_save = 0 
NUM_WORKERS = 24 
IMAGE_HEIGHT = 128 
IMAGE_WIDTH = 128 
TEST_FREQUENCY = 100 
PIN_MEMORY = True 
LOAD_MODEL = True 
# LOAD_PATH = "./keypoints_model/checkpoints/keypoints_model_reaugmented_training.pth.tar"
LOAD_PATH = "./keypoints_model/checkpoints/keypoints_model_reaugmented_training.pth (copy).tar"
MAIN_DIR = "./segmentation_model/data/data_20250330-013534_reaugmented/" 
TRAIN_IMG_DIR = os.path.join(MAIN_DIR, "train", "roi_rgb_reaugmented") 
TRAIN_KEYPOINTS_DIR = os.path.join(MAIN_DIR, "train", "roi_keypoints")
VAL_IMG_DIR = os.path.join(MAIN_DIR, "val", "roi_rgb_reaugmented")  
VAL_KEYPOINTS_DIR = os.path.join(MAIN_DIR, "val", "roi_keypoints")

def train_fn(loader, model, optimizer, loss_fn, scaler): 
    loop = tqdm(loader)
    running_loss = 0

    for batch_idx, (data, targets) in enumerate(loop): 
        data = data.to(device=DEVICE).to(torch.float32).permute(0,3,1,2) 
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

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    to_pil = ToPILImage()
    os.makedirs(folder, exist_ok=True)
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device).to(torch.float32).permute(0,3,1,2) 
        with torch.no_grad():
            preds = model(x)

        for j in range(len(preds)): 
            pred = preds[j].cpu().numpy().reshape(-1, 2)
            # img_rgb = to_pil(x[j])
            # keypoints_image = overlay_points_on_image(image=np.array(img_rgb), pixel_points=pred, radius=1)
            img_array = (x[j].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            keypoints_image = overlay_points_on_image(image=img_array, pixel_points=pred, radius=1)
            save_path = os.path.join(folder, f"pred_{idx}_{j}.png")
            plt.imshow(keypoints_image)
            plt.axis('off')
            plt.title(f'Prediction {idx}_{j}')
            plt.savefig(save_path)
            plt.close()
            wandb.log({f"predictions/pred_{idx}_{j}": wandb.Image(save_path)})

def main():
    wandb.init(project="fiducial-keypoints", name="mobilenetv3-reaugmented", config={
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "image_size": (IMAGE_HEIGHT, IMAGE_WIDTH),
        "train_data": TRAIN_IMG_DIR,
        "val_data": VAL_IMG_DIR,
        "loss": "L1Loss",
        "resume":"auto", 
    })

    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    model = RegressorMobileNetV3().to(DEVICE)
    loss_fn = nn.L1Loss()  # Using L1 loss for keypoint regression
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR, TRAIN_KEYPOINTS_DIR,
        VAL_IMG_DIR, VAL_KEYPOINTS_DIR,
        BATCH_SIZE, train_transform, val_transform,
        NUM_WORKERS, PIN_MEMORY
    )

    if LOAD_MODEL:
        checkpoint = torch.load(LOAD_PATH)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 0)
        print(f"[INFO] Resuming from epoch {start_epoch}")
    else:
        start_epoch = 0

    save_count = 0
    best_mae_loss = float('inf')
    scaler = torch.amp.GradScaler()

    for epoch in range(start_epoch, NUM_EPOCHS):
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        wandb.log({"train/epoch_loss": train_loss}, step=epoch)

        new_mae_loss = evaluate_l1_loss(val_loader, model, device=DEVICE)
        wandb.log({"val/mae_loss": new_mae_loss}, step=epoch)
        print(f"EPOCH: {epoch}. MAE: {new_mae_loss:.4f}") 

        if new_mae_loss < best_mae_loss and epoch > num_epoch_dont_save:
            best_mae_loss = new_mae_loss
            save_checkpoint({
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,  # resume from next epoch
            }, "./keypoints_model/checkpoints/keypoints_model_reaugmented_training.pth.tar")

            # if save_count > TEST_FREQUENCY:
            #     save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=DEVICE)
            #     save_count = 0

        save_count += 1

    wandb.finish()

if __name__ == "__main__": 
    main()