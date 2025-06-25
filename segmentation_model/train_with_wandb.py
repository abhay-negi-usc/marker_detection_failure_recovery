import torch 
import albumentations as A 
from albumentations.pytorch import ToTensorV2 
from tqdm import tqdm 
import torch.nn as nn 
import torch.optim as optim 
from segmentation_model.model import UNET, UNETWithDropout
from segmentation_model.utils import (
    load_checkpoint, 
    save_checkpoint, 
    get_loaders, 
    check_accuracy,
    save_predictions_as_imgs,
)
import os 
import wandb

DATA_DIR = "./segmentation_model/data/data_20250607-214821/" 
TRAIN_IMG_DIR = f"{DATA_DIR}/train/rgb"
TRAIN_MASK_DIR = f"{DATA_DIR}/train/seg"
VAL_IMG_DIR = f"{DATA_DIR}/val/rgb"
VAL_MASK_DIR = f"{DATA_DIR}/val/seg"

LEARNING_RATE = 1e-5 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
BATCH_SIZE = 8 
NUM_EPOCHS = 1000 
num_epoch_dont_save = 0 
NUM_WORKERS = 8
IMAGE_HEIGHT = 480 
IMAGE_WIDTH = 640 
PIN_MEMORY = True 
LOAD_MODEL = True                          

def train_fn(loader, model, optimizer, loss_fn, scaler, epoch): 
    loop = tqdm(loader) # progress bar 
    epoch_loss = 0

    for batch_idx, (data, targets) in enumerate(loop): 
        data = data.to(device=DEVICE) 
        targets = targets.float().unsqueeze(1).to(device=DEVICE) 

        # forward 
        with torch.amp.autocast(device_type=DEVICE): 
            predictions = model(data) 
            loss = loss_fn(predictions, targets) 

        # backward 
        optimizer.zero_grad() 
        scaler.scale(loss).backward() 
        scaler.step(optimizer)
        scaler.update() 

        # accumulate loss
        epoch_loss += loss.item()

        # Log batch loss to wandb
        wandb.log({"batch_loss": loss.item(), "epoch": epoch, "batch_idx": batch_idx})

        # update tqdm loop 
        loop.set_postfix(loss=loss.item())         

        # Save checkpoint every 100 batches
        if batch_idx % 1000 == 0: 
            save_checkpoint({
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            # }, f"./segmentation_model/models/my_checkpoint_multimarker_epoch_{epoch}_batch_{batch_idx}.pth.tar")
            }, f"/home/nom4d/marker_ws/segmentation_checkpoints/my_checkpoint_multimarker_epoch_{epoch}_batch_{batch_idx}.pth.tar")

    # Log average training loss to wandb
    avg_loss = epoch_loss / len(loader)
    wandb.log({"train_loss": avg_loss, "epoch": epoch})

def main(): 
    train_transform = A.Compose(
        [
            # A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            # A.Rotate(limit=35, p=1.0),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.1),
            A.Normalize(
                max_pixel_value=1.0,
            ),
            ToTensorV2(), 
        ]
    )

    val_transform = A.Compose(
        [
            # A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            # A.Rotate(limit=35, p=1.0),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.1),
            A.Normalize(
                max_pixel_value=1.0,
            ),
            ToTensorV2(), 
        ]
    )

    model = UNETWithDropout(in_channels=3, out_channels=1).to(DEVICE) 
    
    loss_fn = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR, 
        TRAIN_MASK_DIR,
        VAL_IMG_DIR, 
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY
    )

    if LOAD_MODEL: 
        # load_checkpoint(torch.load("./segmentation_model/models/my_checkpoint_20250329.pth.tar"), model)
        load_checkpoint(torch.load("/home/nom4d/marker_ws/segmentation_checkpoints/my_checkpoint_multimarker_epoch_0_batch_20000.pth.tar"), model)
        accuracy = 0.0
    else: 
        accuracy = 0.0 

    # check_accuracy(val_loader, model, device=DEVICE) 

    scaler = torch.amp.GradScaler()
    for epoch in range(NUM_EPOCHS): 
        train_fn(train_loader, model, optimizer, loss_fn, scaler, epoch)

        # check accuracy 
        new_accuracy = check_accuracy(val_loader, model, device=DEVICE)  # FIXME: update to output dice score

        # Log accuracy to wandb
        wandb.log({"val_accuracy": new_accuracy, "epoch": epoch})

        if epoch == 0: 
            accuracy = new_accuracy

        if new_accuracy > accuracy and epoch > num_epoch_dont_save: 
            accuracy = new_accuracy 
            save_checkpoint({
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            # }, f"./segmentation_model/models/my_checkpoint_multimarker_epoch_{epoch}.pth.tar")  # Save with epoch and accuracy
            }, f"/home/nom4d/marker_ws/segmentation_checkpoints/my_checkpoint_multimarker_epoch_{epoch}.pth.tar")  # Save with epoch and accuracy

            # Optionally save some predictions
            # saved_images_dir = "saved_images/"
            # os.makedirs(saved_images_dir, exist_ok=True)
            # save_predictions_as_imgs(
            #     val_loader, model, folder=saved_images_dir, device=DEVICE
            # )

if __name__ == "__main__":
    # Initialize wandb
    wandb.init(
        config={
            "wandb_key": "9336a0a286df1f392970fb1192519ef0191ba865",
            "wandb_project": "multimarker_segmentation", 
            "wandb_entity": "abhay-negi-usc-university-of-southern-california", 
            "learning_rate": 1e-4,
            "batch_size": 128,
            "epochs": 1_000_000,
            "image_height": 480,
            "image_width": 640,
            "num_workers": 8,
            "pin_memory": True,
            "train_img_dir": TRAIN_IMG_DIR,
            "train_mask_dir": TRAIN_MASK_DIR,
            "val_img_dir": VAL_IMG_DIR,
            "val_mask_dir": VAL_MASK_DIR,
        }
    )

    main()
