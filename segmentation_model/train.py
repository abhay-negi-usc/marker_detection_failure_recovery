import torch 
import albumentations as A 
from albumentations.pytorch import ToTensorV2 
from tqdm import tqdm 
import torch.nn as nn 
import torch.optim as optim 
from model import UNET, UNETWithDropout
from utils import (
    load_checkpoint, 
    save_checkpoint, 
    get_loaders, 
    check_accuracy,
    save_predictions_as_imgs,
)
import os 

LEARNING_RATE = 1e-8 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
BATCH_SIZE = 12  
NUM_EPOCHS = 1000 
num_epoch_dont_save = 0 
NUM_WORKERS = 30 
IMAGE_HEIGHT = 480 
IMAGE_WIDTH = 640 
PIN_MEMORY = True 
LOAD_MODEL = True                          
TRAIN_IMG_DIR = "/home/anegi/abhay_ws/marker_detection_failure_recovery/segmentation_model/data/data_20250327-173029/train/rgb"
TRAIN_MASK_DIR = "/home/anegi/abhay_ws/marker_detection_failure_recovery/segmentation_model/data/data_20250327-173029/train/seg" 
VAL_IMG_DIR = "/home/anegi/abhay_ws/marker_detection_failure_recovery/segmentation_model/data/data_20250327-173029/val/rgb"
VAL_MASK_DIR = "/home/anegi/abhay_ws/marker_detection_failure_recovery/segmentation_model/data/data_20250327-173029/val/seg" 
    
def train_fn(loader, model, optimizer, loss_fn, scaler): 
    loop = tqdm(loader) # progress bar 

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

        # update tqdm loop 
        loop.set_postfix(loss=loss.item())         

def main(): 
    train_transform = A.Compose(
        [
            # A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            # A.Rotate(limit=35, p=1.0),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.1),
            A.Normalize(
                # mean=[0.0, 0.0, 0.0],
                # std=[1.0, 1.0, 1.0],
                # max_pixel_value=255.0,
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
                # mean=[0.0, 0.0, 0.0],
                # std=[1.0, 1.0, 1.0],
                # max_pixel_value=255.0,
                max_pixel_value=1.0,
            ),
            ToTensorV2(), 
        ]
    )

    # model = UNET(in_channels=3, out_channels=1).to(DEVICE)
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
        load_checkpoint(torch.load("./segmentation_model/models/my_checkpoint.pth.tar"), model)
        # load_checkpoint(torch.load("./my_checkpoint.pth.tar"), model)
        accuracy = 0.965
    else: 
        accuracy = 0.0 

    # check_accuracy(val_loader, model, device=DEVICE) 

    scaler = torch.amp.GradScaler()
    for epoch in range(NUM_EPOCHS): 
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model 
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(), 
        }

        # check accuracy 
        new_accuracy = check_accuracy(val_loader, model, device=DEVICE) # FIXME: update to output dice score 

        if epoch == 0: 
            accuracy = new_accuracy

        if new_accuracy > accuracy and epoch > num_epoch_dont_save: 
            accuracy = new_accuracy 
            save_checkpoint(checkpoint, "./segmentation_model/models/my_checkpoint.pth.tar") # update to save checkpoint with dice score in filename 

            # # print some examples to folder 
            # saved_images_dir = "saved_images/"
            # os.makedirs(saved_images_dir, exist_ok=True)
            # save_predictions_as_imgs(
            #     val_loader, model, folder=saved_images_dir, device=DEVICE
            # )

if __name__ == "__main__": 
    main() 