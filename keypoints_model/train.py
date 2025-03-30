import torch 
import albumentations as A 
from albumentations.pytorch import ToTensorV2 
from tqdm import tqdm 
import torch.nn as nn 
import torch.optim as optim 
from model import RegressorMobileNetV3 
from utils import (
    load_checkpoint, 
    save_checkpoint, 
    get_loaders, 
    evaluate_mse_loss,
    overlay_points_on_image,
)
import os 
from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np 
from torchvision.transforms import ToPILImage
import matplotlib
matplotlib.use('Agg')  # Use Agg backend (non-Qt)


LEARNING_RATE = 1e-6 * 4 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
BATCH_SIZE = 1024       
NUM_EPOCHS = 1000 
num_epoch_dont_save = 0 
NUM_WORKERS = 30 
IMAGE_HEIGHT = 480 
IMAGE_WIDTH = 640 
SAVE_FREQUENCY = 10 
PIN_MEMORY = True 
LOAD_MODEL = True  
MAIN_DIR = "/home/anegi/abhay_ws/marker_detection_failure_recovery/segmentation_model/data/data_20250330-013534/" 
TRAIN_IMG_DIR = os.path.join(MAIN_DIR, "train", "roi_rgb") 
TRAIN_KEYPOINTS_DIR = os.path.join(MAIN_DIR, "train", "roi_keypoints")
VAL_IMG_DIR = os.path.join(MAIN_DIR, "val", "roi_rgb")  
VAL_KEYPOINTS_DIR = os.path.join(MAIN_DIR, "val", "roi_keypoints")
    
def train_fn(loader, model, optimizer, loss_fn, scaler): 
    loop = tqdm(loader) # progress bar 

    for batch_idx, (data, targets) in enumerate(loop): 
        data = data.to(device=DEVICE).to(torch.float32).permute(0,3,1,2) 
        targets = targets.float().to(device=DEVICE) 

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

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    to_pil = ToPILImage()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device).to(torch.float32).permute(0,3,1,2) 
        with torch.no_grad():
            preds = model(x)

    for j in range(len(preds)): 
        pred = preds[j]

        # prediction is a tensor, so convert it to a numpy array 
        pred = pred.cpu().numpy()  # Move the prediction to the CPU and convert to a numpy array 
        # pred = pred.squeeze(0)  # Remove the batch dimension

        # reshape the prediction and true keypoints to (num_keypoints, 2)
        pred = pred.reshape(-1, 2)

        img_rgb = to_pil(x[j])  # Convert the j-th image in the batch to a PIL image
        keypoints_image = overlay_points_on_image(image=np.array(img_rgb), pixel_points=pred, radius=1)
        plt.imshow(keypoints_image)
        plt.axis('off')  # Hide axes
        plt.title(f'Keypoints Image {j}') 
        plt.savefig(os.path.join(folder, f"pred_{j}.png")) 
        plt.close() 
        
        # torchvision.utils.save_image(
        #     preds, f"{folder}/pred_{idx}.png"
        # )
        # torchvision.utils.save_image(x, f"{folder}/rgb_{idx}.png", normalize=True)  
        

def main(): 
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean values
                std=[0.229, 0.224, 0.225],   # ImageNet std values
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean values
                std=[0.229, 0.224, 0.225],   # ImageNet std values
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    model = RegressorMobileNetV3().to(DEVICE) 
    
    loss_fn = nn.MSELoss()  
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR, 
        TRAIN_KEYPOINTS_DIR,
        VAL_IMG_DIR, 
        VAL_KEYPOINTS_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY
    )

    if LOAD_MODEL: 
        load_checkpoint(torch.load("./keypoints_model/models/my_checkpoint.pth.tar"), model)

    save_count = 0 

    scaler = torch.amp.GradScaler()
    for epoch in range(NUM_EPOCHS): 
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model 
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(), 
        }

        new_mse_loss = evaluate_mse_loss(val_loader, model, device=DEVICE) 

        print(f"EPOCH: {epoch}. MSE: {new_mse_loss:.2f}") 

        if epoch == 0: 
            best_mse_loss = new_mse_loss

        if new_mse_loss < best_mse_loss and epoch > num_epoch_dont_save: 
            best_mse_loss = new_mse_loss  
            save_checkpoint(checkpoint, "./keypoints_model/models/my_checkpoint.pth.tar") # update to save checkpoint with dice score in filename 

            if save_count > SAVE_FREQUENCY: 
                # print some examples to folder 
                saved_images_dir = "saved_images/"
                os.makedirs(saved_images_dir, exist_ok=True)
                save_predictions_as_imgs(
                    val_loader, model, folder=saved_images_dir, device=DEVICE
                )
                save_count = 0 
        
        save_count += 1 

if __name__ == "__main__": 
    main() 