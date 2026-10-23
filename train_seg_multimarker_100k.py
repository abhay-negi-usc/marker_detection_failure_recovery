"""
Train the grayscale segmentation model (UNETWithDropoutMini) on the overlay-generated
multi-marker 100K dataset (AprilTag 36h11 ids 0-15), from scratch, logging to wandb.

Every ~15 min it logs a 16-sample validation panel (input | ground-truth | prediction)
to wandb and a scalar val pixel-accuracy on those samples. Only the best-on-val
checkpoint is kept (no intermediates).

Reuses the sibling repo's model / grayscale loaders / utils. Run from this repo root:
    cd /home/rp/abhay_ws/marker_detection_failure_recovery
    python train_seg_multimarker_100k.py
"""
import os
import time
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import wandb

from segmentation_model.model import UNETWithDropoutMini
from segmentation_model.utils import save_checkpoint, check_accuracy, get_loaders

# ---- data: overlay-generated 100K multi-marker set (grayscale loader converts rgb->L) ----
DATA_DIR = "/home/rp/abhay_ws/deep-marker-estimation/data_generation/generated_data/data_100k"
TRAIN_IMG_DIR = f"{DATA_DIR}/train/rgb"
TRAIN_MASK_DIR = f"{DATA_DIR}/train/seg"
VAL_IMG_DIR = f"{DATA_DIR}/val/rgb"
VAL_MASK_DIR = f"{DATA_DIR}/val/seg"

SAVE_DIR = "/home/rp/abhay_ws/deep-marker-estimation/segmentation_model/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)
BEST_CKPT = os.path.join(SAVE_DIR, "seg_mini_multimarker_best.pth.tar")

# ---- hyperparameters ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4          # from-scratch (the fine-tune script used 1e-7)
BATCH_SIZE = 8                # ~8GB at 480x640 on the 12GB GPU
NUM_EPOCHS = 100
NUM_WORKERS = 8
PIN_MEMORY = True
IMAGE_HEIGHT, IMAGE_WIDTH = 480, 640
PRED_INTERVAL = 15 * 60       # seconds between validation-prediction panels
N_VIS = 16                    # samples in the panel

# grayscale normalization (input distribution the eventual grayscale inference must match)
TRANSFORM = A.Compose([A.Normalize(mean=(0.485,), std=(0.229,), max_pixel_value=255.0), ToTensorV2()])


def build_fixed_vis_samples(img_dir, mask_dir, n=N_VIS):
    """Fixed set of n (input_gray, gt_binary, input_tensor) so panels are comparable over time."""
    files = sorted(os.listdir(img_dir))[:n]
    samples = []
    for f in files:
        # grayscale loader maps mask name via replace('img','seg') keeping extension
        mask_name = f.replace("img", "seg").replace("_0.png", ".png")
        gray = np.array(Image.open(os.path.join(img_dir, f)).convert("L"))
        gt = (np.array(Image.open(os.path.join(mask_dir, mask_name)).convert("L")) >= 100).astype(np.uint8)
        x = TRANSFORM(image=gray)["image"].unsqueeze(0)  # (1,1,H,W)
        samples.append((gray, gt, x))
    return samples


def _cell(gray, gt, pred, w=192, h=144):
    """Horizontal [input | GT | pred] tile, grayscale->BGR, with white separators."""
    def to3(a):
        return cv2.cvtColor(cv2.resize(a, (w, h)), cv2.COLOR_GRAY2BGR)
    sep = np.full((h, 3, 3), 255, np.uint8)
    return np.hstack([to3(gray), sep, to3(gt * 255), sep, to3(pred * 255)])


def log_val_predictions(model, samples, step, epoch):
    """Predict on the fixed samples, log a 4x4 panel + mean pixel accuracy to wandb."""
    model.eval()
    tiles, accs = [], []
    with torch.no_grad():
        for gray, gt, x in samples:
            with torch.amp.autocast(device_type=DEVICE):
                logit = model(x.to(DEVICE))
            pred = (torch.sigmoid(logit)[0, 0].float().cpu().numpy() > 0.5).astype(np.uint8)
            accs.append(float((pred == gt).mean()))
            tiles.append(_cell(gray, gt, pred))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    panel = np.vstack(rows)
    panel_rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
    wandb.log({
        "val_predictions": wandb.Image(panel_rgb, caption=f"epoch {epoch} | cols: input | GT | pred"),
        "val16_pixel_acc": float(np.mean(accs)),
        "epoch": epoch,
    }, step=step)
    model.train()
    return float(np.mean(accs))


def train_fn(loader, model, optimizer, loss_fn, scaler, epoch, vis_samples, state):
    loop = tqdm(loader)
    epoch_loss = 0.0
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        with torch.amp.autocast(device_type=DEVICE):
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        state["step"] += 1
        wandb.log({"batch_loss": loss.item(), "epoch": epoch}, step=state["step"])
        loop.set_postfix(loss=loss.item())

        # periodic validation-prediction panel (~every PRED_INTERVAL seconds)
        if time.time() - state["last_pred"] >= PRED_INTERVAL:
            acc = log_val_predictions(model, vis_samples, state["step"], epoch)
            state["last_pred"] = time.time()
            print(f"[VIS] epoch {epoch} step {state['step']}: val16_pixel_acc={acc:.4f}")

    wandb.log({"train_loss": epoch_loss / len(loader), "epoch": epoch}, step=state["step"])


def main():
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR,
        BATCH_SIZE, TRANSFORM, TRANSFORM, NUM_WORKERS, PIN_MEMORY,
    )
    vis_samples = build_fixed_vis_samples(VAL_IMG_DIR, VAL_MASK_DIR, N_VIS)

    model = UNETWithDropoutMini(in_channels=1, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] device={DEVICE}  params={n_params/1e6:.2f}M  "
          f"train={len(train_loader.dataset)}  val={len(val_loader.dataset)}")

    state = {"step": 0, "last_pred": time.time()}
    log_val_predictions(model, vis_samples, state["step"], 0)  # baseline panel at init

    # early stopping: stop once val accuracy stops improving (saturation)
    PATIENCE = 3
    MIN_DELTA = 1e-4
    best_acc = 0.0
    since_improve = 0
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, epoch, vis_samples, state)
        acc = float(check_accuracy(val_loader, model, device=DEVICE))
        wandb.log({"val_accuracy": acc, "epoch": epoch}, step=state["step"])
        print(f"[INFO] epoch {epoch}: val_accuracy={acc}")
        if acc > best_acc + MIN_DELTA:           # only the best checkpoint is kept
            best_acc = acc
            since_improve = 0
            save_checkpoint({"state_dict": model.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "epoch": epoch, "val_accuracy": acc}, BEST_CKPT)
            print(f"[INFO] new best val_accuracy={acc:.5f} -> saved {BEST_CKPT}")
        else:
            since_improve += 1
            print(f"[INFO] no improvement ({since_improve}/{PATIENCE}), best={best_acc:.5f}")
            if since_improve >= PATIENCE:
                print(f"[INFO] SATURATED at epoch {epoch} (best val_accuracy={best_acc:.5f}) -- stopping.")
                break


if __name__ == "__main__":
    wandb.init(
        entity="abhay-negi-usc-university-of-southern-california",
        project="segmentation-multimarker-36h11",
        name="unet-mini-grayscale-100k",
        config={
            "model": "UNETWithDropoutMini", "channels": "grayscale",
            "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS, "image_size": [IMAGE_HEIGHT, IMAGE_WIDTH],
            "data_dir": DATA_DIR, "from_scratch": True,
            "train_size": 90000, "val_size": 10000, "markers": "36h11 ids 0-15",
            "pred_interval_min": PRED_INTERVAL / 60, "checkpoint_policy": "best_only",
        },
    )
    main()
