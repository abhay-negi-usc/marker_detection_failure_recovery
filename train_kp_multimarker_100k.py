"""
Train the keypoint regressor (RegressorMobileNetV3) on the overlay-generated
multi-marker 100K dataset (AprilTag 36h11 ids 0-15), from scratch, logging to wandb.

Every ~15 min it logs a 16-sample validation panel (ROI with GT keypoints in GREEN and
predicted keypoints in RED overlaid) plus a scalar mean keypoint L1 error (px). Only the
best-on-val checkpoint is kept.

Input convention matches inference (build_lbcv_predictor): raw 0-255 RGB ROI, permuted to
CHW, NO normalization; targets are the 121 keypoints in 128-px ROI space (flattened to 242).

Run from this repo root (GPU must be free -- stop seg training first):
    cd /home/rp/abhay_ws/marker_detection_failure_recovery
    python train_kp_multimarker_100k.py
"""
import os
import time
import json
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

from keypoints_model.model import RegressorMobileNetV3
from keypoints_model.utils import save_checkpoint, get_loaders, evaluate_l1_loss

# ---- data: overlay-generated 100K multi-marker set ----
DATA_DIR = "/home/rp/abhay_ws/deep-marker-estimation/data_generation/generated_data/data_100k"
TRAIN_IMG_DIR = f"{DATA_DIR}/train/roi_rgb"
TRAIN_KP_DIR = f"{DATA_DIR}/train/roi_keypoints"
VAL_IMG_DIR = f"{DATA_DIR}/val/roi_rgb"
VAL_KP_DIR = f"{DATA_DIR}/val/roi_keypoints"

SAVE_DIR = "/home/rp/abhay_ws/deep-marker-estimation/keypoints_model/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)
BEST_CKPT = os.path.join(SAVE_DIR, "kp_mobilenetv3_multimarker_best.pth.tar")

# ---- hyperparameters ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4          # from-scratch (the fine-tune script used 1e-5)
BATCH_SIZE = 256              # 128x128 ROI + small MobileNetV3Small backbone
NUM_EPOCHS = 200
NUM_WORKERS = 8
PIN_MEMORY = True
ROI = 128
PRED_INTERVAL = 15 * 60       # seconds between validation-prediction panels
N_VIS = 16


def build_fixed_vis_samples(img_dir, kp_dir, n=N_VIS):
    """Fixed n (roi_rgb HWC uint8, gt 121x2 px, input_tensor 1x3xHxW raw 0-255)."""
    files = sorted(os.listdir(img_dir))[:n]
    samples = []
    for f in files:
        kpf = f.replace("roi", "roi_keypoints").replace(".png", ".json")
        img = np.array(Image.open(os.path.join(img_dir, f)).convert("RGB"))  # HWC uint8
        kd = json.load(open(os.path.join(kp_dir, kpf)))
        gt = np.array([kd[k] for k in kd], dtype=np.float32).reshape(-1, 2)  # 121x2, 128-space
        x = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)      # (1,3,H,W) raw 0-255
        samples.append((img, gt, x))
    return samples


def _draw(img_rgb, gt, pred, up=2):
    """Upscaled ROI (RGB) with GT keypoints green and predicted keypoints red."""
    vis = cv2.resize(img_rgb, (ROI * up, ROI * up), interpolation=cv2.INTER_NEAREST).copy()
    for (x, y) in gt:
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(vis, (int(x * up), int(y * up)), 2, (0, 255, 0), -1)  # GT green (RGB)
    for (x, y) in pred:
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(vis, (int(x * up), int(y * up)), 2, (255, 0, 0), -1)  # pred red (RGB)
    return vis


def log_val_predictions(model, samples, step, epoch):
    model.eval()
    tiles, errs = [], []
    with torch.no_grad():
        for img, gt, x in samples:
            out = model(x.to(DEVICE))                                        # fp32
            pred = out[0].float().cpu().numpy().reshape(-1, 2)               # 121x2, 128-space
            errs.append(float(np.mean(np.linalg.norm(pred - gt, axis=1))))   # mean per-kp px error
            tiles.append(_draw(img, gt, pred))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    panel = np.vstack(rows)
    wandb.log({
        "val_predictions": wandb.Image(panel, caption=f"epoch {epoch} | green=GT  red=pred"),
        "val16_kp_l1_px": float(np.mean(errs)),
        "epoch": epoch,
    }, step=step)
    model.train()
    return float(np.mean(errs))


def train_fn(loader, model, optimizer, loss_fn, epoch, vis_samples, state):
    loop = tqdm(loader)
    epoch_loss = 0.0
    for data, targets in loop:
        data = data.to(device=DEVICE).to(torch.float32).permute(0, 3, 1, 2)  # raw 0-255 CHW
        targets = targets.float().to(DEVICE)
        predictions = model(data)                                            # fp32
        loss = loss_fn(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        state["step"] += 1
        wandb.log({"batch_l1_loss": loss.item(), "epoch": epoch}, step=state["step"])
        loop.set_postfix(loss=loss.item())
        if time.time() - state["last_pred"] >= PRED_INTERVAL:
            err = log_val_predictions(model, vis_samples, state["step"], epoch)
            state["last_pred"] = time.time()
            print(f"[VIS] epoch {epoch} step {state['step']}: val16_kp_l1_px={err:.3f}")
    wandb.log({"train_l1_loss": epoch_loss / len(loader), "epoch": epoch}, step=state["step"])


def main():
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR, TRAIN_KP_DIR, VAL_IMG_DIR, VAL_KP_DIR,
        BATCH_SIZE, None, None, NUM_WORKERS, PIN_MEMORY,   # dataset ignores transform
    )
    vis_samples = build_fixed_vis_samples(VAL_IMG_DIR, VAL_KP_DIR, N_VIS)

    model = RegressorMobileNetV3().to(DEVICE)
    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] device={DEVICE}  params={n_params/1e6:.2f}M  "
          f"train={len(train_loader.dataset)}  val={len(val_loader.dataset)}")

    state = {"step": 0, "last_pred": time.time()}
    log_val_predictions(model, vis_samples, state["step"], 0)  # baseline panel

    # early stopping: stop once val L1 stops improving (saturation)
    PATIENCE = 25
    MIN_DELTA = 1e-3
    best_l1 = float("inf")
    since_improve = 0
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, epoch, vis_samples, state)
        val_l1 = float(evaluate_l1_loss(val_loader, model, device=DEVICE))
        wandb.log({"val_l1_loss": val_l1, "epoch": epoch}, step=state["step"])
        print(f"[INFO] epoch {epoch}: val_l1_loss={val_l1:.5f}")
        if val_l1 < best_l1 - MIN_DELTA:         # only best (lowest val L1) is kept
            best_l1 = val_l1
            since_improve = 0
            save_checkpoint({"state_dict": model.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "epoch": epoch, "val_l1_loss": val_l1}, BEST_CKPT)
            print(f"[INFO] new best val_l1={val_l1:.5f} -> saved {BEST_CKPT}")
        else:
            since_improve += 1
            if since_improve >= PATIENCE:
                print(f"[INFO] SATURATED at epoch {epoch} (best val_l1={best_l1:.5f}) -- stopping.")
                break


if __name__ == "__main__":
    wandb.init(
        entity="abhay-negi-usc-university-of-southern-california",
        project="keypoints-multimarker-36h11",
        name="mobilenetv3-100k-augroi",
        config={
            "model": "RegressorMobileNetV3", "input": "raw 0-255 RGB ROI 128",
            "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS, "loss": "L1", "from_scratch": True,
            "data_dir": DATA_DIR, "train_size": 90000, "val_size": 10000,
            "markers": "36h11 ids 0-15", "num_keypoints": 121,
            "pred_interval_min": PRED_INTERVAL / 60, "checkpoint_policy": "best_only",
        },
    )
    main()
