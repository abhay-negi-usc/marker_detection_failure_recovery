import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm
from hrnet_lite.model import LiteHRNet
import matplotlib.pyplot as plt


def overlay_points_on_image(image, pixel_points, radius=5, color=(0, 0, 255), thickness=-1):
    """
    Overlays a list of pixel points on the input image.

    Parameters:
    - image: The input image (a NumPy array).
    - pixel_points: A list of 2D pixel coordinates [(x1, y1), (x2, y2), ...].
    - radius: The radius of the circle to draw around each point. Default is 5.
    - color: The color of the circle (BGR format). Default is red (0, 0, 255).
    - thickness: The thickness of the circle. Default is -1 to fill the circle.

    Returns:
    - The image with points overlaid.
    """
    # Iterate over each pixel point and overlay it on the image
    for point in pixel_points:
        if point is not None:  # Only overlay valid points
            x, y = int(point[0]), int(point[1])
            # check if the point is within the image bounds
            if x < 0 or x >= image.shape[1] or y < 0 or y >= image.shape[0]:
                continue
            # Draw a filled circle at the pixel coordinates
            cv2.circle(image, (x, y), radius, color, thickness)
    return image

class HRNetLiteKeypoint(torch.nn.Module):
    def __init__(self, num_keypoints, input_width, input_height):
        super().__init__()
        self.backbone = LiteHRNet(num_keypoints=num_keypoints)
        self.input_width = input_width
        self.input_height = input_height

    def forward(self, x):
        return self.backbone(x)
import os
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
from hrnet_lite.model import LiteHRNet

def overlay_points_on_image(image, pixel_points, radius=5, color=(0, 0, 255), thickness=-1):
    for point in pixel_points:
        if point is not None:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                cv2.circle(image, (x, y), radius, color, thickness)
    return image

class HRNetLiteKeypoint(torch.nn.Module):
    def __init__(self, num_keypoints, input_width, input_height):
        super().__init__()
        self.backbone = LiteHRNet(num_keypoints=num_keypoints)
        self.input_width = input_width
        self.input_height = input_height

    def forward(self, x):
        return self.backbone(x)

@torch.no_grad()
def run_inference(
    model_path,
    image_dir,
    save_dir,
    num_keypoints,
    input_width=640,
    input_height=480,
    device='cuda'
):
    transform = transforms.Compose([
        transforms.Resize((input_height, input_width)),
        transforms.ToTensor(),
    ])

    model = HRNetLiteKeypoint(num_keypoints, input_width, input_height).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    os.makedirs(save_dir, exist_ok=True)
    image_paths = sorted([p for p in Path(image_dir).iterdir() if p.suffix.lower() in [".jpg", ".png"]])

    for img_path in image_paths:
        pil_img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = pil_img.size

        img_tensor = transform(pil_img).unsqueeze(0).to(device)  # [1, 3, H, W]
        pred = model(img_tensor).squeeze().cpu().numpy().reshape(-1, 2)

        # Rescale normalized keypoints to original image size
        pred[:, 0] *= orig_w
        pred[:, 1] *= orig_h

        # Save keypoints
        np.save(Path(save_dir) / f"{img_path.stem}_pred.npy", pred)

        # Overlay keypoints using OpenCV
        cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        overlayed_img = overlay_points_on_image(cv2_img.copy(), pred)
        cv2.imwrite(str(Path(save_dir) / f"{img_path.stem}_overlay.png"), overlayed_img)

if __name__ == "__main__":
    run_inference(
        model_path="./hrnet_lite/checkpoints/lite_hrnet_epoch140.pth",
        image_dir="./segmentation_model/data/data_20250603-201339/val/rgb",
        save_dir="./hrnet_lite/inference_results",
        num_keypoints=(6+1)**2,  # Set according to your model
        input_width=640,
        input_height=480,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
