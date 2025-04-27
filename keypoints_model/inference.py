import numpy as np 
import torch
from torchvision import transforms
from PIL import Image
from model import RegressorMobileNetV3 
import os 
import albumentations as A 
from albumentations.pytorch import ToTensorV2 
import time 
import json 
import cv2 
import matplotlib.pyplot as plt
import math 
from utils import compute_2D_gridpoints, overlay_points_on_image  
# from utils import * 
from PIL import Image 

import matplotlib
matplotlib.use('Agg')  # Use Agg backend (non-Qt)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
IMAGE_HEIGHT = 480 
IMAGE_WIDTH = 640 

model = RegressorMobileNetV3().to(DEVICE) 
model.eval()    
load_checkpoint(torch.load("./keypoints_model/models/my_checkpoint_keypoints_20250330.pth.tar", map_location=torch.device(DEVICE)), model) 

transform = A.Compose(
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

# camera parameters 
width = 640 
height = 480 
focal_length = 24.0 
horiz_aperture = 20.955
# Pixels are square so we can do:
vert_aperture = height/width * horiz_aperture
fov = 2 * math.atan(horiz_aperture / (2 * focal_length))
# compute focal point and center
fx = width * focal_length / horiz_aperture
fy = height * focal_length / vert_aperture
cx = width / 2
cy = height /2 
camera_matrix = np.array([
    [fx,0,cx],
    [0,fy,cy],
    [0,0,1]
])
dist_coeffs = np.zeros((5, 1))

# marker frame keypoints 
keypoints_tag_frame = compute_2D_gridpoints(N=10, s=0.100)  

# dataset_dir = "/home/rp/abhay_ws/marker_detection_failure_recovery/test_data/data_20250328-144918/train/" 
dataset_dir = "/home/rp/abhay_ws/marker_detection_failure_recovery/test_data/data_20250329-002702/train/"
image_dir = os.path.join(dataset_dir, "rgb") 
all_images = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]  # Filter out directories
output_dir = os.path.join(dataset_dir,f"keypoints_predictions_{time.strftime('%Y%m%d-%H%M%S')}")
os.makedirs(output_dir, exist_ok=True)  
os.makedirs(os.path.join(output_dir, "json"), exist_ok=True)  
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)  
os.makedirs(os.path.join(output_dir, "analysis"), exist_ok=True)  
os.makedirs(os.path.join(output_dir, "pose"), exist_ok=True) 

for i in range(len(all_images)): 
    image_path = os.path.join(image_dir, all_images[i]) 
    image = Image.open(image_path).convert("RGB")  # Open the image and convert to RGB
    image = np.array(image)  # Convert the image to a numpy array

    # Apply transformation (e.g., resizing, normalization, etc.)
    transformed = transform(image=image)['image']
    image_tensor = transformed.unsqueeze(0).to(DEVICE)  # Add batch dimension and move to device
    image_tensor = image_tensor.to(torch.float32)  

    # Perform inference # TODO: do this in parallel using a DataLoader 
    with torch.no_grad():  # Disable gradient calculation for inference
        pred = model(image_tensor)  # Get the model's prediction

    # prediction is a tensor, so convert it to a numpy array 
    pred = pred.cpu().numpy()  # Move the prediction to the CPU and convert to a numpy array 
    pred = pred.squeeze(0)  # Remove the batch dimension

    # save the prediction as a json file 
    json_path = os.path.join(output_dir, "json", all_images[i].replace(".png", ".json")) 
    with open(json_path, "w") as f:
        json.dump(pred.tolist(), f)

    # read the true keypoints values 
    img_filename = os.path.basename(all_images[i]) 
    keypoints_filename = all_images[i][:img_filename.rfind('_')] + '.json' 
    keypoints_filename = keypoints_filename.replace('img', 'keypoints')  
    true_keypoints_path = os.path.join(dataset_dir, "keypoints", keypoints_filename) 
    with open(true_keypoints_path, "r") as f:
        true_keypoints_data = json.load(f)

    true_keypoints_list = [] 
    for key in true_keypoints_data.keys():
        true_keypoints_list.append(np.array(true_keypoints_data[key])) 
    true_keypoints = np.array(true_keypoints_list) 

    # reshape the prediction and true keypoints to (num_keypoints, 2)
    pred = pred.reshape(-1, 2)

    # compute the pixel distance error between the prediction and true keypoints
    pixel_distance_error = np.linalg.norm(pred - true_keypoints, axis=1)
    mean_pixel_distance_error = np.mean(pixel_distance_error) 
    print(f"Mean Pixel Distance Error: {mean_pixel_distance_error}") 

    # compute MSE 
    mse = np.mean((pred - true_keypoints)**2) 
    print(f"MSE: {mse}")

    # save pixel_distance_error and mean_pixel_distance_error to a json file 
    error_dict = {"pixel_distance_error": pixel_distance_error.tolist(), "mean_pixel_distance_error": mean_pixel_distance_error} 
    error_json_path = os.path.join(output_dir, "json", all_images[i].replace(".png", "_error.json"))
    with open(error_json_path, "w") as f:
        json.dump(error_dict, f) 

    # # save the image with the true keypoints overlaid with x marker and the prediction overlaid with o marker 
    # image_with_keypoints = image.copy()
    # for j in range(len(true_keypoints)):
    #     x, y = true_keypoints[j]
    #     cv2.drawMarker(image_with_keypoints, (int(x), int(y)), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_AA)
    # for j in range(len(pred)):
    #     x, y = pred[j]
    #     cv2.drawMarker(image_with_keypoints, (int(x), int(y)), (0, 0, 255), markerType=cv2.MARKER_SQUARE, markerSize=10, thickness=2, line_type=cv2.LINE_AA)
    # image_with_keypoints_path = os.path.join(output_dir, "images", all_images[i])
    # cv2.imwrite(image_with_keypoints_path, image_with_keypoints)

    # keypoints image 
    img_rgb = Image.open(image_path) 

    keypoints_image = overlay_points_on_image(image=np.array(img_rgb), pixel_points=true_keypoints, radius=1)
    keypoints_image = overlay_points_on_image(image=np.array(keypoints_image), pixel_points=pred, radius=1, color=(255, 0, 0))  
    plt.imshow(keypoints_image)
    plt.axis('off')  # Hide axes
    plt.title(f'Keypoints Image {i}') 
    plt.savefig(os.path.join(output_dir, "images", all_images[i].replace(".png", "_keypoints.png")))
    plt.close() 

    # save histogram plot of pixel_distance_error 
    plt.hist(pixel_distance_error, bins=20)
    plt.xlabel("Pixel Distance Error")
    plt.ylabel("Frequency")
    plt.title("Pixel Distance Error Histogram")
    plt.savefig(os.path.join(output_dir, "analysis", all_images[i].replace(".png", "_histogram.png")))
    plt.close()

    # # convert keypoints to 2D points
    # keypoints = pred.reshape(-1, 2).astype(np.float32)
    # # convert 3D points to 2D points
    # gridpoints = np.array(keypoints_tag_frame).astype(np.float32)
    # # solvePnP algorithm
    # _, rvec, tvec, inliers = cv2.solvePnPRansac(gridpoints, keypoints, camera_matrix, dist_coeffs)
    # # save the rotation and translation vectors to a json file
    # pose_dict = {"rvec": rvec.tolist(), "tvec": tvec.tolist()}
    # pose_json_path = os.path.join(output_dir, "pose", all_images[i].replace(".png", "_pose.json"))
    # with open(pose_json_path, "w") as f:
    #     json.dump(pose_dict, f) 




    