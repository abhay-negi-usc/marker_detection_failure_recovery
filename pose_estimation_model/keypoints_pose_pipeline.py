import numpy as np
import torch 
from torchvision import transforms 
from PIL import Image
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
import sys
import cv2 
import math 
import matplotlib.pyplot as plt 
import matplotlib 
matplotlib.use('Agg')  # Use Agg backend (non-Qt)

# module imports
sys.path.append('/home/rp/abhay_ws/marker_detection_failure_recovery')
from segmentation_model.utils import *
from segmentation_model.model import UNETWithDropout  
from keypoints_model.utils import * 
from keypoints_model.model import RegressorMobileNetV3 
from pose_estimation_model.utils import * 

# constants 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
MAIN_DIR = "/home/rp/abhay_ws/marker_detection_failure_recovery/test_data/data_20250329-002702/train/"  
SEGMENTATION_MODEL_PATH = "/home/rp/abhay_ws/marker_detection_failure_recovery/segmentation_model/models/my_checkpoint_20250329.pth.tar"

# class definition 
class PoseEstimator():
    def __init__(self):
        self.segmentation_model = None 
        self.keypoints_model = None

    def load_segmentation_model(self, model_path):
        self.segmentation_model = UNETWithDropout(in_channels=3, out_channels=1).to(DEVICE) 
        load_checkpoint(torch.load(model_path, map_location=torch.device(DEVICE)), self.segmentation_model) 
        self.segmentation_model.eval()
        print("Segmentation model loaded.")  

    def load_keypoints_model(self, model_path): 
        self.keypoints_model = RegressorMobileNetV3().to(DEVICE) 
        load_checkpoint(torch.load(model_path, map_location=torch.device(DEVICE)), self.keypoints_model) 
        self.keypoints_model.eval() 
        print("Keypoints model loaded.") 

    def set_camera_properties(self): 
        pass 

    def set_marker_properties(self): 
        pass

    def load_input_data(self, dir_images): 
        self.rgb_list = sorted([os.path.join(dir_images, img) for img in os.listdir(dir_images) if img.endswith(".png")])

    def run_segmentation_inference(self, image):
        # transform image
        transform = A.Compose([
                A.Resize(height=480, width=640),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet mean values
                    std=[0.229, 0.224, 0.225],   # ImageNet std values
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
        ])
        img_rgb = cv2.imread(image)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        img_rgb = transform(image=img_rgb)["image"]
        img_rgb = img_rgb.unsqueeze(0).to(DEVICE)
        # run inference
        with torch.no_grad():
            segmentation_preds = self.segmentation_model(img_rgb)
            segmentation_preds = (segmentation_preds > 0.5).float()
        return segmentation_preds 

    def compute_roi(self, segmentation):      

        padding = 5 
        roi_size = 128 

        image_border_size = np.max([np.array(seg).shape[0], np.array(seg).shape[1]]) 

        # get pixel info of seg 
        seg = np.array(seg) 
        seg = cv2.copyMakeBorder(seg, image_border_size, image_border_size, image_border_size, image_border_size, cv2.BORDER_CONSTANT, value=0) 
        tag_pixels = np.argwhere(seg == 255)
        seg_tag_min_x = np.min(tag_pixels[:,1])
        seg_tag_max_x = np.max(tag_pixels[:,1])
        seg_tag_min_y = np.min(tag_pixels[:,0])
        seg_tag_max_y = np.max(tag_pixels[:,0])
        seg_height = seg_tag_max_y - seg_tag_min_y  
        seg_width = seg_tag_max_x - seg_tag_min_x 
        seg_center_x = (seg_tag_min_x + seg_tag_max_x) // 2
        seg_center_y = (seg_tag_min_y + seg_tag_max_y) // 2 

        # get pixel info of rgb 
        rgb = np.array(Image.open(self.rgb_filepath))
        rgb = cv2.copyMakeBorder(rgb, image_border_size, image_border_size, image_border_size, image_border_size, cv2.BORDER_CONSTANT, value=0) 
        rgb_side = max(seg_height, seg_width) + 2*padding 
        rgb_tag_min_x = seg_center_x - rgb_side // 2
        rgb_tag_max_x = seg_center_x + rgb_side // 2
        rgb_tag_min_y = seg_center_y - rgb_side // 2
        rgb_tag_max_y = seg_center_y + rgb_side // 2
        roi_img = rgb[rgb_tag_min_y:rgb_tag_max_y, rgb_tag_min_x:rgb_tag_max_x, :]
        roi_img = cv2.resize(roi_img, (roi_size, roi_size))
        W = rgb.shape[1] 
        H = rgb.shape[0]
        roi_coordinates = np.array([rgb_tag_min_x-W/2, rgb_tag_max_x-W/2, rgb_tag_min_y-H/2, rgb_tag_max_y-H/2]) # FIXME: there is some issue here # image (x,y) coordinates (origin at image center) 

        return roi_img, roi_coordinates  

    def run_keypoints_inference(self, roi):
        # transform image
        transform = A.Compose(
            [
                A.Resize(height=480, width=640),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet mean values
                    std=[0.229, 0.224, 0.225],   # ImageNet std values
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )
        img_rgb = cv2.imread(roi)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        img_rgb = transform(image=img_rgb)["image"]
        img_rgb = img_rgb.unsqueeze(0).to(DEVICE)
        # run inference
        with torch.no_grad():
            keypoints_preds = self.keypoints_model(img_rgb)
        return keypoints_preds

    def perform_pose_estimation(self, keypoints):
        pass



    # read in data 

    # run segmentation inference on images 

    # use segmentation output to compute roi 

    # run keypoints inference on roi 

    # return keypoints 

    # perform pose estimation using PnP algorithm on keypoints 
        

PE = PoseEstimator()
# PE.load_segmentation_model(SEGMENTATION_MODEL_PATH) 
# PE.load_keypoints_model("./keypoints_model/models/my_checkpoint_20250330.pth.tar") 


