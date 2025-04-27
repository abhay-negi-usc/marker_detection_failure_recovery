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
import json 

# module imports
sys.path.append('/home/rp/abhay_ws/marker_detection_failure_recovery')
from segmentation_model.utils import *
from segmentation_model.model import UNETWithDropout  
from keypoints_model.utils import * 
from keypoints_model.model import RegressorMobileNetV3 
# from pose_estimation_model.utils import * 

# constants 
# MAIN_DIR = "/home/rp/abhay_ws/marker_detection_failure_recovery/test_data/data_20250329-002702/train/"  
MAIN_DIR = "/home/rp/abhay_ws/marker_detection_failure_recovery/test_data/sdg_markers_20250330-181934/"
# MAIN_DIR = "/home/rp/abhay_ws/marker_detection_failure_recovery/test_data/test_images_real/"

SEGMENTATION_MODEL_PATH = "/home/rp/abhay_ws/marker_detection_failure_recovery/segmentation_model/models/my_checkpoint_20250329.pth.tar"
KEYPOINTS_MODEL_PATH = "/home/rp/abhay_ws/marker_detection_failure_recovery/keypoints_model/models/my_checkpoint.pth.tar"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 

# class definition 
class PoseEstimator():
    def __init__(self):
        self.segmentation_model = None 
        self.keypoints_model = None

    def set_out_dir(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True) 

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

    def set_camera_properties(self, camera_matrix=None, dist_coeffs=None):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs 
        self.tf_W_Ccv = np.array([
            [1,0,0,0],
            [0,-1,0,0],
            [0,0,-1,0],
            [0,0,0,1]
        ]) # opencv camera coordinates wrt isaac sim world coordinates (equivalent to isaac sim camera coordinates) 

    def set_marker_properties(self, marker_image, marker_side_length, keypoints_tag_frame):
        self.marker_image = marker_image
        self.marker_side_length = marker_side_length
        self.keypoints_tag_frame = keypoints_tag_frame 
        self.T_Mcv_Mis = np.array([
            [-1,0,0,0],
            [0,1,0,0],
            [0,0,-1,0],
            [0,0,0,1]
        ]) # isaac sim marker coordinates wrt opencv marker coordinates  

    def load_input_data(self, dir_images): 
        self.rgb_list = sorted([os.path.join(dir_images, img) for img in os.listdir(dir_images) if img.endswith(".png")])
        print(f"Loaded {len(self.rgb_list)} images.") 

    def run_segmentation_inference(self, image):
        # transform image
        IMAGE_HEIGHT = 480 
        IMAGE_WIDTH = 640 
        transform = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Normalize(
                    max_pixel_value=1.0,
                ),
                ToTensorV2(), 
            ]
        )
        transformed = transform(image=image)  # Apply the transform
        img_rgb = transformed["image"].unsqueeze(0).to(DEVICE)  # Add batch dimension and move to device

        # run inference
        with torch.no_grad():
            segmentation_preds = torch.sigmoid(self.segmentation_model(img_rgb))
            segmentation_preds = (segmentation_preds > 0.5).float()
        segmentation_preds = transforms.ToPILImage()(segmentation_preds.squeeze(0).cpu())
        return segmentation_preds 

    def compute_roi(self, seg, rgb):      

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
        if isinstance(rgb, str):
            rgb = np.array(cv2.imread(rgb))
        if isinstance(rgb, Image.Image):
            rgb = np.array(rgb)
        if isinstance(rgb, np.ndarray):
            rgb = rgb
        rgb = cv2.copyMakeBorder(rgb, image_border_size, image_border_size, image_border_size, image_border_size, cv2.BORDER_CONSTANT, value=0) 
        rgb_side = max(seg_height, seg_width) + 2*padding 
        rgb_tag_min_x = seg_center_x - rgb_side // 2
        rgb_tag_max_x = seg_center_x + rgb_side // 2
        rgb_tag_min_y = seg_center_y - rgb_side // 2
        rgb_tag_max_y = seg_center_y + rgb_side // 2
        roi_img = rgb[rgb_tag_min_y:rgb_tag_max_y, rgb_tag_min_x:rgb_tag_max_x, :]
        roi_img = cv2.resize(roi_img, (roi_size, roi_size))
        roi_coordinates = np.array([rgb_tag_min_x, rgb_tag_max_x, rgb_tag_min_y, rgb_tag_max_y]) - image_border_size 

        return roi_img, roi_coordinates  

    def run_keypoints_inference(self, roi, roi_coordinates):
        # TODO: clean up data handling 

        # transform image
        # IMAGE_HEIGHT = 480 
        # IMAGE_WIDTH = 640 
        transform = A.Compose(
            [
                # A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                # A.Normalize(
                #     mean=[0.485, 0.456, 0.406],  # ImageNet mean values
                #     std=[0.229, 0.224, 0.225],   # ImageNet std values
                #     max_pixel_value=255.0,
                # ),
                ToTensorV2(),
            ]
        )

        image = roi 
        transformed = transform(image=image)['image']
        image_tensor = transformed.unsqueeze(0).to(DEVICE)  # Add batch dimension and move to device
        image_tensor = image_tensor.to(torch.float32)  
        img_rgb = image_tensor 

        # run inference
        with torch.no_grad():
            keypoints_roi_preds = self.keypoints_model(img_rgb)
            keypoints_roi_preds = keypoints_roi_preds.cpu().numpy().reshape(-1, 2) 

        # convert to image coordinates 
        s = np.array(roi.shape[:2]) 
        img_roi_center_x = (roi_coordinates[0] + roi_coordinates[1]) / 2 
        img_roi_center_y = (roi_coordinates[2] + roi_coordinates[3]) / 2 
        roi_center = np.array([img_roi_center_x, img_roi_center_y]) 
        w = roi_coordinates[1] - roi_coordinates[0]
        h = roi_coordinates[3] - roi_coordinates[2] 
        m = s / np.array([w,h])

        # keypoints_img_preds = (keypoints_roi_preds - s/2)/m + roi_center 
        
        keypoints_img_preds = [] 
        for i in range(keypoints_roi_preds.shape[0]): 
            kp_roi = keypoints_roi_preds[i,:].reshape(2)  
            kp_img = (kp_roi - s/2)/m + roi_center 
            keypoints_img_preds.append(kp_img) 
        keypoints_img_preds = np.array(keypoints_img_preds)

        return keypoints_roi_preds, keypoints_img_preds 

    def perform_pose_estimation(self, keypoints):
        # TODO: try with RANSAC 
        _, rvec, tvec = cv2.solvePnP(objectPoints=self.keypoints_tag_frame, imagePoints=keypoints, cameraMatrix=self.camera_matrix, distCoeffs=self.dist_coeffs) 
        pose_marker = rvectvec_to_xyzabc(rvec, tvec) 
        tf_marker = self.tf_W_Ccv @ xyzabc_to_tf(pose_marker) 
        return tf_marker, pose_marker 

    def run_pose_estimation_pipeline(self): 
        # TODO: make datatypes consistent and label in comments

        os.makedirs(os.path.join(self.out_dir, "pred_seg"), exist_ok=True)
        # os.makedirs(os.path.join(self.out_dir, "pred_roi"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "pred_keypoints_roi_img"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "pred_keypoints_roi_json"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "pred_keypoints_orig_img"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "pred_keypoints_orig_json"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "pred_pose"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "pred_summary"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "analysis"), exist_ok=True)

        # lists 
        pose_errors = [] 
        
        for i, rgb_filepath in enumerate(self.rgb_list): 
            # print(f"Processing image {i+1}/{len(self.rgb_list)}") 

            # rgb 
            rgb = np.array(Image.open(rgb_filepath).convert("RGB"))

            # segmentation 
            # TODO: run in batch rather than one by one 
            segmentation_preds = self.run_segmentation_inference(rgb) 
            pred_seg_filename = os.path.basename(rgb_filepath).replace(".png", "_pred_seg.png")
            pred_seg_rgb = segmentation_preds.convert("RGB")
            pred_seg_rgb.save(os.path.join(self.out_dir, "pred_seg", pred_seg_filename)) 

            if np.array(segmentation_preds).max() == 0:
                print(f"Index: {i}. No tag detected in image. Skipping further processing.")
                continue

            # roi 
            roi_img, roi_coordinates = self.compute_roi(pred_seg_rgb, rgb)            
            # roi_img_filename = os.path.basename(rgb_filepath).replace(".png", "_roi.png") 
            # roi_img = Image.fromarray(roi_img)
            # roi_img.save(os.path.join(self.out_dir, "pred_roi", roi_img_filename)) 

            # keypoints 
            # TODO: run in batch rather than one by one 
            keypoints_roi_preds, keypoints_img_preds = self.run_keypoints_inference(roi_img, roi_coordinates) 

            keypoints_roi_img = overlay_points_on_image(image=roi_img, pixel_points=keypoints_roi_preds, radius=2) 
            keypoints_roi_img = overlay_points_on_image(image=roi_img, pixel_points=keypoints_roi_preds[0,:].reshape(1,2), radius=3, color=(255,0,0)) 
            keypoints_roi_img_filename = os.path.basename(rgb_filepath).replace(".png", "_keypoints_roi.png")
            Image.fromarray(keypoints_roi_img).save(os.path.join(self.out_dir, "pred_keypoints_roi_img", keypoints_roi_img_filename))
            keypoints_roi_json_filename = os.path.basename(rgb_filepath).replace(".png", "_keypoints_roi.json")
            with open(os.path.join(self.out_dir, "pred_keypoints_roi_json", keypoints_roi_json_filename), "w") as f:
                json.dump(keypoints_roi_preds.tolist(), f)

            keypoints_orig_img = overlay_points_on_image(image=rgb, pixel_points=keypoints_img_preds, radius=2)
            keypoints_orig_img = overlay_points_on_image(image=rgb, pixel_points=keypoints_img_preds[0,:].reshape(1,2), radius=3, color=(255,0,0))
            keypoints_orig_img_filename = os.path.basename(rgb_filepath).replace(".png", "_keypoints_orig.png")
            Image.fromarray(keypoints_orig_img).save(os.path.join(self.out_dir, "pred_keypoints_orig_img", keypoints_orig_img_filename))
            keypoints_orig_json_filename = os.path.basename(rgb_filepath).replace(".png", "_keypoints_orig.json")
            with open(os.path.join(self.out_dir, "pred_keypoints_orig_json", keypoints_orig_json_filename), "w") as f:
                json.dump(keypoints_img_preds.tolist(), f) 

            # pose 
            tf_marker, pose_marker = self.perform_pose_estimation(keypoints_img_preds)

            # read true pose 
            pose_json_filename = os.path.basename(rgb_filepath).replace("rgb","pose").replace(".png", ".json")
            pose_json_path = os.path.join(MAIN_DIR, "pose", pose_json_filename) 
            with open(pose_json_path, "r") as f:
                pose_json = json.load(f)
            tf_marker_true = np.array(pose_json["tag"]) 
            if tf_marker_true[0,3]==0 and tf_marker_true[1,3]==0 and tf_marker_true[2,3]==0 and tf_marker_true[3,3]==1 and tf_marker_true[3,:3].sum() != 0: 
                tf_marker_true = tf_marker_true.transpose() 
            tf_marker_true *= np.array([
                                [10,10,10,1],
                                [10,10,10,1],
                                [10,10,10,1],
                                [1,1,1,1]
                            ]) # rescale the tag 
            # compute error 
            tf_error = np.linalg.inv(tf_marker_true) @ tf_marker 
            pose_error = tf_to_xyzabc(tf_error) 
            pose_errors.append(pose_error) 

        pose_errors = np.array(pose_errors) 

        pose_labels = ["X", "Y", "Z", "A", "B", "C"]
        units_labels = ["m", "m", "m", "deg", "deg", "deg"] 
        # 2x3 violion subplots of pose errors
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        for i in range(6):
            row = i // 3
            col = i % 3
            axs[row, col].violinplot(pose_errors[:, i], showmeans=True)
            axs[row, col].set_title(f"{pose_labels[i]} error")
            axs[row, col].set_ylabel(units_labels[i])
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "analysis", "pose_errors.png"))
        plt.close()


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
dist_coeffs = np.zeros(5) 

# marker properties 
marker_image_path = "./synthetic_data_generation/assets/tags/tag36h11_0.png" 
marker_image = Image.open(marker_image_path).convert("RGB") 
marker_side_length = 0.100 # meters 
marker_num_squares = 10  
keypoints_tag_frame = np.array(compute_2D_gridpoints(N=marker_num_squares, s=marker_side_length) )

PE = PoseEstimator()
PE.set_out_dir(os.path.join(MAIN_DIR, f"out")) 
PE.set_camera_properties(camera_matrix, dist_coeffs)
PE.set_marker_properties(marker_image, marker_side_length, keypoints_tag_frame) 
PE.load_segmentation_model(SEGMENTATION_MODEL_PATH) 
PE.load_keypoints_model(KEYPOINTS_MODEL_PATH) 
PE.load_input_data(os.path.join(MAIN_DIR, "rgb")) 
PE.run_pose_estimation_pipeline() 

