import numpy as np 
import os 
from real_data_processing.utils import (
    DataPoint, 
    opencv_marker_pose, 
    get_marker_segmentation, 
    seg_IOU
)
import cv2 
import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from segmentation_model.model import UNETWithDropout
import matplotlib.pyplot as plt

class Processor(): 
    def __init__(self, config):
        self.dir_path = config["dir_path"]
        self.K = config["camera_intrinsic_matrix"]
        self.dist_coeffs = config["camera_dist_coeffs"]
        self.aruco_dict = config["aruco_dict"]
        self.marker_length = config["marker_length"]
        self.out_dir = config["out_dir"]

        self.datapoints = self.read_frames() 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def read_frames(self):
        frames = [] 
        datapoints = [] 
        files = os.listdir(self.dir_path) 
        files = [f for f in files if f.endswith('.png') or f.endswith('.jpg')]  # Filter for image files 
        for file in files: 
            dp = DataPoint(os.path.join(self.dir_path, file)) 
            datapoints.append(dp) 
            dp.set_marker_length(self.marker_length)
            dp.set_camera_matrix(self.K)
        return datapoints 
    
    def run_CCV_detection(self):
        self.CCV_pose_list = []  
        self.CCV_detected_list = [] 
        self.CCV_marker_brightness_list = [] 
        for dp in self.datapoints: 
            image = np.array(cv2.imread(dp.image_path)) 
            tf_c_m, xyzabc_c_m, corners = opencv_marker_pose(image, self.K, self.dist_coeffs, self.marker_length, self.aruco_dict) 
            if tf_c_m is not None:
                self.CCV_pose_list.append(tf_c_m) 
                self.CCV_detected_list.append(True) 
            else: 
                self.CCV_pose_list.append(None) 
                self.CCV_detected_list.append(False)

            self.CCV_pose_list_filtered = [pose for pose in self.CCV_pose_list if pose is not None] 
            self.CCV_pose_mean = np.mean(self.CCV_pose_list_filtered, axis=0) 

        self.CCV_detection_rate = np.sum(self.CCV_detected_list) / len(self.CCV_detected_list) 
        print(f"CCV detection rate: {self.CCV_detection_rate:.2f}")  

    def get_marker_brightness(self): 
        marker_brightness = [] 
        for idx, dp in enumerate(self.datapoints): 
            self.datapoints[idx].set_pose("OPTK", self.CCV_pose_mean) 
            marker_brightness.append(dp.get_marker_brightness()) 
        self.marker_brightness_list = marker_brightness 

    def get_ground_truth_mask(self): 
        # get mask by projecting CCV_pose_mean 
        # onto the image plane and creating a mask
        # for the marker area
        self.true_seg = get_marker_segmentation(self.datapoints[0].get_image(), self.CCV_pose_mean, self.marker_length, self.K) / 255 

    def load_seg_model(self, seg_model_path): 
        # Load the segmentation model from the specified path
        self.seg_model = UNETWithDropout(in_channels=3, out_channels=1).to(self.device)
        self.seg_model.load_state_dict((torch.load(seg_model_path, map_location=self.device)["state_dict"])) 
        self.seg_model.eval() 

    def run_LBCV_detection(self, seg_thresh=0.5, save_segmentation=False):
        seg_size = (640, 480)
        seg_transform = A.Compose([
            A.Normalize(max_pixel_value=1.0),
            ToTensorV2()
        ])

        self.LBCV_detected_list = [] 
        self.LBCV_IOU_list = [] 

        # resize the true segmentation mask to the same size as the input image 
        true_seg = cv2.resize(self.true_seg, seg_size)  # shape (H, W, 3) 

        for dp in self.datapoints: 
            image = np.array(cv2.imread(dp.image_path)) 
            resized_rgb = cv2.resize(image, seg_size)  # shape (H, W, 3)
            transformed = seg_transform(image=resized_rgb)
            img_tensor = transformed["image"].unsqueeze(0).to(self.device)

            with torch.no_grad():
                seg_mask = torch.sigmoid(self.seg_model(img_tensor))
                seg_mask = (seg_mask > seg_thresh).float().cpu()

            seg_mask_img = np.array(transforms.ToPILImage()(seg_mask.squeeze(0)))/255 # shape matches resized_rgb 
            self.LBCV_IOU_list.append(seg_IOU(true_seg, seg_mask_img)) 

            if save_segmentation:
                os.makedirs(os.path.join(self.out_dir, "LBCV_segmentation"), exist_ok=True)
                seg_mask_img = (seg_mask_img * 255).astype(np.uint8)  # Convert to uint8 for saving
                cv2.imwrite(os.path.join(self.out_dir, "LBCV_segmentation", os.path.basename(dp.image_path)), seg_mask_img)

            if seg_mask_img.sum() > 100:  # Check if any pixels are segmented
                self.LBCV_detected_list.append(True) 
            else: 
                self.LBCV_detected_list.append(False) 

    def compare_detection_rates(self): 
        # Compare detection rates between CCV and LBCV
        CCV_detection_rate = np.sum(self.CCV_detected_list) / len(self.CCV_detected_list) 
        LBCV_detection_rate = np.sum(self.LBCV_detected_list) / len(self.LBCV_detected_list) 

        print(f"CCV detection rate: {CCV_detection_rate:.2f}")  
        print(f"LBCV detection rate: {LBCV_detection_rate:.2f}") 

        # plot LBCV IOU vs marker brightness coplotted with CCV detection rate vs marker brightness 
        plt.figure(figsize=(10, 5))
        plt.scatter(self.marker_brightness_list, self.LBCV_IOU_list, label="LBCV IOU", color="blue")
        plt.scatter(self.marker_brightness_list, self.CCV_detected_list, label="CCV Detection", color="red")
        plt.xlabel("Marker Brightness")
        plt.ylabel("Detection Bool / IOU")
        plt.title("Detection Rate Comparison")
        plt.grid() 
        plt.ylim(0, 1)
        plt.legend()
        plt.savefig(os.path.join(self.out_dir, "detection_rate_comparison.png"))
        plt.close()
                
        
def main(): 

    fx, fy, cx, cy, dist_coeffs = (1363.85, 1365.40, 958.58, 552.25, np.array([0.1693, -0.4755, 0.0018, 0.0023, 0.4114]))

    config = {
        "dir_path": "./real_data_processing/raw_data/controlled_tests/TEST",
        "camera_intrinsic_matrix": np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
        "camera_dist_coeffs": dist_coeffs,
        "aruco_dict": cv2.aruco.DICT_APRILTAG_36h11,
        "marker_length": 0.0798,
        "out_dir": f"./real_data_processing/raw_data/controlled_tests/TEST/results",
    }

    processor = Processor(config)
    processor.load_seg_model("./segmentation_model/models/my_checkpoint_20250329.pth.tar")  # Load the segmentation model 
    processor.run_CCV_detection()
    processor.get_ground_truth_mask()  
    processor.get_marker_brightness()
    processor.run_LBCV_detection(seg_thresh=0.001, save_segmentation=True)  # Save segmentation results
    processor.compare_detection_rates()
    # "./keypoints_model/models/my_checkpoint_keypoints_20250401.pth"
    print("Processing complete.")

if __name__ == "__main__":
    main()