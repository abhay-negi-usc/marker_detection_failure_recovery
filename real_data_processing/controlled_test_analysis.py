import numpy as np 
import os 
import json
import csv
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

        os.makedirs(self.out_dir, exist_ok=True)

        self.datapoints = self.read_frames() 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def read_frames(self):
        datapoints = [] 
        files = os.listdir(self.dir_path) 
        files = [f for f in files if f.endswith('.png') or f.endswith('.jpg')]  
        for file in files: 
            dp = DataPoint(os.path.join(self.dir_path, file)) 
            dp.set_marker_length(self.marker_length)
            dp.set_camera_matrix(self.K)
            datapoints.append(dp) 
        return datapoints 
    
    def run_CCV_detection(self):
        self.CCV_pose_list = []  
        self.CCV_detected_list = [] 
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
        self.CCV_pose_mean = np.mean(self.CCV_pose_list_filtered, axis=0) if self.CCV_pose_list_filtered else None
        self.CCV_detection_rate = np.sum(self.CCV_detected_list) / len(self.CCV_detected_list) 
        print(f"CCV detection rate: {self.CCV_detection_rate:.2f}")  

    def get_marker_brightness(self): 
        marker_brightness = []
        marker_areas = []  
        for idx, dp in enumerate(self.datapoints): 
            dp.set_pose("OPTK", self.CCV_pose_mean) 
            marker_brightness.append(dp.get_marker_brightness())
            marker_areas.append(dp.get_marker_area())
        self.marker_brightness_list = marker_brightness
        self.marker_areas = marker_areas  

    def get_ground_truth_mask(self): 
        self.true_seg = get_marker_segmentation(self.datapoints[0].get_image(), self.CCV_pose_mean, self.marker_length, self.K) / 255 

    def load_seg_model(self, seg_model_path): 
        self.seg_model = UNETWithDropout(in_channels=3, out_channels=1).to(self.device)
        self.seg_model.load_state_dict(torch.load(seg_model_path, map_location=self.device)["state_dict"]) 
        self.seg_model.eval() 

    def run_LBCV_detection(self, seg_thresh=0.5, save_segmentation=False):
        seg_size = (640, 480)
        seg_transform = A.Compose([
            A.Normalize(max_pixel_value=1.0),
            ToTensorV2()
        ])

        self.LBCV_detected_list = [] 
        self.LBCV_IOU_list = [] 
        true_seg = cv2.resize(self.true_seg, seg_size)  

        for dp in self.datapoints: 
            image = np.array(cv2.imread(dp.image_path)) 
            resized_rgb = cv2.resize(image, seg_size)
            transformed = seg_transform(image=resized_rgb)
            img_tensor = transformed["image"].unsqueeze(0).to(self.device)

            with torch.no_grad():
                seg_mask = torch.sigmoid(self.seg_model(img_tensor))
                seg_mask = (seg_mask > seg_thresh).float().cpu()

            seg_mask_img = np.array(transforms.ToPILImage()(seg_mask.squeeze(0))) / 255
            self.LBCV_IOU_list.append(seg_IOU(true_seg, seg_mask_img)) 

            if save_segmentation:
                seg_dir = os.path.join(self.out_dir, "LBCV_segmentation")
                os.makedirs(seg_dir, exist_ok=True)
                seg_mask_img = (seg_mask_img * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(seg_dir, os.path.basename(dp.image_path)), seg_mask_img)

            self.LBCV_detected_list.append(seg_mask_img.sum() > 100)

    def compare_detection_rates(self): 
        CCV_detection_rate = np.sum(self.CCV_detected_list) / len(self.CCV_detected_list) 
        LBCV_detection_rate = np.sum(self.LBCV_detected_list) / len(self.LBCV_detected_list) 
        print(f"CCV detection rate: {CCV_detection_rate:.2f}")  
        print(f"LBCV detection rate: {LBCV_detection_rate:.2f}") 

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

    def save_results(self, filename="results"):
        csv_path = os.path.join(self.out_dir, f"{filename}.csv")
        json_path = os.path.join(self.out_dir, f"{filename}.json")

        rows = []
        for i, dp in enumerate(self.datapoints):
            rows.append({
                "filename": os.path.basename(dp.image_path),
                "CCV_detected": int(self.CCV_detected_list[i]),
                "LBCV_detected": int(self.LBCV_detected_list[i]),
                "marker_brightness": float(self.marker_brightness_list[i]),
                "marker_area": float(self.marker_areas[i]),
                "LBCV_IOU": float(self.LBCV_IOU_list[i])
            })

        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        summary = {
            "CCV_detection_rate": float(self.CCV_detection_rate),
            "LBCV_detection_rate": float(np.sum(self.LBCV_detected_list) / len(self.LBCV_detected_list)),
            "CCV_pose_mean": self.CCV_pose_mean.tolist() if self.CCV_pose_mean is not None else None,
        }

        with open(json_path, 'w') as jsonfile:
            json.dump(summary, jsonfile, indent=2)


def main(): 
    fx, fy, cx, cy = 1363.85, 1365.40, 958.58, 552.25
    dist_coeffs = np.array([0.1693, -0.4755, 0.0018, 0.0023, 0.4114])

    config = {
        "dir_path": "./real_data_processing/raw_data/controlled_tests/dark_test_3",
        "camera_intrinsic_matrix": np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
        "camera_dist_coeffs": dist_coeffs,
        "aruco_dict": cv2.aruco.DICT_APRILTAG_36h11,
        "marker_length": 0.0798,
        "out_dir": "./real_data_processing/raw_data/controlled_tests/TEST/results",
    }

    processor = Processor(config)
    processor.load_seg_model("./segmentation_model/models/my_checkpoint_20250329.pth.tar")
    processor.run_CCV_detection()
    processor.get_ground_truth_mask()  
    processor.get_marker_brightness()
    processor.run_LBCV_detection(seg_thresh=0.001, save_segmentation=False)
    processor.compare_detection_rates()
    processor.save_results(filename="dark_test_3")
    print("Processing complete. Results saved.")

if __name__ == "__main__":
    main()
