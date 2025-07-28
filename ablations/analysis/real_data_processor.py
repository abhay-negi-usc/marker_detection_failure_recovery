import os 
import json 
from pathlib import Path
import cv2
import numpy as np
import pandas as pd 
import logging
import torch 
from PIL import Image
import albumentations as A
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2
import yaml 

from segmentation_model.model import UNETWithDropout
from segmentation_model.utils import load_checkpoint as load_seg_ckpt
from keypoints_model.model import RegressorMobileNetV3
from keypoints_model.utils import load_checkpoint as load_kp_ckpt
from real_data_processing.utils import marker_pose_estimation_estimatePoseSingleMarkers, get_marker_segmentation
from keypoints_model.utils import compute_2D_gridpoints
from ablations.analysis.utils import * 
from keypoints_model.utils import xyzabc_to_tf, rvectvec_to_xyzabc
from pose_estimation_model.utils import compute_segmentation_IOU 
from pose_estimation_model.utils import * 
from utils.pose_estimation_utils import * 

logger = logging.getLogger(__name__)

class DataPoint():
    def __init__(self, idx):
        self.idx = idx

    def set_image_path(self, image_path):
        self.image_path = image_path

    def set_metadata(self, data):
        # parse the data and set the metadata attributes
        self.metadata = data

    def set_camera_matrix(self, camera_matrix):
        self.camera_matrix = camera_matrix

    def set_marker_length(self, marker_length):
        self.marker_length = marker_length
        self.marker_length_without_border = marker_length * 0.8 # FIXME: hardcoded for now, should be set in config based on marker pattern 
        self.marker_corners_black_border = np.array([
            [+self.marker_length_without_border/2, +self.marker_length_without_border/2, 0],
            [-self.marker_length_without_border/2, +self.marker_length_without_border/2, 0],
            [-self.marker_length_without_border/2, -self.marker_length_without_border/2, 0],
            [+self.marker_length_without_border/2, -self.marker_length_without_border/2, 0]
        ])
        self.marker_corners = np.array([
            [+marker_length/2, -marker_length/2, 0],
            [-marker_length/2, -marker_length/2, 0],
            [-marker_length/2, +marker_length/2, 0],
            [+marker_length/2, +marker_length/2, 0]
        ])

    def set_true_pose(self, tf): 
        self.tf_true = tf  

    def set_tf_CCV(self, tf):
        self.tf_CCV = tf 
        # if self.tf_true exists 
        if hasattr(self, 'tf_true') and self.tf_true is not None and tf is not None:
            self.tf_error_CCV = compute_tf_error(self.tf_true, self.tf_CCV) 
            self.pose_error_CCV = tf_to_pose(self.tf_error_CCV) if self.tf_error_CCV is not None else None 

    def set_corners_CCV(self, corners):
        self.corners_CCV = corners
    
    def set_detected_CCV(self, bool_detected):
        self.detected_CCV = bool_detected

    def set_tf_LBCV(self, tf):
        self.tf_LBCV = tf 
        # if self.tf_true exists 
        if hasattr(self, 'tf_true') and self.tf_true is not None and tf is not None:
            self.tf_error_LBCV = compute_tf_error(self.tf_true, self.tf_LBCV) 
            self.pose_error_LBCV = tf_to_pose(self.tf_error_LBCV) if self.tf_error_LBCV is not None else None 

    def set_keypoints_LBCV(self, keypoints):
        self.keypoints_LBCV = keypoints 
        if keypoints is not None: 
            len_keypoints = len(keypoints)
            len_keypoints_side = int(np.sqrt(len_keypoints)) 
            corners_idx = np.array([0, len_keypoints_side-1, len_keypoints-1, len_keypoints-len_keypoints_side])
            corners = keypoints[corners_idx, :2]  # Extract only x, y coordinates
            self.set_corners_LBCV(corners) 
        else: 
            self.keypoints_LBCV = None 
            self.corners_LBCV = None 
            self.corners_error_LBCV = None 
            self.mean_corners_error_LBCV = None

    def set_corners_LBCV(self, corners):
        self.corners_LBCV = corners
        if hasattr(self, 'corners_true'): 
            # compute mean corners error in pixel distance by finding closest corners in self.corners_true 
            distances = np.linalg.norm(self.corners_true[:, np.newaxis, :] - self.corners_LBCV[np.newaxis, :, :], axis=-1)  # shape (4, 4)
            closest_indices = np.argmin(distances, axis=1)
            self.corners_error_LBCV = np.linalg.norm(self.corners_true - self.corners_LBCV[closest_indices], axis=1)  # shape (4,) # FIXME: should use the appropriate indices for the corners 
            self.mean_corners_error_LBCV = np.mean(self.corners_error_LBCV)  # scalar value 
    
    def set_detected_LBCV(self, bool_detected):
        self.detected_LBCV = bool_detected

    def set_LBCV_IOU(self, iou):
        self.LBCV_IOU = iou

    def set_LBCV_mean_mask_score(self, mean_mask_score):
        self.LBCV_mean_mask_score = mean_mask_score

    def set_PBCV_IOU(self, iou):
        self.PBCV_IOU = iou

    def set_fraction_marker_viewable(self, fraction):
        self.fraction_marker_visible = fraction 

    def set_detected_HCV(self, bool_detected):
        self.detected_HCV = bool_detected

    def set_tf_HCV(self, tf):
        self.tf_HCV = tf 
        # if self.tf_true exists 
        if hasattr(self, 'tf_true') and self.tf_true is not None and tf is not None:
            self.tf_error_HCV = compute_tf_error(self.tf_true, self.tf_HCV) 
            self.pose_error_HCV = tf_to_pose(self.tf_error_HCV) if self.tf_error_HCV is not None else None
        else: 
            self.tf_error_HCV = None 
            self.pose_error_HCV = None
    
    def set_corners_HCV(self, corners):
        self.corners_HCV = corners

    def set_tf_PBCV(self, tf):
        self.tf_PBCV = tf 
        # if self.tf_true exists 
        if hasattr(self, 'tf_true') and self.tf_true is not None and tf is not None:
            self.tf_error_PBCV = compute_tf_error(self.tf_true, self.tf_PBCV) 
            self.pose_error_PBCV = tf_to_pose(self.tf_error_PBCV) if self.tf_error_PBCV is not None else None 
            if hasattr(self, 'corners_true'): 
                self.corners_PBCV = project_points_array_to_image(C=self.camera_matrix,T=self.tf_PBCV, P_array=self.marker_corners, convert_cam_is2cv=True)
                self.corners_error_PBCV = np.linalg.norm(self.corners_true - self.corners_PBCV, axis=1)  # shape (4,) # FIXME: should use the appropriate indices for the corners
                self.mean_corners_error_PBCV = np.mean(self.corners_error_PBCV)
        else: 
            self.tf_error_PBCV = None 
            self.pose_error_PBCV = None

    def set_keypoints_PBCV(self, keypoints):
        self.keypoints_PBCV = keypoints

    def set_detected_PBCV(self, bool_detected):
        self.detected_PBCV = bool_detected

    def set_detection_scores(self, harris_corner_response_score=None, num_valid_proj_points=None, keypoint_residual_score=None, detection_score=None):
        self.harris_corner_response_score = harris_corner_response_score
        self.num_valid_proj_points = num_valid_proj_points 
        self.keypoint_residual_score = keypoint_residual_score
        self.detection_score = detection_score

    def set_image_similarity_score(self, image_similarity_score):
        self.image_similarity_score = image_similarity_score

    def get_segmentation(self, square_length, camera_matrix): 
        if hasattr(self, 'image_path') and hasattr(self, 'tf_true') and self.image_path is not None and self.tf_true is not None:
            image = cv2.imread(self.image_path)
            image_segmentation = get_marker_segmentation(
                image = image, 
                tf = self.tf_true, 
                square_length = square_length, 
                K = camera_matrix 
            )
            del image # free memory
            return image_segmentation
        else:
            raise ValueError("Image path not set for this DataPoint.")
        
    def get_corners_true(self, square_length, camera_matrix):
        if hasattr(self, 'tf_true') and self.tf_true is not None:
            square_corners_3d = np.array([
                [square_length/2, square_length/2, 0],
                [-square_length/2, square_length/2, 0],
                [-square_length/2, -square_length/2, 0],
                [square_length/2, -square_length/2, 0]
            ])  # shape (4, 3)

            # Extract rotation and translation
            R_wc = self.tf_true[:3, :3]
            t_wc = self.tf_true[:3, 3]

            # Transform corners to camera frame
            square_corners_cam = (R_wc @ square_corners_3d.T + t_wc.reshape(3, 1)).T  # shape (4, 3)

            # Project to 2D using intrinsic matrix
            square_corners_2d = (camera_matrix @ square_corners_cam.T).T  # shape (4, 3)
            square_corners_2d = square_corners_2d[:, :2] / square_corners_2d[:, 2:3]  # normalize
            self.corners_true = square_corners_2d  # shape (4, 2)

            return self.corners_true 
class DataProcessor(): 
    def __init__(self, config):
        self.config = config
        self.set_directories(self.config["data_path"]) 
        self.set_parameters() 
        self.create_list_datapoints() 
        self.read_pose() 

    def set_parameters(self):
        self.max_num_datapoints = self.config["max_num_datapoints"] if "max_num_datapoints" in self.config else None
        self.camera_parameters = self.config["camera_parameters"]
        self.marker_parameters = self.config["marker_parameters"]
        fx = self.camera_parameters["fx"]
        fy = self.camera_parameters["fy"]
        cx = self.camera_parameters["cx"]
        cy = self.camera_parameters["cy"]
        width = self.camera_parameters["width"]
        height = self.camera_parameters["height"]

        self.camera_matrix = np.array([[fx, 0, cx],
                                       [0, fy, cy],
                                       [0, 0, 1]], dtype=np.float32)
        
        self.camera_matrix_resized = np.array([[fx * 640 / width , 0, cx * 640 / width ],
                                               [0, fy * 480 / height, cy * 480 / height],
                                               [0, 0, 1]], dtype=np.float32)

        self.dist_coeffs = np.array(self.camera_parameters["distortion_coefficients"], dtype=np.float32)
        if len(self.dist_coeffs) == 0:
            self.dist_coeffs = np.zeros(5, dtype=np.float32)
        elif len(self.dist_coeffs) != 5:
            raise ValueError("Distortion coefficients must be a list of 5 elements.")
        self.aruco_dict = self.marker_parameters["aruco_dict"]
        self.marker_length = self.marker_parameters["marker_length"]
        self.marker_length_without_border = self.marker_parameters["marker_length_without_border"]
        self.num_squares = self.marker_parameters["num_squares"] 

        self.segmentation_model_path = self.config.get("seg_model_path", None)
        self.keypoints_model_path = self.config.get("kp_model_path", None)
        self.device = self.config.get("device", "cpu") 

    def set_directories(self, directory):
        self.directory = directory
        self.dir_images = os.path.join(directory, "images")
        self.tf_m_c_path = os.path.join(directory, "tf_m_c.csv")

    def create_list_datapoints(self):
        # create a list of datapoint objects from the metadata directory
        self.datapoints = [] 
        if self.max_num_datapoints is not None:
            self.max_num_datapoints = min(self.max_num_datapoints, len(os.listdir(self.dir_images))) 
        else:
            self.max_num_datapoints = len(os.listdir(self.dir_images))
        for idx in range(self.max_num_datapoints): 
            datapoint = DataPoint(idx)
            datapoint.set_image_path(os.path.join(self.dir_images, f"picture_{idx}.png"))
            datapoint.set_camera_matrix(self.camera_matrix)
            datapoint.set_marker_length(self.marker_length)
            self.datapoints.append(datapoint)

        # check if tf_c_m.csv exists 
        if not os.path.exists(self.tf_m_c_path):
            tf = None 
            for datapoint in self.datapoints:
                image = cv2.imread(datapoint.image_path) 
                ids, rvecs, tvecs, corners_tuple = marker_pose_estimation_estimatePoseSingleMarkers(
                    image,
                    self.camera_matrix,
                    self.dist_coeffs,
                    self.aruco_dict,
                    self.marker_length_without_border,
                    show=False
                )
                del image 
                if ids is not None and rvecs is not None and tvecs is not None:
                    R_matrix, _ = cv2.Rodrigues(rvecs[0])
                    tf = np.eye(4)
                    tf[:3, :3] = R_matrix
                    tf[:3, 3] = tvecs[0].reshape(3)
                    break 
            if tf is None:
                logger.warning("[CCV] No marker detected in any image. Cannot create tf_m_c.csv.")
                tf = np.eye(4)  # Default to identity matrix if no marker is detected
            # save tf to tf_m_c.csv 
            list_tf = [tf.flatten() for _ in range(len(self.datapoints))]  # Initialize with identity matrices
            df_tf_m_c = pd.DataFrame(list_tf, columns=["m00", "m01", "m02", "m03",  
                                                        "m10", "m11", "m12", "m13",
                                                        "m20", "m21", "m22", "m23",
                                                        "m30", "m31", "m32", "m33"])
            df_tf_m_c.to_csv(self.tf_m_c_path, index=False) 

    def read_pose(self):
        
        df_tf_m_c = pd.read_csv(self.tf_m_c_path) 


        array_tf_m_c = df_tf_m_c.values.reshape(-1, 4, 4) 

        assert(len(self.datapoints) == len(array_tf_m_c), "[CCV] Number of datapoints does not match number of poses in tf_m_c.csv")

        self.tf_marker = []


        for idx, datapoint in enumerate(self.datapoints):
            tf_marker = np.linalg.inv(array_tf_m_c[idx])  # Inverse to get marker wrt camera transform
            # tf_marker = array_tf_c_m[idx] # Inverse to get marker wrt camera transform

            self.tf_marker.append(tf_marker)
            self.datapoints[idx].set_true_pose(tf_marker) 
            self.datapoints[idx].get_corners_true(self.marker_length, self.camera_matrix)

            if self.max_num_datapoints is not None and len(self.tf_marker) >= self.max_num_datapoints:
                break

    def run_opencv_fiducial_marker_detection(self, save_results=False):
        
        output_dir = os.path.join(self.directory, "CCV_results")
        os.makedirs(output_dir, exist_ok=True)

        # self.image_paths = sorted([f for f in os.listdir(self.dir_rgb) if f.endswith(".png") or f.endswith(".jpg")]) 
        # self.image_paths = self.image_paths[:self.max_num_datapoints] if self.max_num_datapoints is not None else self.image_paths

        # for idx, image_path in enumerate(self.image_paths):
        for idx, datapoint in enumerate(self.datapoints):
            image_path = datapoint.image_path 
            image = cv2.imread(image_path)
            # image = cv2.imread(os.path.join(self.dir_rgb, str(image_path)))
            if image is None:
                logger.warning(f"[CCV] Could not read image {image_path}")
                continue

            ids, rvecs, tvecs, corners_tuple = marker_pose_estimation_estimatePoseSingleMarkers(
                image,
                self.camera_matrix,
                self.dist_coeffs,
                self.aruco_dict,
                self.marker_length_without_border,
                show=False
            )

            if ids is not None and rvecs is not None and tvecs is not None:
                R_matrix, _ = cv2.Rodrigues(rvecs[0])
                tf = np.eye(4)
                tf[:3, :3] = R_matrix
                tf[:3, 3] = tvecs[0].reshape(3)
                corners = np.array([corner.reshape(-1, 2) for corner in corners_tuple]) 
                self.datapoints[idx].CCV_detected = True 
                self.datapoints[idx].set_tf_CCV(tf) 
                self.datapoints[idx].set_corners_CCV(corners)
            else: 
                self.datapoints[idx].CCV_detected = False 
                self.datapoints[idx].set_tf_CCV(None)
                self.datapoints[idx].set_corners_CCV(None)

            if save_results:
                out_img = image.copy()
                if ids is not None:
                    out_img = cv2.aruco.drawDetectedMarkers(out_img, corners_tuple, ids)
                outpath = os.path.join(output_dir, f"CCV_{idx:05d}.png")
                cv2.imwrite(str(outpath), out_img)

    def setup_models(self): 
        self.keypoints_ref = np.array(
            compute_2D_gridpoints(N=self.num_squares, s=self.marker_length)
        )
        # self.corners_ref = np.array([
        #     [+self.marker_length, +self.marker_length, 0],
        #     [-self.marker_length, +self.marker_length, 0],
        #     [-self.marker_length, -self.marker_length, 0],
        #     [+self.marker_length, -self.marker_length, 0],
        # ])
        self.corners_ref = np.array([
            [+self.marker_length/2, -self.marker_length/2, 0],
            [-self.marker_length/2, -self.marker_length/2, 0],
            [-self.marker_length/2, +self.marker_length/2, 0],
            [+self.marker_length/2, +self.marker_length/2, 0],
        ])
        self.seg_transform = Compose([Normalize(max_pixel_value=1.0), ToTensorV2()])
        self.seg_model = UNETWithDropout(in_channels=3, out_channels=1).to(self.device)
        load_seg_ckpt(torch.load(self.segmentation_model_path, map_location=self.device), self.seg_model)
        self.seg_model.eval()
        self.kp_transform = A.Compose([ToTensorV2()]) 
        self.kp_model = RegressorMobileNetV3().to(self.device)
        load_kp_ckpt(torch.load(self.keypoints_model_path, map_location=self.device), self.kp_model)
        self.kp_model.eval()

    def run_LBCV_segmentation(self, image, detection_threshold=1000):         
        img_tensor = self.seg_transform(image=image)["image"].unsqueeze(0).to(self.device)
        with torch.no_grad():
            seg_mask = torch.sigmoid(self.seg_model(img_tensor))
            mean_mask_score = seg_mask.mean().item()  # Get the mean score of the segmentation mask
            seg_mask = (seg_mask > 0.5).float().cpu()
            seg_mask_img = Image.fromarray(seg_mask.squeeze().numpy().astype(np.uint8) * 255)
        if np.count_nonzero(np.array(seg_mask_img)) > detection_threshold:
            bool_detected = True  
        else:
            bool_detected = False

        return seg_mask_img, bool_detected, mean_mask_score
    
    def compute_roi(self, seg, rgb):

        padding = 5
        roi_size = 128
        image_border_size = np.max([np.array(seg).shape[0], np.array(seg).shape[1]])

        seg = np.array(seg)
        seg = cv2.copyMakeBorder(seg, image_border_size, image_border_size, image_border_size, image_border_size, cv2.BORDER_CONSTANT, value=0)
        # only keep largest blob in seg
        num_labels, labels_im = cv2.connectedComponents(seg.astype(np.uint8), connectivity=8)
        largest_label = 1 + np.argmax(np.bincount(labels_im.flat)[1:])
        seg = (labels_im == largest_label).astype(np.uint8) * 255
        tag_pixels = np.argwhere(seg == 255)
        if tag_pixels.size < 1000: # min number of pixels to consider a tag, value from training data filtering 
            return None, None

        seg_tag_min_x = np.min(tag_pixels[:, 1])
        seg_tag_max_x = np.max(tag_pixels[:, 1])
        seg_tag_min_y = np.min(tag_pixels[:, 0])
        seg_tag_max_y = np.max(tag_pixels[:, 0])
        seg_height = seg_tag_max_y - seg_tag_min_y
        seg_width = seg_tag_max_x - seg_tag_min_x
        seg_center_x = (seg_tag_min_x + seg_tag_max_x) // 2
        seg_center_y = (seg_tag_min_y + seg_tag_max_y) // 2

        if isinstance(rgb, str):
            rgb = np.array(cv2.imread(rgb))
        if isinstance(rgb, Image.Image):
            rgb = np.array(rgb)
        if isinstance(rgb, np.ndarray):
            rgb = rgb
        rgb = cv2.copyMakeBorder(rgb, image_border_size, image_border_size, image_border_size, image_border_size, cv2.BORDER_CONSTANT, value=0)

        rgb_side = max(seg_height, seg_width) + 2 * padding
        rgb_tag_min_x = seg_center_x - rgb_side // 2
        rgb_tag_max_x = seg_center_x + rgb_side // 2
        rgb_tag_min_y = seg_center_y - rgb_side // 2
        rgb_tag_max_y = seg_center_y + rgb_side // 2
        roi_img = rgb[rgb_tag_min_y:rgb_tag_max_y, rgb_tag_min_x:rgb_tag_max_x, :]
        roi_img = cv2.resize(roi_img, (roi_size, roi_size))
        roi_coordinates = np.array([rgb_tag_min_x, rgb_tag_max_x, rgb_tag_min_y, rgb_tag_max_y]) - image_border_size 

        return roi_img, roi_coordinates

    def run_LBCV_keypoints_estimation(self, image_roi): 
        roi_tensor = self.kp_transform(image=image_roi)["image"].unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            keypoints_roi = self.kp_model(roi_tensor).cpu().numpy().reshape(-1, 2)
        return keypoints_roi 

    def scale_keypoints_to_original_image(self, keypoints_roi, image_roi, coords_roi, image): 
        roi_height, roi_width = image_roi.shape[:2]
        w = coords_roi[1] - coords_roi[0]
        h = coords_roi[3] - coords_roi[2]

        scale_x = w / roi_width
        scale_y = h / roi_height

        origin_x = coords_roi[0]
        origin_y = coords_roi[2]

        keypoints_in_resized_rgb = np.stack([
            keypoints_roi[:, 0] * scale_x + origin_x,
            keypoints_roi[:, 1] * scale_y + origin_y
        ], axis=1)

        # Step 2: remap from resized RGB (640×480) to original image
        H_orig, W_orig = image.shape[:2]
        H_resized, W_resized = image.shape[:2]

        scale_x_back = W_orig / W_resized
        scale_y_back = H_orig / H_resized

        keypoints_img = np.stack([
            keypoints_in_resized_rgb[:, 0] * scale_x_back,
            keypoints_in_resized_rgb[:, 1] * scale_y_back
        ], axis=1)

        return keypoints_img  

    def estimate_tf_from_keypoints(self, keypoints_ref, keypoints_est): 
        # Check if keypoints_est is None or invalid
        if keypoints_est is None:
            return None
        
        # Check if keypoints_ref is None or invalid
        if keypoints_ref is None:
            return None
        
        # Ensure keypoints_est is a numpy array with correct shape and type
        if not isinstance(keypoints_est, np.ndarray):
            return None
        
        # Ensure keypoints_ref is a numpy array with correct shape and type
        if not isinstance(keypoints_ref, np.ndarray):
            return None
        
        # Ensure it's the correct shape (N, 2) for image points and (N, 3) for object points
        if keypoints_est.ndim != 2 or keypoints_est.shape[1] != 2:
            return None
        
        if keypoints_ref.ndim != 2 or keypoints_ref.shape[1] != 3:
            return None
        
        # Ensure they have the same number of points
        if keypoints_est.shape[0] != keypoints_ref.shape[0]:
            return None
        
        # Ensure it's float32/float64 type as expected by cv2.solvePnP
        keypoints_est = keypoints_est.astype(np.float32)
        keypoints_ref = keypoints_ref.astype(np.float32)

        success, rvec, tvec = cv2.solvePnP(
            objectPoints=keypoints_ref,
            imagePoints=keypoints_est,
            cameraMatrix=self.camera_matrix,
            distCoeffs=self.dist_coeffs,
        )
        if not success:
            return None 
        else: 
            pose_marker = rvectvec_to_xyzabc(rvec, tvec)
            tf_marker = xyzabc_to_tf(pose_marker)
            return tf_marker 

    def find_closest_symmetric_pose(self, tf_est, tf_ref): 
        tf_z_90 = np.array([
            [0, -1, 0, 0],
            [1,  0, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ])
        tf_z_180 = tf_z_90 @ tf_z_90
        tf_z_270 = tf_z_180 @ tf_z_90

        tf_candidates = [
            tf_est,
            tf_est @ tf_z_90,
            tf_est @ tf_z_180,
            tf_est @ tf_z_270
        ]

        min_error = float('inf')
        best_tf = None
        for tf_candidate in tf_candidates:
            error = np.linalg.norm(tf_candidate[:3, :3] - tf_ref[:3, :3]) # angular error 
            if error < min_error:
                min_error = error
                best_tf = tf_candidate

        if best_tf is None:
            logger.warning("[LBCV] No valid symmetric pose found.")
            return None 
        else:
            return best_tf

    def get_true_segmentation(self, datapoint): 
        image_segmentation = datapoint.get_segmentation(
            square_length=self.marker_length, 
            camera_matrix=self.camera_matrix
        )
        return image_segmentation 

    def run_LBCV_fiducial_marker_detection(self, save_results=False, run_corners_HCV=False, run_PBCV=False, use_precomputed_segmentation=False): 
        self.setup_models() 
        # for idx, image_path in enumerate(self.image_paths):
        for idx, datapoint in enumerate(self.datapoints):
            image_path = datapoint.image_path
            image = np.array(cv2.imread(image_path))
            # image = np.array(cv2.imread(os.path.join(self.dir_rgb, str(image_path)))) 

            if use_precomputed_segmentation: 
                seg_path = os.path.join(os.path.dirname(datapoint.image_path).replace("images","LBCV_segmentation_results"), f"LBCV_segmentation_{datapoint.idx:05d}.png")
                image_segmentation = Image.open(seg_path).convert("L")  # Load as grayscale
                bool_detected = np.count_nonzero(np.array(image_segmentation)) > 100  # threshold for detection, can be adjusted, #NOTE: this may have to be adjusted 
            else: 
                image_segmentation, bool_detected, mean_mask_score = self.run_LBCV_segmentation(image) 

            if bool_detected: 
                # compute segmentation IOU
                image_seg_est_np = np.array(image_segmentation) 
                image_seg_true_np = self.get_true_segmentation(self.datapoints[idx]) 
                IOU = compute_segmentation_IOU(image_seg_est_np, image_seg_true_np) 
                image_roi, coords_roi = self.compute_roi(image_segmentation, image) 
                if image_roi is None or coords_roi is None: #or IOU < 0.25: # FIXME: using IOU < 0.25 as a threshold for detection, this is not ideal 
                    bool_detected = False
                    keypoints_est = None 
                    tf_est = None 
                    IOU = None
                else:
                    keypoints_roi = self.run_LBCV_keypoints_estimation(image_roi) 
                    keypoints_est = self.scale_keypoints_to_original_image(keypoints_roi, image_roi, coords_roi, image)
                    tf_est = self.estimate_tf_from_keypoints(self.keypoints_ref, keypoints_est) 
                    tf_correction = np.array([
                        [-1,0,0,0],
                        [0,-1,0,0],
                        [0,0,1,0],
                        [0,0,0,1] 
                    ])
                    tf_est = tf_est @ tf_correction 
                    # tf_est = tf_est 
            else: 
                keypoints_est = None 
                tf_est = None 
                IOU = None 
            self.datapoints[idx].set_detected_LBCV(bool_detected) 
            self.datapoints[idx].set_tf_LBCV(tf_est) 
            self.datapoints[idx].set_keypoints_LBCV(keypoints_est)
            self.datapoints[idx].set_LBCV_IOU(IOU)
            self.datapoints[idx].set_LBCV_mean_mask_score(mean_mask_score) 

            # hybrid method 
            if run_corners_HCV: 
                img_marker_path = "./synthetic_data_generation/assets/tags/tag36h11_0.png"
                img_marker = cv2.imread(img_marker_path)
                keypoints_marker_image_space = find_keypoints(img_marker)
                bool_detected_hybrid = False
                if bool_detected: 
                    # check if no segmentation points within margin of border 
                    margin = 10 # units: pixels 
                    if np.any(image_seg_est_np[:margin, :]) or \
                    np.any(image_seg_est_np[-margin:, :]) or \
                    np.any(image_seg_est_np[:, :margin]) or \
                    np.any(image_seg_est_np[:, -margin:]): 
                        bool_detected_hybrid = False 
                        self.datapoints[idx].set_detected_HCV(bool_detected_hybrid) 
                        self.datapoints[idx].set_corners_HCV(None) 
                        self.datapoints[idx].set_tf_HCV(None) 
                        
                    else: 
                        bool_detected_hybrid = True
                        # fit quadrilateral and find corners of segmentation mask 
                        corners_est, area_ratio = find_segmentation_four_corners(image_seg_est_np)
                        # solve for pose using the corners 
                        tf_est_hcv = self.estimate_tf_from_keypoints(self.corners_ref, corners_est)
                        
                        # Check if pose estimation was successful
                        if tf_est_hcv is None:
                            bool_detected_hybrid = False
                            self.datapoints[idx].set_detected_HCV(bool_detected_hybrid) 
                            self.datapoints[idx].set_corners_HCV(None) 
                            self.datapoints[idx].set_tf_HCV(None)
                        else:
                            tf_est_corrected = tf_est
                            tf_est_hcv = self.find_closest_symmetric_pose(tf_est_hcv, tf_est_corrected)
                            self.datapoints[idx].set_detected_HCV(bool_detected_hybrid) 
                            self.datapoints[idx].set_corners_HCV(corners_est) 
                            self.datapoints[idx].set_tf_HCV(tf_est_hcv) 

            # pattern hybrid method 
            if run_PBCV: 
                print(idx)
                # corners, area_ratio = find_segmentation_four_corners(image_seg_est_np, bound_box=False)
                # if corners is None or image is None or image_seg_est_np is None or area_ratio<0.5 or np.count_nonzero(image_seg_est_np) < 1000:
                #     tf_PBCV = None 
                    # continue  
                # quad_seg = fill_segmentation_from_polygon(image_seg_est_np.shape, corners)
                # keypoints_rgb_image_space = find_keypoints(image, quad_seg)
                seg_mask_img_np = segmentation_biggest_blob_filter(image_seg_est_np, min_area=1000)
                keypoints_rgb_image_space = find_keypoints(image, seg_mask_img_np)
                keypoints_marker_cartesian_space = convert_marker_keypoints_to_cartesian(
                    keypoints_marker_image_space, image_size=(img_marker.shape[0], img_marker.shape[1]), marker_size=(0.1, 0.1)
                )
                if keypoints_rgb_image_space is not None and seg_mask_img_np is not None: 
                    tf_PBCV, residual = refine_pose_icp_3d2d_auto_match(
                        np.array(image), keypoints_marker_cartesian_space, keypoints_rgb_image_space, self.camera_matrix,
                        tf_est, max_iterations=10, show_iteration_images=False, max_keypoints_est_2d=72, output_final_image=True,
                    )
                    import pdb;pdb.set_trace()

                    if tf_PBCV is not None:
                        harris_corner_response_score, num_valid_proj_points, keypoint_residual_score, detection_score = compute_detection_score(
                            image, keypoints_marker_image_space, keypoints_marker_cartesian_space, tf_PBCV, self.camera_matrix, self.dist_coeffs, 
                            harris_corner_response_weight=1.0, keypoint_residual_score_weight=1.0
                        )
                        image_similarity_score = compute_image_similarity_score(image, img_marker, self.marker_length, tf_PBCV, self.camera_matrix, self.dist_coeffs)
                        PBCV_seg_mask_img_np = get_marker_segmentation(
                            image=image, 
                            tf=tf_PBCV, 
                            square_length=self.marker_length, 
                            K=self.camera_matrix
                        )
                        PBCV_IOU = compute_segmentation_IOU(PBCV_seg_mask_img_np, image_seg_est_np)
                        self.datapoints[idx].set_detection_scores(harris_corner_response_score, num_valid_proj_points, keypoint_residual_score, detection_score) 
                        self.datapoints[idx].set_tf_PBCV(tf_PBCV) 
                        self.datapoints[idx].set_keypoints_PBCV(keypoints_rgb_image_space) 
                        self.datapoints[idx].set_detected_PBCV(True) 
                        self.datapoints[idx].set_image_similarity_score(image_similarity_score)
                        self.datapoints[idx].set_PBCV_IOU(PBCV_IOU)
                        # NOTE: filtering out bad estimates based on scores 
                        bool_detected_PBCV = (harris_corner_response_score > 0.001) and (num_valid_proj_points > 2) and (tf_PBCV[2,3] < 10) and (image_similarity_score > 20_000) 
                        # self.datapoints[idx].set_detected_PBCV(bool_detected_PBCV)
                    else:
                        self.datapoints[idx].set_tf_PBCV(None)
                        self.datapoints[idx].set_keypoints_PBCV(None)
                        self.datapoints[idx].set_detected_PBCV(False)
                        self.datapoints[idx].set_detection_scores(None) 
                        self.datapoints[idx].set_image_similarity_score(None)

                else:
                    self.datapoints[idx].set_tf_PBCV(None)
                    self.datapoints[idx].set_keypoints_PBCV(None)
                    self.datapoints[idx].set_detected_PBCV(False)
                    self.datapoints[idx].set_detection_scores(None) 
                    self.datapoints[idx].set_image_similarity_score(None)

                # FIXME: this can be made more elegant 
                if not bool_detected: self.datapoints[idx].set_detected_PBCV(False) 

            if save_results:
                output_dir = os.path.join(self.directory, "LBCV_keypoints_results")
                os.makedirs(output_dir, exist_ok=True)
                out_img = image.copy()
                if bool_detected:
                    for kp in keypoints_est:
                        cv2.circle(out_img, tuple(kp.astype(int)), 3, (0, 255, 0), -1)
                outpath = os.path.join(output_dir, f"LBCV_{idx:05d}.png")
                cv2.imwrite(str(outpath), out_img)

                if run_corners_HCV:
                    output_dir = os.path.join(self.directory, "HCV_corners_results")
                    os.makedirs(output_dir, exist_ok=True)
                    out_img = image.copy()
                    if bool_detected_hybrid:
                        for kp in corners_est:
                            cv2.circle(out_img, tuple(kp.astype(int)), 3, (0, 255, 0), -1)
                    outpath = os.path.join(output_dir, f"HCV_{idx:05d}.png")
                    cv2.imwrite(str(outpath), out_img)

                output_dir = os.path.join(self.directory, "LBCV_segmentation_results") 
                os.makedirs(output_dir, exist_ok=True)
                out_segmentation_path = os.path.join(output_dir, f"LBCV_segmentation_{idx:05d}.png")
                image_segmentation.save(out_segmentation_path) 

                if run_PBCV:
                    output_dir = os.path.join(self.directory, "PBCV_keypoints_results")
                    os.makedirs(output_dir, exist_ok=True)
                    out_img = image.copy()
                    if self.datapoints[idx].detected_PBCV:
                        for kp in self.datapoints[idx].keypoints_PBCV:
                            cv2.circle(out_img, tuple(kp.astype(int)), 3, (0, 255, 0), -1)
                    outpath = os.path.join(output_dir, f"PBCV_{idx:05d}.png")
                    cv2.imwrite(str(outpath), out_img)


            del image 

    def compute_values(self): 
        for idx, datapoint in enumerate(self.datapoints): 
            image_segmentation = self.get_true_segmentation(datapoint) 
            original_image_segmentation = image_segmentation.copy()  # Keep original for later use
            if image_segmentation is not None:
                border_width = max(image_segmentation.shape)
                image_segmentation = cv2.copyMakeBorder(
                    image_segmentation, 
                    border_width, border_width, border_width, border_width, 
                    cv2.BORDER_CONSTANT, 
                    value=0
                ) 

            # 3D marker corners in marker frame (Z=0 plane)
            corners_marker = np.array([
                [ self.marker_length / 2,  self.marker_length / 2, 0],
                [-self.marker_length / 2,  self.marker_length / 2, 0],
                [-self.marker_length / 2, -self.marker_length / 2, 0],
                [ self.marker_length / 2, -self.marker_length / 2, 0]
            ])

            tf_true = datapoint.tf_true # 4x4
            Rot = tf_true[:3, :3]
            t = tf_true[:3, 3].reshape(3, 1)

            corners_camera = (Rot @ corners_marker.T + t).T  # shape (4, 3)

            # Project into image space using camera matrix
            corners_homog = corners_camera @ self.camera_matrix.T  # shape (4, 3)
            corners_image = corners_homog[:, :2] / corners_homog[:, 2:]

            # Create binary mask from projected marker corners
            mask_shape = image_segmentation.shape if image_segmentation is not None else (480, 640)
            marker_mask = np.zeros(mask_shape, dtype=np.uint8)

            # Convert to int pixel coords
            corners_int = np.round(corners_image).astype(np.int32)

            # Offset polygon if needed due to earlier border added
            offset = border_width if image_segmentation is not None else 0
            corners_int += offset

            # Draw filled polygon (projected marker area)
            cv2.fillConvexPoly(marker_mask, corners_int, 1)

            # Count pixels in projected marker area
            marker_pixel_count = marker_mask.sum()

            # Count visible marker pixels in segmentation
            visible_pixels = np.logical_and(marker_mask == 1, image_segmentation > 0).sum()

            # Compute visible fraction
            if marker_pixel_count > 0:
                fraction_visible = visible_pixels / marker_pixel_count
            else:
                fraction_visible = 0.0

            # get indices of visible pixels in segmentation
            visible_indices = np.argwhere(original_image_segmentation > 0) 
            # compute mean brightness of visible pixels in rgb image 
            rgb_image = cv2.imread(datapoint.image_path)
            visible_rgb_pixels = rgb_image[visible_indices[:, 0], visible_indices[:, 1]] # shape (N, 3) where N is number of visible pixels 
            mean_brightness = np.mean(visible_rgb_pixels, axis=0)
            mean_brightness = np.mean(mean_brightness)  # Average over RGB channels 

            num_saturated_high_pixels = np.sum(visible_rgb_pixels.mean(axis=1) > 250)  # Count pixels with brightness > 250 
            fraction_saturated_high_pixels = num_saturated_high_pixels / visible_rgb_pixels.shape[0] if visible_rgb_pixels.shape[0] > 0 else 0.0

            num_saturated_low_pixels = np.sum(visible_rgb_pixels.mean(axis=1) < 5)  # Count pixels with brightness < 5
            fraction_saturated_low_pixels = num_saturated_low_pixels / visible_rgb_pixels.shape[0] if visible_rgb_pixels.shape[0] > 0 else 0.0

            # Store in datapoint
            datapoint.set_fraction_marker_viewable(fraction_visible) 

            distance_to_camera = np.linalg.norm(datapoint.tf_true[:3,3])  # Distance from camera to marker center
            lateral_position = datapoint.tf_true[0, 3]  # Lateral position in camera frame (X-axis) 
            euler_angles = R.from_matrix(tf_true[:3,:3]).as_euler('xyz', degrees=True)  # Convert rotation matrix to Euler angles
            pitch = euler_angles[0]  # Pitch angle (rotation around X-axis)
            yaw = euler_angles[1]  # Yaw angle (rotation around Y-axis)
            roll = euler_angles[2]  # Roll angle (rotation around Z-axis) 

            datapoint_metadata = {
                "distance_to_camera": distance_to_camera, 
                "lateral": lateral_position,
                "pitch": pitch,
                "yaw": yaw,
                "roll": roll,
                # "truncation": datapoint.metadata.get("truncation", None),
                # "ambient_light_intensity": datapoint.metadata.get("ambient_light_intensity", None),
                # "underexposure": datapoint.metadata.get("underexposure", None),
                "mean_marker_pixel_brightness": mean_brightness,
                "fraction_saturated_high_pixels": fraction_saturated_high_pixels,
                "fraction_saturated_low_pixels": fraction_saturated_low_pixels,
            }
            datapoint.set_metadata(datapoint_metadata)

    def compile_results(self, save_results=False): 
        self.dict_results = [] # for storing unpacked results in a list of dictionaries
        self.df_results = pd.DataFrame(columns=[
            "idx", 
            "image_path",
            "background_id",
            "ambient_light_intensity", 
            "distance_to_camera", 
            "tf_true_Rxx",
            "tf_true_Rxy",
            "tf_true_Rxz",
            "tf_true_Ryx",
            "tf_true_Ryy",
            "tf_true_Ryz",
            "tf_true_Rzx",
            "tf_true_Rzy",
            "tf_true_Rzz",
            "tf_true_tx",
            "tf_true_ty",
            "tf_true_tz", 
            "detected_CCV", 
            "tf_error_CCV_Rxx",
            "tf_error_CCV_Rxy",
            "tf_error_CCV_Rxz",
            "tf_error_CCV_Ryx",
            "tf_error_CCV_Ryy",
            "tf_error_CCV_Ryz",
            "tf_error_CCV_Rzx",
            "tf_error_CCV_Rzy",
            "tf_error_CCV_Rzz",
            "tf_error_CCV_tx",
            "tf_error_CCV_ty",
            "tf_error_CCV_tz",
            "pose_error_CCV_x",
            "pose_error_CCV_y",
            "pose_error_CCV_z",
            "pose_error_CCV_a",
            "pose_error_CCV_b",
            "pose_error_CCV_c",
            "detected_LBCV",
            "tf_error_LBCV_Rxx",
            "tf_error_LBCV_Rxy",
            "tf_error_LBCV_Rxz",
            "tf_error_LBCV_Ryx",
            "tf_error_LBCV_Ryy",
            "tf_error_LBCV_Ryz",
            "tf_error_LBCV_Rzx",
            "tf_error_LBCV_Rzy",
            "tf_error_LBCV_Rzz",
            "tf_error_LBCV_tx",
            "tf_error_LBCV_ty",
            "tf_error_LBCV_tz",
            "pose_error_LBCV_x",
            "pose_error_LBCV_y",
            "pose_error_LBCV_z",
            "pose_error_LBCV_a",
            "pose_error_LBCV_b",
            "pose_error_LBCV_c",
            "detected_HCV",
            "tf_error_HCV_Rxx",
            "tf_error_HCV_Rxy",
            "tf_error_HCV_Rxz",
            "tf_error_HCV_Ryx",
            "tf_error_HCV_Ryy",
            "tf_error_HCV_Ryz",
            "tf_error_HCV_Rzx",
            "tf_error_HCV_Rzy",
            "tf_error_HCV_Rzz",
            "tf_error_HCV_tx",
            "tf_error_HCV_ty",
            "tf_error_HCV_tz",
            "pose_error_HCV_x",
            "pose_error_HCV_y",
            "pose_error_HCV_z",
            "pose_error_HCV_a",
            "pose_error_HCV_b",
            "pose_error_HCV_c",
            "lateral",
            "fraction_marker_visible",
            "mean_marker_pixel_brightness",
            "fraction_saturated_high_pixels",
            "fraction_saturated_low_pixels",
            "skew", 
            "pitch",
            "yaw",
            "roll",
            "detected_PBCV",
            "tf_error_PBCV_Rxx",
            "tf_error_PBCV_Rxy",
            "tf_error_PBCV_Rxz",
            "tf_error_PBCV_Ryx",    
            "tf_error_PBCV_Ryy",
            "tf_error_PBCV_Ryz",
            "tf_error_PBCV_Rzx",
            "tf_error_PBCV_Rzy",
            "tf_error_PBCV_Rzz",
            "tf_error_PBCV_tx", 
            "tf_error_PBCV_ty",
            "tf_error_PBCV_tz",
            "pose_error_PBCV_x",
            "pose_error_PBCV_y",
            "pose_error_PBCV_z",
            "pose_error_PBCV_a",
            "pose_error_PBCV_b",
            "pose_error_PBCV_c",
        ]) # for storing packed results in a pandas DataFrame format 

        for idx, datapoint in enumerate(self.datapoints): 
            self.df_results.loc[idx,"idx"] = datapoint.idx 
            self.df_results.loc[idx, "image_path"] = datapoint.image_path
            self.df_results.loc[idx, "fraction_marker_visible"] = datapoint.fraction_marker_visible 
            self.df_results.loc[idx, "detected_CCV"] = datapoint.CCV_detected 
            self.df_results.loc[idx, "tf_true_Rxx"] = datapoint.tf_true[0, 0]
            self.df_results.loc[idx, "tf_true_Rxy"] = datapoint.tf_true[0, 1]
            self.df_results.loc[idx, "tf_true_Rxz"] = datapoint.tf_true[0, 2]
            self.df_results.loc[idx, "tf_true_Ryx"] = datapoint.tf_true[1, 0]
            self.df_results.loc[idx, "tf_true_Ryy"] = datapoint.tf_true[1, 1]
            self.df_results.loc[idx, "tf_true_Ryz"] = datapoint.tf_true[1, 2]
            self.df_results.loc[idx, "tf_true_Rzx"] = datapoint.tf_true[2, 0]
            self.df_results.loc[idx, "tf_true_Rzy"] = datapoint.tf_true[2, 1]
            self.df_results.loc[idx, "tf_true_Rzz"] = datapoint.tf_true[2, 2]
            self.df_results.loc[idx, "tf_true_tx"] = datapoint.tf_true[0, 3]
            self.df_results.loc[idx, "tf_true_ty"] = datapoint.tf_true[1, 3]
            self.df_results.loc[idx, "tf_true_tz"] = datapoint.tf_true[2, 3]
            self.df_results.loc[idx, "detected_LBCV"] = datapoint.detected_LBCV
            self.df_results.loc[idx, "LBCV_mean_mask_score"] = datapoint.LBCV_mean_mask_score if hasattr(datapoint, 'LBCV_mean_mask_score') else None
            self.df_results.loc[idx, "detected_HCV"] = datapoint.detected_LBCV # FIXME 
            self.df_results.loc[idx, "background_id"] = datapoint.metadata.get("background_id", None)
            self.df_results.loc[idx, "ambient_light_intensity"] = datapoint.metadata.get("ambient_light_intensity", None)
            self.df_results.loc[idx, "distance_to_camera"] = datapoint.metadata.get("distance_to_camera", None)
            self.df_results.loc[idx, "lateral"] = datapoint.metadata.get("lateral", None)
            self.df_results.loc[idx,    "skew"] = datapoint.metadata.get("skew", None)
            self.df_results.loc[idx, "pitch"] = datapoint.metadata.get("pitch", None)
            self.df_results.loc[idx, "yaw"] = datapoint.metadata.get("yaw", None)
            self.df_results.loc[idx, "roll"] = datapoint.metadata.get("roll", None)
            self.df_results.loc[idx, "mean_marker_pixel_brightness"] = datapoint.metadata.get("mean_marker_pixel_brightness", None)
            self.df_results.loc[idx, "fraction_saturated_high_pixels"] = datapoint.metadata.get("fraction_saturated_high_pixels", None)
            self.df_results.loc[idx, "fraction_saturated_low_pixels"] = datapoint.metadata.get("fraction_saturated_low_pixels", None)
            self.df_results.loc[idx, "mean_corners_error_LBCV"] = datapoint.mean_corners_error_LBCV if hasattr(datapoint, 'mean_corners_error_LBCV') else None
            self.df_results.loc[idx, "mean_corners_error_PBCV"] = datapoint.mean_corners_error_PBCV if hasattr(datapoint, 'mean_corners_error_PBCV') else None

            if datapoint.CCV_detected:
                self.df_results.loc[idx, "tf_error_CCV_Rxx"] = datapoint.tf_error_CCV[0, 0] 
                self.df_results.loc[idx, "tf_error_CCV_Rxy"] = datapoint.tf_error_CCV[0, 1]
                self.df_results.loc[idx, "tf_error_CCV_Rxz"] = datapoint.tf_error_CCV[0, 2]
                self.df_results.loc[idx, "tf_error_CCV_Ryx"] = datapoint.tf_error_CCV[1, 0]
                self.df_results.loc[idx, "tf_error_CCV_Ryy"] = datapoint.tf_error_CCV[1, 1]
                self.df_results.loc[idx, "tf_error_CCV_Ryz"] = datapoint.tf_error_CCV[1, 2]
                self.df_results.loc[idx, "tf_error_CCV_Rzx"] = datapoint.tf_error_CCV[2, 0]   
                self.df_results.loc[idx, "tf_error_CCV_Rzy"] = datapoint.tf_error_CCV[2, 1]
                self.df_results.loc[idx, "tf_error_CCV_Rzz"] = datapoint.tf_error_CCV[2, 2]
                self.df_results.loc[idx, "tf_error_CCV_tx"] = datapoint.tf_error_CCV[0, 3]
                self.df_results.loc[idx, "tf_error_CCV_ty"] = datapoint.tf_error_CCV[1, 3]
                self.df_results.loc[idx, "tf_error_CCV_tz"] = datapoint.tf_error_CCV[2, 3]
                
                self.df_results.loc[idx, "pose_error_CCV_x"] = datapoint.pose_error_CCV[0]
                self.df_results.loc[idx, "pose_error_CCV_y"] = datapoint.pose_error_CCV[1]
                self.df_results.loc[idx, "pose_error_CCV_z"] = datapoint.pose_error_CCV[2]
                self.df_results.loc[idx, "pose_error_CCV_a"] = datapoint.pose_error_CCV[3]
                self.df_results.loc[idx, "pose_error_CCV_b"] = datapoint.pose_error_CCV[4]
                self.df_results.loc[idx, "pose_error_CCV_c"] = datapoint.pose_error_CCV[5] 
            else:
                self.df_results.loc[idx, "tf_error_CCV_Rxx"] = None
                self.df_results.loc[idx, "tf_error_CCV_Rxy"] = None
                self.df_results.loc[idx, "tf_error_CCV_Rxz"] = None
                self.df_results.loc[idx, "tf_error_CCV_Ryx"] = None
                self.df_results.loc[idx, "tf_error_CCV_Ryy"] = None
                self.df_results.loc[idx, "tf_error_CCV_Ryz"] = None
                self.df_results.loc[idx, "tf_error_CCV_Rzx"] = None
                self.df_results.loc[idx, "tf_error_CCV_Rzy"] = None
                self.df_results.loc[idx, "tf_error_CCV_Rzz"] = None
                self.df_results.loc[idx, "tf_error_CCV_tx"] = None
                self.df_results.loc[idx, "tf_error_CCV_ty"] = None
                self.df_results.loc[idx, "tf_error_CCV_tz"] = None
                self.df_results.loc[idx, "pose_error_CCV_x"] = None 
                self.df_results.loc[idx, "pose_error_CCV_y"] = None 
                self.df_results.loc[idx, "pose_error_CCV_z"] = None 
                self.df_results.loc[idx, "pose_error_CCV_a"] = None 
                self.df_results.loc[idx, "pose_error_CCV_b"] = None 
                self.df_results.loc[idx, "pose_error_CCV_c"] = None  

            if datapoint.detected_LBCV:

                self.df_results.loc[idx, "tf_LBCV_Rxx"] = datapoint.tf_LBCV[0, 0]
                self.df_results.loc[idx, "tf_LBCV_Rxy"] = datapoint.tf_LBCV[0, 1]
                self.df_results.loc[idx, "tf_LBCV_Rxz"] = datapoint.tf_LBCV[0, 2]     
                self.df_results.loc[idx, "tf_LBCV_Ryx"] = datapoint.tf_LBCV[1, 0]
                self.df_results.loc[idx, "tf_LBCV_Ryy"] = datapoint.tf_LBCV[1, 1]
                self.df_results.loc[idx, "tf_LBCV_Ryz"] = datapoint.tf_LBCV[1, 2]
                self.df_results.loc[idx, "tf_LBCV_Rzx"] = datapoint.tf_LBCV[2, 0]
                self.df_results.loc[idx, "tf_LBCV_Rzy"] = datapoint.tf_LBCV[2, 1]
                self.df_results.loc[idx, "tf_LBCV_Rzz"] = datapoint.tf_LBCV[2, 2]
                self.df_results.loc[idx, "tf_LBCV_tx"] = datapoint.tf_LBCV[0, 3]
                self.df_results.loc[idx, "tf_LBCV_ty"] = datapoint.tf_LBCV[1, 3]
                self.df_results.loc[idx, "tf_LBCV_tz"] = datapoint.tf_LBCV[2, 3]

                self.df_results.loc[idx, "tf_error_LBCV_Rxx"] = datapoint.tf_error_LBCV[0, 0] 
                self.df_results.loc[idx, "tf_error_LBCV_Rxy"] = datapoint.tf_error_LBCV[0, 1]
                self.df_results.loc[idx, "tf_error_LBCV_Rxz"] = datapoint.tf_error_LBCV[0, 2]
                self.df_results.loc[idx, "tf_error_LBCV_Ryx"] = datapoint.tf_error_LBCV[1, 0]
                self.df_results.loc[idx, "tf_error_LBCV_Ryy"] = datapoint.tf_error_LBCV[1, 1]
                self.df_results.loc[idx, "tf_error_LBCV_Ryz"] = datapoint.tf_error_LBCV[1, 2]
                self.df_results.loc[idx, "tf_error_LBCV_Rzx"] = datapoint.tf_error_LBCV[2, 0]   
                self.df_results.loc[idx, "tf_error_LBCV_Rzy"] = datapoint.tf_error_LBCV[2, 1]
                self.df_results.loc[idx, "tf_error_LBCV_Rzz"] = datapoint.tf_error_LBCV[2, 2]
                self.df_results.loc[idx, "tf_error_LBCV_tx"] = datapoint.tf_error_LBCV[0, 3]
                self.df_results.loc[idx, "tf_error_LBCV_ty"] = datapoint.tf_error_LBCV[1, 3]
                self.df_results.loc[idx, "tf_error_LBCV_tz"] = datapoint.tf_error_LBCV[2, 3]

                self.df_results.loc[idx, "pose_error_LBCV_x"] = datapoint.pose_error_LBCV[0]
                self.df_results.loc[idx, "pose_error_LBCV_y"] = datapoint.pose_error_LBCV[1]
                self.df_results.loc[idx, "pose_error_LBCV_z"] = datapoint.pose_error_LBCV[2]
                self.df_results.loc[idx, "pose_error_LBCV_a"] = datapoint.pose_error_LBCV[3]
                self.df_results.loc[idx, "pose_error_LBCV_b"] = datapoint.pose_error_LBCV[4]
                self.df_results.loc[idx, "pose_error_LBCV_c"] = datapoint.pose_error_LBCV[5]
                self.df_results.loc[idx, "LBCV_IOU"] = datapoint.LBCV_IOU if hasattr(datapoint, 'LBCV_IOU') else None
                if hasattr(datapoint, 'tf_error_HCV'): 
                    if datapoint.tf_error_HCV is not None and datapoint.pose_error_HCV is not None:
                        self.df_results.loc[idx, "tf_error_HCV_Rxx"] = datapoint.tf_error_HCV[0, 0]
                        self.df_results.loc[idx, "tf_error_HCV_Rxy"] = datapoint.tf_error_HCV[0, 1]
                        self.df_results.loc[idx, "tf_error_HCV_Rxz"] = datapoint.tf_error_HCV[0, 2]
                        self.df_results.loc[idx, "tf_error_HCV_Ryx"] = datapoint.tf_error_HCV[1, 0]
                        self.df_results.loc[idx, "tf_error_HCV_Ryy"] = datapoint.tf_error_HCV[1, 1]
                        self.df_results.loc[idx, "tf_error_HCV_Ryz"] = datapoint.tf_error_HCV[1, 2]
                        self.df_results.loc[idx, "tf_error_HCV_Rzx"] = datapoint.tf_error_HCV[2, 0]
                        self.df_results.loc[idx, "tf_error_HCV_Rzy"] = datapoint.tf_error_HCV[2, 1]
                        self.df_results.loc[idx, "tf_error_HCV_Rzz"] = datapoint.tf_error_HCV[2, 2]
                        self.df_results.loc[idx, "tf_error_HCV_tx"] = datapoint.tf_error_HCV[0, 3]
                        self.df_results.loc[idx, "tf_error_HCV_ty"] = datapoint.tf_error_HCV[1, 3]
                        self.df_results.loc[idx, "tf_error_HCV_tz"] = datapoint.tf_error_HCV[2, 3]
                        self.df_results.loc[idx, "pose_error_HCV_x"] = datapoint.pose_error_HCV[0]
                        self.df_results.loc[idx, "pose_error_HCV_y"] = datapoint.pose_error_HCV[1]
                        self.df_results.loc[idx, "pose_error_HCV_z"] = datapoint.pose_error_HCV[2]
                        self.df_results.loc[idx, "pose_error_HCV_a"] = datapoint.pose_error_HCV[3]
                        self.df_results.loc[idx, "pose_error_HCV_b"] = datapoint.pose_error_HCV[4]
                        self.df_results.loc[idx, "pose_error_HCV_c"] = datapoint.pose_error_HCV[5]
                else:
                    for col in [
                        "tf_error_HCV_Rxx", "tf_error_HCV_Rxy", "tf_error_HCV_Rxz",
                        "tf_error_HCV_Ryx", "tf_error_HCV_Ryy", "tf_error_HCV_Ryz",
                        "tf_error_HCV_Rzx", "tf_error_HCV_Rzy", "tf_error_HCV_Rzz",
                        "tf_error_HCV_tx",  "tf_error_HCV_ty",  "tf_error_HCV_tz",
                        "pose_error_HCV_x", "pose_error_HCV_y", "pose_error_HCV_z",
                        "pose_error_HCV_a", "pose_error_HCV_b", "pose_error_HCV_c"
                    ]:
                        self.df_results.loc[idx, col] = None
            else:
                self.df_results.loc[idx, "tf_error_LBCV_Rxx"] = None
                self.df_results.loc[idx, "tf_error_LBCV_Rxy"] = None
                self.df_results.loc[idx, "tf_error_LBCV_Rxz"] = None
                self.df_results.loc[idx, "tf_error_LBCV_Ryx"] = None
                self.df_results.loc[idx, "tf_error_LBCV_Ryy"] = None
                self.df_results.loc[idx, "tf_error_LBCV_Ryz"] = None
                self.df_results.loc[idx, "tf_error_LBCV_Rzx"] = None
                self.df_results.loc[idx, "tf_error_LBCV_Rzy"] = None
                self.df_results.loc[idx, "tf_error_LBCV_Rzz"] = None
                self.df_results.loc[idx, "tf_error_LBCV_tx"] = None
                self.df_results.loc[idx, "tf_error_LBCV_ty"] = None
                self.df_results.loc[idx, "tf_error_LBCV_tz"] = None
                self.df_results.loc[idx, "pose_error_LBCV_x"] = None 
                self.df_results.loc[idx, "pose_error_LBCV_y"] = None 
                self.df_results.loc[idx, "pose_error_LBCV_z"] = None 
                self.df_results.loc[idx, "pose_error_LBCV_a"] = None 
                self.df_results.loc[idx, "pose_error_LBCV_b"] = None 
                self.df_results.loc[idx, "pose_error_LBCV_c"] = None
                self.df_results.loc[idx, "tf_error_HCV_Rxx"] = None
                self.df_results.loc[idx, "tf_error_HCV_Rxy"] = None
                self.df_results.loc[idx, "tf_error_HCV_Rxz"] = None
                self.df_results.loc[idx, "tf_error_HCV_Ryx"] = None
                self.df_results.loc[idx, "tf_error_HCV_Ryy"] = None
                self.df_results.loc[idx, "tf_error_HCV_Ryz"] = None
                self.df_results.loc[idx, "tf_error_HCV_Rzx"] = None
                self.df_results.loc[idx, "tf_error_HCV_Rzy"] = None
                self.df_results.loc[idx, "tf_error_HCV_Rzz"] = None
                self.df_results.loc[idx, "tf_error_HCV_tx"] = None
                self.df_results.loc[idx, "tf_error_HCV_ty"] = None
                self.df_results.loc[idx, "tf_error_HCV_tz"] = None
                self.df_results.loc[idx, "pose_error_HCV_x"] = None 
                self.df_results.loc[idx, "pose_error_HCV_y"] = None 
                self.df_results.loc[idx, "pose_error_HCV_z"] = None 
                self.df_results.loc[idx, "pose_error_HCV_a"] = None 
                self.df_results.loc[idx, "pose_error_HCV_b"] = None 
                self.df_results.loc[idx, "pose_error_HCV_c"] = None
                self.df_results.loc[idx, "LBCV_IOU"] = None

            if hasattr(datapoint, 'detected_PBCV'): 
                if datapoint.tf_error_PBCV is not None and datapoint.pose_error_PBCV is not None:
                    self.df_results.loc[idx, "detected_PBCV"] = datapoint.detected_PBCV
                    self.df_results.loc[idx, "tf_PBCV_Rxx"] = datapoint.tf_PBCV[0, 0]
                    self.df_results.loc[idx, "tf_PBCV_Rxy"] = datapoint.tf_PBCV[0, 1]
                    self.df_results.loc[idx, "tf_PBCV_Rxz"] = datapoint.tf_PBCV[0, 2]     
                    self.df_results.loc[idx, "tf_PBCV_Ryx"] = datapoint.tf_PBCV[1, 0]
                    self.df_results.loc[idx, "tf_PBCV_Ryy"] = datapoint.tf_PBCV[1, 1]
                    self.df_results.loc[idx, "tf_PBCV_Ryz"] = datapoint.tf_PBCV[1, 2]
                    self.df_results.loc[idx, "tf_PBCV_Rzx"] = datapoint.tf_PBCV[2, 0]
                    self.df_results.loc[idx, "tf_PBCV_Rzy"] = datapoint.tf_PBCV[2, 1]
                    self.df_results.loc[idx, "tf_PBCV_Rzz"] = datapoint.tf_PBCV[2, 2]
                    self.df_results.loc[idx, "tf_PBCV_tx"] = datapoint.tf_PBCV[0, 3]
                    self.df_results.loc[idx, "tf_PBCV_ty"] = datapoint.tf_PBCV[1, 3]
                    self.df_results.loc[idx, "tf_PBCV_tz"] = datapoint.tf_PBCV[2, 3]
                    self.df_results.loc[idx, "tf_error_PBCV_Rxx"] = datapoint.tf_error_PBCV[0, 0]
                    self.df_results.loc[idx, "tf_error_PBCV_Rxy"] = datapoint.tf_error_PBCV[0, 1]
                    self.df_results.loc[idx, "tf_error_PBCV_Rxz"] = datapoint.tf_error_PBCV[0, 2]
                    self.df_results.loc[idx, "tf_error_PBCV_Ryx"] = datapoint.tf_error_PBCV[1, 0]
                    self.df_results.loc[idx, "tf_error_PBCV_Ryy"] = datapoint.tf_error_PBCV[1, 1]
                    self.df_results.loc[idx, "tf_error_PBCV_Ryz"] = datapoint.tf_error_PBCV[1, 2]
                    self.df_results.loc[idx, "tf_error_PBCV_Rzx"] = datapoint.tf_error_PBCV[2, 0]
                    self.df_results.loc[idx, "tf_error_PBCV_Rzy"] = datapoint.tf_error_PBCV[2, 1]
                    self.df_results.loc[idx, "tf_error_PBCV_Rzz"] = datapoint.tf_error_PBCV[2, 2]
                    self.df_results.loc[idx, "tf_error_PBCV_tx"] = datapoint.tf_error_PBCV[0, 3]
                    self.df_results.loc[idx, "tf_error_PBCV_ty"] = datapoint.tf_error_PBCV[1, 3]
                    self.df_results.loc[idx, "tf_error_PBCV_tz"] = datapoint.tf_error_PBCV[2, 3]
                    self.df_results.loc[idx, "pose_error_PBCV_x"] = datapoint.pose_error_PBCV[0]
                    self.df_results.loc[idx, "pose_error_PBCV_y"] = datapoint.pose_error_PBCV[1]
                    self.df_results.loc[idx, "pose_error_PBCV_z"] = datapoint.pose_error_PBCV[2]
                    self.df_results.loc[idx, "pose_error_PBCV_a"] = datapoint.pose_error_PBCV[3]
                    self.df_results.loc[idx, "pose_error_PBCV_b"] = datapoint.pose_error_PBCV[4]
                    self.df_results.loc[idx, "pose_error_PBCV_c"] = datapoint.pose_error_PBCV[5]
                    self.df_results.loc[idx, "harris_corner_response_score"] = datapoint.harris_corner_response_score if hasattr(datapoint, 'harris_corner_response_score') else None
                    self.df_results.loc[idx, "num_valid_proj_points"] = datapoint.num_valid_proj_points if hasattr(datapoint, 'num_valid_proj_points') else None
                    self.df_results.loc[idx, "keypoint_residual_score"] = datapoint.keypoint_residual_score if hasattr(datapoint, 'keypoint_residual_score') else None
                    self.df_results.loc[idx, "detection_score"] = datapoint.detection_score if hasattr(datapoint, 'detection_score') else None
                    self.df_results.loc[idx, "image_similarity_score"] = datapoint.image_similarity_score if hasattr(datapoint, 'image_similarity_score') else None
                    self.df_results.loc[idx, "PBCV_IOU"] = datapoint.PBCV_IOU if hasattr(datapoint, 'PBCV_IOU') else None

                else:
                    for col in [
                        "tf_error_PBCV_Rxx", "tf_error_PBCV_Rxy", "tf_errorPBCV_Rxz",
                        "tf_error_PBCV_Ryx", "tf_error_PBCV_Ryy", "tf_errorPBCV_Ryz",
                        "tf_error_PBCV_Rzx", "tf_error_PBCV_Rzy", "tf_errorPBCV_Rzz",
                        "tf_error_PBCV_tx",  "tf_error_PBCV_ty",  "tf_errorPBCV_tz",
                        "pose_error_PBCV_x", "pose_error_PBCV_y", "pose_error_PBCV_z",
                        "pose_error_PBCV_a", "pose_error_PBCV_b", "pose_error_PBCV_c",
                        "detected_PBCV",
                        "harris_corner_response_score",
                        "keypoint_residual_score",
                        "detection_score",
                        "image_similarity_score",
                    ]:
                        self.df_results.loc[idx, col] = None
            

        if save_results: 
            output_dir = os.path.join(self.directory, "results")
            os.makedirs(output_dir, exist_ok=True) 
            output_path = os.path.join(output_dir, "results.json")
            with open(output_path, 'w') as f:
                json.dump(self.dict_results, f, indent=4)

            df_output_path = os.path.join(output_dir, "results.csv")
            self.df_results.to_csv(df_output_path, index=False)


def main(): 

    # camera_parameters = {
    #     "width": 1280,
    #     "height": 720,
    #     "fx": 886.643,
    #     "fy": 886.643,
    #     "cx": 631.834,
    #     "cy": 367.724,
    #     "distortion_coefficients": np.zeros(5),
    # }

    camera_parameters = {
        "width": 1280,
        "height": 720,
        "fx": 906.995,
        "fy": 906.995,
        "cx": 638.235,
        "cy": 360.533,
        "distortion_coefficients": np.array([0,0,0,0,0], dtype=float),
        # "distortion_coefficients": np.array([0.17328606, -0.52955904, -0.00090532,  0.00268294,  0.46284461], dtype=float)
    } # realsense calibration 

    # camera_parameters = {
    #     "width": 1920,
    #     "height": 1080,
    #     "fx": 1363.85,
    #     "fy": 1365.40,
    #     "cx": 958.58,
    #     "cy": 552.25,
    #     "distortion_coefficients": np.array([0.1693, -0.4755, 0.0018, 0.0023, 0.4114], dtype=float),
    # } # from charuco calibration 

    # camera_parameters = {
    #     "width": 1920,
    #     "height": 1080,
    #     "fx": 1360.49,
    #     "fy": 1360.49,
    #     "cx": 957.355,
    #     "cy": 540.8,
    #     "distortion_coefficients": np.array([0.1693, -0.4755, 0.0018, 0.0023, 0.4114], dtype=float),
    # } # from realsense

    marker_parameters = {
        "marker_length": 0.100,  # units: meters
        "marker_length_without_border": 0.080,  # units: meters
        "num_squares": 10, # including border 
        "aruco_dict": cv2.aruco.DICT_APRILTAG_36h11, 
    }

    # get ablation data path 
    # results in: distance_20250712, skew_20250712, truncation_20250712, underexposure_20250712, shadow_20250712, glare_20250712 
    ablation = "underexposure_20250712"  # options: "underexposure", "ambient_light_intensity", "truncation", "skew", "lateral", "pitch", "yaw", "roll"
    data_yaml_path = "./ablations/real_exp_data_description.yaml" 
    with open(data_yaml_path, 'r') as f:
        data_description = yaml.safe_load(f) 
    data_path = data_description[ablation]["data_path"] 

    config = {
        "data_path": data_path, 
        "max_num_datapoints": None, 
        "camera_parameters": camera_parameters,
        "marker_parameters": marker_parameters, 
        "seg_model_path":"./segmentation_model/models/my_checkpoint_20250329.pth.tar",
        # "kp_model_path": "./keypoints_model/models/my_checkpoint_keypoints_20250330.pth.tar", 
        "kp_model_path": "./keypoints_model/models/my_checkpoint_keypoints_20250401.pth.tar", 
        "device": "cuda" if torch.cuda.is_available() else "cpu", 
    }

    processor = DataProcessor(config)
    processor.run_opencv_fiducial_marker_detection(save_results=False) 
    processor.run_LBCV_fiducial_marker_detection(save_results=False, run_corners_HCV=True, run_PBCV=True, use_precomputed_segmentation=False) 
    processor.compute_values() 
    processor.compile_results(save_results=True)

if __name__ == "__main__":
    main() 