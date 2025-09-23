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
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from segmentation_model.model import UNETWithDropout, UNETWithDropoutMini
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

    def set_keypoints_PBCV_matched(self, matched_keypoints):
        """
        Set the PBCV keypoints that were successfully matched during RANSAC pose estimation.
        
        Args:
            matched_keypoints: Array of matched keypoints in image space (N x 2)
        """
        self.keypoints_PBCV_matched = matched_keypoints

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

    def _draw_alpha_polygon(self, image, points, color, thickness, alpha=0.7):
        """
        Draw a polygon with alpha transparency so overlapping borders are visible.
        
        Args:
            image: The image to draw on
            points: Array of points forming the polygon (shape: N x 2)
            color: BGR color tuple
            thickness: Line thickness
            alpha: Alpha transparency value (0.0 = transparent, 1.0 = opaque)
        """
        # Create an overlay image for alpha blending
        overlay = image.copy()
        
        # Draw the polygon on the overlay
        points = np.array(points, dtype=np.int32)
        cv2.polylines(overlay, [points], isClosed=True, color=color, thickness=thickness)
        
        # Blend the overlay with the original image using alpha
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    def draw_marker_borders_from_poses(self, output_path=None, draw_lbcv=True, draw_pbcv=True, 
                                     lbcv_color=(0, 255, 0), pbcv_color=(0, 0, 255), 
                                     true_color=(255, 0, 0), line_thickness=2, alpha=0.7):
        """
        Draw marker borders on the original image using reprojection from LBCV and PBCV pose estimates.
        Uses alpha transparency so overlapping borders are visible.
        
        Args:
            output_path (str, optional): Path to save the output image. If None, returns the image array.
            draw_lbcv (bool): Whether to draw LBCV pose estimate border (default: True)
            draw_pbcv (bool): Whether to draw PBCV pose estimate border (default: True)
            lbcv_color (tuple): BGR color for LBCV border (default: green)
            pbcv_color (tuple): BGR color for PBCV border (default: red)
            true_color (tuple): BGR color for CCV mean corners border (default: blue)
            line_thickness (int): Thickness of the border lines (default: 2)
            alpha (float): Alpha transparency value (0.0 = transparent, 1.0 = opaque, default: 0.7)
            
        Returns:
            numpy.ndarray: Image with drawn borders (if output_path is None)
        """
        if not hasattr(self, 'image_path') or self.image_path is None:
            raise ValueError("Image path not set for this DataPoint.")
        
        # Load the original image
        image = cv2.imread(self.image_path)
        if image is None:
            raise ValueError(f"Could not load image from {self.image_path}")
        
        # Draw CCV mean corners border if available (used as reference/true pose)
        if hasattr(self, 'mean_corners_CCV') and self.mean_corners_CCV is not None:
            corners_mean_ccv_int = np.array(self.mean_corners_CCV, dtype=np.int32)
            self._draw_alpha_polygon(image, corners_mean_ccv_int, true_color, line_thickness, alpha)
        
        # Draw LBCV pose estimate border
        if draw_lbcv and hasattr(self, 'tf_LBCV') and self.tf_LBCV is not None:
            # Project marker corners using LBCV pose estimate
            corners_lbcv = project_points_array_to_image(
                C=self.camera_matrix, 
                T=self.tf_LBCV, 
                P_array=self.marker_corners, 
                convert_cam_is2cv=True
            )
            corners_lbcv_int = np.array(corners_lbcv, dtype=np.int32)
            self._draw_alpha_polygon(image, corners_lbcv_int, lbcv_color, line_thickness, alpha)
        
        # Draw PBCV pose estimate border
        if draw_pbcv and hasattr(self, 'tf_PBCV') and self.tf_PBCV is not None:
            # Project marker corners using PBCV pose estimate
            corners_pbcv = project_points_array_to_image(
                C=self.camera_matrix, 
                T=self.tf_PBCV, 
                P_array=self.marker_corners, 
                convert_cam_is2cv=True
            )
            corners_pbcv_int = np.array(corners_pbcv, dtype=np.int32)
            self._draw_alpha_polygon(image, corners_pbcv_int, pbcv_color, line_thickness, alpha)
        
        # Save or return the image
        if output_path is not None:
            cv2.imwrite(output_path, image)
            print(f"Image with marker borders saved to: {output_path}")
        else:
            return image

    def draw_pose_comparison_visualization(self, output_path=None, 
                                         ccv_color=(0, 0, 255), lbcv_color=(255, 0, 0), pbcv_color=(0, 255, 0),
                                         keypoint_color=(0, 255, 0), line_thickness=2, keypoint_radius=3, alpha=0.7):
        """
        Draw a comprehensive pose comparison visualization with:
        - Red borders for CCV pose estimates
        - Blue borders for LBCV pose estimates  
        - Green borders for PBCV pose estimates
        - Green keypoints overlay for PBCV matched keypoints
        Uses alpha transparency so overlapping borders are visible.
        
        Args:
            output_path (str, optional): Path to save the output image. If None, returns the image array.
            ccv_color (tuple): BGR color for CCV border (default: red)
            lbcv_color (tuple): BGR color for LBCV border (default: blue)
            pbcv_color (tuple): BGR color for PBCV border (default: green)
            keypoint_color (tuple): BGR color for PBCV keypoints (default: green)
            line_thickness (int): Thickness of the border lines (default: 2)
            keypoint_radius (int): Radius of keypoint circles (default: 3)
            alpha (float): Alpha transparency value (0.0 = transparent, 1.0 = opaque, default: 0.7)
            
        Returns:
            numpy.ndarray: Image with drawn borders and keypoints (if output_path is None)
        """
        if not hasattr(self, 'image_path') or self.image_path is None:
            raise ValueError("Image path not set for this DataPoint.")
        
        # Load the original image
        image = cv2.imread(self.image_path)
        if image is None:
            raise ValueError(f"Could not load image from {self.image_path}")
        
        # Draw CCV pose border (red)
        if hasattr(self, 'tf_CCV') and self.tf_CCV is not None:
            corners_ccv = project_points_array_to_image(
                C=self.camera_matrix, 
                T=self.tf_CCV, 
                P_array=self.marker_corners, 
                convert_cam_is2cv=True
            )
            corners_ccv_int = np.array(corners_ccv, dtype=np.int32)
            self._draw_alpha_polygon(image, corners_ccv_int, ccv_color, line_thickness, alpha)
        
        # Draw LBCV pose border (blue)
        if hasattr(self, 'tf_LBCV') and self.tf_LBCV is not None:
            corners_lbcv = project_points_array_to_image(
                C=self.camera_matrix, 
                T=self.tf_LBCV, 
                P_array=self.marker_corners, 
                convert_cam_is2cv=True
            )
            corners_lbcv_int = np.array(corners_lbcv, dtype=np.int32)
            self._draw_alpha_polygon(image, corners_lbcv_int, lbcv_color, line_thickness, alpha)
        
        # Draw PBCV pose border (green)
        if hasattr(self, 'tf_PBCV') and self.tf_PBCV is not None:
            corners_pbcv = project_points_array_to_image(
                C=self.camera_matrix, 
                T=self.tf_PBCV, 
                P_array=self.marker_corners, 
                convert_cam_is2cv=True
            )
            corners_pbcv_int = np.array(corners_pbcv, dtype=np.int32)
            self._draw_alpha_polygon(image, corners_pbcv_int, pbcv_color, line_thickness, alpha)
        
        # Overlay PBCV keypoints (green circles)
        if hasattr(self, 'keypoints_PBCV') and self.keypoints_PBCV is not None:
            for keypoint in self.keypoints_PBCV:
                # keypoint should be in format [x, y] or [x, y, confidence]
                if len(keypoint) >= 2:
                    x, y = int(keypoint[0]), int(keypoint[1])
                    # Check if keypoint is within image bounds
                    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                        cv2.circle(image, (x, y), keypoint_radius, keypoint_color, -1)  # -1 for filled circle
        
        # Save or return the image
        if output_path is not None:
            cv2.imwrite(output_path, image)
            print(f"Pose comparison visualization saved to: {output_path}")
        else:
            return image

    def draw_pbcv_matched_keypoints_visualization(self, output_path=None, 
                                                border_color=(0, 255, 0), keypoint_color=(0, 255, 0),
                                                line_thickness=2, keypoint_radius=4, alpha=0.8):
        """
        Draw visualization showing only PBCV matched keypoints (after RANSAC) along with 
        the marker border based on PBCV pose prediction.
        
        Args:
            output_path (str, optional): Path to save the output image. If None, returns the image array.
            border_color (tuple): BGR color for PBCV pose border (default: green)
            keypoint_color (tuple): BGR color for matched keypoints (default: green)
            line_thickness (int): Thickness of the border lines (default: 2)
            keypoint_radius (int): Radius of keypoint circles (default: 4)
            alpha (float): Alpha transparency value (0.0 = transparent, 1.0 = opaque, default: 0.8)
            
        Returns:
            numpy.ndarray: Image with drawn border and matched keypoints (if output_path is None)
        """
        if not hasattr(self, 'image_path') or self.image_path is None:
            raise ValueError("Image path not set for this DataPoint.")
        
        # Load the original image
        image = cv2.imread(self.image_path)
        if image is None:
            raise ValueError(f"Could not load image from {self.image_path}")
        
        # Draw PBCV pose border if available
        if hasattr(self, 'tf_PBCV') and self.tf_PBCV is not None:
            corners_pbcv = project_points_array_to_image(
                C=self.camera_matrix, 
                T=self.tf_PBCV, 
                P_array=self.marker_corners, 
                convert_cam_is2cv=True
            )
            corners_pbcv_int = np.array(corners_pbcv, dtype=np.int32)
            self._draw_alpha_polygon(image, corners_pbcv_int, border_color, line_thickness, alpha)
        
        # Draw matched keypoints if available
        if hasattr(self, 'keypoints_PBCV_matched') and self.keypoints_PBCV_matched is not None:
            for keypoint in self.keypoints_PBCV_matched:
                if len(keypoint) >= 2:
                    x, y = int(keypoint[0]), int(keypoint[1])
                    # Check if keypoint is within image bounds
                    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                        cv2.circle(image, (x, y), keypoint_radius, keypoint_color, -1)  # -1 for filled circle
        elif hasattr(self, 'keypoints_PBCV') and self.keypoints_PBCV is not None:
            # Fallback to all PBCV keypoints if matched keypoints not available
            for keypoint in self.keypoints_PBCV:
                if len(keypoint) >= 2:
                    x, y = int(keypoint[0]), int(keypoint[1])
                    # Check if keypoint is within image bounds
                    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                        cv2.circle(image, (x, y), keypoint_radius, keypoint_color, -1)
        
        # Save or return the image
        if output_path is not None:
            cv2.imwrite(output_path, image)
            print(f"PBCV matched keypoints visualization saved to: {output_path}")
        else:
            return image

    def create_combined_2x2_visualization(self, experiment_name=None, experiment_variable=None, 
                                        experiment_value=None, output_path=None, figsize=(24, 14)):
        """
        Create a 2x2 combined image visualization showing:
        - Top left: CCV pose estimate with red border (Classical Detection & Pose Estimation)
        - Top right: Segmentation prediction image (Learning-Based Segmentation)
        - Bottom left: LBCV keypoints in blue with blue border (Learning-Based Keypoints & Pose Estimation)
        - Bottom right: PBCV matched keypoints in green with green border (Pattern-Based Pose Estimation)
        
        Args:
            experiment_name (str, optional): Name of the experiment for the suptitle
            experiment_variable (str, optional): Name of the experiment variable (e.g., 'mean_marker_pixel_brightness')
            experiment_value (float, optional): Value of the experiment variable
            output_path (str, optional): Path to save the output image. If None, returns the figure.
            figsize (tuple): Figure size in inches (default: (12, 10))
            
        Returns:
            matplotlib.figure.Figure: The created figure (if output_path is None)
        """
        if not hasattr(self, 'image_path') or self.image_path is None:
            raise ValueError("Image path not set for this DataPoint.")
        
        # Load the original image
        original_image = cv2.imread(self.image_path)
        if original_image is None:
            raise ValueError(f"Could not load image from {self.image_path}")
        
        # Convert BGR to RGB for matplotlib
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Create the figure with 2x2 subplots with no spacing
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(self._create_suptitle(experiment_name, experiment_variable, experiment_value), 
                    fontsize=18, fontweight='bold', y=0.92)  # Increased from 14 to 18
        
        # Remove all spacing between subplots including vertical spacing
        plt.subplots_adjust(left=0, bottom=0, right=1, top=0.90, wspace=0, hspace=0)
        
        # Top left: Classical Detection & Pose Estimation (CCV)
        ccv_image = self._create_ccv_visualization_with_title(original_image_rgb.copy(), 
                                                             "Classical Detection & Pose Estimation")
        axes[0, 0].imshow(ccv_image)
        axes[0, 0].axis('off')
        
        # Top right: Learning-Based Segmentation
        seg_image = self._create_segmentation_visualization_with_title(original_image_rgb.copy(),
                                                                     "Learning-Based Segmentation")
        axes[0, 1].imshow(seg_image, cmap='gray' if len(seg_image.shape) == 2 else None)
        axes[0, 1].axis('off')
        
        # Bottom left: Learning-Based Keypoints & Pose Estimation (LBCV)
        lbcv_image = self._create_lbcv_visualization_with_title(original_image_rgb.copy(),
                                                              "Learning-Based Keypoints &\nPose Estimation (LBCV)")
        axes[1, 0].imshow(lbcv_image)
        axes[1, 0].axis('off')
        
        # Bottom right: Pattern-Based Pose Estimation (PBCV)
        pbcv_image = self._create_pbcv_visualization_with_title(original_image_rgb.copy(),
                                                              "Pattern-Based Pose Estimation\n(PBCV)")
        axes[1, 1].imshow(pbcv_image)
        axes[1, 1].axis('off')
        
        # Save or return the figure
        if output_path is not None:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
            print(f"Combined 2x2 visualization saved to: {output_path}")
            plt.close(fig)
        else:
            return fig

    def _create_suptitle(self, experiment_name, experiment_variable, experiment_value):
        """Create the suptitle string for the combined visualization."""
        # Dictionary for relabeling variable names to more readable versions
        variable_name_map = {
            'mean_marker_pixel_brightness': 'Mean Marker Brightness',
            'ambient_light_intensity': 'Ambient Light Intensity',
            'truncation_percentage': 'Truncation Percentage',
            'skew_angle': 'Skew Angle',
            'distance_to_marker': 'Distance to Marker',
            'rotation_angle': 'Rotation Angle',
            'pitch_angle': 'Pitch Angle',
            'yaw_angle': 'Yaw Angle',
            'roll_angle': 'Roll Angle',
            'lateral_offset': 'Lateral Offset',
            'blur_intensity': 'Blur Intensity',
            'noise_level': 'Noise Level',
            'exposure_time': 'Exposure Time',
            'camera_gain': 'Camera Gain'
        }
        
        if experiment_name and experiment_variable and experiment_value is not None:
            # Get readable variable name from dictionary, fallback to original if not found
            readable_variable = variable_name_map.get(experiment_variable, experiment_variable)
            
            # Truncate numerical values to 2 decimal places
            if isinstance(experiment_value, (int, float)):
                formatted_value = f"{experiment_value:.2f}"
            else:
                formatted_value = str(experiment_value)
            return f"{experiment_name} - {readable_variable}: {formatted_value}"
        elif experiment_name:
            return experiment_name
        else:
            return f"Marker Detection & Pose Estimation Comparison (Image {self.idx})"

    def _create_ccv_visualization_with_title(self, image, title):
        """Create the CCV visualization with red border and overlay title."""
        # Convert RGB back to BGR for OpenCV operations
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw CCV pose border (red)
        if hasattr(self, 'tf_CCV') and self.tf_CCV is not None:
            corners_ccv = project_points_array_to_image(
                C=self.camera_matrix, 
                T=self.tf_CCV, 
                P_array=self.marker_corners, 
                convert_cam_is2cv=True
            )
            corners_ccv_int = np.array(corners_ccv, dtype=np.int32)
            self._draw_alpha_polygon(image_bgr, corners_ccv_int, (0, 0, 255), 3, 0.8)  # Red border
        
        # Add title overlay
        self._add_title_overlay(image_bgr, title)
        
        # Convert back to RGB
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def _create_segmentation_visualization_with_title(self, image, title):
        """Create the segmentation visualization with overlay title."""
        # For segmentation, we need to get the segmentation mask
        if hasattr(self, 'segmentation_mask') and self.segmentation_mask is not None:
            # If we have a stored segmentation mask, use it
            seg_image = np.array(self.segmentation_mask)
            # Convert to 3-channel if it's grayscale for text overlay
            if len(seg_image.shape) == 2:
                seg_image = cv2.cvtColor(seg_image, cv2.COLOR_GRAY2BGR)
        elif hasattr(self, 'processor') and hasattr(self.processor, 'run_LBCV_segmentation'):
            # If we have access to the processor and can run segmentation
            try:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                seg_mask_img, _, _ = self.processor.run_LBCV_segmentation(image_bgr)
                seg_image = np.array(seg_mask_img)
                # Convert to 3-channel if it's grayscale for text overlay
                if len(seg_image.shape) == 2:
                    seg_image = cv2.cvtColor(seg_image, cv2.COLOR_GRAY2BGR)
            except Exception as e:
                logger.warning(f"Could not run segmentation: {e}")
                # Fallback to grayscale
                gray_image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
                seg_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        else:
            # Return the original image as a grayscale placeholder
            gray_image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
            seg_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        
        # Add title overlay
        self._add_title_overlay(seg_image, title)
        
        # Convert back to RGB for matplotlib
        return cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB)

    def _create_lbcv_visualization_with_title(self, image, title):
        """Create the LBCV visualization with blue keypoints and blue border with overlay title."""
        # Convert RGB back to BGR for OpenCV operations
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Check if LBCV IOU meets threshold (> 0.5)
        show_lbcv_results = False
        if hasattr(self, 'LBCV_IOU') and self.LBCV_IOU is not None and self.LBCV_IOU > 0.5:
            show_lbcv_results = True
        
        if show_lbcv_results:
            # Draw LBCV pose border (blue)
            if hasattr(self, 'tf_LBCV') and self.tf_LBCV is not None:
                corners_lbcv = project_points_array_to_image(
                    C=self.camera_matrix, 
                    T=self.tf_LBCV, 
                    P_array=self.marker_corners, 
                    convert_cam_is2cv=True
                )
                corners_lbcv_int = np.array(corners_lbcv, dtype=np.int32)
                self._draw_alpha_polygon(image_bgr, corners_lbcv_int, (255, 0, 0), 3, 0.8)  # Blue border
            
            # Draw LBCV keypoints (blue circles)
            if hasattr(self, 'keypoints_LBCV') and self.keypoints_LBCV is not None:
                for keypoint in self.keypoints_LBCV:
                    if len(keypoint) >= 2:
                        x, y = int(keypoint[0]), int(keypoint[1])
                        # Check if keypoint is within image bounds
                        if 0 <= x < image_bgr.shape[1] and 0 <= y < image_bgr.shape[0]:
                            cv2.circle(image_bgr, (x, y), 4, (255, 0, 0), -1)  # Blue filled circle
        
        # Add title overlay
        self._add_title_overlay(image_bgr, title)
        
        # Convert back to RGB
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def _create_pbcv_visualization_with_title(self, image, title):
        """Create the PBCV visualization with green matched keypoints and green border with overlay title."""
        # Convert RGB back to BGR for OpenCV operations
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Check if PBCV IOU meets threshold (> 0.5)
        show_pbcv_results = False
        if hasattr(self, 'PBCV_IOU') and self.PBCV_IOU is not None and self.PBCV_IOU > 0.5:
            show_pbcv_results = True
        
        if show_pbcv_results:
            # Draw PBCV pose border (green)
            if hasattr(self, 'tf_PBCV') and self.tf_PBCV is not None:
                corners_pbcv = project_points_array_to_image(
                    C=self.camera_matrix, 
                    T=self.tf_PBCV, 
                    P_array=self.marker_corners, 
                    convert_cam_is2cv=True
                )
                corners_pbcv_int = np.array(corners_pbcv, dtype=np.int32)
                self._draw_alpha_polygon(image_bgr, corners_pbcv_int, (0, 255, 0), 3, 0.8)  # Green border
            
            # Draw PBCV matched keypoints (green circles)
            if hasattr(self, 'keypoints_PBCV_matched') and self.keypoints_PBCV_matched is not None:
                for keypoint in self.keypoints_PBCV_matched:
                    if len(keypoint) >= 2:
                        x, y = int(keypoint[0]), int(keypoint[1])
                        # Check if keypoint is within image bounds
                        if 0 <= x < image_bgr.shape[1] and 0 <= y < image_bgr.shape[0]:
                            cv2.circle(image_bgr, (x, y), 4, (0, 255, 0), -1)  # Green filled circle
            elif hasattr(self, 'keypoints_PBCV') and self.keypoints_PBCV is not None:
                # Fallback to all PBCV keypoints if matched keypoints not available
                for keypoint in self.keypoints_PBCV:
                    if len(keypoint) >= 2:
                        x, y = int(keypoint[0]), int(keypoint[1])
                        # Check if keypoint is within image bounds
                        if 0 <= x < image_bgr.shape[1] and 0 <= y < image_bgr.shape[0]:
                            cv2.circle(image_bgr, (x, y), 4, (0, 255, 0), -1)  # Green filled circle
        
        # Add title overlay
        self._add_title_overlay(image_bgr, title)
        
        # Convert back to RGB
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def _add_title_overlay(self, image, title):
        """Add a title overlay to the top center of an image."""
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Split title into lines if it contains newlines
        lines = title.split('\n')
        
        # Font settings - increased font scale and thickness
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0  # Increased from 0.7
        font_thickness = 2  # Increased from 2
        text_color = (255, 255, 255)  # White text
        outline_color = (0, 0, 0)    # Black outline
        outline_thickness = 6  # Increased from 4
        
        # Calculate text size for positioning
        line_heights = []
        line_widths = []
        for line in lines:
            (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, font_thickness)
            line_heights.append(text_height + baseline)
            line_widths.append(text_width)
        
        # Calculate starting position (top center)
        total_height = sum(line_heights) + (len(lines) - 1) * 8  # Increased spacing between lines from 5 to 8
        start_y = 35  # Increased from 25 pixels from top
        
        # Draw each line
        for i, line in enumerate(lines):
            text_width = line_widths[i]
            text_x = (width - text_width) // 2  # Center horizontally
            text_y = start_y + sum(line_heights[:i+1]) + i * 8
            
            # Draw text outline (black)
            cv2.putText(image, line, (text_x, text_y), font, font_scale, outline_color, outline_thickness)
            # Draw text (white)
            cv2.putText(image, line, (text_x, text_y), font, font_scale, text_color, font_thickness)

    def _create_ccv_visualization(self, image):
        """Create the CCV visualization with red border."""
        # Convert RGB back to BGR for OpenCV operations
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw CCV pose border (red)
        if hasattr(self, 'tf_CCV') and self.tf_CCV is not None:
            corners_ccv = project_points_array_to_image(
                C=self.camera_matrix, 
                T=self.tf_CCV, 
                P_array=self.marker_corners, 
                convert_cam_is2cv=True
            )
            corners_ccv_int = np.array(corners_ccv, dtype=np.int32)
            self._draw_alpha_polygon(image_bgr, corners_ccv_int, (0, 0, 255), 3, 0.8)  # Red border
        
        # Convert back to RGB
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def _create_segmentation_visualization(self, image):
        """Create the segmentation visualization."""
        # For segmentation, we need to get the segmentation mask
        # This would typically come from the LBCV segmentation step
        if hasattr(self, 'segmentation_mask') and self.segmentation_mask is not None:
            # If we have a stored segmentation mask, use it
            return np.array(self.segmentation_mask)
        elif hasattr(self, 'processor') and hasattr(self.processor, 'run_LBCV_segmentation'):
            # If we have access to the processor and can run segmentation
            try:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                seg_mask_img, _, _ = self.processor.run_LBCV_segmentation(image_bgr)
                return np.array(seg_mask_img)
            except Exception as e:
                logger.warning(f"Could not run segmentation: {e}")
                # Fallback to grayscale
                gray_image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
                return gray_image
        else:
            # Return the original image as a grayscale placeholder
            # In practice, you would run the segmentation model here
            gray_image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
            return gray_image

    def _create_lbcv_visualization(self, image):
        """Create the LBCV visualization with blue keypoints and blue border."""
        # Convert RGB back to BGR for OpenCV operations
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw LBCV pose border (blue)
        if hasattr(self, 'tf_LBCV') and self.tf_LBCV is not None:
            corners_lbcv = project_points_array_to_image(
                C=self.camera_matrix, 
                T=self.tf_LBCV, 
                P_array=self.marker_corners, 
                convert_cam_is2cv=True
            )
            corners_lbcv_int = np.array(corners_lbcv, dtype=np.int32)
            self._draw_alpha_polygon(image_bgr, corners_lbcv_int, (255, 0, 0), 3, 0.8)  # Blue border
        
        # Draw LBCV keypoints (blue circles)
        if hasattr(self, 'keypoints_LBCV') and self.keypoints_LBCV is not None:
            for keypoint in self.keypoints_LBCV:
                if len(keypoint) >= 2:
                    x, y = int(keypoint[0]), int(keypoint[1])
                    # Check if keypoint is within image bounds
                    if 0 <= x < image_bgr.shape[1] and 0 <= y < image_bgr.shape[0]:
                        cv2.circle(image_bgr, (x, y), 4, (255, 0, 0), -1)  # Blue filled circle
        
        # Convert back to RGB
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def _create_pbcv_visualization(self, image):
        """Create the PBCV visualization with green matched keypoints and green border."""
        # Convert RGB back to BGR for OpenCV operations
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw PBCV pose border (green)
        if hasattr(self, 'tf_PBCV') and self.tf_PBCV is not None:
            corners_pbcv = project_points_array_to_image(
                C=self.camera_matrix, 
                T=self.tf_PBCV, 
                P_array=self.marker_corners, 
                convert_cam_is2cv=True
            )
            corners_pbcv_int = np.array(corners_pbcv, dtype=np.int32)
            self._draw_alpha_polygon(image_bgr, corners_pbcv_int, (0, 255, 0), 3, 0.8)  # Green border
        
        # Draw PBCV matched keypoints (green circles)
        if hasattr(self, 'keypoints_PBCV_matched') and self.keypoints_PBCV_matched is not None:
            for keypoint in self.keypoints_PBCV_matched:
                if len(keypoint) >= 2:
                    x, y = int(keypoint[0]), int(keypoint[1])
                    # Check if keypoint is within image bounds
                    if 0 <= x < image_bgr.shape[1] and 0 <= y < image_bgr.shape[0]:
                        cv2.circle(image_bgr, (x, y), 4, (0, 255, 0), -1)  # Green filled circle
        elif hasattr(self, 'keypoints_PBCV') and self.keypoints_PBCV is not None:
            # Fallback to all PBCV keypoints if matched keypoints not available
            for keypoint in self.keypoints_PBCV:
                if len(keypoint) >= 2:
                    x, y = int(keypoint[0]), int(keypoint[1])
                    # Check if keypoint is within image bounds
                    if 0 <= x < image_bgr.shape[1] and 0 <= y < image_bgr.shape[0]:
                        cv2.circle(image_bgr, (x, y), 4, (0, 255, 0), -1)  # Green filled circle
        
        # Convert back to RGB
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

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
        
        # Compute mean corners from all CCV detections
        self.compute_mean_corners_CCV()

    def compute_mean_corners_CCV(self):
        """
        Compute the mean corners across all CCV pose results, excluding None results.
        Stores the result in self.mean_corners_CCV for each datapoint.
        """
        # Collect all valid CCV corners
        valid_corners = []
        for datapoint in self.datapoints:
            if hasattr(datapoint, 'corners_CCV') and datapoint.corners_CCV is not None:
                # corners_CCV is a numpy array of shape (1, 4, 2) from the CCV detection
                # We need to reshape it to (4, 2) to get the 4 corner points
                corners = datapoint.corners_CCV.reshape(-1, 2)  # Shape: (4, 2)
                if corners.shape[0] == 4:  # Ensure we have exactly 4 corners
                    valid_corners.append(corners)
        
        if len(valid_corners) > 0:
            # Stack all valid corners and compute mean
            all_corners = np.stack(valid_corners, axis=0)  # Shape: (N, 4, 2)
            mean_corners = np.mean(all_corners, axis=0)    # Shape: (4, 2)
            
            # Set mean_corners_CCV for all datapoints
            for datapoint in self.datapoints:
                datapoint.mean_corners_CCV = mean_corners
            
            logger.info(f"Computed mean corners from {len(valid_corners)} valid CCV detections")
        else:
            # No valid CCV corners found, set to None for all datapoints
            for datapoint in self.datapoints:
                datapoint.mean_corners_CCV = None
            logger.warning("No valid CCV corners found to compute mean")

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
        if self.config.get("seg_mini_model", False):
            self.seg_model = UNETWithDropoutMini(in_channels=1, out_channels=1).to(self.device)
        load_seg_ckpt(torch.load(self.segmentation_model_path, map_location=self.device), self.seg_model)
        self.seg_model.eval()
        self.kp_transform = A.Compose([ToTensorV2()]) 
        self.kp_model = RegressorMobileNetV3().to(self.device)
        load_kp_ckpt(torch.load(self.keypoints_model_path, map_location=self.device), self.kp_model)
        self.kp_model.eval()

    def _extract_matched_keypoints(self, keypoints_3d, keypoints_2d, tf_estimate, camera_matrix, max_reprojection_error=5.0):
        """
        Extract matched keypoints based on reprojection error after pose estimation.
        This simulates RANSAC-like filtering by finding closest matches between detected keypoints
        and projected marker corners, then keeping those with low reprojection error.
        
        Args:
            keypoints_3d: 3D keypoints in marker coordinate system (N x 3) - typically marker corners
            keypoints_2d: 2D keypoints in image space (M x 2) - detected keypoints
            tf_estimate: Estimated transformation matrix (4 x 4)
            camera_matrix: Camera intrinsic matrix (3 x 3)
            max_reprojection_error: Maximum allowed reprojection error in pixels
            
        Returns:
            numpy.ndarray: Matched 2D keypoints (K x 2) where K <= min(N, M)
        """
        if keypoints_3d is None or keypoints_2d is None or tf_estimate is None:
            return None
            
        keypoints_3d = np.array(keypoints_3d)
        keypoints_2d = np.array(keypoints_2d)
        
        # Project 3D keypoints (marker corners) using estimated pose
        projected_2d = project_points_array_to_image(
            C=camera_matrix,
            T=tf_estimate, 
            P_array=keypoints_3d,
            convert_cam_is2cv=True
        )
        
        print(f"Shapes: detected keypoints {keypoints_2d.shape}, projected corners {projected_2d.shape}")
        
        # Find closest matches between detected keypoints and projected corners
        matched_keypoints = []
        matched_corners = []
        
        for i, projected_corner in enumerate(projected_2d):
            # Calculate distances from this projected corner to all detected keypoints
            distances = np.linalg.norm(keypoints_2d - projected_corner, axis=1)
            closest_idx = np.argmin(distances)
            min_distance = distances[closest_idx]
            
            # Only consider it a match if the distance is below threshold
            if min_distance < max_reprojection_error:
                matched_keypoints.append(keypoints_2d[closest_idx])
                matched_corners.append(projected_corner)
        
        matched_keypoints = np.array(matched_keypoints) if matched_keypoints else None
        
        # if matched_keypoints is not None:
        #     print(f"Matched {len(matched_keypoints)} out of {len(projected_2d)} projected corners with detected keypoints (threshold: {max_reprojection_error} pixels)")
        # else:
        #     print(f"No matches found between projected corners and detected keypoints (threshold: {max_reprojection_error} pixels)")
        
        return matched_keypoints

    def run_LBCV_segmentation(self, image, detection_threshold=1000):         
        if self.config.get("seg_mini_model", False):
            input_size = (480, 640)  # For the mini model, we use a fixed input size, FIXME: avoid hardcoding
            img_tensor = self.seg_transform(image=image)["image"].unsqueeze(0).to(self.device)
            # For the mini model, we need to convert the image to grayscale
            img_tensor = img_tensor.mean(dim=1, keepdim=True)  # Convert to grayscale by averaging channels
            # Tile original image
            image_tiles, image_tiles_coords = split_image_by_aspect_ratio(
                image, M=input_size[0], N=input_size[1]
            )

            seg_tiles = []

            for tile in image_tiles:
                orig_h, orig_w = tile.shape[:2]

                # Resize to model input size
                tile_resized = cv2.resize(tile, (input_size[1], input_size[0]), interpolation=cv2.INTER_LINEAR)

                # Prepare input tensor
                seg_transform = A.Compose([
                    A.Normalize(max_pixel_value=1.0),
                    ToTensorV2(),
                ])
                transformed = seg_transform(image=tile_resized)
                tile_tensor = transformed["image"].unsqueeze(0).to(self.device)

                # convert to grayscale 
                if tile_tensor.shape[1] == 3:
                    tile_tensor = tile_tensor.mean(dim=1, keepdim=True)

                # Predict
                with torch.no_grad():
                    seg_output = torch.sigmoid(self.seg_model(tile_tensor))  # shape: (1, 1, H, W)
                    seg_mask = seg_output.squeeze().cpu().numpy()  # shape: (H, W)
                    torch.cuda.empty_cache()

                # Threshold (optional)
                seg_mask = (seg_mask > 0.5).astype(np.uint8)

                # Resize segmentation output back to original tile size
                seg_mask_resized = cv2.resize(seg_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

                seg_tiles.append(seg_mask_resized)

            # Stitch back
            seg_mask = combine_tiles_and_coords(seg_tiles, image_tiles_coords, image.shape[:2])
            mean_mask_score = seg_mask.mean().item()  # Get the mean score of the segmentation mask
            seg_mask_img = Image.fromarray(seg_mask.squeeze().astype(np.uint8) * 255)
            if np.count_nonzero(np.array(seg_mask_img)) > detection_threshold:
                bool_detected = True  
            else:
                bool_detected = False

            return seg_mask_img, bool_detected, mean_mask_score

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

                if bool_detected:

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

                        if tf_PBCV is not None:
                            # Extract matched keypoints based on reprojection error
                            matched_keypoints = self._extract_matched_keypoints(
                                keypoints_marker_cartesian_space, keypoints_rgb_image_space, 
                                tf_PBCV, self.camera_matrix, max_reprojection_error=5.0
                            )
                            
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
                            self.datapoints[idx].set_keypoints_PBCV_matched(matched_keypoints)
                            self.datapoints[idx].set_detected_PBCV(True) 
                            self.datapoints[idx].set_image_similarity_score(image_similarity_score)
                            self.datapoints[idx].set_PBCV_IOU(PBCV_IOU)
                            # NOTE: filtering out bad estimates based on scores 
                            bool_detected_PBCV = (harris_corner_response_score > 0.001) and (num_valid_proj_points > 2) and (tf_PBCV[2,3] < 10) and (image_similarity_score > 20_000) 
                            # self.datapoints[idx].set_detected_PBCV(bool_detected_PBCV)
                        else:
                            self.datapoints[idx].set_tf_PBCV(None)
                            self.datapoints[idx].set_keypoints_PBCV(None)
                            self.datapoints[idx].set_keypoints_PBCV_matched(None)
                            self.datapoints[idx].set_detected_PBCV(False)
                            self.datapoints[idx].set_detection_scores(None) 
                            self.datapoints[idx].set_image_similarity_score(None)

                    else:
                        self.datapoints[idx].set_tf_PBCV(None)
                        self.datapoints[idx].set_keypoints_PBCV(None)
                        self.datapoints[idx].set_keypoints_PBCV_matched(None)
                        self.datapoints[idx].set_detected_PBCV(False)
                        self.datapoints[idx].set_detection_scores(None) 
                        self.datapoints[idx].set_image_similarity_score(None)

                    # FIXME: this can be made more elegant 
                    if not bool_detected: self.datapoints[idx].set_detected_PBCV(False) 
                
                else:
                    self.datapoints[idx].set_tf_PBCV(None)
                    self.datapoints[idx].set_keypoints_PBCV(None)
                    self.datapoints[idx].set_keypoints_PBCV_matched(None)
                    self.datapoints[idx].set_detected_PBCV(False)
                    self.datapoints[idx].set_detection_scores(None) 
                    self.datapoints[idx].set_image_similarity_score(None)

            if save_results:
                output_dir = os.path.join(self.directory, "LBCV_keypoints_results")
                if self.config.get("seg_mini_model", False):
                    output_dir = os.path.join(self.directory, "LBCV_keypoints_results_minimodel")
                os.makedirs(output_dir, exist_ok=True)
                out_img = image.copy()
                if bool_detected:
                    for kp in keypoints_est:
                        cv2.circle(out_img, tuple(kp.astype(int)), 3, (0, 255, 0), -1)
                outpath = os.path.join(output_dir, f"LBCV_{idx:05d}.png")
                cv2.imwrite(str(outpath), out_img)

                if run_corners_HCV:
                    output_dir = os.path.join(self.directory, "HCV_corners_results")
                    if self.config.get("seg_mini_model", False):
                        output_dir = os.path.join(self.directory, "HCV_corners_results_minimodel")
                    os.makedirs(output_dir, exist_ok=True)
                    out_img = image.copy()
                    if bool_detected_hybrid:
                        for kp in corners_est:
                            cv2.circle(out_img, tuple(kp.astype(int)), 3, (0, 255, 0), -1)
                    outpath = os.path.join(output_dir, f"HCV_{idx:05d}.png")
                    cv2.imwrite(str(outpath), out_img)

                output_dir = os.path.join(self.directory, "LBCV_segmentation_results") 
                if self.config.get("seg_mini_model", False):
                    output_dir = os.path.join(self.directory, "LBCV_segmentation_results_minimodel")
                os.makedirs(output_dir, exist_ok=True)
                out_segmentation_path = os.path.join(output_dir, f"LBCV_segmentation_{idx:05d}.png")
                image_segmentation.save(out_segmentation_path) 

                if run_PBCV:
                    output_dir = os.path.join(self.directory, "PBCV_keypoints_results")
                    if self.config.get("seg_mini_model", False):
                        output_dir = os.path.join(self.directory, "PBCV_keypoints_results_minimodel")
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
            if self.config.get("seg_mini_model", False):
                output_dir += "_seg_mini"
            os.makedirs(output_dir, exist_ok=True) 
            output_path = os.path.join(output_dir, "results.json")
            with open(output_path, 'w') as f:
                json.dump(self.dict_results, f, indent=4)

            df_output_path = os.path.join(output_dir, "results.csv")
            self.df_results.to_csv(df_output_path, index=False)

    def draw_marker_borders_batch(self, datapoint_indices=None, output_dir=None, 
                                draw_lbcv=True, draw_pbcv=True, 
                                lbcv_color=(0, 255, 0), pbcv_color=(0, 0, 255), 
                                true_color=(255, 0, 0), line_thickness=2, alpha=0.7):
        """
        Draw marker borders for multiple datapoints using pose estimates.
        Uses alpha transparency so overlapping borders are visible.
        
        Args:
            datapoint_indices (list, optional): List of datapoint indices to process. If None, processes all datapoints.
            output_dir (str, optional): Directory to save output images. If None, creates 'marker_borders_vis' in data directory.
            draw_lbcv (bool): Whether to draw LBCV pose estimate borders (default: True)
            draw_pbcv (bool): Whether to draw PBCV pose estimate borders (default: True)
            lbcv_color (tuple): BGR color for LBCV borders (default: green)
            pbcv_color (tuple): BGR color for PBCV borders (default: red)
            true_color (tuple): BGR color for true pose borders (default: blue)
            line_thickness (int): Thickness of the border lines (default: 2)
            alpha (float): Alpha transparency value (0.0 = transparent, 1.0 = opaque, default: 0.7)
        """
        if datapoint_indices is None:
            datapoint_indices = range(len(self.datapoints))
        
        if output_dir is None:
            output_dir = os.path.join(self.directory, "marker_borders_vis")
        os.makedirs(output_dir, exist_ok=True)
        
        processed_count = 0
        for idx in datapoint_indices:
            if idx >= len(self.datapoints):
                print(f"Warning: Index {idx} is out of range. Skipping.")
                continue
                
            datapoint = self.datapoints[idx]
            
            # Check if we have at least one pose estimate to draw
            has_poses = False
            if draw_lbcv and hasattr(datapoint, 'tf_LBCV') and datapoint.tf_LBCV is not None:
                has_poses = True
            if draw_pbcv and hasattr(datapoint, 'tf_PBCV') and datapoint.tf_PBCV is not None:
                has_poses = True
            
            if not has_poses:
                print(f"Skipping datapoint {idx}: No pose estimates available")
                continue
            
            try:
                output_path = os.path.join(output_dir, f"picture_{idx}_borders.png")
                datapoint.draw_marker_borders_from_poses(
                    output_path=output_path,
                    draw_lbcv=draw_lbcv,
                    draw_pbcv=draw_pbcv,
                    lbcv_color=lbcv_color,
                    pbcv_color=pbcv_color,
                    true_color=true_color,
                    line_thickness=line_thickness,
                    alpha=alpha
                )
                processed_count += 1
            except Exception as e:
                print(f"Error processing datapoint {idx}: {str(e)}")
        
        print(f"Successfully processed {processed_count} datapoints. Images saved to: {output_dir}")

    def draw_pose_comparison_batch(self, datapoint_indices=None, output_dir=None,
                                 ccv_color=(0, 0, 255), lbcv_color=(255, 0, 0), pbcv_color=(0, 255, 0),
                                 keypoint_color=(0, 255, 0), line_thickness=2, keypoint_radius=3, alpha=0.7):
        """
        Draw pose comparison visualizations for multiple datapoints with CCV, LBCV, PBCV borders and PBCV keypoints.
        Uses alpha transparency so overlapping borders are visible.
        
        Args:
            datapoint_indices (list, optional): List of datapoint indices to process. If None, processes all datapoints.
            output_dir (str, optional): Directory to save output images. If None, creates 'pose_comparison_vis' in data directory.
            ccv_color (tuple): BGR color for CCV borders (default: red)
            lbcv_color (tuple): BGR color for LBCV borders (default: blue)
            pbcv_color (tuple): BGR color for PBCV borders (default: green)
            keypoint_color (tuple): BGR color for PBCV keypoints (default: green)
            line_thickness (int): Thickness of the border lines (default: 2)
            keypoint_radius (int): Radius of keypoint circles (default: 3)
            alpha (float): Alpha transparency value (0.0 = transparent, 1.0 = opaque, default: 0.7)
            
        Returns:
            numpy.ndarray: Image with drawn borders and keypoints (if output_path is None)
        """
        if datapoint_indices is None:
            datapoint_indices = range(len(self.datapoints))
        
        if output_dir is None:
            output_dir = os.path.join(self.directory, "pose_comparison_vis")
        os.makedirs(output_dir, exist_ok=True)
        
        processed_count = 0
        for idx in datapoint_indices:
            if idx >= len(self.datapoints):
                print(f"Warning: Index {idx} is out of range. Skipping.")
                continue
                
            datapoint = self.datapoints[idx]
            
            # Check if we have at least one pose estimate to draw
            has_poses = False
            if hasattr(datapoint, 'tf_CCV') and datapoint.tf_CCV is not None:
                has_poses = True
            if hasattr(datapoint, 'tf_LBCV') and datapoint.tf_LBCV is not None:
                has_poses = True
            if hasattr(datapoint, 'tf_PBCV') and datapoint.tf_PBCV is not None:
                has_poses = True
            
            if not has_poses:
                print(f"Skipping datapoint {idx}: No pose estimates available")
                continue
            
            try:
                output_path = os.path.join(output_dir, f"picture_{idx}_pose_comparison.png")
                datapoint.draw_pose_comparison_visualization(
                    output_path=output_path,
                    ccv_color=ccv_color,
                    lbcv_color=lbcv_color,
                    pbcv_color=pbcv_color,
                    keypoint_color=keypoint_color,
                    line_thickness=line_thickness,
                    keypoint_radius=keypoint_radius,
                    alpha=alpha
                )
                processed_count += 1
            except Exception as e:
                print(f"Error processing datapoint {idx}: {str(e)}")
        
        print(f"Successfully processed {processed_count} pose comparison visualizations. Images saved to: {output_dir}")
    
    def draw_pbcv_matched_keypoints_batch(self, datapoint_indices=None, output_dir=None,
                                        border_color=(0, 255, 0), keypoint_color=(0, 255, 0),
                                        line_thickness=2, keypoint_radius=4, alpha=0.8):
        """
        Draw PBCV matched keypoints visualizations for multiple datapoints.
        Shows only matched keypoints (after RANSAC) with PBCV pose-based marker border.
        
        Args:
            datapoint_indices (list, optional): List of datapoint indices to process. If None, processes all datapoints.
            output_dir (str, optional): Directory to save output images. If None, creates 'pbcv_matched_keypoints_vis' in data directory.
            border_color (tuple): BGR color for PBCV pose border (default: green)
            keypoint_color (tuple): BGR color for matched keypoints (default: green)
            line_thickness (int): Thickness of the border lines (default: 2)
            keypoint_radius (int): Radius of keypoint circles (default: 4)
            alpha (float): Alpha transparency value (0.0 = transparent, 1.0 = opaque, default: 0.8)
        """
        if datapoint_indices is None:
            datapoint_indices = range(len(self.datapoints))
        
        if output_dir is None:
            output_dir = os.path.join(self.directory, "pbcv_matched_keypoints_vis")
        os.makedirs(output_dir, exist_ok=True)
        
        processed_count = 0
        for idx in datapoint_indices:
            if idx >= len(self.datapoints):
                print(f"Warning: Index {idx} is out of range. Skipping.")
                continue
                
            datapoint = self.datapoints[idx]
            
            # Check if we have PBCV pose estimate and keypoints
            has_pbcv_data = False
            if hasattr(datapoint, 'tf_PBCV') and datapoint.tf_PBCV is not None:
                has_pbcv_data = True
            
            # Check for keypoints (either matched or all PBCV keypoints)
            has_keypoints = False
            if hasattr(datapoint, 'keypoints_PBCV_matched') and datapoint.keypoints_PBCV_matched is not None:
                has_keypoints = True
            elif hasattr(datapoint, 'keypoints_PBCV') and datapoint.keypoints_PBCV is not None:
                has_keypoints = True
            
            if not has_pbcv_data or not has_keypoints:
                print(f"Skipping datapoint {idx}: No PBCV pose estimate or keypoints available")
                continue
            
            try:
                output_path = os.path.join(output_dir, f"picture_{idx}_pbcv_matched_keypoints.png")
                datapoint.draw_pbcv_matched_keypoints_visualization(
                    output_path=output_path,
                    border_color=border_color,
                    keypoint_color=keypoint_color,
                    line_thickness=line_thickness,
                    keypoint_radius=keypoint_radius,
                    alpha=alpha
                )
                processed_count += 1
            except Exception as e:
                print(f"Error processing datapoint {idx}: {str(e)}")
        
        print(f"Successfully processed {processed_count} PBCV matched keypoints visualizations. Images saved to: {output_dir}")
        
    def create_combined_2x2_visualizations_batch(self, datapoint_indices=None, output_dir=None, 
                                               experiment_name=None, experiment_variable=None, 
                                               experiment_values=None, figsize=(18, 12),
                                               ablation_config=None):
        """
        Create combined 2x2 visualizations for a batch of datapoints.
        
        Args:
            datapoint_indices (list, optional): List of datapoint indices to process. If None, processes all datapoints.
            output_dir (str, optional): Directory to save the output images. If None, creates 'combined_2x2_visualizations' directory.
            experiment_name (str, optional): Name of the experiment for the suptitle
            experiment_variable (str, optional): Name of the experiment variable (e.g., 'mean_marker_pixel_brightness')
            experiment_values (list, optional): List of experiment values corresponding to each datapoint
            figsize (tuple): Figure size in inches (default: (18, 12))
            ablation_config (dict, optional): Configuration from YAML file for this ablation
        """
        # If ablation_config is provided, extract variable name and filter data
        if ablation_config is not None:
            experiment_variable = ablation_config.get("ablation_variable")
            
            # Apply filtering based on min/max values from YAML config
            filtered_indices = self._filter_datapoints_by_ablation_range(ablation_config)
            
            # Extract experiment values from filtered datapoints
            experiment_values = []
            for idx in filtered_indices:
                if idx < len(self.datapoints):
                    datapoint = self.datapoints[idx]
                    if hasattr(datapoint, 'metadata') and datapoint.metadata and experiment_variable in datapoint.metadata:
                        experiment_values.append(datapoint.metadata[experiment_variable])
                    else:
                        experiment_values.append(None)
            
            # Use filtered indices if no specific indices were provided
            if datapoint_indices is None:
                datapoint_indices = filtered_indices
        
        # Set default indices if not provided
        if datapoint_indices is None:
            datapoint_indices = list(range(len(self.datapoints)))
        
        # Set default output directory if not provided
        if output_dir is None:
            output_dir = os.path.join(self.directory, "combined_2x2_visualizations")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        processed_count = 0
        
        for i, idx in enumerate(datapoint_indices):
            if idx >= len(self.datapoints):
                print(f"Warning: Datapoint index {idx} is out of range. Skipping.")
                continue
            
            try:
                datapoint = self.datapoints[idx]
                
                # Set processor reference for segmentation access
                datapoint.processor = self
                
                # Get experiment value for this datapoint
                experiment_value = experiment_values[i] if experiment_values and i < len(experiment_values) else None
                
                # Create output filename
                output_filename = f"combined_2x2_datapoint_{idx:04d}.png"
                output_path = os.path.join(output_dir, output_filename)
                
                # Create the combined visualization
                datapoint.create_combined_2x2_visualization(
                    experiment_name=experiment_name,
                    experiment_variable=experiment_variable,
                    experiment_value=experiment_value,
                    output_path=output_path,
                    figsize=figsize
                )
                processed_count += 1
            except Exception as e:
                print(f"Error processing datapoint {idx}: {str(e)}")
        
        print(f"Successfully processed {processed_count} combined 2x2 visualizations. Images saved to: {output_dir}")

    def _filter_datapoints_by_ablation_range(self, ablation_config):
        """
        Filter datapoints based on ablation variable min/max values from YAML config.
        Similar to the filtering done in real_data_plotter.py.
        
        Args:
            ablation_config (dict): Configuration from YAML file for this ablation
            
        Returns:
            list: List of filtered datapoint indices
        """
        filtered_indices = []
        ablation_variable = ablation_config.get("ablation_variable")
        min_val = ablation_config.get("ablation_variable_min")
        max_val = ablation_config.get("ablation_variable_max")
        
        # If no filtering specified, return all indices
        if min_val is None or max_val is None:
            return list(range(len(self.datapoints)))
        
        for idx, datapoint in enumerate(self.datapoints):
            if hasattr(datapoint, 'metadata') and datapoint.metadata and ablation_variable in datapoint.metadata:
                value = datapoint.metadata[ablation_variable]
                if value is not None and min_val <= value <= max_val:
                    filtered_indices.append(idx)
        
        print(f"Filtered {len(filtered_indices)} datapoints out of {len(self.datapoints)} based on {ablation_variable} range [{min_val}, {max_val}]")
        return filtered_indices

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
    ablation = "shadow_20250712"  # options: "underexposure", "ambient_light_intensity", "truncation", "skew", "lateral", "pitch", "yaw", "roll"
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
    # processor.draw_marker_borders_batch(draw_lbcv=True, draw_pbcv=False, line_thickness=2)
    # processor.draw_pose_comparison_batch(line_thickness=2, keypoint_radius=3)
    # processor.draw_pbcv_matched_keypoints_batch()
    
    # Create combined 2x2 visualizations
    # Get the ablation configuration from the data description
    ablation_config = data_description[ablation]
    
    processor.create_combined_2x2_visualizations_batch(
        experiment_name=ablation.replace("_20250712", "").title() + " Experiment",
        ablation_config=ablation_config
    )

if __name__ == "__main__":
    main()