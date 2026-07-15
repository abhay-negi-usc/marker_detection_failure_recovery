#!/usr/bin/env python3
"""
Example script demonstrating the new pose comparison visualization function.

This script shows how to use the new draw_pose_comparison_visualization function which displays:
- Red borders for CCV pose estimates
- Blue borders for LBCV pose estimates  
- Green borders for PBCV pose estimates
- Green keypoints overlay for PBCV matched keypoints
"""

import os
import sys
import cv2
import numpy as np
import torch
import yaml

# Add the project root to Python path
sys.path.append('/home/rp/abhay_ws/marker_detection_failure_recovery')

from ablations.analysis.real_data_processor import DataProcessor

def main():
    # Example configuration (adjust paths and parameters as needed)
    camera_parameters = {
        "width": 1280,
        "height": 720,
        "fx": 886.643,
        "fy": 886.643,
        "cx": 631.834,
        "cy": 367.724,
        "distortion_coefficients": np.zeros(5),
    }

    marker_parameters = {
        "marker_length": 0.100,  # units: meters
        "marker_length_without_border": 0.080,  # units: meters
        "num_squares": 10, # including border 
        "aruco_dict": cv2.aruco.DICT_APRILTAG_36h11, 
    }

    config = {
        "data_path": "/path/to/your/data",  # Update this path
        "max_num_datapoints": 10,  # Process first 10 images
        "camera_parameters": camera_parameters,
        "marker_parameters": marker_parameters, 
        "seg_model_path": "./segmentation_model/models/my_checkpoint.pth.tar",
        "kp_model_path": "./keypoints_model/models/my_checkpoint_keypoints.pth.tar",
        "device": "cuda" if torch.cuda.is_available() else "cpu", 
        "seg_mini_model": True, 
    }

    # Initialize processor
    processor = DataProcessor(config)
    
    # Run all pose estimation methods
    print("Step 1: Running CCV detection...")
    processor.run_opencv_fiducial_marker_detection(save_results=False)
    
    print("Step 2: Running LBCV and PBCV detection...")
    processor.run_LBCV_fiducial_marker_detection(save_results=False, run_corners_HCV=True, run_PBCV=True) 
    
    # Example 1: Single datapoint pose comparison visualization
    print("\nExample 1: Creating pose comparison visualization for datapoint 0")
    if len(processor.datapoints) > 0:
        datapoint = processor.datapoints[0]
        
        output_path = "example_pose_comparison_single.png"
        try:
            datapoint.draw_pose_comparison_visualization(
                output_path=output_path,
                ccv_color=(0, 0, 255),      # Red for CCV
                lbcv_color=(255, 0, 0),     # Blue for LBCV
                pbcv_color=(0, 255, 0),     # Green for PBCV
                keypoint_color=(0, 255, 0), # Green for keypoints
                line_thickness=3,
                keypoint_radius=4
            )
            print(f"Single pose comparison visualization saved to: {output_path}")
        except Exception as e:
            print(f"Error creating single visualization: {e}")
    
    # Example 2: Batch pose comparison visualizations
    print("\nExample 2: Creating batch pose comparison visualizations")
    try:
        processor.draw_pose_comparison_batch(
            datapoint_indices=[0, 1, 2, 3],  # Process first 4 datapoints
            output_dir="example_pose_comparison_batch",
            ccv_color=(0, 0, 255),      # Red for CCV
            lbcv_color=(255, 0, 0),     # Blue for LBCV
            pbcv_color=(0, 255, 0),     # Green for PBCV
            keypoint_color=(0, 255, 0), # Green for keypoints
            line_thickness=2,
            keypoint_radius=3
        )
    except Exception as e:
        print(f"Error in batch pose comparison: {e}")
    
    # Example 3: Custom colors for different visualization style
    print("\nExample 3: Creating visualizations with custom colors")
    try:
        processor.draw_pose_comparison_batch(
            datapoint_indices=[0, 1],
            output_dir="example_custom_colors",
            ccv_color=(0, 255, 255),    # Yellow for CCV
            lbcv_color=(255, 0, 255),   # Magenta for LBCV
            pbcv_color=(255, 255, 0),   # Cyan for PBCV
            keypoint_color=(255, 255, 0), # Cyan for keypoints
            line_thickness=4,
            keypoint_radius=5
        )
    except Exception as e:
        print(f"Error with custom colors: {e}")

    print("\nPose comparison visualization examples completed!")
    print("\nStandard color coding:")
    print("  - Red semi-transparent borders: CCV pose estimates")
    print("  - Blue semi-transparent borders: LBCV pose estimates")
    print("  - Green semi-transparent borders: PBCV pose estimates")
    print("  - Green circles: PBCV matched keypoints")
    print("\nThese visualizations use alpha transparency so overlapping borders")
    print("are visible, allowing you to compare all three pose estimation methods")
    print("simultaneously and see how well the PBCV keypoints align with the detected features.")

if __name__ == "__main__":
    main()
