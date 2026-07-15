#!/usr/bin/env python3
"""
Example script demonstrating how to use the new marker border drawing functionality.

This script shows how to:
1. Load a DataProcessor with pose estimates
2. Draw marker borders for individual datapoints
3. Draw marker borders for multiple datapoints in batch
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
    # Camera parameters (example - adjust based on your setup)
    camera_parameters = {
        "width": 1280,
        "height": 720,
        "fx": 886.643,
        "fy": 886.643,
        "cx": 631.834,
        "cy": 367.724,
        "distortion_coefficients": np.zeros(5),
    }

    # Marker parameters (example - adjust based on your setup)
    marker_parameters = {
        "marker_length": 0.100,  # units: meters
        "marker_length_without_border": 0.080,  # units: meters
        "num_squares": 10, # including border 
        "aruco_dict": cv2.aruco.DICT_APRILTAG_36h11, 
    }

    # Example configuration
    config = {
        "data_path": "/path/to/your/data",  # Update this path
        "max_num_datapoints": 10,  # Process only first 10 images for demo
        "camera_parameters": camera_parameters,
        "marker_parameters": marker_parameters, 
        "seg_model_path": "./segmentation_model/models/my_checkpoint.pth.tar",  # Update path
        "kp_model_path": "./keypoints_model/models/my_checkpoint_keypoints.pth.tar",  # Update path
        "device": "cuda" if torch.cuda.is_available() else "cpu", 
        "seg_mini_model": True, 
    }

    # Initialize processor
    processor = DataProcessor(config)
    
    # Run pose estimation methods to get LBCV and PBCV estimates
    processor.run_opencv_fiducial_marker_detection(save_results=False) 
    processor.run_LBCV_fiducial_marker_detection(save_results=False, run_corners_HCV=True, run_PBCV=True) 
    
    # Example 1: Draw borders for a single datapoint
    print("Example 1: Drawing borders for datapoint 0")
    if len(processor.datapoints) > 0:
        datapoint = processor.datapoints[0]
        
        # Draw borders and save to file
        output_path = "example_single_datapoint_borders.png"
        try:
            datapoint.draw_marker_borders_from_poses(
                output_path=output_path,
                draw_lbcv=True,
                draw_pbcv=True,
                lbcv_color=(0, 255, 0),  # Green for LBCV
                pbcv_color=(0, 0, 255),  # Red for PBCV
                true_color=(255, 0, 0),  # Blue for ground truth
                line_thickness=3
            )
            print(f"Single datapoint image saved to: {output_path}")
        except Exception as e:
            print(f"Error drawing borders for single datapoint: {e}")
    
    # Example 2: Draw borders for multiple datapoints in batch
    print("\nExample 2: Drawing borders for multiple datapoints")
    try:
        processor.draw_marker_borders_batch(
            datapoint_indices=[0, 1, 2],  # Process first 3 datapoints
            output_dir="example_batch_borders",
            draw_lbcv=True,
            draw_pbcv=True,
            lbcv_color=(0, 255, 0),  # Green for LBCV
            pbcv_color=(0, 0, 255),  # Red for PBCV
            true_color=(255, 0, 0),  # Blue for ground truth
            line_thickness=2
        )
    except Exception as e:
        print(f"Error in batch processing: {e}")
    
    # Example 3: Draw only LBCV borders
    print("\nExample 3: Drawing only LBCV borders")
    try:
        processor.draw_marker_borders_batch(
            datapoint_indices=[0, 1],
            output_dir="example_lbcv_only",
            draw_lbcv=True,
            draw_pbcv=False,  # Only LBCV
            lbcv_color=(0, 255, 255),  # Yellow for LBCV
            line_thickness=3
        )
    except Exception as e:
        print(f"Error drawing LBCV only: {e}")

    print("\nExample script completed!")
    print("\nColor coding:")
    print("  - Blue semi-transparent borders: CCV mean corners (reference pose)")
    print("  - Green semi-transparent borders: LBCV pose estimate")
    print("  - Red semi-transparent borders: PBCV pose estimate")
    print("\nThe alpha transparency allows overlapping borders to be visible,")
    print("making it easier to compare pose estimation accuracy.")

if __name__ == "__main__":
    main()
