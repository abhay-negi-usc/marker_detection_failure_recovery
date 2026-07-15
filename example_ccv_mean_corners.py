#!/usr/bin/env python3
"""
Example demonstrating the use of mean CCV corners as reference in marker border visualization.

This script shows how the new functionality works:
1. Runs CCV detection on all images
2. Computes mean corners from all valid CCV detections
3. Uses these mean corners as the reference/true pose in visualization
4. Draws LBCV and PBCV borders for comparison
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
    
    # Step 1: Run CCV detection (this will also compute mean corners automatically)
    print("Step 1: Running CCV detection and computing mean corners...")
    processor.run_opencv_fiducial_marker_detection(save_results=False)
    
    # Step 2: Run LBCV and PBCV detection
    print("Step 2: Running LBCV and PBCV detection...")
    processor.run_LBCV_fiducial_marker_detection(save_results=False, run_corners_HCV=True, run_PBCV=True) 
    
    # Step 3: Check if mean corners were computed
    sample_datapoint = processor.datapoints[0]
    if hasattr(sample_datapoint, 'mean_corners_CCV') and sample_datapoint.mean_corners_CCV is not None:
        print(f"✓ Mean CCV corners computed successfully!")
        print(f"  Mean corners shape: {sample_datapoint.mean_corners_CCV.shape}")
        print(f"  Mean corners:\n{sample_datapoint.mean_corners_CCV}")
    else:
        print("✗ No mean CCV corners available (no valid CCV detections)")
        return
    
    # Step 4: Visualize borders using mean CCV corners as reference
    print("Step 3: Creating visualizations with CCV mean corners as reference...")
    
    # Draw borders for first few datapoints
    processor.draw_marker_borders_batch(
        datapoint_indices=[0, 1, 2],
        output_dir="ccv_mean_corners_visualization",
        draw_lbcv=True,
        draw_pbcv=True,
        lbcv_color=(0, 255, 0),    # Green for LBCV
        pbcv_color=(0, 0, 255),    # Red for PBCV  
        true_color=(255, 0, 0),    # Blue for CCV mean corners (reference)
        line_thickness=2
    )
    
    print("\nVisualization complete!")
    print("\nIn the output images:")
    print("  - Blue semi-transparent borders: CCV mean corners (computed from all valid CCV detections)")
    print("  - Green semi-transparent borders: LBCV pose estimates")
    print("  - Red semi-transparent borders: PBCV pose estimates")
    print("\nThe semi-transparent borders allow overlapping pose estimates to be visible,")
    print("making it easy to compare the accuracy of different methods.")
    print("The blue borders represent the mean/average marker position")
    print("computed from all successful CCV detections across the dataset.")

if __name__ == "__main__":
    main()
