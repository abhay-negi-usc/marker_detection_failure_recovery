#!/usr/bin/env python3
"""
Example script demonstrating the new PBCV matched keypoints visualization functionality.

This script shows how to:
1. Run PBCV pose estimation which extracts matched keypoints after RANSAC-like filtering
2. Visualize only the matched keypoints along with the marker border based on PBCV pose
3. Compare matched vs all keypoints
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
    
    # Run pose estimation methods (this will extract matched keypoints automatically)
    print("Step 1: Running CCV detection...")
    processor.run_opencv_fiducial_marker_detection(save_results=False)
    
    print("Step 2: Running LBCV and PBCV detection (with matched keypoints extraction)...")
    processor.run_LBCV_fiducial_marker_detection(save_results=False, run_corners_HCV=True, run_PBCV=True) 
    
    # Example 1: Single datapoint PBCV matched keypoints visualization
    print("\nExample 1: Creating PBCV matched keypoints visualization for datapoint 0")
    if len(processor.datapoints) > 0:
        datapoint = processor.datapoints[0]
        
        # Check if we have matched keypoints
        if hasattr(datapoint, 'keypoints_PBCV_matched') and datapoint.keypoints_PBCV_matched is not None:
            print(f"✓ Found {len(datapoint.keypoints_PBCV_matched)} matched keypoints")
            
            output_path = "example_pbcv_matched_keypoints_single.png"
            try:
                datapoint.draw_pbcv_matched_keypoints_visualization(
                    output_path=output_path,
                    border_color=(0, 255, 0),      # Green border
                    keypoint_color=(0, 255, 0),    # Green keypoints
                    line_thickness=3,
                    keypoint_radius=5,
                    alpha=0.8
                )
                print(f"PBCV matched keypoints visualization saved to: {output_path}")
            except Exception as e:
                print(f"Error creating single visualization: {e}")
        else:
            print("✗ No matched keypoints found for datapoint 0")
            
            # Check if we have any PBCV keypoints at all
            if hasattr(datapoint, 'keypoints_PBCV') and datapoint.keypoints_PBCV is not None:
                print(f"  Note: Found {len(datapoint.keypoints_PBCV)} total PBCV keypoints")
                print("  This likely means no keypoints passed the reprojection error threshold")
            else:
                print("  No PBCV keypoints detected at all")
    
    # Example 2: Batch PBCV matched keypoints visualizations
    print("\nExample 2: Creating batch PBCV matched keypoints visualizations")
    try:
        processor.draw_pbcv_matched_keypoints_batch(
            datapoint_indices=[0, 1, 2, 3],  # Process first 4 datapoints
            output_dir="example_pbcv_matched_keypoints_batch",
            border_color=(0, 255, 0),      # Green border
            keypoint_color=(0, 255, 0),    # Green keypoints
            line_thickness=2,
            keypoint_radius=4,
            alpha=0.8
        )
    except Exception as e:
        print(f"Error in batch processing: {e}")
    
    # Example 3: Comparison with all keypoints vs matched keypoints
    print("\nExample 3: Creating comparison visualizations")
    if len(processor.datapoints) > 0:
        datapoint = processor.datapoints[0]
        
        # Draw all PBCV keypoints
        try:
            all_keypoints_img = datapoint.draw_pose_comparison_visualization(
                output_path="comparison_all_keypoints.png",
                ccv_color=(0, 0, 255),        # Red for CCV
                lbcv_color=(255, 0, 0),       # Blue for LBCV  
                pbcv_color=(0, 255, 0),       # Green for PBCV
                keypoint_color=(255, 255, 0), # Cyan for all keypoints
                line_thickness=2,
                keypoint_radius=3,
                alpha=0.7
            )
            print("All keypoints comparison saved to: comparison_all_keypoints.png")
        except Exception as e:
            print(f"Error creating all keypoints comparison: {e}")
            
        # Draw only matched keypoints
        try:
            matched_keypoints_img = datapoint.draw_pbcv_matched_keypoints_visualization(
                output_path="comparison_matched_keypoints.png",
                border_color=(0, 255, 0),     # Green border
                keypoint_color=(255, 255, 0), # Cyan for matched keypoints
                line_thickness=2,
                keypoint_radius=4,
                alpha=0.8
            )
            print("Matched keypoints only saved to: comparison_matched_keypoints.png")
        except Exception as e:
            print(f"Error creating matched keypoints visualization: {e}")

    print("\nPBCV matched keypoints visualization examples completed!")
    print("\nWhat was done:")
    print("  1. Ran PBCV pose estimation with automatic matched keypoints extraction")
    print("  2. Filtered keypoints based on reprojection error (simulating RANSAC inliers)")
    print("  3. Created visualizations showing only the high-quality matched keypoints")
    print("  4. Added marker borders based on PBCV pose estimates")
    print("\nVisualization features:")
    print("  - Green border: PBCV pose-based marker boundary")
    print("  - Green/Cyan circles: PBCV matched keypoints (after filtering)")
    print("  - Semi-transparent overlay for clear visualization")
    print("\nThis helps evaluate the quality of PBCV pose estimation by showing")
    print("only the keypoints that contributed to the final pose estimate.")

if __name__ == "__main__":
    main()
