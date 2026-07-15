#!/usr/bin/env python3
"""
Example script demonstrating how to use the create_combined_2x2_visualization function.

This script shows how to create a 2x2 combined image visualization showing:
- Top left: Classical Detection & Pose Estimation (CCV) with red border
- Top right: Learning-Based Segmentation 
- Bottom left: Learning-Based Keypoints & Pose Estimation (LBCV) with blue keypoints and border
- Bottom right: Pattern-Based Pose Estimation (PBCV) with green matched keypoints and border
"""

import os
import cv2
import numpy as np
import yaml
import torch
from ablations.analysis.real_data_processor import DataProcessor

def example_create_combined_visualization():
    """Example function showing how to create a combined 2x2 visualization."""
    
    # Camera parameters (example values - replace with your actual camera calibration)
    camera_parameters = {
        "fx": 800.0,
        "fy": 800.0, 
        "cx": 320.0,
        "cy": 240.0,
        "width": 640,
        "height": 480,
        "distortion_coefficients": [0.1, -0.2, 0.001, 0.001, 0.0]
    }

    # Marker parameters  
    marker_parameters = {
        "marker_length": 0.100,  # units: meters
        "marker_length_without_border": 0.080,  # units: meters
        "num_squares": 10, # including border 
        "aruco_dict": cv2.aruco.DICT_APRILTAG_36h11, 
    }

    # Configure the experiment data path
    ablation = "underexposure_20250712"  # Change this to your experiment
    data_yaml_path = "./ablations/real_exp_data_description.yaml"
    
    # Load experiment data configuration
    with open(data_yaml_path, 'r') as f:
        data_description = yaml.safe_load(f) 
    data_path = data_description[ablation]["data_path"]

    # Configure the data processor
    config = {
        "data_path": data_path, 
        "max_num_datapoints": 5,  # Process only first 5 images for this example
        "camera_parameters": camera_parameters,
        "marker_parameters": marker_parameters, 
        "seg_model_path": "./segmentation_model/models/my_checkpoint_20250329.pth.tar",
        "kp_model_path": "./keypoints_model/models/my_checkpoint_keypoints_20250401.pth.tar", 
        "device": "cuda" if torch.cuda.is_available() else "cpu", 
    }

    # Initialize the processor
    processor = DataProcessor(config)
    
    # Run the detection and pose estimation pipelines
    print("Running classical detection...")
    processor.run_opencv_fiducial_marker_detection(save_results=False)
    
    print("Running learning-based detection...")
    processor.run_LBCV_fiducial_marker_detection(
        save_results=False, 
        run_corners_HCV=True, 
        run_PBCV=True, 
        use_precomputed_segmentation=False
    )
    
    print("Computing values...")
    processor.compute_values()
    processor.compile_results(save_results=True)

    # Create combined visualizations for all processed datapoints
    print("Creating combined 2x2 visualizations...")
    
    # For underexposure experiment, extract mean pixel brightness values
    experiment_values = []
    for datapoint in processor.datapoints:
        if hasattr(datapoint, 'metadata') and datapoint.metadata and 'mean_marker_pixel_brightness' in datapoint.metadata:
            experiment_values.append(datapoint.metadata['mean_marker_pixel_brightness'])
        else:
            experiment_values.append(None)
    
    # Create the combined visualizations
    processor.create_combined_2x2_visualizations_batch(
        experiment_name="Underexposure Experiment",
        experiment_variable="mean_marker_pixel_brightness",
        experiment_values=experiment_values,
        figsize=(14, 12)  # Larger figure size for better visibility
    )
    
    print("Combined visualizations created successfully!")
    print(f"Check the output directory: {os.path.join(processor.directory, 'combined_2x2_visualizations')}")


def example_single_datapoint():
    """Example showing how to create a visualization for a single datapoint."""
    
    # (Same setup as above - abbreviated for brevity)
    camera_parameters = {
        "fx": 800.0, "fy": 800.0, "cx": 320.0, "cy": 240.0,
        "width": 640, "height": 480, "distortion_coefficients": []
    }
    
    marker_parameters = {
        "marker_length": 0.100,
        "marker_length_without_border": 0.080,
        "num_squares": 10,
        "aruco_dict": cv2.aruco.DICT_APRILTAG_36h11,
    }
    
    config = {
        "data_path": "./test_data",  # Replace with your data path
        "max_num_datapoints": 1,
        "camera_parameters": camera_parameters,
        "marker_parameters": marker_parameters,
        "device": "cpu",
    }
    
    processor = DataProcessor(config)
    
    # Process the data
    processor.run_opencv_fiducial_marker_detection(save_results=False)
    processor.run_LBCV_fiducial_marker_detection(save_results=False, run_PBCV=True)
    
    # Create visualization for the first datapoint
    datapoint = processor.datapoints[0]
    datapoint.processor = processor  # Give datapoint access to processor for segmentation
    
    # Create the visualization
    fig = datapoint.create_combined_2x2_visualization(
        experiment_name="Test Experiment",
        experiment_variable="test_variable", 
        experiment_value=42.0,
        output_path="./combined_visualization_example.png"
    )
    
    print("Single datapoint visualization saved as: combined_visualization_example.png")


if __name__ == "__main__":
    print("Choose an example to run:")
    print("1. Batch processing example")
    print("2. Single datapoint example")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        example_create_combined_visualization()
    elif choice == "2":
        example_single_datapoint()
    else:
        print("Invalid choice. Running batch example by default.")
        example_create_combined_visualization()
