#!/usr/bin/env python3
"""
Example script demonstrating the combined pose visualization function.
This creates a 2x2 grid showing all pose estimation methods in one image.
"""

import os
import sys

# Add the parent directory to the path to import from ablations.analysis
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from ablations.analysis.real_data_processor import DataProcessor

def main():
    """
    Example of creating combined pose visualizations showing all methods in a 2x2 grid.
    """
    
    # Example configuration - adjust paths as needed
    config = {
        "data_path": "./real_data_processing/real_exp/", 
        "max_num_datapoints": 5,  # Process only first 5 images for example
        "camera_parameters": {
            "fx": 800, "fy": 800, "cx": 320, "cy": 240,
            "k1": 0, "k2": 0, "p1": 0, "p2": 0, "k3": 0
        },
        "marker_parameters": {
            "marker_size": 0.1  # 10cm marker
        }
    }
    
    # Initialize data processor
    processor = DataProcessor(config)
    
    # Run pose estimation to populate the data
    print("Running pose estimation...")
    processor.run_LBCV_fiducial_marker_detection(
        save_results=False, 
        run_corners_HCV=True, 
        run_PBCV=True, 
        use_precomputed_segmentation=False
    )
    
    # Example 1: Single combined visualization
    print("\n=== Single Combined Visualization ===")
    datapoint = processor.datapoints[0]
    
    # Create experiment variable from metadata if available
    experiment_variable = None
    if hasattr(datapoint, 'metadata') and datapoint.metadata:
        if 'mean_marker_pixel_brightness' in datapoint.metadata:
            brightness = datapoint.metadata['mean_marker_pixel_brightness']
            experiment_variable = {"mean_marker_pixel_brightness": f"{brightness:.1f}"}
    
    output_path = "./combined_pose_example.png"
    datapoint.draw_combined_pose_visualization(
        output_path=output_path,
        experiment_name="Underexposure Experiment",
        experiment_variable=experiment_variable,
        keypoint_radius=4,
        line_thickness=3,
        alpha=0.8
    )
    print(f"Single combined visualization saved to: {output_path}")
    
    # Example 2: Batch processing
    print("\n=== Batch Combined Visualizations ===")
    
    # Process first 3 datapoints
    processor.draw_combined_pose_visualization_batch(
        datapoint_indices=[0, 1, 2],
        output_dir="./combined_pose_batch_output/",
        experiment_name="Lighting Robustness Study",
        experiment_variable_key="mean_marker_pixel_brightness",  # Will extract from metadata
        keypoint_radius=3,
        line_thickness=2,
        alpha=0.7
    )
    
    # Example 3: Different experiment types
    print("\n=== Different Experiment Visualization ===")
    
    # Simulate different experiment metadata
    if len(processor.datapoints) > 1:
        datapoint2 = processor.datapoints[1]
        
        # Example for occlusion experiment
        occlusion_variable = {"occlusion_percentage": "25%"}
        output_path2 = "./combined_pose_occlusion_example.png"
        datapoint2.draw_combined_pose_visualization(
            output_path=output_path2,
            experiment_name="Occlusion Robustness Study",
            experiment_variable=occlusion_variable,
            keypoint_radius=3,
            line_thickness=2,
            alpha=0.9,
            font_scale=0.6,
            title_font_scale=1.0
        )
        print(f"Occlusion experiment visualization saved to: {output_path2}")
    
    print("\n=== All visualizations completed! ===")
    print("The combined visualization shows:")
    print("  Top Left: Classical Detection & Pose Estimation (CCV) - Red border")
    print("  Top Right: Learning-Based Segmentation - Colored overlay")
    print("  Bottom Left: Learning-Based Keypoints & Pose (LBCV) - Blue keypoints & border")
    print("  Bottom Right: Pattern-Based Pose Estimation (PBCV) - Green keypoints & border")
    print("  Suptitle: Shows experiment name and variable value")

if __name__ == "__main__":
    main()
