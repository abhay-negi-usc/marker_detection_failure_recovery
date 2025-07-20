import numpy as np 
import cv2 
import torch 
from pose_estimation_model.optimal_overlap_torch import * 

def main():
    p = Processor(
        dir_seg="output/sdg_markers_20250323-031913/seg/",
        dir_rgb="output/sdg_markers_20250323-031913/rgb/",
        marker_path="synthetic_data_generation/assets/tags/4x4_1000-31.png"
    )

    width = 640 
    height = 480 
    focal_length = 24.0 
    horiz_aperture = 20.955
    vert_aperture = height/width * horiz_aperture
    fx = width * focal_length / horiz_aperture
    fy = height * focal_length / vert_aperture
    cx = width / 2
    cy = height / 2

    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    marker_corners_2d = np.array([
        [0, 0],
        [0, p.marker.shape[0]],
        [p.marker.shape[1], p.marker.shape[0]],
        [p.marker.shape[1], 0]
    ], dtype=np.float32)

    marker_length = 0.10  # meters
    marker_corners_3d = np.array([
        [0, 0, 0],
        [marker_length, 0, 0],
        [marker_length, marker_length, 0],
        [0, marker_length, 0]
    ], dtype=np.float32)

    dp = p.datapoints[0]
    seg_gt = dp.get_seg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pose_initial = estimate_initial_pose_from_segmentation(seg_gt, camera_matrix, marker_length, marker_corners_2d, marker_corners_3d, p.marker, image_size=(height, width), device=device)
    print("Initial pose:", pose_initial.detach().cpu().numpy())

    pose_initial = (pose_initial.clone().detach() +
                torch.tensor([0.0, -0.0, 0.0, 0.0, 0, 10], device=pose_initial.device)).requires_grad_()

    use_gradient_free = True  
    dim_ranges = np.array([0.10, 0.10, 1.0, 10.0, 10.0, 10.0])

    if use_gradient_free:
        optimized_pose, loss_history = p._optimize_pose_gradient_free(
            p.marker, marker_corners_2d, marker_corners_3d, seg_gt, camera_matrix, pose_initial, device, steps=1000, perturb_eps=dim_ranges*1e-1, num_perturb=25, lr=1e-13)
    else:
        optimized_pose, loss_history = p._optimize_pose_custom(
            p.marker, marker_corners_2d, marker_corners_3d, seg_gt, camera_matrix, pose_initial, device, lr=1e-4, steps=1000)

    print("Final pose:", optimized_pose.numpy())
    p.plot_loss_curve(loss_history)

    with torch.no_grad():
        initial_rendered = marker_reprojection_differentiable(p.marker, marker_corners_2d, marker_corners_3d, pose_initial.to(device), camera_matrix)
        final_rendered = marker_reprojection_differentiable(p.marker, marker_corners_2d, marker_corners_3d, optimized_pose.to(device), camera_matrix)

    rgb = cv2.resize(dp.get_rgb(), (final_rendered.shape[-1], final_rendered.shape[-2]))
    rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    initial_np = (initial_rendered.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    final_np = (final_rendered.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    blended_initial = cv2.addWeighted(rgb, 0.4, initial_np, 0.6, 0)
    blended_final = cv2.addWeighted(rgb, 0.4, final_np, 0.6, 0)

    comparison = np.hstack([blended_initial, blended_final])
    cv2.imwrite("comparison_overlay.png", comparison)
    print("Saved initial and final overlay comparison to comparison_overlay.png")

if __name__ == "__main__":
    main()