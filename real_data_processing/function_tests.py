import cv2 
import matplotlib.pyplot as plt 
from pose_estimation_model.utils import * 

fx, fy, cx, cy, dist_coeffs = 1363.85, 1365.40, 958.58, 552.25, np.array([0.1693, -0.4755, 0.0018, 0.0023, 0.4114])
sx, sy = 640 / 1920, 480 / 1080
camera_matrix_resized = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]]) 

idx = 85
img_path = f"./test_data/realsense/realsense_6_frames/realsense_6_frame_{idx:05d}.png"
seg_path = f"./test_data/realsense/realsense_6_frames_LBCV/segmentation_masks/LBCV_seg_{idx:05d}.png"
img_marker_path = "/home/rp/abhay_ws/marker_detection_failure_recovery/synthetic_data_generation/assets/tags/tag36h11_0.png"
 # read segmentation mask
img_rgb = cv2.imread(img_path, cv2.IMREAD_COLOR)
img_seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
img_rgb = cv2.resize(img_rgb, (640, 480))
img_seg = cv2.resize(img_seg, (640, 480))
img_rgb_masked = crop_rgb_using_seg(img_rgb, img_seg)
corners, area_ratio = find_segmentation_four_corners(img_seg, bound_box=False)

if corners is not None: 
    img_rgb_with_corners = overlay_points_on_image(img_rgb.copy(),corners)
    plt.imshow(img_rgb_with_corners[:, :, ::-1]) 
    plt.show() 

plt.imshow(img_seg) 
plt.title(f"area ratio: {area_ratio}")
plt.show() 

# fill in segmentation image from polygon vertices 
quad_seg = fill_segmentation_from_polygon(img_seg.shape, corners)
tf_candidates = compute_tf_candidates_from_corners(corners, marker_size=(0.1, 0.1), camera_matrix=camera_matrix_resized)
keypoints_rgb_image_space = find_keypoints(img_rgb, quad_seg)
img_marker = cv2.imread(img_marker_path)
keypoints_marker_image_space = find_keypoints(img_marker)

