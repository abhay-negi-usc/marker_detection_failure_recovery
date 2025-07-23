import numpy as np 
import cv2 
from pose_estimation_model.optimal_overlap_torch import * 

def compute_detection_score(image, detected_keypoints, reference_keypoints_3D, tf_est, camera_matrix, dist_coeffs, harris_corner_response_weight=0.1, keypoint_residual_score_weight=1.0):
    """
    Computes a detection score based on Harris corner response and keypoint residuals.
    Parameters:
    - image: The input image (a NumPy array).
    - detected_keypoints: A list of detected keypoints in image coordinates.
    - reference_keypoints_3D: A list of 3D reference keypoints in the marker frame.
    - tf_est: The estimated transformation matrix from the marker frame to the camera frame.
    - camera_matrix: The camera intrinsic matrix.
    - dist_coeffs: The distortion coefficients of the camera.
    - harris_corner_response_weight: Weight for the Harris corner response score.
    - keypoint_residual_score_weight: Weight for the keypoint residual score.
    Returns:
    - harris_corner_response_score: The Harris corner response score.
    - keypoint_residual_score: The keypoint residual score.
    - detection_score: The combined detection score.
    """
    image = np.array(image)
    rvec, _ = cv2.Rodrigues(tf_est[:3, :3])
    tvec = tf_est[:3,3].reshape(3, 1).astype(np.float32)
    projected_keypoints, _ = cv2.projectPoints(reference_keypoints_3D, rvec.copy(), tvec.copy(), camera_matrix, dist_coeffs)
    projected_keypoints = projected_keypoints.reshape(-1, 2)
    harris_corner_response_score, num_valid_proj_points = compute_harris_corner_response_score(image, projected_keypoints) 
    keypoint_residual_score = compute_keypoint_residual_score(detected_keypoints, projected_keypoints)
    detection_score = harris_corner_response_weight * harris_corner_response_score + keypoint_residual_score_weight * keypoint_residual_score 
    return harris_corner_response_score, num_valid_proj_points, keypoint_residual_score, detection_score

import cv2
import numpy as np

def compute_harris_corner_response_score(image, keypoints, radius=5):
    """
    Computes the normalized Harris corner response score for a set of keypoints in an image.

    Parameters:
    - image: The input image (a NumPy array, BGR).
    - keypoints: A list of 2D keypoint coordinates [(x1, y1), (x2, y2), ...].

    Returns:
    - avg_normalized_score: The average normalized Harris response (0 to 1).
    - num_strong_corners: Number of keypoints with normalized score > 0.01 (good corners).
    """
    image = np.array(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = np.float32(gray_image)

    # Compute the Harris corner response for the entire image once
    harris_response = cv2.cornerHarris(gray_image, blockSize=2, ksize=3, k=0.04)

    # Normalize response to [0, 1]
    harris_response = np.maximum(harris_response, 0)
    max_response = harris_response.max()
    if max_response > 0:
        harris_response /= max_response

    scores = []
    num_strong_corners = 0

    for (x, y) in keypoints:
        x, y = int(x), int(y)
        if 0 <= x < gray_image.shape[1] and 0 <= y < gray_image.shape[0]:
            # find highest response in a radius around the keypoint
            y_min = max(0, y - radius)
            y_max = min(gray_image.shape[0], y + radius)
            x_min = max(0, x - radius)
            x_max = min(gray_image.shape[1], x + radius)
            harris_response_region = harris_response[y_min:y_max, x_min:x_max]
            score = harris_response_region.max()
            scores.append(score)
            if score > 0.01:
                num_strong_corners += 1

    harris_corner_response_score = np.mean(scores) if scores else 0.0
    return harris_corner_response_score, num_strong_corners
     

def compute_keypoint_residual_score(detected_keypoints, projected_keypoints):
    """
    Computes the residual score for keypoints by comparing detected keypoints with projected keypoints.
    The residual is the Euclidean distance between each detected keypoint and its closest projected keypoint

    Parameters:
    - detected_keypoints: List of detected keypoints in image coordinates.
    - projected_keypoints: List of projected keypoints in image coordinates.
    Returns:
    - keypoint_residual_score: The average residual distance between detected and projected keypoints.
    """
    residuals = np.empty(len(detected_keypoints))
    for idx, detected_kp in enumerate(detected_keypoints):
        closest_projected_keypoints = projected_keypoints[np.argmin(np.linalg.norm(projected_keypoints - detected_kp, axis=1))]
        residuals[idx] = np.linalg.norm(detected_kp - closest_projected_keypoints)
    keypoint_residual_score = -np.mean(residuals) # lower residual for higher score 
    return keypoint_residual_score  

def compute_image_similarity_score(image, image_marker, marker_length, tf, camera_matrix, dist_coeffs): 
    marker_corners_2d = np.array([
        [0, 0],
        [0, image_marker.shape[0]],
        [image_marker.shape[1], image_marker.shape[0]],
        [image_marker.shape[1], 0]
    ], dtype=np.float32)
    marker_corners_3d = np.array([
        [0, 0, 0],
        [marker_length, 0, 0],
        [marker_length, marker_length, 0],
        [0, marker_length, 0]
    ], dtype=np.float32)
    pose = np.zeros(6)
    pose[:3] = tf[:3, 3]
    pose[3:] = R.from_matrix(tf[:3, :3]).as_euler('xyz', degrees=True)
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    pose_tensor = torch.tensor(pose, dtype=torch.float32, device=device)
    marker_seg = image.copy()
    marker_seg[:, :, :] = marker_seg.max()
    # FIXME: remove unnecessary GPU operations 
    rendered = marker_reprojection_differentiable(image_marker, marker_corners_2d, marker_corners_3d, pose_tensor, camera_matrix, image_size=(image.shape[0],image.shape[1])) 
    rendered_seg = marker_reprojection_differentiable(marker_seg, marker_corners_2d, marker_corners_3d, pose_tensor, camera_matrix, image_size=(image.shape[0],image.shape[1]))
    rendered = rendered.cpu().numpy().squeeze(0).transpose(1, 2, 0) * 255
    rendered_seg = rendered_seg.cpu().numpy().squeeze(0).transpose(1, 2, 0) * 255 
    rendered_seg = rendered_seg[:,:,0] # Use only one channel for segmentation 
    similarity_score = _image_similarity_score(image, rendered, rendered_seg) 
    return similarity_score

def _image_similarity_score(image, rendered, rendered_seg, threshold=0.25): 
    if rendered_seg.sum() == 0:
        return 0 
    # turn images black and white 
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rendered_gray = cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY)
    seg = rendered_seg 
    # get coordinates of non-zero pixels in the rendered segmentation
    non_zero_coords = np.column_stack(np.where(seg > 0))
    # normalize images at non_zero_coords 
    image_marker_region = image_gray[non_zero_coords[:, 0], non_zero_coords[:, 1]]
    render_marker_region = rendered_gray[non_zero_coords[:, 0], non_zero_coords[:, 1]]
    image_marker_region_normalized = image_marker_region / (image_marker_region.max() - image_marker_region.min())  # Normalize to [0, 1]
    render_marker_region_normalized = render_marker_region / (render_marker_region.max() - render_marker_region.min()) # Normalize to [0, 1]
    similarity_score = 0 
    for idx in range(len(image_marker_region_normalized)):
        if abs(image_marker_region_normalized[idx] - render_marker_region_normalized[idx]) < threshold:
            similarity_score += 1 
    return similarity_score 