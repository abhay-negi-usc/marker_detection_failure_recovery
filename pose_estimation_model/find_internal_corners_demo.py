import os
import cv2
import numpy as np  
import matplotlib.pyplot as plt
from pose_estimation_model.utils import * 

dir_rgb = "/home/rp/abhay_ws/marker_detection_failure_recovery/real_data_processing/raw_data/controlled_tests/dark_test_3_frames/"
dir_seg = "/home/rp/abhay_ws/marker_detection_failure_recovery/real_data_processing/raw_data/controlled_tests/dark_test_3_frames/results/LBCV_segmentation/"

# list all images files in dir_rgb 
files_rgb = [f for f in os.listdir(dir_rgb) if f.endswith('.png') or f.endswith('.jpg')]
# get files_seg by replacing the rgb directory with seg directory 
files_seg = [f.replace(dir_rgb, dir_seg) for f in files_rgb]

idx = 100 
# get images 
img_rgb = cv2.imread(os.path.join(dir_rgb, files_rgb[idx]))
img_seg = cv2.imread(os.path.join(dir_seg, files_seg[idx]))
# resize images to 480 x 640
img_rgb = cv2.resize(img_rgb, (640, 480))
img_seg = cv2.resize(img_seg, (640, 480))
# use seg to crop rgb, set pixels not in seg to black
mask = img_seg > 0  # Assuming the segmentation mask is binary (0 for background, 1 for foreground)
img_rgb_masked = np.zeros_like(img_rgb)
img_rgb_masked[mask] = img_rgb[mask]
img_combined = np.hstack((img_rgb, img_seg, img_rgb_masked))
# show image using matplotlib
plt.imshow(cv2.cvtColor(img_combined, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# use opencv to find internal corners in the masked rgb image
gray = cv2.cvtColor(img_rgb_masked, cv2.COLOR_BGR2GRAY)
# find internal corners 
corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10, blockSize=3)
# draw corners on the masked rgb image
if corners is not None:
    corners = corners.astype(int)  
    for i in corners:
        x, y = i.ravel()
        cv2.circle(img_rgb_masked, (x, y), 3, 255, -1)
# show the image with corners
plt.imshow(cv2.cvtColor(img_rgb_masked, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# get marker image 
img_marker_path = "./synthetic_data_generation/assets/marker_images/tag36h11-0.png"
# find internal corners in the marker image
img_marker = cv2.imread(img_marker_path)
img_marker = cv2.resize(img_marker, (640, 480))
gray_marker = cv2.cvtColor(img_marker, cv2.COLOR_BGR2GRAY)
corners_marker = cv2.goodFeaturesToTrack(gray_marker, maxCorners=100, qualityLevel=0.01, minDistance=10, blockSize=3)
# draw corners on the marker image
if corners_marker is not None:
    corners_marker = corners_marker.astype(int)  
    for i in corners_marker:
        x, y = i.ravel()
        cv2.circle(img_marker, (x, y), 3, 255, -1)  
# show the marker image with corners
plt.imshow(cv2.cvtColor(img_marker, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

import numpy as np
import cv2
from scipy.spatial import cKDTree

def match_points_and_find_homography(pts1, pts2, ransac_thresh=5.0, max_distance=20.0):
    """
    Match 2D points between two sets using nearest neighbors and compute homography using RANSAC.

    Args:
        pts1 (np.ndarray): Points in image 1, shape (N1, 2).
        pts2 (np.ndarray): Points in image 2, shape (N2, 2).
        ransac_thresh (float): RANSAC reprojection threshold.
        max_distance (float): Maximum distance to consider points as potential matches.

    Returns:
        H (np.ndarray): Homography matrix (3x3) or None.
        pts1_matched (np.ndarray): Matched points from image 1.
        pts2_matched (np.ndarray): Corresponding matched points from image 2.
    """

    if pts1.shape[0] < 4 or pts2.shape[0] < 4:
        print("Not enough points to match.")
        return None, None, None

    # Build KD-tree for pts2
    tree = cKDTree(pts2)

    # For each point in pts1, find the closest point in pts2
    distances, indices = tree.query(pts1, distance_upper_bound=max_distance)

    # Filter valid matches
    valid_mask = distances < max_distance
    pts1_matched = pts1[valid_mask]
    pts2_matched = pts2[indices[valid_mask]]

    # Need at least 4 points to compute homography
    if pts1_matched.shape[0] < 4:
        print("Not enough valid matches to compute homography.")
        return None, None, None

    # Compute homography using RANSAC
    H, mask = cv2.findHomography(pts1_matched, pts2_matched, cv2.RANSAC, ransac_thresh)

    if H is None:
        print("Homography estimation failed.")
        return None, None, None

    # Filter final inlier points
    inlier_mask = mask.ravel() > 0
    pts1_inliers = pts1_matched[inlier_mask]
    pts2_inliers = pts2_matched[inlier_mask]

    return H, pts1_inliers, pts2_inliers

# Before passing to match_points_and_find_homography
if corners is not None and corners_marker is not None:
    corners = corners.reshape(-1, 2)
    corners_marker = corners_marker.reshape(-1, 2)

    H, pts1_inliers, pts2_inliers = match_points_and_find_homography(corners, corners_marker)
    if H is not None:
        print("Homography computed. Number of inliers:", len(pts1_inliers))
    else:
        print("Failed to compute homography.")
else:
    print("One of the corner arrays is None.")


if H is not None:
    print("Homography matrix:\n", H)
    print("Matched points in image 1:\n", pts1_inliers)
    print("Matched points in image 2:\n", pts2_inliers)
else:
    print("Homography estimation failed or not enough matches found.")

# show corresponding points between the two images
for pt1, pt2 in zip(pts1_inliers, pts2_inliers):
    pt1 = tuple(pt1.astype(int))
    pt2 = tuple(pt2.astype(int))
    cv2.circle(img_rgb_masked, pt1, 5, (0, 255, 0), -1)  # Draw in image 1
    cv2.circle(img_marker, pt2, 5, (0, 255, 0), -1)  # Draw in image 2
# Show the images with matched points
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img_rgb_masked, cv2.COLOR_BGR2RGB))
plt.title('Image 1 with Matched Points')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_marker, cv2.COLOR_BGR2RGB))
plt.title('Image 2 with Matched Points')
plt.axis('off')
plt.show()
