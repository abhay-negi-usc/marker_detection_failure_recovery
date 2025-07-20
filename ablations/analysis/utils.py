from scipy.spatial.transform import Rotation as R
import numpy as np 
import cv2
import seaborn as sns
import pandas as pd

def compute_tf_error(tf_ref, tf_est):
    tf_err = np.linalg.inv(tf_ref) @ tf_est 
    return tf_err 

def tf_to_pose(tf): 
    """
    Convert a transformation matrix to a pose (position and orientation).
    """
    position = tf[:3, 3]
    euler = R.from_matrix(tf[:3, :3]).as_euler('xyz', degrees=True) 
    pose = np.concatenate((position, euler))
    return pose 

def find_segmentation_four_corners(segmentation):
    contours, _ = cv2.findContours(
        segmentation.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    # Choose the largest contour by area
    contour = max(contours, key=cv2.contourArea)

    # Approximate polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 4:
        corners = approx.reshape(4, 2)
    elif len(approx) > 4:
        # Use convex hull and select 4 corners via bounding box
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        corners = np.int8(box)
    else:
        # Fall back to bounding box if not enough corners
        x, y, w, h = cv2.boundingRect(contour)
        corners = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.int32)

    # Ensure corners are in a consistent order (counterclockwise)
    corners = corners[np.argsort(np.arctan2(corners[:, 1] - np.mean(corners[:, 1]), 
                                             corners[:, 0] - np.mean(corners[:, 0])))]
    corners = corners.reshape(4, 2).astype(np.float32)
    return corners


def segmentation_biggest_blob_filter(segmentation_mask, min_area=1000):
    """
    Filters the segmentation mask to keep only the largest connected component (blob).
    
    Args:
        segmentation_mask (numpy.ndarray): Binary segmentation mask.
        min_area (int): Minimum area of the blob to keep.
    
    Returns:
        numpy.ndarray: Filtered segmentation mask with only the largest blob.
    """
    if segmentation_mask is None or not np.any(segmentation_mask):
        return None
    
    # Find contours
    contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(largest_contour) < min_area:
        return None
    
    # Create a new mask for the largest blob
    filtered_mask = np.zeros_like(segmentation_mask)
    cv2.drawContours(filtered_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    return filtered_mask