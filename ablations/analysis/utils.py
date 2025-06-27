from scipy.spatial.transform import Rotation as R
import numpy as np 
import cv2

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
        corners = np.int0(box)
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
