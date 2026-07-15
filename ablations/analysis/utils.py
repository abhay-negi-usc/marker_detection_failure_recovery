from scipy.spatial.transform import Rotation as R
import numpy as np 
import cv2
import seaborn as sns
import pandas as pd
from math import gcd

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


def split_image_by_aspect_ratio(image, M, N, stride=None):
    """
    Splits an image into overlapping tiles of size P×Q such that:
    - P/Q = M/N (aspect ratio match)
    - P ≥ M, Q ≥ N
    - Entire image is covered by tiles (possibly overlapping)

    Args:
        image (np.ndarray): Input image of shape (H, W, C) or (H, W)
        M (int): Height component of desired aspect ratio
        N (int): Width component of desired aspect ratio
        stride (tuple, optional): (vertical_stride, horizontal_stride). If None, defaults to (P//2, Q//2)

    Returns:
        tiles (List[np.ndarray]): List of image tiles
        coords (List[Tuple[int, int]]): List of (y, x) top-left coordinates in original image
    """
    H, W = image.shape[:2]

    # Simplify aspect ratio
    d = gcd(M, N)
    m, n = M // d, N // d

    # Choose largest P, Q that fit the aspect ratio and image
    scale = min(H // m, W // n)
    P, Q = m * scale, n * scale

    # Define stride
    if stride is None:
        stride_y, stride_x = P // 2, Q // 2
    else:
        stride_y, stride_x = stride

    tiles = []
    coords = []

    # Compute y and x positions to ensure full coverage
    y_positions = list(range(0, H - P + 1, stride_y))
    x_positions = list(range(0, W - Q + 1, stride_x))

    # Add final row/column if needed to reach image edge
    if y_positions[-1] + P < H:
        y_positions.append(H - P)
    if x_positions[-1] + Q < W:
        x_positions.append(W - Q)

    for y in y_positions:
        for x in x_positions:
            tile = image[y:y + P, x:x + Q]
            tiles.append(tile)
            coords.append((y, x))

    return tiles, coords

def combine_tiles_and_coords(tiles, coords, image_shape):
    """
    Combines tiles back into a full image using given coordinates.
    Handles overlapping tiles by averaging pixel values.

    Args:
        tiles (List[np.ndarray]): List of image tiles (H, W) or (H, W, C) or (1, H, W)
        coords (List[Tuple[int, int]]): Top-left (y, x) coordinates for each tile
        image_shape (Tuple[int, int] or Tuple[int, int, int]): Shape of final image (H, W[, C])

    Returns:
        np.ndarray: Combined image
    """
    # Determine if grayscale or color
    is_color = tiles[0].ndim == 3 and tiles[0].shape[-1] in [1, 3]

    # Prepare accumulation and count arrays
    combined_image = np.zeros(image_shape, dtype=np.float32)
    count_image = np.zeros(image_shape, dtype=np.float32)

    for tile, (y, x) in zip(tiles, coords):
        # Convert (1, H, W) → (H, W)
        if tile.ndim == 3 and tile.shape[0] == 1:
            tile = tile[0]  # from (1, H, W) to (H, W)

        h, w = tile.shape[:2]

        # Add tile to combined image and update count for averaging
        combined_image[y:y + h, x:x + w] += tile
        count_image[y:y + h, x:x + w] += 1.0

    # Avoid divide-by-zero
    count_image[count_image == 0] = 1.0
    combined_image = combined_image / count_image

    return combined_image.astype(tiles[0].dtype)