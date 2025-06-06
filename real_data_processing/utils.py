# transformations 

import numpy as np 
from scipy.spatial.transform import Rotation as R, Slerp 
import cv2 
import os 
from pathlib import Path
from typing import Optional, Tuple, List
import logging

import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import logging

def draw_overlay_square(image: np.ndarray, tf: np.ndarray, square_length: float, K: np.ndarray) -> np.ndarray:
    """
    Projects and overlays a square of known size on the image using a 4x4 pose matrix and camera intrinsics.
    """
    # Define 3D square corners in marker frame
    square_corners_3d = np.array([
        [square_length/2, square_length/2, 0],
        [-square_length/2, square_length/2, 0],
        [-square_length/2, -square_length/2, 0],
        [square_length/2, -square_length/2, 0]
    ])  # shape (4, 3)

    # Extract rotation and translation
    R_wc = tf[:3, :3]
    t_wc = tf[:3, 3]

    # Transform corners to camera frame
    square_corners_cam = (R_wc @ square_corners_3d.T + t_wc.reshape(3, 1)).T  # shape (4, 3)

    # Project to 2D using intrinsic matrix
    square_corners_2d = (K @ square_corners_cam.T).T  # shape (4, 3)
    square_corners_2d = square_corners_2d[:, :2] / square_corners_2d[:, 2:3]  # normalize

    # Draw the square on the image
    overlay_img = image.copy()
    pts = square_corners_2d.astype(int).reshape(-1, 1, 2)
    cv2.polylines(overlay_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    return overlay_img

def get_marker_segmentation(image, tf, square_length, K):
    """
    Segments the marker from the image using the pose and camera intrinsics.
    """
    # Define 3D square corners in marker frame
    square_corners_3d = np.array([
        [square_length/2, square_length/2, 0],
        [-square_length/2, square_length/2, 0],
        [-square_length/2, -square_length/2, 0],
        [square_length/2, -square_length/2, 0]
    ])  # shape (4, 3)

    # Extract rotation and translation
    R_wc = tf[:3, :3]
    t_wc = tf[:3, 3]

    # Transform corners to camera frame
    square_corners_cam = (R_wc @ square_corners_3d.T + t_wc.reshape(3, 1)).T  # shape (4, 3)

    # Project to 2D using intrinsic matrix
    square_corners_2d = (K @ square_corners_cam.T).T  # shape (4, 3)
    square_corners_2d = square_corners_2d[:, :2] / square_corners_2d[:, 2:3]  # normalize

    # Create a mask for the marker
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    pts = square_corners_2d.astype(int).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], color=(255))

    return mask

def seg_IOU(seg1, seg2):
    """
    Computes the Intersection over Union (IoU) between two segmentation masks.
    
    Args:
        seg1: First segmentation mask (binary).
        seg2: Second segmentation mask (binary).

    Returns:
        IoU value.
    """
    intersection = np.logical_and(seg1, seg2)
    union = np.logical_or(seg1, seg2)
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
    return iou

class DataPoint:
    def __init__(self, image_path: Path):
        self.image_path = image_path
        self.image: Optional[np.ndarray] = None
        self.time: Optional[float] = None

    def get_image(self) -> Optional[np.ndarray]:
        self.image = cv2.imread(str(self.image_path))
        if self.image is None:
            logging.warning(f"Failed to read image: {self.image_path}")
        return self.image

    def forget_image(self):
        self.image = None

    def set_time(self, time: float):
        self.time = time

    def _set_pose(self, pose: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
        if pose is None or not isinstance(pose, np.ndarray):
            return None, None, False
        if pose.shape == (4, 4):
            tf = pose
            xyzabc = tf_to_xyzabc(tf)
            return tf, xyzabc, True
        elif pose.shape == (6,):
            xyzabc = pose
            tf = xyzabc_to_tf(xyzabc)
            return tf, xyzabc, True
        else:
            raise ValueError(f"Invalid pose shape {pose.shape}. Expected (4, 4) or (6,)")

    def set_pose(self, method: str, pose: np.ndarray):
        tf, xyzabc, detected = self._set_pose(pose)
        setattr(self, f"{method}_tf", tf)
        setattr(self, f"{method}_pose", xyzabc)
        setattr(self, f"{method}_detected", detected)

    def save_overlay(self, method: str, square_length: float, K: np.ndarray, output_dir: Path):
        """
        Saves an image with an overlayed quadrilateral corresponding to the pose estimated by a given method.

        Args:
            method: Name of the method (used to retrieve pose from attributes).
            square_length: Side length of the square marker in meters.
            K: Camera intrinsics matrix (3x3).
            output_dir: Directory to save the output image.
        """
        if not hasattr(self, f"{method}_tf"):
            logging.warning(f"No pose found for method '{method}' in {self.image_path}")
            return

        tf = getattr(self, f"{method}_tf")
        if tf is None:
            logging.warning(f"Pose is None for method '{method}' in {self.image_path}")
            return

        image = self.get_image()
        if image is None:
            return

        overlay_img = draw_overlay_square(image, tf, square_length, K)

        image_filename = self.image_path.name 
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, image_filename)
        cv2.imwrite(str(output_path), overlay_img)
        # logging.info(f"Saved overlay image to {output_path}")

    def set_marker_length(self, marker_length: float):
        self.marker_length = marker_length
    
    def set_camera_matrix(self, K: np.ndarray):
        self.K = K

    def get_marker_brightness(self): 
        # use OPTK_tf to segment the marker pixels 
        image = cv2.imread(str(self.image_path))
        # if don't have attribute marker_pixels 
        if not hasattr(self, "marker_pixels"):
            seg = get_marker_segmentation(image, self.OPTK_tf, self.marker_length, self.K)
            self.marker_pixels = image[seg == 255]
        self.marker_brightness = np.mean(self.marker_pixels)
        return self.marker_brightness  
    
    def get_marker_area(self): 
        # use OPTK_tf to segment the marker pixels 
        image = cv2.imread(str(self.image_path))
        # if don't have attribute marker_pixels 
        if not hasattr(self, "marker_pixels"):
            seg = get_marker_segmentation(image, self.OPTK_tf, self.marker_length, self.K)
            self.marker_pixels = image[seg == 255]
        self.marker_area = np.sum(self.marker_pixels > 0)
        return self.marker_area
    
    def compute_errors(self): 
        for method in ["CCV", "LBCV"]: 
            if not getattr(self, f"{method}_detected", False):
                setattr(self, f"{method}_error", None) 
                continue 
            tf = getattr(self, f"{method}_tf") 
            if tf is None:
                setattr(self, f"{method}_error", None)
                continue
            tf_true = self.OPTK_tf 
            error_tf = np.linalg.inv(tf_true) @ tf
            setattr(self, f"{method}_error_tf", error_tf)
            # compute the error in translation and rotation 
            error_translation = np.linalg.norm(error_tf[:3, 3])
            error_rotation = np.degrees(np.arccos((np.trace(error_tf[:3, :3]) - 1) / 2)) 
            error_in_plane_translation = np.linalg.norm(error_tf[:2, 3]) 
            error_out_plane_translation = error_tf[2, 3]
            error_in_plane_rotation = np.degrees(np.arctan2(error_tf[1, 0], error_tf[0, 0]))
            error_out_plane_rotation = np.degrees(np.arctan2(error_tf[2, 1], error_tf[2, 2]))
            error = {
                "translation": error_translation,
                "rotation": error_rotation,
                "in_plane_translation": error_in_plane_translation,
                "out_plane_translation": error_out_plane_translation,
                "in_plane_rotation": error_in_plane_rotation,
                "out_plane_rotation": error_out_plane_rotation
            }
            setattr(self, f"{method}_error", error) 
        return self.CCV_error, self.LBCV_error 

    def set_ccv_corners(self, corners: np.ndarray):
        self.CCV_corners = corners

    def get_ccv_corners(self) -> Optional[np.ndarray]:
        return getattr(self, "CCV_corners", None)

    def set_lbcv_keypoints(self, keypoints: np.ndarray):
        self.LBCV_keypoints = keypoints

    def get_lbcv_keypoints(self) -> Optional[np.ndarray]:
        return getattr(self, "LBCV_keypoints", None)


    def __repr__(self):
        return f"DataPoint(path={self.image_path.name}, time={self.time})"

def xyzquat_to_tf(xyzquat: np.ndarray, input_qw_first: bool) -> np.ndarray:
    """
    convert vector of [x,y,z,qw,qx,qy,qz] to homogeneous transformation matrix 
    """    
    tf = np.eye(4) 
    tf[:3, 3] = xyzquat[:3]  # Set translation part (x, y, z) 
    if input_qw_first:  
        quat = xyzquat[[4,5,6,3]] # Reorder to [qx, qy, qz, qw] for scipy   
    else: 
        quat = xyzquat[3:7] # keep order of [qx, qy, qz, qw] 
    tf[:3,:3] = R.from_quat(quat).as_matrix() # scipy uses qw last convention  
    return tf 

def tf_to_xyzabc(tf: np.ndarray) -> np.ndarray:
    """
    Convert a homogeneous transformation matrix to [x, y, z, alpha, beta, gamma].
    
    Args:
        tf: A 4x4 homogeneous transformation matrix.

    Returns:
        A numpy array containing [x, y, z, alpha, beta, gamma].
    """
    # Extract translation
    x = tf[0, 3]
    y = tf[1, 3]
    z = tf[2, 3]

    # Extract rotation matrix
    R_mat = tf[:3, :3]
    
    # Convert rotation matrix to Euler angles (roll, pitch, yaw)
    r = R.from_matrix(R_mat)
    a, b, c = r.as_euler('xyz', degrees=True)  # in degrees

    return np.array([x, y, z, a, b, c])

def xyzabc_to_tf(xyzabc: np.ndarray) -> np.ndarray:
    """
    Convert [x, y, z, alpha, beta, gamma] to a homogeneous transformation matrix.
    
    Args:
        xyzabc: A numpy array containing [x, y, z, alpha, beta, gamma].

    Returns:
        A 4x4 homogeneous transformation matrix.
    """
    if xyzabc.shape == (1,6): 
        xyzabc = xyzabc.reshape(6) 

    x, y, z = xyzabc[:3]
    a, b, c = np.radians(xyzabc[3:])  # Convert degrees to radians

    # Create rotation matrix from Euler angles
    R_mat = R.from_euler('xyz', [a, b, c]).as_matrix()

    # Create the transformation matrix
    tf = np.eye(4)
    tf[:3, :3] = R_mat
    tf[:3, 3] = [x, y, z]

    return tf

def split_video_to_frames(video_path, output_folder, get_timestamps=False):
    video_filename = os.path.basename(video_path).replace(".mp4","")

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None 

    # Get video details
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames

    print(f"Total frames: {total_frames}")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    timestamps = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Generate filename for each frame
        frame_filename = os.path.join(output_folder, f"{video_filename}_frame_{frame_count:05d}.png")
        
        # Save the frame as PNG
        cv2.imwrite(frame_filename, frame)
        
        timestamp = frame_count / fps
        timestamps.append(timestamp)
        frame_count += 1
        # print(f"Processing frame {frame_count}/{total_frames}")
    
    # Release the video capture object
    cap.release()

    if get_timestamps:
        # Return the list of timestamps if requested
        return np.array(timestamps) 

    # print("Video processing completed.")

def marker_pose_estimation_estimatePoseSingleMarkers(
    image,
    camera_matrix,
    dist_coeffs,
    aruco_dict=cv2.aruco.DICT_APRILTAG_36h11,
    marker_length=0.1,
    show=False
):
    """
    Detect ArUco markers in an image and estimate their poses using estimatePoseSingleMarkers.
    
    Args:
        image: Input image (numpy array).
        camera_matrix: Camera intrinsic matrix (numpy array).
        dist_coeffs: Camera distortion coefficients (numpy array).
        aruco_dict: ArUco dictionary to use (default: DICT_APRILTAG_36h11).
        marker_length: Side length of the marker in meters.
        show: If True, draw markers and pose axes on the image.

    Returns:
        marker_ids: Detected marker IDs (flattened array).
        rotation_vectors: List of rotation vectors.
        translation_vectors: List of translation vectors.
        corners: List of detected marker corners.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the dictionary
    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
    
    # Detect markers
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, dictionary)

    if ids is None:
        return None, None, None, None

    # Pose estimation
    rotation_vectors, translation_vectors, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners,
        marker_length,
        camera_matrix,
        dist_coeffs
    )

    if show:
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(image, corners, ids)

        # Draw axes for each marker
        for rvec, tvec in zip(rotation_vectors, translation_vectors):
            cv2.aruco.drawAxis(image, camera_matrix, dist_coeffs, rvec, tvec, marker_length * 0.5)

    return ids.flatten(), rotation_vectors, translation_vectors, corners

def find_corners(binary_img, debug=False, epsilon=0.01):
    """
    Finds corners in a binary image 

    Args:
        binary_img (np.ndarray): Binary image with white object on black background.
        debug (bool): If True, returns the image with drawn lines.

    Returns:
        corners (np.ndarray): Array of corner points (shape: (4, 2))
        debug_img (np.ndarray): (Optional) Image with edges drawn.
    """
    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found")

    # Assume the largest contour is the object
    contour = max(contours, key=cv2.contourArea)

    # Approximate it to reduce number of points
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon * peri, True)
    approx = approx[:, 0, :]  # reshape to (N, 2)

    # Compute all pairwise edges in order
    edges = []
    for i in range(len(approx)):
        pt1 = approx[i]
        pt2 = approx[(i + 1) % len(approx)]
        length = np.linalg.norm(pt2 - pt1)
        edges.append((length, tuple(pt1), tuple(pt2)))

    # Extract unique corner points from edges
    corner_pts = set()
    for _, p1, p2 in edges:
        corner_pts.add(p1)
        corner_pts.add(p2)

    corners = np.array(list(corner_pts), dtype=np.float32)

    if debug:
        debug_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        for _, p1, p2 in edges:
            cv2.line(debug_img, p1, p2, (0, 0, 255), 2)
        for corner in corners:
            cv2.circle(debug_img, tuple(corner.astype(int)), 5, (0, 255, 0), -1)
        return corners, debug_img

    return corners

def refine_corners(corners, seg):

    # list corners that are near the edges of the image and group by which edge they are near 
    h, w = seg.shape[:2] 
    corners_near_border = []
    tol = 0.05  
    for i, corner in enumerate(corners): 
        x, y = corner 
        if x < tol * w: 
            corners_near_border.append((corner, "left")) 
        elif x > 1-tol * w: 
            corners_near_border.append((corner, "right")) 
        if y < tol * h: 
            corners_near_border.append((corner, "top")) 
        elif y > 1-tol * h: 
            corners_near_border.append((corner, "bottom"))

    borders = set([b for _, b in corners_near_border]) 
    for border in borders: 
        border_corners = [corner for corner, b in corners_near_border if b == border] 
        if len(border_corners) > 1: 
            # find the intersection of the two lines formed by the edges 
            p1 = border_corners[0]
            p2 = border_corners[1]
            p3 = border_corners[2]
            p4 = border_corners[3]
            A = np.array([p2 - p1, p4 - p3]).T
            b = np.array([p3 - p1])
            if np.linalg.matrix_rank(A) == 2:
                intersection = np.linalg.solve(A, b)
                corners = np.vstack((corners, intersection))
                corners = np.delete(corners, [0, 1], axis=0)
            else:
                # If lines are parallel, keep the original corners
                pass
        elif len(border_corners) == 1:
            # If only one corner is near the border, keep it
            pass
        else:
            # If no corners are near the border, do nothing
            pass

        img = seg 

    return corners, img 

def opencv_marker_pose(
        image, 
        K,
        D,
        marker_size,
        aruco_dict=cv2.aruco.DICT_APRILTAG_36h11,
        show=False, 
        out_path=None 
): 
    """
    Detects ArUco markers in an image and estimates their poses using OpenCV's ArUco module.
    Args:
        image: Input image (numpy array).
        K: Camera intrinsic matrix (3x3).
        D: Camera distortion coefficients (1D array).
        marker_size: Size of the marker in meters.
        aruco_dict: ArUco dictionary to use (default: DICT_APRILTAG_36h11).
        show: If True, draw markers and pose axes on the image.
    Returns:
        tf: 4x4 transformation matrix of the detected marker.
        xyzabc: Pose in [x, y, z, alpha, beta, gamma] format.
        corners: List of detected marker corners. 
    """

    # Convert to grayscale 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Load the dictionary
    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
    # Detect markers
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, dictionary)
    if ids is None:
        return None, None, None
    # Pose estimation
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners,
        marker_size,
        K,
        D
    )
    # Draw detected markers
    if show:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        for rvec, tvec in zip(rvecs, tvecs):
            cv2.drawFrameAxes(image, K, D, rvec, tvec, marker_size * 0.5)
        # Display the image
        cv2.imshow("Detected Markers", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # Convert rotation vector to rotation matrix
    R = np.zeros((3, 3))
    cv2.Rodrigues(rvecs[0], R)
    # Create the transformation matrix
    tf = np.eye(4)
    tf[:3, :3] = R
    tf[:3, 3] = tvecs[0].reshape(3)
    # Convert to [x, y, z, alpha, beta, gamma]
    xyzabc = tf_to_xyzabc(tf)
    # Convert corners to numpy array
    corners = np.array(corners[0], dtype=np.float32)

    if out_path is not None:
        overlay_img = draw_overlay_square(image, tf, marker_size, K)
        cv2.imwrite(out_path, overlay_img)
    
    return tf, xyzabc, corners 

def sanitize_for_json(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    else:
        return obj


def main(): 
    seg_path = "/home/rp/abhay_ws/marker_detection_failure_recovery/real_data_processing/raw_data/realsense/realsense_5_frames_LBCV/segmentation_masks/LBCV_seg_00055.png" 
    seg = np.array(cv2.cvtColor(cv2.imread(seg_path))) 
    corners, debug_img = find_corners(seg, debug=True, epsilon=0.01)
    print(corners)
    cv2.imshow("Debug Image", debug_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    corners, img = refine_corners(corners, seg)

if __name__ == "__main__": 
    main() 