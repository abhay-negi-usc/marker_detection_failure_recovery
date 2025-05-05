# transformations 

import numpy as np 
from scipy.spatial.transform import Rotation as R, Slerp 
import cv2 
import os 
from pathlib import Path
from typing import Optional, Tuple, List
import logging

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
