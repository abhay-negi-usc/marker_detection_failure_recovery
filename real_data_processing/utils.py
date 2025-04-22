# transformations 

import numpy as np 
from scipy.spatial.transform import Rotation as R 
import cv2 
import os 

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
def classical_marker_pose_estimation(image, camera_matrix, dist_coeffs, aruco_dict=cv2.aruco.DICT_APRILTAG_36h11, marker_length=0.1, show=False):
    """
    Detect ArUco markers in the input image and compute their poses.
    
    Args:
        image: The input image (numpy array) containing the ArUco marker(s).
        camera_matrix: Camera intrinsic matrix (numpy array).
        dist_coeffs: Camera distortion coefficients (numpy array).
        aruco_dict: The ArUco dictionary to use (default is DICT_6X6_250).
        marker_length: The length of the ArUco marker's side (in meters).

    Returns:
        marker_ids: List of detected ArUco marker IDs.
        rotation_vectors: List of rotation vectors of the markers.
        translation_vectors: List of translation vectors of the markers.
    """
    # Convert the image to grayscale (necessary for ArUco marker detection)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load the ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
    
    # Detect ArUco markers in the image
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict)
    
    # If no markers detected, return None's for all outputs 
    if ids is None:
        return None, None, None, None 
    
    # Estimate pose of each marker
    rotation_vectors = []
    translation_vectors = []
    marker_frame_corners = np.array([[[-marker_length/2, -marker_length/2, 0], [marker_length/2, -marker_length/2, 0], [marker_length/2, marker_length/2, 0], [-marker_length/2, marker_length/2, 0]]])  # Define the marker frame corners

    # Iterate over all detected markers
    for i in range(len(ids)):
        # Compute the pose (rotation and translation vectors)
        ret = cv2.solvePnP(marker_frame_corners, corners[i], camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)  # Use the marker frame corners as object points 
        
        # Unpack the results (rotation and translation vectors)
        # rotation_vector, translation_vector = ret[0][0], ret[1][0]
        rotation_vector, translation_vector = ret[1], ret[2] 
        
        rotation_vectors.append(rotation_vector)
        translation_vectors.append(translation_vector)
        
        if show: 
            # Optionally: Draw the marker and its pose on the image
            cv2.aruco.drawDetectedMarkers(image, corners)
            
            # Manually draw the axes using projectPoints
            axis_points = np.float32([[marker_length, 0, 0], [0, marker_length, 0], [0, 0, marker_length]]).reshape(-1, 3)
            img_points, _ = cv2.projectPoints(axis_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            
            # Draw the 3D axis on the image
            corner = corners[i].reshape(-1, 2)
            for j in range(3):
                # Ensure the points are formatted as tuples of integers
                pt1 = tuple(corner[0].astype(int))
                pt2 = tuple(img_points[j].ravel().astype(int))
                cv2.line(image, pt1, pt2, (0, 255, 0), 5)

    return ids.flatten(), rotation_vectors, translation_vectors, corners 