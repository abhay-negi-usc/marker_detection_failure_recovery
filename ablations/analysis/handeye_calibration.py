import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt 

# === Utility functions ===

def marker_pose_estimation_estimatePoseSingleMarkers(
    image,
    camera_matrix,
    dist_coeffs,
    aruco_dict=cv2.aruco.DICT_APRILTAG_36h11,
    marker_length=0.1,
    show=False,
):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)

    if ids is None:
        return None, None, None, None

    rotation_vectors, translation_vectors, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, marker_length, camera_matrix, dist_coeffs
    )

    if show:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        for rvec, tvec in zip(rotation_vectors, translation_vectors):
            cv2.aruco.drawAxis(image, camera_matrix, dist_coeffs, rvec, tvec, marker_length * 0.5)

    return ids.flatten(), rotation_vectors, translation_vectors, corners

def detect_fiducial_pose_opencv(
    image_input,
    camera_matrix,
    dist_coeffs,
    aruco_dict,
    marker_length,
):
    if isinstance(image_input, (str, Path)):
        image = cv2.imread(str(image_input))
        if image is None:
            return {"pose_c_m": None, "corners": None}
    else:
        image = image_input
        if image is None:
            return {"pose_c_m": None, "corners": None}

    marker_ids, rvecs, tvecs, corners_tuple = marker_pose_estimation_estimatePoseSingleMarkers(
        image, camera_matrix, dist_coeffs, aruco_dict, marker_length, show=False
    )

    pose_c_m, corners = None, None
    if rvecs is not None and tvecs is not None and corners_tuple is not None:
        tf = np.eye(4)
        tf[:3, :3] = cv2.Rodrigues(rvecs[0])[0]
        tf[:3, 3] = tvecs[0].reshape(3)
        pose_c_m = tf
        corners = corners_tuple[0].reshape(-1, 2)

    return {"pose_c_m": pose_c_m, "corners": corners}

def compute_residuals(A_list, B_list, X):
    """
    Compute residuals for the equation AX = XB.
    A_list: list of 4x4 numpy arrays representing A_i
    B_list: list of 4x4 numpy arrays representing B_i
    X: 4x4 numpy array representing the transformation from camera to gripper
    Returns:
        residuals: list of residuals for each pair (A_i, B_i)
    """
    residuals = []
    for i in range(len(A_list)):
        A = A_list[i]
        B = B_list[i]
        tf_left = A @ X 
        tf_right = X @ B 
        tf_err = np.linalg.inv(tf_left) @ tf_right 
        position_err = tf_err[:3, 3]
        rotation_err = R.from_matrix(tf_err[:3, :3]).as_euler('xyz', degrees=True)
        pose_err = np.concatenate((position_err, rotation_err))
        residuals.append(pose_err)
    print("Residuals (position, rotation):" , np.mean(np.abs(residuals), axis=0))
    return residuals

def main(): 
    # === Load data ===

    # flange_poses_path = "/home/rp/dhanush_ws/sunrise-wrapper/data/marker_data_collection/handeye_calibration_v6/tf_b_f.csv"
    flange_poses_path = "/home/rp/dhanush_ws/sunrise-wrapper/data/marker_data_collection/july12/skew/tf_b_f.csv"
    flange_poses = np.loadtxt(flange_poses_path, delimiter=',', skiprows=1).reshape(-1, 4, 4)

    # checkerboard calibration 
    # camera_matrix = np.array([
    #     [913.63976705, 0, 638.73567244],
    #     [0, 914.60688165, 358.8395136],
    #     [0, 0, 1]
    # ], dtype=float)
    # distortion_coefficients = np.array([0.1693, -0.4755, 0.0018, 0.0023, 0.4114], dtype=float)

    # # realsense calibration 
    camera_matrix = np.array([
        [906.995, 0, 638.235],
        [0, 906.995, 360.533],
        [0, 0, 1]
    ], dtype=float)
    distortion_coefficients = np.array([0,0,0,0,0], dtype=float)

    # dir_images = Path("/home/rp/dhanush_ws/sunrise-wrapper/data/marker_data_collection/handeye_calibration_v6/images")
    dir_images = Path("/home/rp/dhanush_ws/sunrise-wrapper/data/marker_data_collection/july12/skew/images")
    image_files = sorted(dir_images.glob("*.png"), key=lambda x: x.stat().st_mtime)

    marker_poses = [] 
    for image_file in image_files: 
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"Failed to read image: {image_file}")
            continue

        # Detect marker pose
        result = detect_fiducial_pose_opencv(
            image,
            camera_matrix,
            distortion_coefficients,
            cv2.aruco.DICT_APRILTAG_36h11,
            marker_length=0.1
        )

        if result["pose_c_m"] is not None:
            marker_poses.append(result["pose_c_m"])
        else:
            marker_poses.append(None)

    valid_indices = [i for i in range(len(marker_poses)) if marker_poses[i] is not None]
    tf_c_m = [marker_poses[i] for i in valid_indices]
    tf_b_f = [flange_poses[i] for i in valid_indices]
    print(f"Number of valid pairs of total: {len(tf_c_m)} / {len(flange_poses)}")

    tf_c_m = np.array(tf_c_m)
    tf_b_f = np.array(tf_b_f)
    R_gripper2base = tf_b_f[:, :3, :3] 
    t_gripper2base = tf_b_f[:, :3, 3].reshape(-1, 3, 1)
    R_target2cam = tf_c_m[:, :3, :3]
    t_target2cam = tf_c_m[:, :3, 3].reshape(-1, 3, 1)

    # === Solve hand-eye calibration ===

    calib_methods = [cv2.CALIB_HAND_EYE_PARK, cv2.CALIB_HAND_EYE_TSAI, cv2.CALIB_HAND_EYE_HORAUD]

    for method in calib_methods:

        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base,
            R_target2cam, t_target2cam,
            method=method
        )

        tf_f_c = np.eye(4)
        tf_f_c[:3, :3] = R_cam2gripper
        tf_f_c[:3, 3] = t_cam2gripper.flatten()

        print("Transformation of camera with respect to flange (tf_f_c):")
        print(tf_f_c)

        rot = R.from_matrix(tf_f_c[:3, :3]).as_euler('xyz', degrees=True)
        print(f"Rotation (roll, pitch, yaw) [deg]: {rot}")

        A_list = []
        B_list = []
        tf_m_c = [] 
        for i in range(tf_c_m.shape[0]): tf_m_c.append(np.linalg.inv(tf_c_m[i])) 
        tf_m_c = np.array(tf_m_c)
        for i in range(len(tf_b_f)-1): 
            A = np.linalg.inv(tf_b_f[i]) @ tf_b_f[i+1] 
            A_list.append(A) 
        for i in range(len(tf_m_c)):
            if i == 0: continue 
            B = np.linalg.inv(tf_m_c[i-1]) @ tf_m_c[i] 
            B_list.append(B) 

        res = compute_residuals(A_list, B_list, tf_f_c)  
        res = np.array(res)

        # plot figure of 2x3 histogram of residuals with mean and std in title 
        labels = ['x', 'y', 'z', 'pitch', 'yaw', 'roll']
        plt.figure(figsize=(12, 6))
        for i in range(6):
            plt.subplot(2, 3, i+1)
            plt.hist(res[:, i], bins=20, alpha=0.7, color='blue', edgecolor='black')
            plt.title(f"{labels[i]} residuals\nmean: {np.mean(res[:, i]):.4f}, MAE: {np.mean(np.abs(res[:, i])):.4f}, std: {np.std(res[:, i]):.4f}")
            plt.xlabel(labels[i])
            plt.ylabel('Frequency')
            plt.grid(True)
        plt.tight_layout()
        plt.show()

        # compute marker wrt base 

        tf_b_m = [] 
        for i in range(tf_b_f.shape[0]): 
            tf_b_m.append(tf_b_f[i] @ tf_f_c @ tf_c_m[i])
        tf_b_m = np.array(tf_b_m)

        tf_b_m_position = tf_b_m[:, :3, 3]
        tf_b_m_rotation = np.array([R.from_matrix(tf_b_m[i][:3, :3]).as_euler('xyz', degrees=True) for i in range(tf_b_m.shape[0])])

        tf_b_m_position_mean = np.mean(tf_b_m_position, axis=0)
        tf_b_m_rotation_mean = np.mean(tf_b_m_rotation, axis=0)
        tf_b_m_position_std = np.std(tf_b_m_position, axis=0)
        tf_b_m_rotation_std = np.std(tf_b_m_rotation, axis=0)
        print(f"Marker position wrt base: {tf_b_m_position_mean} m, std: {tf_b_m_position_std} m")
        print(f"Marker rotation wrt base (roll, pitch, yaw) [deg]: {tf_b_m_rotation_mean}, std: {tf_b_m_rotation_std} deg")

        tf_b_m_mean = np.mean(tf_b_m, axis=0)
        print("Mean transformation of marker with respect to base (tf_b_m_mean):")
        print(tf_b_m_mean)
        
if __name__ == "__main__":
    main()