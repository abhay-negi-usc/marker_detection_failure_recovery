import cv2
import numpy as np
import glob
import time
import os


def load_images(image_dir, max_images):
    images = glob.glob(os.path.join(image_dir, '*.png'))
    original_num_images = len(images)

    if original_num_images > max_images:
        step = original_num_images // max_images
        images = images[::step]

    print(f"Number of images used for calibration: {len(images)} out of {original_num_images}")
    return images


def detect_charuco_corners(images, board, detector):
    all_corners = []
    all_ids = []

    for idx, img_path in enumerate(images):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)

        if len(corners) > 0:
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)

            if retval and charuco_corners is not None and charuco_ids is not None and len(charuco_ids) >= 4:
                corners_np = np.array(charuco_corners, dtype=np.float32).reshape(-1, 2)
                ids_np = np.array(charuco_ids, dtype=np.float32)

                if np.any(np.isnan(corners_np)) or np.any(np.isnan(ids_np)):
                    print(f"Skipping {img_path}: contains NaNs.")
                    continue

                if cv2.contourArea(corners_np) < 1e-3:
                    print(f"Skipping {img_path}: low area (possible degeneracy).")
                    continue

                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)

    print(f"Number of valid frames: {len(all_corners)}")
    return all_corners, all_ids, gray.shape[::-1]


def detect_checkerboard_corners(images, pattern_size):
    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    for idx, img_path in enumerate(images):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    print(f"Number of valid frames: {len(objpoints)}")
    return objpoints, imgpoints, gray.shape[::-1]


def calibrate_camera_charuco(all_corners, all_ids, board, image_size, camera_matrix_init=None, dist_coeffs_init=None, use_intrinsic_guess=False):
    if len(all_corners) < 10:
        print("Not enough valid detections for calibration (need at least 10).")
        return None

    flags = cv2.CALIB_USE_INTRINSIC_GUESS if use_intrinsic_guess else 0

    print("Starting ChArUco calibration...")
    time_start = time.time()
    ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=camera_matrix_init,
        distCoeffs=dist_coeffs_init,
        flags=flags
    )
    time_end = time.time()

    print(f"Calibration completed in {time_end - time_start:.2f} seconds.")
    print("Calibration RMS error:", ret)
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)

    return mtx, dist


def calibrate_camera_checkerboard(objpoints, imgpoints, image_size):
    print("Starting checkerboard calibration...")
    time_start = time.time()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None
    )
    time_end = time.time()

    print(f"Calibration completed in {time_end - time_start:.2f} seconds.")
    print("Calibration RMS error:", ret)
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)

    return mtx, dist


def save_calibration(file_path, camera_matrix, distortion_coeffs):
    np.savez(file_path, camera_matrix=camera_matrix, distortion_coeffs=distortion_coeffs)
    print(f"Calibration saved to {file_path}")


def main():
    # Parameters
    image_dir = 'C:/Users/NegiA/Desktop/abhay_ws/marker_detection_failure_recovery/real_data_processing/raw_data/realsense415_charuco_calibration_frames_v2'
    output_file = './real_data_processing/raw_data/camera_calibration.npz'
    chessboard_size = (5, 7) 
    pattern_size = (4, 6) # Number of inner corners (not squares)
    square_length = 0.07750 
    marker_length = 0.03875
    max_images = 250
    use_charuco = False  # Set to False to use checkerboard calibration

    images = load_images(image_dir, max_images)

    if use_charuco:
        aruco_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_5X5_250, 5)  # Specify marker size
        board = cv2.aruco.CharucoBoard(chessboard_size, square_length, marker_length, aruco_dict)
        detector_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

        all_corners, all_ids, image_size = detect_charuco_corners(images, board, detector)

        if len(all_corners) < 10:
            print("Calibration aborted: not enough valid frames.")
            return

        camera_matrix_init = np.array([[1400.0, 0.0, 950.0], [0.0, 1400.0, 530.0], [0, 0, 1]], dtype=np.float32)
        dist_coeffs_init = np.zeros((5, 1), dtype=np.float32)

        mtx, dist = calibrate_camera_charuco(
            all_corners,
            all_ids,
            board,
            image_size,
            camera_matrix_init,
            dist_coeffs_init,
            use_intrinsic_guess=True
        )

    else:
        objpoints, imgpoints, image_size = detect_checkerboard_corners(images, pattern_size) 
        if len(objpoints) < 10:
            print("Calibration aborted: not enough valid frames.")
            return

        mtx, dist = calibrate_camera_checkerboard(
            objpoints,
            imgpoints,
            image_size
        )

    if mtx is not None:
        save_calibration(output_file, mtx, dist)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
