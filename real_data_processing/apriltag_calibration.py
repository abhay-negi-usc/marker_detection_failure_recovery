import cv2
import numpy as np
import glob
import time
import os


def load_images(image_dir, max_images):
    images = glob.glob(os.path.join(image_dir, '*.png'))
    original_num_images = len(images)

    if original_num_images > max_images:
        step = max(1, original_num_images // max_images)
        images = images[::step]

    print(f"Number of images used for calibration: {len(images)} out of {original_num_images}")
    return images


def detect_apriltag_corners(images, tag_size, camera_matrix_init=None, dist_coeffs_init=None):
    objpoints = []  # 3D points in tag frame
    imgpoints = []  # 2D image points

    # 3D object points for the corners of the tag
    half_size = tag_size / 2.0
    objp = np.array([
        [-half_size,  half_size, 0],
        [ half_size,  half_size, 0],
        [ half_size, -half_size, 0],
        [-half_size, -half_size, 0]
    ], dtype=np.float32)

    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))

    for idx, img_path in enumerate(images):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)

        if corners and ids is not None and len(corners) > 0:
            # Use only first tag (assuming single tag)
            img_corners = corners[0].reshape(-1, 2)

            # Optional: skip if tag area is too small
            if cv2.contourArea(img_corners) < 1e-3:
                print(f"Skipping {img_path}: low area.")
                continue

            objpoints.append(objp)
            imgpoints.append(img_corners)

    print(f"Number of valid frames: {len(objpoints)}")
    return objpoints, imgpoints, gray.shape[::-1]


def calibrate_camera_apriltag(objpoints, imgpoints, image_size, camera_matrix_init=None, dist_coeffs_init=None, use_intrinsic_guess=False):
    if len(objpoints) < 10:
        print("Not enough valid detections for calibration (need at least 10).")
        return None, None

    flags = cv2.CALIB_USE_INTRINSIC_GUESS if use_intrinsic_guess else 0

    print("Starting AprilTag-based calibration...")
    time_start = time.time()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        camera_matrix_init,
        dist_coeffs_init,
        flags=flags
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
    image_dir = '/home/rp/dhanush_ws/sunrise-wrapper/data/marker_data_collection/handeye_calibration_v2/images/'
    output_file = './camera_calibration_apriltag.npz'
    tag_size = 0.080  # Tag side length in meters
    max_images = 250

    images = load_images(image_dir, max_images)

    # Initial guess (optional)
    camera_matrix_init = np.array([[886.643, 0.0, 631.834],
                                   [0.0, 886.643, 367.724],
                                   [0.0,    0.0, 1.0]], dtype=np.float32)
    dist_coeffs_init = np.zeros((5, 1), dtype=np.float32)
    use_intrinsic_guess = True

    objpoints, imgpoints, image_size = detect_apriltag_corners(images, tag_size, camera_matrix_init, dist_coeffs_init)

    if len(objpoints) < 10:
        print("Calibration aborted: not enough valid frames.")
        return

    mtx, dist = calibrate_camera_apriltag(
        objpoints,
        imgpoints,
        image_size,
        camera_matrix_init,
        dist_coeffs_init,
        use_intrinsic_guess
    )

    if mtx is not None:
        save_calibration(output_file, mtx, dist)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
