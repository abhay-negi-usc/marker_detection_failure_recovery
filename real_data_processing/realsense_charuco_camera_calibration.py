import cv2
import numpy as np
import realsense_launch as rs_launch  # Import your RealSense module


def define_checkerboard_world_frame(pattern_size, square_size, origin="center"):
    """
    Defines 3D object points for a checkerboard with a configurable origin.

    Args:
        pattern_size (tuple): Number of inner corners (rows, columns) in the checkerboard.
        square_size (float): Size of each square in meters.
        origin (str): Desired origin location. Options are:
            - "center": Origin at the geometric center of the checkerboard.
            - "top-left": Origin at the top-left corner.
            - "top-right": Origin at the top-right corner.
            - "bottom-left": Origin at the bottom-left corner.
            - "bottom-right": Origin at the bottom-right corner.

    Returns:
        numpy.ndarray: 3D object points with origin shifted to the desired location.
    """
    # Generate grid of points for corners
    obj_points = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)
    obj_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    obj_points *= square_size

    # Compute offsets for different origins
    if origin == "center":
        offset = np.array([(pattern_size[0] - 1) * square_size / 2,
                           (pattern_size[1] - 1) * square_size / 2,
                           0])
    elif origin == "top-left":
        offset = np.array([0, 0, 0])
    elif origin == "top-right":
        offset = np.array([(pattern_size[0] - 1) * square_size,
                           0,
                           0])
    elif origin == "bottom-left":
        offset = np.array([0,
                           (pattern_size[1] - 1) * square_size,
                           0])
    elif origin == "bottom-right":
        offset = np.array([(pattern_size[0] - 1) * square_size,
                           (pattern_size[1] - 1) * square_size,
                           0])
    else:
        raise ValueError(f"Invalid origin '{origin}'. Valid options are: 'center', 'top-left', 'top-right', 'bottom-left', 'bottom-right'.")

    # Shift origin to desired location
    obj_points -= offset

    return obj_points


def detect_checkerboard(image, pattern_size=(8, 6)):
    """
    Detects the checkerboard corners in the image and computes 2D image points.

    Args:
        image (numpy.ndarray): Input image from the RealSense camera.
        pattern_size (tuple): Number of inner corners (rows, columns) in the checkerboard.

    Returns:
        img_points (numpy.ndarray): 2D points in the image.
        ret (bool): True if corners are detected successfully, False otherwise.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret:
        # Refine corner locations for better accuracy
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        return corners, ret
    else:
        return None, False


def calculate_3d_points(img_points, depth_image, intrinsics):
    """
    Converts 2D image points to 3D points using depth data and camera intrinsics.

    Args:
        img_points (numpy.ndarray): Detected 2D points in image space.
        depth_image (numpy.ndarray): Depth map from RealSense camera.
        intrinsics (dict): Camera intrinsic parameters.

    Returns:
        numpy.ndarray: Corresponding 3D points in camera space.
    """
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["ppx"], intrinsics["ppy"]

    points_3d = []
    for point in img_points:
        u, v = int(point[0][0]), int(point[0][1])  # Pixel coordinates
        z = depth_image[v, u] * intrinsics["depth_scale"]  # Depth value in meters
        if z == 0:  # Skip invalid depth values
            continue
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points_3d.append([x, y, z])

    return np.array(points_3d)


def compare_and_choose_points(obj_points_manual, obj_points_depth_based):
    """
    Compares manually calculated object points with depth-based object points and chooses the best.

    Args:
        obj_points_manual (numpy.ndarray): Manually calculated object points.
        obj_points_depth_based (numpy.ndarray): Depth-based object points.

    Returns:
        numpy.ndarray: The chosen set of object points based on validity and accuracy.
    """
    if len(obj_points_depth_based) == len(obj_points_manual):
        print("Using depth-based object points.")
        return obj_points_depth_based
    else:
        print("Depth data is incomplete or invalid. Falling back to manually calculated object points.")
        return obj_points_manual


def calculate_camera_to_target_transform(obj_points, img_points, camera_matrix, dist_coeffs):
    """
    Calculates rotation and translation matrices from camera to target using solvePnP.

    Args:
        obj_points (numpy.ndarray): 3D points in the checkerboard's coordinate system.
        img_points (numpy.ndarray): 2D points in the image.
        camera_matrix (numpy.ndarray): Camera intrinsic matrix.
        dist_coeffs (numpy.ndarray): Camera distortion coefficients.

    Returns:
        R_cam_to_target (numpy.ndarray): Rotation matrix from camera to target.
        t_cam_to_target (numpy.ndarray): Translation vector from camera to target.
    """
    # SolvePnP to compute rotation and translation
    ret, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)

    if ret:
        # Convert rotation vector to rotation matrix
        R_cam_to_target, _ = cv2.Rodrigues(rvec)
        return R_cam_to_target, tvec
    else:
        raise Exception("Failed to compute pose using solvePnP.")


def draw_world_axes(image, R_cam_to_target, t_cam_to_target, camera_matrix, dist_coeffs):
    """
    Draws world frame axes on the image based on the transformation matrix.

    Args:
        image (numpy.ndarray): The RGB image on which to draw.
        R_cam_to_target (numpy.ndarray): Rotation matrix from camera to target.
        t_cam_to_target (numpy.ndarray): Translation vector from camera to target.
        camera_matrix (numpy.ndarray): Camera intrinsic matrix.
        dist_coeffs (numpy.ndarray): Camera distortion coefficients.

    Returns:
        numpy.ndarray: The image with world frame axes drawn.
    """
    # Define world frame axes in 3D space
    axis_length = 0.1  # Length of axes in meters
    axes_3d = np.float32([
        [0, 0, 0],               # Origin
        [axis_length, 0, 0],     # X-axis
        [0, axis_length, 0],     # Y-axis
        [0, 0, axis_length]      # Z-axis
    ])

    # Project the 3D axes onto the image plane
    axes_2d, _ = cv2.projectPoints(axes_3d,
                                   cv2.Rodrigues(R_cam_to_target)[0],
                                   t_cam_to_target,
                                   camera_matrix,
                                   dist_coeffs)

    # Convert to integer coordinates for drawing
    axes_2d = np.int32(axes_2d).reshape(-1, 2)

    # Draw the axes on the image
    origin = tuple(axes_2d[0])   # Origin point
    x_axis = tuple(axes_2d[1])   # X-axis endpoint
    y_axis = tuple(axes_2d[2])   # Y-axis endpoint
    z_axis = tuple(axes_2d[3])   # Z-axis endpoint

    cv2.line(image, origin, x_axis, (0, 0, 255), 3)   # X-axis in red
    cv2.line(image, origin, y_axis, (0, 255, 0), 3)   # Y-axis in green
    cv2.line(image, origin, z_axis, (255, 0, 0), 3)   # Z-axis in blue

    return image


def main():
    # Get serial numbers of connected devices using realsense_launch module
    serial_numbers = rs_launch.get_serial_numbers()

    if not serial_numbers:
        print("No RealSense devices detected!")
        return

    # Set up pipeline for the first connected RealSense device
    pipeline = rs_launch.setup_pipeline(serial_numbers[0])
    # device = pipeline.get_active_profile().get_device()
    # sensors = device.query_sensors()

    # Get camera intrinsics using realsense_launch module
    intrinsics = rs_launch.get_intrinsics(pipeline.get_active_profile().get_device().query_sensors()[0], "depth")
    camera_matrix = np.array([[intrinsics["fx"], 0.0, intrinsics["ppx"]],
                                [0.0, intrinsics["fy"], intrinsics["ppy"]],
                                [0.0, 0.0, 1.0]])
    dist_coeffs = np.array([intrinsics["k1"], intrinsics["k2"],
                            intrinsics["p1"], intrinsics["p2"], intrinsics["k3"]])
    try:
        while True:
            # Capture frames using realsense_launch module
            frames_dict = rs_launch.capture_frames([pipeline])
            color_image = frames_dict[0][0]  # Extract RGB image from first camera
            depth_image = frames_dict[0][1]  # Extract Depth image from first camera

            # Define checkerboard pattern size and square size
            pattern_size = (8, 6)          # Checkerboard with inner corners: rows x cols
            square_size = 0.073            # Square size in meters

            # Define manually calculated object points with origin at center of checkerboard
            obj_points_manual = define_checkerboard_world_frame(pattern_size=pattern_size,
                                                                square_size=square_size,
                                                                origin="bottom-right")

            # Detect checkerboard corners in RGB image
            img_points, ret = detect_checkerboard(color_image,
                                                  pattern_size=pattern_size)
            
            if ret:
                print("Checkerboard detected successfully!")


                # Compute corresponding object points using depth data
                obj_points_depth_based = calculate_3d_points(img_points, depth_image, intrinsics)

                # Compare and choose between manual and depth-based object points
                chosen_obj_points = compare_and_choose_points(obj_points_manual, obj_points_depth_based)
                # chosen_obj_points = obj_points_manual

                print("Chosen Object Points:\n", chosen_obj_points)

                # Perform pose estimation using chosen object points and detected image points
                R_cam_to_target, t_cam_to_target = calculate_camera_to_target_transform(chosen_obj_points,
                                                                                        img_points,
                                                                                        camera_matrix,
                                                                                        dist_coeffs)

                print("Rotation Matrix (Camera to Target):\n", R_cam_to_target)
                print("Translation Vector (Camera to Target):\n", t_cam_to_target)

                # Draw world frame axes on live RGB stream
                color_image = draw_world_axes(color_image,
                                              R_cam_to_target,
                                              t_cam_to_target,
                                              camera_matrix,
                                              dist_coeffs)

                # Visualize detected corners on live RGB stream
                cv2.drawChessboardCorners(color_image,
                                          pattern_size,
                                          img_points,
                                          ret)

            else:
                print("Failed to detect checkerboard!")

            # Display live RGB stream with world frame axes and detected corners overlayed
            cv2.imshow("RealSense Live Viewer with World Frame", color_image)

            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        rs_launch.stop_pipelines([pipeline])
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
