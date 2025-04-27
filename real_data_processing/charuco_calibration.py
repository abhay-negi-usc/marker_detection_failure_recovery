import cv2
import numpy as np
import glob
import time 

# Parameters for the ChArUco board
chessboard_size = (5, 7)  # (number of internal corners in chessboard grid)
square_length = 0.0730  # Length of the square in meters (adjust accordingly)
marker_length = 0.0365  # Length of the marker in meters (adjust accordingly)
max_images = 100 # Maximum number of images to use for calibration 

# Prepare the ChArUco board
# Define the dictionary for the ArUco markers
aruco_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250, 6)  # Specify marker size

# Create a ChArUco board using the correct method (CharucoBoard_create)
charuco_board = cv2.aruco.CharucoBoard(chessboard_size,square_length, marker_length, aruco_dict)

# Set up termination criteria for the calibration (e.g., 30 iterations or 0.1 precision)
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

# Prepare object points (3D points in the world coordinate system)
object_points = []  # 3D points in the real world
all_charuco_corners = []   # 2D points in the image plane
charuco_ids_list = [] 

# Detector setup
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Find all the calibration images (ensure you have a set of images with the ChArUco board in them)
# images = glob.glob('C:/Users/NegiA/Desktop/abhay_ws/marker_detection_failure_recovery/real_data_processing/raw_data/charuco_calibration_frames/*.png')  # Adjust the path to your images
images = glob.glob('C:/Users/NegiA/Desktop/abhay_ws/marker_detection_failure_recovery/real_data_processing/raw_data/realsense415_charuco_calibration_frames/*.png')  # Adjust the path to your images

# uniformly downsample the images to max_images 
original_num_images = len(images)
if len(images) > max_images:
    step = len(images) // max_images
    images = images[::step] # Downsample the list to max_images 
print(f"Number of images used for calibration: {len(images)} out of {original_num_images} total images") 

# Process each image
for idx, image_file in enumerate(images):
    # Read the image
    img = cv2.imread(image_file)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect markers and interpolate the ChArUco corners
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, rejected_img_points = detector.detectMarkers(gray)
    
    if len(corners) > 0:
        # Interpolate the ChArUco corners
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, charuco_board)

        if retval and charuco_corners is not None and charuco_ids is not None and len(charuco_ids) >= 4:
            # Convert to NumPy for safety checks
            corners_np = np.array(charuco_corners, dtype=np.float32).reshape(-1, 2)
            ids_np = np.array(charuco_ids, dtype=np.float32)

            # Skip NaNs
            if np.any(np.isnan(corners_np)) or np.any(np.isnan(ids_np)):
                print(f"Skipping {image_file}: contains NaNs.")
                continue

            # Reject degenerate cases (low area = almost collinear points)
            if cv2.contourArea(corners_np) < 1e-3:
                print(f"Skipping {image_file}: low area (possible degeneracy).")
                continue

            all_charuco_corners.append(charuco_corners)
            charuco_ids_list.append(charuco_ids)

print(f"Num valid frames: {len(all_charuco_corners)}")
if len(all_charuco_corners) > 0:
    print(f"First frame corners: {all_charuco_corners[0].shape}")
    print(f"First frame ids: {charuco_ids_list[0].shape}")
else:
    print("No valid frames collected. Exiting.")
    exit()

# Only calibrate if enough valid views
if len(all_charuco_corners) >= 10:
    # Initial guess for intrinsics (optional)
    cameraMatrixInit = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
    distCoeffsInit = np.zeros((5, 1), dtype=np.float32)

    ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_charuco_corners,
        charucoIds=charuco_ids_list,
        board=charuco_board,
        imageSize=gray.shape[::-1],
        cameraMatrix=cameraMatrixInit,
        distCoeffs=distCoeffsInit,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS)

    print("✅ Calibration successful")
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)

    # Save to file
    np.savez("camera_calibration.npz", camera_matrix=mtx, distortion_coeffs=dist)
else:
    print("❌ Not enough valid detections for calibration (need at least 10).")

cv2.destroyAllWindows()

        # Save visualized image
        image_name = os.path.basename(image_file)
        output_path = os.path.join(output_dir, f"charuco_{idx:03d}_{image_name}")
        cv2.imwrite(output_path, img_marked)


# Perform the camera calibration
cameraMatrixInit = np.array([[1400.0,0.0,950.0],[0.0,1400.0,530.0],[0,0,1]], dtype=np.float32) 
distCoeffsInit = np.zeros((5, 1), dtype=np.float32)  # Assuming no initial distortion coefficients
rvecs_empty = None # np.zeros((len(image_points), 3), dtype=np.float32)  # Initialize rotation vectors
tvecs_empty = None # np.zeros((len(image_points), 3), dtype=np.float32)  # Initialize translation vectors
# import pdb; pdb.set_trace() 
# ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
#     all_charuco_corners, charuco_ids_list, charuco_board, gray.shape[::-1], 
#     None, None) 
time_start = time.time()    
ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    all_charuco_corners, charuco_ids_list, charuco_board, gray.shape[::-1], 
    cameraMatrixInit, distCoeffsInit)
time_end = time.time() 
time_taken = time_end - time_start  
print(f"Time taken for calibration: {len(images)} images in  {time_taken:.2f} seconds")  
# Output the results
print("Calibration was successful: ", ret)
print("Camera matrix: \n", mtx) 
print("Distortion coefficients: \n", dist)
# print("Rotation vectors: \n", rvecs)
# print("Translation vectors: \n", tvecs)
# print("Rotation vectors: \n", rvecs)
# print("Translation vectors: \n", tvecs)

# Save the calibration parameters
np.savez("./real_data_processing/raw_data/camera_calibration.npz", camera_matrix=mtx, distortion_coeffs=dist)

# Close all OpenCV windows
cv2.destroyAllWindows()
