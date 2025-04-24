import cv2
import numpy as np
import glob
import os 

# Parameters for the ChArUco board
chessboard_size = (5, 7)  # (number of internal corners in chessboard grid)
square_length = 0.0730  # Length of the square in meters (adjust accordingly)
marker_length = 0.0365  # Length of the marker in meters (adjust accordingly)

# Create a directory to save visualized results
output_dir = "./real_data_processing/raw_data/charuco_detected_images"
os.makedirs(output_dir, exist_ok=True)

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

# Find all the calibration images (ensure you have a set of images with the ChArUco board in them)
images = glob.glob('./real_data_processing/raw_data/charuco_calibration_frames/*.png')  # Adjust the path to your images

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

        # Draw detected markers
        img_marked = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)

        # # If we found enough corners, add them to the list
        # if retval >= 4:
        #     all_charuco_corners.append(charuco_corners)
        #     # object_points.append(np.zeros_like(charuco_corners, dtype=np.float32)) 
        #     charuco_ids_list.append(charuco_ids.flatten())  # Store the charuco ids for each image

        if retval is not None and retval >= 6:
            all_charuco_corners.append(charuco_corners)
            charuco_ids_list.append(charuco_ids.flatten())

            img_marked = cv2.aruco.drawDetectedCornersCharuco(img_marked, charuco_corners, charuco_ids) 

            # Draw detected markers and the ChArUco corners
            # img_with_markers = cv2.aruco.drawDetectedMarkers(img, corners, ids)
            # img_with_charuco = cv2.aruco.drawDetectedCornersCharuco(img_with_markers, charuco_corners, charuco_ids)
            # cv2.imshow('ChArUco Calibration', img_with_charuco)
            # cv2.waitKey(1)

        # Save visualized image
        image_name = os.path.basename(image_file)
        output_path = os.path.join(output_dir, f"charuco_{idx:03d}_{image_name}")
        cv2.imwrite(output_path, img_marked)


# Perform the camera calibration
cameraMatrixInit = np.zeros((3,3)) # np.array([[400,0,320],[0,400,240],[0,0,1]], dtype=np.float32) 
distCoeffsInit = np.zeros((5, 1), dtype=np.float32)  # Assuming no initial distortion coefficients
rvecs_empty = None # np.zeros((len(image_points), 3), dtype=np.float32)  # Initialize rotation vectors
tvecs_empty = None # np.zeros((len(image_points), 3), dtype=np.float32)  # Initialize translation vectors
# import pdb; pdb.set_trace() 
fx, fy, cx, cy = 722, 698, 310, 272 
camera_matrix_init = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)  # Initial guess for camera matrix
distortion_coeffs_init = np.zeros((5, 1), dtype=np.float64)  # Initial guess for distortion coefficients 
ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    all_charuco_corners, charuco_ids_list, charuco_board, gray.shape[::-1], 
    camera_matrix_init, distortion_coeffs_init, flags=cv2.CALIB_USE_INTRINSIC_GUESS) 
# Output the results
print("Calibration was successful: ", ret)
print("Camera matrix: \n", mtx) 
print("Distortion coefficients: \n", dist)
# print("Rotation vectors: \n", rvecs)
# print("Translation vectors: \n", tvecs)

# Save the calibration parameters
np.savez("./real_data_processing/raw_data/camera_calibration.npz", camera_matrix=mtx, distortion_coeffs=dist)

# Close all OpenCV windows
cv2.destroyAllWindows()
