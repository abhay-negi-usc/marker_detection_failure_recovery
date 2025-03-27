# IMPORTS 
import os 
import matplotlib.pyplot as plt 
import numpy as np 
import json
import cv2 
from PIL import Image 
import random 
import math 
from scipy.spatial.transform import Rotation as R

# HELPER FUNCTIONS 
def project_point_to_image(C,T,P): 
    P_H = np.array([[P[0]],[P[1]],[P[2]],[1]]) 
    T_H = T[:3,:4]  
    uv = C @ T_H @ P_H 
    uv = uv / uv[2] 
    uv = uv[:2] 
    uv = uv.reshape((2)) 
    return uv 

def project_point_list_to_image(C,T,P_list,convert_cam_is2cv=True): 
    if convert_cam_is2cv: 
        T = T @ np.array([
            [1,0,0,0],
            [0,-1,0,0],
            [0,0,-1,0],
            [0,0,0,1]
        ])  
    n = len(P_list)
    uv_list = []  
    for P in P_list: 
        uv = project_point_to_image(C,T,P) 
        uv_list.append(uv) 
    return uv_list   

def transform_pts(pts, T):  
    pts_transformed = [] 
    for pt in pts: 
        pt = pt.reshape(3,1) 
        pt = np.vstack((pt, 1))  
        pt_transformed = T @ pt  
        pts_transformed.append(pt_transformed[:3]) 
    return pts_transformed

def overlay_points_on_image(image, pixel_points, radius=5, color=(0, 0, 255), thickness=-1):
    """
    Overlays a list of pixel points on the input image.

    Parameters:
    - image: The input image (a NumPy array).
    - pixel_points: A list of 2D pixel coordinates [(x1, y1), (x2, y2), ...].
    - radius: The radius of the circle to draw around each point. Default is 5.
    - color: The color of the circle (BGR format). Default is red (0, 0, 255).
    - thickness: The thickness of the circle. Default is -1 to fill the circle.

    Returns:
    - The image with points overlaid.
    """
    # Iterate over each pixel point and overlay it on the image
    for point in pixel_points:
        if point is not None:  # Only overlay valid points
            x, y = int(point[0]), int(point[1])
            # Draw a filled circle at the pixel coordinates
            cv2.circle(image, (x, y), radius, color, thickness)

    return image

def compute_2D_gridpoints(N=10,s=0.1): 
    # N = num squares, s = side length  
    u = np.linspace(-s/2, +s/2, N+1) 
    v = np.linspace(-s/2, +s/2, N+1) 
    gridpoints = [] 
    for uu in u:
        for vv in v: 
            gridpoints.append(np.array([uu,vv,0])) 
    return gridpoints 

def lambertian_reflection(I_incident, N, L):
    # Calculate diffuse reflection (Lambertian model)
    return I_incident * max(np.dot(N, L), 0)

def phong_reflection(I_incident, N, L, V, shininess):
    # Calculate the reflection vector
    R = (2 * np.dot(N, L) * N) - L
    # Calculate specular reflection using the Phong model
    return I_incident * max(np.dot(R, V), 0) ** shininess

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

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        # Convert np.ndarray to list
        return obj.tolist()
    elif isinstance(obj, np.generic):
        # Convert numpy scalars (like np.float32) to native Python types
        return obj.item()
    elif isinstance(obj, list):
        # Convert each item in the list to a serializable type
        return [convert_to_serializable(item) for item in obj]
    else:
        # Return other objects as they are
        return obj

def compute_segmentation_IOU(segmentation_mask, ground_truth_mask):
    """
    Compute the Intersection over Union (IoU) between a segmentation mask and a ground truth mask.

    Args:
        segmentation_mask (numpy.ndarray): The predicted segmentation mask.
        ground_truth_mask (numpy.ndarray): The ground truth segmentation mask.

    Returns:
        float: The IoU score between the two masks.
    """
    # Ensure both masks are binary (0 or 1)
    segmentation_mask = (segmentation_mask > 0).astype(np.uint8)
    ground_truth_mask = (ground_truth_mask > 0).astype(np.uint8)

    # Compute intersection and union
    intersection = np.logical_and(segmentation_mask, ground_truth_mask).sum()
    union = np.logical_or(segmentation_mask, ground_truth_mask).sum()

    # Compute IoU
    iou = intersection / union if union != 0 else 0.0

    return iou

def marker_reprojection(image, pred, marker_image, marker_corners_2d, marker_corners_3d, rvec, tvec, camera_matrix, dist_coeffs, alpha):
    # Load the image and marker image 
    if isinstance(image, str):
        image = cv2.imread(image) 
    image = np.array(image) 
    if image.shape[2] == 4: 
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB) 
    marker_image = np.array(marker_image) 

    if isinstance(pred, str):
        pred = cv2.imread(pred)
    # if pred is a binary image convert it to rgb 
    if len(pred.shape) == 2: 
        pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB) 

    # Project the 3D model points of the marker to the image
    image_points, _ = cv2.projectPoints(marker_corners_3d, rvec, tvec, camera_matrix, dist_coeffs)

    # Convert the image points to integer
    image_points = np.round(image_points).astype(int)

    # Draw the marker on the image 
    reprojected_marker_mask = cv2.fillPoly(image*0, [image_points], (255, 255, 255)) 

    image = cv2.addWeighted(image, alpha, pred, 1 - alpha, 0) 

    marker_image_corners = np.array([
        [0, 0],
        [marker_image.shape[1], 0],
        [marker_image.shape[1], marker_image.shape[0]],
        [0, marker_image.shape[0]] 
    ], dtype=np.float32)

    image_points = np.array(image_points).reshape(-1, 2).astype(np.float32)
    M = cv2.getPerspectiveTransform(marker_image_corners, image_points) 

    # define overlay to be each pixel of the marker image transformed using H 
    reprojected_marker_image = cv2.warpPerspective(marker_image, M, (image.shape[1], image.shape[0])) 
    blended_image = cv2.addWeighted(image, 1-alpha, reprojected_marker_image, alpha, 0) 
    
    return reprojected_marker_image, reprojected_marker_mask, blended_image   

def detect_corners(seg_mask, area_threshold=100, epsilon_factor=0.02):
    """
    Detects corners of a polygon in a segmentation mask.
    
    Parameters:
        seg_mask (numpy.ndarray): The binary segmentation mask.
        area_threshold (int): Minimum area threshold to ignore small contours (default is 100).
        epsilon_factor (float): Factor to control the approximation accuracy (default is 0.02).
    
    Returns:
        numpy.ndarray: Detected corners of the polygon (could be any polygon with > 3 points), or None if no valid contour is found.
    """
    # Find contours in the segmentation mask
    contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # assume one contour 

        # Check if the contour is large enough to be considered
        area = cv2.contourArea(contour)
        if area < area_threshold:
            # print(f"Contour area {area} is smaller than threshold {area_threshold}. Ignoring this contour.") 
            continue

        # Approximate the contour to a polygon (reduce epsilon for higher accuracy)
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # this is for forcing a quadrilateral fit 
        # # If the contour has more than 4 points, apply convex hull and approximate again
        # if len(approx) > 4:
        #     hull = cv2.convexHull(contour)
        #     epsilon = epsilon_factor * cv2.arcLength(hull, True)
        #     approx = cv2.approxPolyDP(hull, epsilon, True)
        
        # Return the detected corners (polygon with at least 3 points)
        return approx.reshape(-1, 2)  # Flatten the points and return as a numpy array

    # Return None if no valid contour was found
    return None

# Function to find N closest point distances
def closest_point_distances(point_list_1, point_list_2): 
    assert(len(point_list_1) > 0 and len(point_list_2) > 0), "Both point lists must be non-empty." 
    if isinstance(point_list_1, np.ndarray): 
        point_list_1 = [row for row in point_list_1] 
    if isinstance(point_list_2, np.ndarray): 
        point_list_2 = [row for row in point_list_2]  
    
    distances = []
    # Iterate through each point in list1
    for point1 in point_list_1: 
        closest_distance = float('inf')  # Initialize with a large number
        # Iterate through each point in list2 to find the closest point
        for point2 in point_list_2:
            distance = np.linalg.norm(np.array(point1) - np.array(point2)) 
            if distance < closest_distance:
                closest_distance = distance
        # Append the closest distance for this point
        distances.append(closest_distance) 
    return distances

# DATAPOINT CLASS DEFINITION 
class datapoint:
    def __init__(self, metadata_filepath, pose_filepath, rgb_filepath, seg_png_filepath, seg_json_filepath):
        # Store the filepaths
        self.metadata_filepath = metadata_filepath
        self.pose_filepath = pose_filepath
        self.rgb_filepath = rgb_filepath
        self.seg_png_filepath = seg_png_filepath
        self.seg_json_filepath = seg_json_filepath
        
        self.read_files()
        self.read_pose_data() 
        self.compute_diffusion_reflectance() 
        self.compute_keypoints() 

    def read_files(self): 
        # Read the actual data from files and store it
        self.metadata = self._read_json(self.metadata_filepath) if self.metadata_filepath else None
        self.pose = self._read_json(self.pose_filepath) if self.pose_filepath else None
        self.rgb = self._read_rgb(self.rgb_filepath) if self.rgb_filepath else None
        self.seg_png = self._read_segmentation_png(self.seg_png_filepath) if self.seg_png_filepath else None
        self.seg_json = self._read_segmentation_json(self.seg_json_filepath) if self.seg_json_filepath else None 

    def read_pose_data(self): 
        # read pose data from pose json file 
        # self.cam_pose = np.array([
        #                     [1, 0, 0, 0],
        #                     [0, -1, 0, 0],
        #                     [0, 0, -1, 0],
        #                     [0, 0, 0, 1]
        #                 ]) # NOTE: cam pose from isaac sim appears to be offset 
        self.cam_pose = np.eye(4) # FIXME: DEBUG 

        self.tag_pose = np.array(self.pose["tag"]).transpose()  
        self.tag_pose *= np.array([
                            [10,10,10,1],
                            [10,10,10,1],
                            [10,10,10,1],
                            [1,1,1,1]
                        ]) # rescale the tag, FIXME: avoid hardcoding tag scale value 
        
        # DEBUG 
        # tf_correction = np.array([
        #     [1,0,0,0],
        #     [0,-1,0,0],
        #     [0,0,-1,0],
        #     [0,0,0,1]
        # ]) 
        # import pdb; pdb.set_trace()
        # self.tag_pose = self.tag_pose @ tf_correction 
        # DEBUG 

        self.light_pose = self.pose["light"] 
        
    def _read_json(self, filepath):
        """Read and parse JSON files."""
        with open(filepath, 'r') as file:
            return json.load(file)

    def _read_rgb(self, filepath):
        """Placeholder for reading RGB image files."""
        return filepath  # Placeholder: returning the file path to avoid memory overload

    def _read_segmentation_png(self, filepath):
        """Placeholder for reading segmentation PNG image files."""
        return filepath  # Placeholder: returning the file path to avoid memory overload

    def _read_segmentation_json(self, filepath):
        """Read segmentation JSON files."""
        with open(filepath, 'r') as file:
            return json.load(file)

    def compute_diffusion_reflectance(self): 
        """Compute the diffuse reflection based on pose and metadata."""
        N = np.array(self.tag_pose)[:3,2] 
        L = np.array(self.light_pose)[:3,2] 
        V = np.array(self.cam_pose)[:3,2] 
        light_exposure = self.metadata["light"]["exposure"] 
        shininess = 1.0  # Placeholder value 
        I_incident = 2**light_exposure 
        shininess = 1.0 # NOTE: placeholder value 
        self.diffuse_reflection = lambertian_reflection(I_incident, N, L)     
        self.specular_reflection = phong_reflection(I_incident, N, L, V, shininess)

    def compute_keypoints(self): 
        # FIXME: avoid hardcoding and take in as arguments 
        # camera parameters 
        width = 640 
        height = 480 
        focal_length = 24.0 
        horiz_aperture = 20.955
        # Pixels are square so we can do:
        vert_aperture = height/width * horiz_aperture
        fov = 2 * math.atan(horiz_aperture / (2 * focal_length))
        # compute focal point and center
        fx = width * focal_length / horiz_aperture
        fy = height * focal_length / vert_aperture
        cx = width / 2
        cy = height /2 

        self.C = np.array([
            [fx,0,cx],
            [0,fy,cy],
            [0,0,1]
        ])

        s = 0.1 # side length of marker 
        self.keypoints_tag_frame = compute_2D_gridpoints(N=10, s=s) 
        self.outter_corners_tag_frame = compute_2D_gridpoints(N=1, s=s)
        # self.inner_corners_tag_frame = compute_2D_gridpoints(N=1, s=s*0.8)
        self.inner_corners_tag_frame = compute_2D_gridpoints(N=1, s=s*1.0)

        # transformations 
        tf_w_t = self.tag_pose 
        tf_w_c = self.cam_pose 
        tf_c_w = np.linalg.inv(tf_w_c) 

        keypoints_world_frame = [] 
        for kp_t in self.keypoints_tag_frame: 
            kp_t_homog = np.hstack((kp_t,np.array([1]))).reshape(4,1)
            kp_w_homog = tf_w_t @ kp_t_homog 
            keypoints_world_frame.append(kp_w_homog[:3].reshape(3)) 
        
        outter_corners_world_frame = [] 
        for kp_t in self.outter_corners_tag_frame: 
            kp_t_homog = np.hstack((kp_t,np.array([1]))).reshape(4,1)
            kp_w_homog = tf_w_t @ kp_t_homog 
            outter_corners_world_frame.append(kp_w_homog[:3].reshape(3))
        
        inner_corners_world_frame = [] 
        for kp_t in self.inner_corners_tag_frame: 
            kp_t_homog = np.hstack((kp_t,np.array([1]))).reshape(4,1)
            kp_w_homog = tf_w_t @ kp_t_homog 
            inner_corners_world_frame.append(kp_w_homog[:3].reshape(3))           

        self.keypoints_image_space = project_point_list_to_image(self.C,tf_c_w,keypoints_world_frame) 
        self.outter_corners_image_space = project_point_list_to_image(self.C,tf_c_w,outter_corners_world_frame) 
        self.inner_corners_image_space = project_point_list_to_image(self.C,tf_c_w,inner_corners_world_frame) 

    def __repr__(self):
        """Custom representation for the datapoint object."""
        # return f"datapoint(metadata_filepath={self.metadata_filepath}, pose_filepath={self.pose_filepath}, rgb_filepath={self.rgb_filepath}, seg_png_filepath={self.seg_png_filepath}, seg_json_filepath={self.seg_json_filepath})"
        description = [
            f"lighting_exposure={self.metadata["light"]["exposure"]:.2f}",
            f"lighting_color=({self.metadata["light"]["color"][0]:.2f},{self.metadata["light"]["color"][1]:.2f},{self.metadata["light"]["color"][2]:.2f})", # FIXME: reduce to two decimal places 
            f"diffuse reflection={self.diffuse_reflection:.2f}", 
            f"specular reflection={self.specular_reflection:.2f}", 
        ]
        return "\n".join(description) 

# DATAPROCESSOR CLASS DEFINITION 
class DataProcessor:

    def __init__(self, data_folders, out_dir):
        self.data_folders = data_folders
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)  
        self.datapoints = []
        self.datapoints_train = []
        self.datapoints_val = []

        # TODO: find a more appropriate place for this 
        self.tf_W_Ccv = np.array([
            [1,0,0,0],
            [0,-1,0,0],
            [0,0,-1,0],
            [0,0,0,1]
        ])
        self.T_Mcv_Mis = np.array([
            [-1,0,0,0],
            [0,1,0,0],
            [0,0,-1,0],
            [0,0,0,1]
        ]) 

    def _get_files_in_subfolder(self, folder, file_extension=None):
        """Helper method to get files in a subfolder, with an optional file extension filter."""
        files_list = os.listdir(folder)
        if file_extension:
            files_list = [file for file in files_list if file.endswith(file_extension)]
        # Order files_list by date created
        files_list = sorted(files_list, key=lambda x: os.path.getctime(os.path.join(folder, x)))  # Assumes creation dates are synchronized
        return files_list

    def process_folders(self):
        """Process the folders and create datapoint objects."""
        for data_folder in self.data_folders:
            metadata_subfolder = os.path.join(data_folder, "metadata")
            pose_subfolder = os.path.join(data_folder, "pose")
            rgb_subfolder = os.path.join(data_folder, "rgb")
            seg_subfolder = os.path.join(data_folder, "seg")

            # List files in subfolders 
            metadata_files = self._get_files_in_subfolder(metadata_subfolder, file_extension=".json")
            pose_files = self._get_files_in_subfolder(pose_subfolder, file_extension=".json")
            rgb_files = self._get_files_in_subfolder(rgb_subfolder, file_extension=".png")
            seg_png_files = self._get_files_in_subfolder(seg_subfolder, file_extension=".png")
            seg_json_files = self._get_files_in_subfolder(seg_subfolder, file_extension=".json")

            # Make sure the files are indexed and aligned properly (by index) across the subfolders
            max_length = max(len(metadata_files), len(pose_files), len(rgb_files), len(seg_png_files), len(seg_json_files))

            # Verify that the lengths are the same
            if not all(len(files) == max_length for files in [metadata_files, pose_files, rgb_files, seg_png_files, seg_json_files]):
                print(f"Lengths do not match for folder: {data_folder}")
                continue

            for i in range(max_length):
                # Use index 'i' to fetch corresponding files. If a file doesn't exist, use None.
                metadata_filepath = os.path.join(metadata_subfolder, metadata_files[i]) if i < len(metadata_files) else None
                pose_filepath = os.path.join(pose_subfolder, pose_files[i]) if i < len(pose_files) else None
                rgb_filepath = os.path.join(rgb_subfolder, rgb_files[i]) if i < len(rgb_files) else None
                seg_png_filepath = os.path.join(seg_subfolder, seg_png_files[i]) if i < len(seg_png_files) else None
                seg_json_filepath = os.path.join(seg_subfolder, seg_json_files[i]) if i < len(seg_json_files) else None

                # Create a datapoint object for each corresponding file
                data_point = datapoint(metadata_filepath, pose_filepath, rgb_filepath, seg_png_filepath, seg_json_filepath)
                self.datapoints.append(data_point)

    def get_datapoints(self):
        """Return the list of datapoint objects."""
        return self.datapoints
    
    def get_datapoints_filtered(self):
        """Return the list of filtered datapoint objects."""
        return self.datapoints_filtered 

    def filter_datapoints(self): 
        """Compute the diffusion reflectance and only keep datapoints with positive values."""
        self.datapoints_filtered = [] 
        for dp in self.datapoints:
            dp.compute_diffusion_reflectance() 
            if dp.diffuse_reflection > 0: 
                self.datapoints_filtered.append(dp)

    def split_train_val(self, filter=True, frac_train=0.8):
        """Split the datapoints into training and validation datasets."""
        if filter: 
            self.datapoints_train = random.sample(self.datapoints_filtered, int(frac_train * len(self.datapoints_filtered)))
            self.datapoints_val = [dp for dp in self.datapoints_filtered if dp not in self.datapoints_train]
        else:
            self.datapoints_train = random.sample(self.datapoints, int(frac_train * len(self.datapoints)))
            self.datapoints_val = [dp for dp in self.datapoints if dp not in self.datapoints_train]

    def create_directories(self):
        """Create directories for training and validation data."""
        dir_train = os.path.join(self.out_dir, "train")
        dir_val = os.path.join(self.out_dir, "val")
        dir_train_rgb = os.path.join(dir_train, "rgb")
        dir_train_seg = os.path.join(dir_train, "seg")
        dir_val_rgb = os.path.join(dir_val, "rgb")
        dir_val_seg = os.path.join(dir_val, "seg")

        os.makedirs(dir_train_rgb, exist_ok=True)
        os.makedirs(dir_train_seg, exist_ok=True)
        os.makedirs(dir_val_rgb, exist_ok=True)
        os.makedirs(dir_val_seg, exist_ok=True)

        return dir_train_rgb, dir_train_seg, dir_val_rgb, dir_val_seg

    def preprocess_rgb(self, img_path):  
        """Preprocess RGB image by resizing it."""
        # new_size = (480, 270)  # Define the new size
        img = Image.open(img_path)
        # img_resized = img.resize(new_size)
        img_resized = img # don't resize image 
        return img_resized

    def preprocess_seg_img(self, seg_img_path, seg_json_path, tag_seg_color=None):
        """
        Preprocesses the segmentation image by resizing and converting it to a binary mask based on tag color.
        """
        # Validate that the segmentation image file exists
        if not os.path.exists(seg_img_path):
            raise FileNotFoundError(f"Segmentation image file not found: {seg_img_path}")

        # Validate that the JSON file exists
        if not os.path.exists(seg_json_path):
            raise FileNotFoundError(f"Segmentation JSON file not found: {seg_json_path}")

        # Load the segmentation JSON data if tag_seg_color is not provided
        if tag_seg_color is None:
            with open(seg_json_path, 'r') as json_file:
                seg_json = json.load(json_file)

            # Find the tag color from the JSON data
            for key, val in seg_json.items(): 
                if val.get("class") == "tag0":  
                    # Convert the key (which is a string representing a tuple) into an actual tuple
                    tag_seg_color = tuple(map(int, key.strip('()').split(', ')))  # Convert string '(140, 25, 255, 255)' into a tuple (140, 25, 255, 255)
                    break
            else:
                # raise ValueError("Tag with class 'tag0' not found in JSON.")
                tag_seg_color = tuple([-1,-1,-1,-1]) # impossible color value # FIXME: this is a workaround which can be turned into something more elegant 

        # Load and resize the segmentation image
        seg_img = Image.open(seg_img_path)
        # new_size = (480, 270)
        # seg_img_resized = seg_img.resize(new_size)
        seg_img_resized = seg_img # don't resize 

        # Convert the resized image to a NumPy array
        seg_img_resized = np.array(seg_img_resized)

        # Check if the image is RGB (3 channels) or RGBA (4 channels) or grayscale (1 channel)
        if len(seg_img_resized.shape) == 3:
            if seg_img_resized.shape[2] == 3:  # RGB image
                # Compare each pixel to the tag color (e.g., RGB triplet)
                seg_img_resized = np.all(seg_img_resized == tag_seg_color[:3], axis=-1)  # Create binary mask for RGB image
            elif seg_img_resized.shape[2] == 4:  # RGBA image
                # Compare each pixel to the tag color (RGBA)
                seg_img_resized = np.all(seg_img_resized == tag_seg_color, axis=-1)  # Create binary mask for RGBA image
        else:  # If it's a single channel (grayscale), use it directly
            seg_img_resized = seg_img_resized == tag_seg_color  # Compare pixel values directly

        # Convert the binary mask to uint8 type (0 or 1)
        seg_img_resized = (seg_img_resized).astype(np.uint8) * 255  # Multiply by 255 to match image range

        # Convert the binary mask back to an image
        seg_img_resized = Image.fromarray(seg_img_resized)

        return seg_img_resized

    def save_preprocessed_images(self, frac_train=0.8):
        """Loop through train and val datapoints and save preprocessed images and segmentation masks."""
        dir_train_rgb, dir_train_seg, dir_val_rgb, dir_val_seg = self.create_directories()

        for i, dp in enumerate(self.datapoints_train): 
            img = self.preprocess_rgb(dp.rgb_filepath) 
            seg = self.preprocess_seg_img(dp.seg_png_filepath, dp.seg_json_filepath) 
            img.save(os.path.join(dir_train_rgb, f"img_{i}.png")) 
            seg.save(os.path.join(dir_train_seg, f"seg_{i}.png"))

        for i, dp in enumerate(self.datapoints_val):
            img = self.preprocess_rgb(dp.rgb_filepath) 
            seg = self.preprocess_seg_img(dp.seg_png_filepath, dp.seg_json_filepath) 
            img.save(os.path.join(dir_val_rgb, f"img_{i}.png")) 
            seg.save(os.path.join(dir_val_seg, f"seg_{i}.png"))

    def save_summary_images(self): 
        self.summary_images_dir = os.path.join(self.out_dir, "summary_images") 
        os.makedirs(self.summary_images_dir, exist_ok=True) 

        for idx, dp in enumerate(self.datapoints): 

            # Read data from files
            rgb_image = cv2.imread(dp.rgb_filepath)
            seg_image = cv2.imread(dp.seg_png_filepath, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

            # Check if images are loaded correctly
            if rgb_image is None:
                raise ValueError(f"RGB image at {dp.rgb_filepath} could not be loaded.")
            if seg_image is None:
                raise ValueError(f"Segmentation image at {dp.seg_filepath} could not be loaded.")

            # Convert from BGR (OpenCV default) to RGB (for matplotlib)
            image_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

            # Create a new figure for each image
            plt.figure(figsize=(12, 6))  # Adjust figure size to make space for metadata

            # Subplot for RGB image
            plt.subplot(2, 2, 1)  # 2 row, 2 columns, 1st subplot
            plt.imshow(image_rgb)
            plt.axis('off')  # Hide axes
            plt.title(f'RGB Image {idx}')

            # Subplot for segmentation image
            plt.subplot(2, 2, 2)  # 2 row, 2 columns, 2nd subplot
            plt.imshow(seg_image, cmap='viridis')  # Use a colormap for better visualization
            plt.axis('off')  # Hide axes
            plt.title(f'Segmentation Image {idx}')

            # Subplot for RGB image - keypoints 
            keypoints_image = overlay_points_on_image(image=image_rgb, pixel_points=dp.keypoints_image_space, radius=1) 
            plt.subplot(2, 2, 3)  # 2 row, 2 columns, 1st subplot
            plt.imshow(keypoints_image)
            plt.axis('off')  # Hide axes
            plt.title(f'RGB Image {idx}')

            # Display metadata as text in a separate area 
            metadata_str = dp.__repr__()  

            # Create a new subplot for metadata
            # plt.subplot(2, 2, 4)
            plt.text(1.05, 0.5, metadata_str, fontsize=12, ha='left', va='center', transform=plt.gca().transAxes, 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=1'))

            # Adjust layout to avoid overlap and make space for metadata
            plt.tight_layout()  # Adjust layout
            plt.subplots_adjust(right=0.8)  # Make space for metadata on the right

            # Save the image to the summary_images folder
            save_path = os.path.join(self.summary_images_dir, f"summary_image_{idx}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)  # Save with high resolution
            plt.close()  # Close the plot to free up memory 

            if len(self.datapoints) < 10 or (idx + 1) % (len(self.datapoints) // 10) == 0:
                print(f"Saved image {idx} to {save_path}") 

    def set_camera_calibration(self, camera_matrix=None, distortion_coefficients=None):
        """Set the camera calibration parameters for the DataProcessor."""
        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients
        if self.camera_matrix is None: 
            # assume isaac sim default 
            width = 640 
            height = 480 
            focal_length = 24.0 
            horiz_aperture = 20.955
            # Pixels are square so we can do:
            vert_aperture = height/width * horiz_aperture
            fov = 2 * math.atan(horiz_aperture / (2 * focal_length))
            # compute focal point and center
            fx = width * focal_length / horiz_aperture
            fy = height * focal_length / vert_aperture
            cx = width / 2
            cy = height /2 

            self.camera_matrix = np.array([
                [fx,0,cx],
                [0,fy,cy],
                [0,0,1]
            ])
        if self.distortion_coefficients is None: 
            # assume no distortion  
            self.distortion_coefficients = np.array([0,0,0,0,0])  # Assuming no distortion for simplicity 

    def set_marker(self, marker_image_filepath=None, marker_side_length=0.1): 
        self.marker_image = cv2.imread(marker_image_filepath)
        self.marker_side_length = marker_side_length  # Set the marker length (in meters) for pose estimation 
        marker_corners_3d = [
            [+marker_side_length/2,+marker_side_length/2,0], 
            [-marker_side_length/2,+marker_side_length/2,0], 
            [-marker_side_length/2,-marker_side_length/2,0], 
            [+marker_side_length/2,-marker_side_length/2,0]
        ] 
        self.marker_corners_3d = np.array(marker_corners_3d, dtype=np.float32)  
        # self.marker_inner_corners_3d = self.marker_corners_3d * 0.8 
        self.marker_inner_corners_3d = self.marker_corners_3d * 1.0 

    def run_classical_marker_pose_estimation(self, save_pose=False, save_image=False): 
        if save_pose: 
            dir_save_pose = os.path.join(self.out_dir, "classical_CV_pose") 
            os.makedirs(dir_save_pose, exist_ok=True) 
        if save_image: 
            dir_save_image = os.path.join(self.out_dir, "classical_CV_bbox") 
            os.makedirs(dir_save_image, exist_ok=True) 

        for idx, dp in enumerate(self.datapoints): 
            if dp.rgb is not None and self.camera_matrix is not None and self.distortion_coefficients is not None:
                # Perform classical marker pose estimation
                marker_ids, rotation_vectors, translation_vectors, corners = classical_marker_pose_estimation(
                    image=cv2.imread(dp.rgb_filepath), 
                    camera_matrix=self.camera_matrix, 
                    dist_coeffs=self.distortion_coefficients, 
                    # aruco_dict=cv2.aruco.DICT_APRILTAG_36h11, 
                    aruco_dict=cv2.aruco.DICT_4X4_50    ,
                    marker_length=0.080,  # 80 mm for black edges, 100 mm for white edges 
                )

                # compute transform 
                # tf_CCV = np.eye(4)  # Initialize as identity matrix 
                tf_Ccv_Mcv = np.eye(4)  # marker in openCV coordinates wrt openCV camera coordinates 
                if rotation_vectors is None or translation_vectors is None:
                    # tf_CCV = None 
                    tf_W_Mis = None 
                    corners = None 
                else: 
                    tf_Ccv_Mcv[:3,:3] = cv2.Rodrigues(rotation_vectors[0])[0] # Convert rotation vector to rotation matrix
                    tf_Ccv_Mcv[:3,3] = translation_vectors[0].reshape(3)  # Set translation vector 
                    
                    # # DEBUG 
                    # # correction 
                    # tf_correction = np.array([
                    #     [1,0,0,0],
                    #     [0,-1,0,0],
                    #     [0,0,-1,0],
                    #     [0,0,0,1]
                    # ])
                    # tf_CCV = tf_correction @ tf_CCV # apply correction 
                    # tf_correction_2 = np.array([
                    #     [-1,0,0,0],
                    #     [0,1,0,0],
                    #     [0,0,-1,0],
                    #     [0,0,0,1]
                    # ]) 
                    # tf_CCV = tf_CCV @ tf_correction_2 # apply correction 
                    # # DEBUG 

                    tf_W_Mis = self.tf_W_Ccv @ tf_Ccv_Mcv @ self.T_Mcv_Mis 

                # Store the results in the datapoint object
                self.datapoints[idx].CCV_marker_ids = marker_ids
                # self.datapoints[idx].CCV_tag_pose = tf_CCV 
                self.datapoints[idx].CCV_tag_pose = tf_W_Mis  
                if corners is not None:
                    self.datapoints[idx].CCV_corners = corners[0][0] # note that these are inner (black) corners 
                else: 
                    self.datapoints[idx].CCV_corners = None
            else:
                print(f"Skipping pose estimation for datapoint {dp} due to missing data.") 
            
            if save_pose: 
                # Save the pose data to a file (e.g., JSON)
                filename = os.path.basename(dp.rgb_filepath).replace(".png","")  # Extract the filename from the path 
                pose_data = {
                    "marker_ids": convert_to_serializable(dp.CCV_marker_ids),
                    # "rotation_vectors": convert_to_serializable(rotation_vectors),
                    # "translation_vectors": convert_to_serializable(translation_vectors)
                    "tag_pose": convert_to_serializable(dp.CCV_tag_pose), 
                }
                with open(os.path.join(dir_save_pose, f"{filename}_pose.json"), 'w') as f:
                    json.dump(pose_data, f) 

            if save_image: 
                # Save the image with bounding boxes drawn on it
                rgb_image = cv2.imread(dp.rgb_filepath)
                if corners is None: 
                    # save original rgb image to dir_save_image 
                    save_path = os.path.join(dir_save_image, f"{filename}_bbox.png")
                    cv2.imwrite(save_path, rgb_image) 
                    continue 
                for i, corner in enumerate(corners): 
                    if i in marker_ids: 
                        # Draw the bounding box on the image
                        cv2.polylines(rgb_image, [np.int32(corner)], isClosed=True, color=(0, 255, 0), thickness=2) 
                # Save the image with bounding boxes 
                save_path = os.path.join(dir_save_image, f"{filename}_bbox.png") 
                cv2.imwrite(save_path, rgb_image) 

        # compute fraction of successful detections 
        num_success = 0 
        for dp in self.datapoints: 
            if dp.CCV_marker_ids is not None: 
                num_success += len(dp.CCV_marker_ids) 
        num_total = len(self.datapoints) 
        self.CCV_detection_fraction = num_success / num_total         
        print(f"CCV: Fraction of successful detections: {self.CCV_detection_fraction:.2f} ({num_success}/{num_total})")

    def compute_CCV_pose_error(self): 
        print(f"\n")
        for idx, dp in enumerate(self.datapoints): 
            if dp.CCV_tag_pose is not None: 
                # compute pose error between CCV and true tag pose 
                tf_true = dp.tag_pose 
                tf_CCV = dp.CCV_tag_pose
                                
                CCV_tf_error = np.linalg.inv(tf_true) @ tf_CCV 
                self.datapoints[idx].CCV_tf_error = CCV_tf_error  
                # compute error in translation and rotation 
                xyz_error = CCV_tf_error[:3,3]  # translation error 
                abc_error = R.from_matrix(CCV_tf_error[:3,:3]).as_euler('xyz', degrees=True)  
                self.datapoints[idx].CCV_pose_error = np.hstack((xyz_error, abc_error))  # concatenate translation and rotation error 
                
                # compute errors in corner prediction 
                self.datapoints[idx].CCV_corner_error = closest_point_distances(dp.CCV_corners, dp.inner_corners_image_space)  

            else: 
                self.datapoints[idx].CCV_tf_error = None 
                self.datapoints[idx].CCV_pose_error = None 
                self.datapoints[idx].CCV_corner_error = None 
        
        # compute mean and std of pose error 
        
        self.CCV_pose_errors = np.array([dp.CCV_pose_error for dp in self.datapoints if dp.CCV_pose_error is not None]) # shape: (N images,6)
        self.CCV_corner_errors = np.array([dp.CCV_corner_error for dp in self.datapoints if dp.CCV_corner_error is not None]) # shape: (N images, 4 corners) 

        self.CCV_pose_error_MAE = np.mean(np.abs(self.CCV_pose_errors), axis=0)  
        self.CCV_pose_error_std = np.std(self.CCV_pose_errors, axis=0) 
        print(f"CCV: Pose MAE: {self.CCV_pose_error_MAE}") 
        print(f"CCV: Pose error Std: {self.CCV_pose_error_std}") 
        self.CCV_MAE_position_error = np.mean(np.abs(self.CCV_pose_errors[:,:3]), axis=1) 
        self.CCV_MAE_rotation_error = np.mean(np.abs(self.CCV_pose_errors[:,3:]), axis=1) 

        # compute covariance matrix of pose error
        self.CCV_pose_error_cov = np.cov(self.CCV_pose_errors.T)  # Transpose to get correct shape
        # print(f"CCV: Covariance matrix of pose error: {self.CCV_pose_error_cov}")

        # compute mean corner error 
        self.CCV_corner_error_MAE = np.mean(np.abs(self.CCV_corner_errors)) # shape: (1,)
        self.CCV_MAE_corner_error = np.mean(np.abs(self.CCV_corner_errors), axis=1) # shape: (N images,)  
        print(f"CCV: corner error MAE: {self.CCV_corner_error_MAE}") 

        # save plot of mean position error vs mean corner error 
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))  
        axs[0].scatter(self.CCV_MAE_corner_error, self.CCV_MAE_position_error) 
        axs[0].set_title("Mean Position Error vs Mean Corner Error") 
        axs[0].set_xlabel("Mean Corner Error") 
        axs[0].set_ylabel("Mean Position Error") 
        axs[0].grid(True) 
        axs[1].scatter(self.CCV_MAE_corner_error, self.CCV_MAE_rotation_error) 
        axs[1].set_title("Mean Rotation Error vs Mean Corner Error") 
        axs[1].set_xlabel("Mean Corner Error") 
        axs[1].set_ylabel("Mean Rotation Error") 
        axs[1].grid(True) 

        # save figure 
        fig.savefig(os.path.join(self.out_dir, "CCV_pose_error_vs_corner_error.png"), dpi=300) 

        # # save violin plots of pose error distributions for each dimension (x,y,z,a,b,c) 
        # errors = [dp.CCV_pose_error for dp in self.datapoints if dp.CCV_pose_error is not None] 
        # if len(errors) == 0: 
        #     print("No pose errors to plot.")
        #     return 
        # errors = np.array(errors)  # Convert to a NumPy array for easier manipulation 
        # fig, axs = plt.subplots(1, 6, figsize=(20, 5))  # Create subplots for each dimension 
        # labels = ['x', 'y', 'z', 'a', 'b', 'c']  # Labels for each dimension 
        # for i in range(6):  # Iterate over each dimension 
        #     axs[i].violinplot(errors[:, i], showmeans=True)  # Create a violin plot for the current dimension
        #     axs[i].set_title(f'Pose Error Distribution - {labels[i]}')  # Set the title for the subplot
        #     axs[i].set_xlabel(labels[i])  # Set the x-label for the subplot
        #     axs[i].set_ylabel('Error')  # Set the y-label for the subplot
        #     axs[i].grid(True)  # Add grid for better readability
        # plt.tight_layout()  # Adjust layout to prevent overlap
        # plt.savefig(os.path.join(self.out_dir, "CCV_pose_error_distribution.png"), dpi=300)  # Save the figure as an image
        # plt.close()

        # Collect the pose errors for each data point, ensuring there are no None values
        errors = [dp.CCV_pose_error for dp in self.datapoints if dp.CCV_pose_error is not None] 
        
        # Check if there are errors to plot
        if len(errors) == 0: 
            print("No pose errors to plot.")
            return 
        
        # Convert errors to a NumPy array for easier manipulation
        errors = np.array(errors)  
        
        # Create subplots for each dimension (2 rows, 3 columns)
        fig, axs = plt.subplots(2, 3, figsize=(20, 10))  # 2x3 grid for the violin plots
        labels = ['x', 'y', 'z', 'a', 'b', 'c']  # Labels for each dimension 

        # Flatten the axes array to make it easier to iterate over
        axs = axs.flatten()  # Now axs will be a 1D array with 6 elements
        
        # Iterate over each dimension (6 in total)
        for i in range(6):
            axs[i].violinplot(errors[:, i], showmeans=True)  # Create a violin plot for the current dimension
            axs[i].set_title(f'Pose Error Distribution - {labels[i]}')  # Set the title for the subplot
            axs[i].set_xlabel(labels[i])  # Set the x-label for the subplot
            axs[i].set_ylabel('Error')  # Set the y-label for the subplot
            axs[i].grid(True)  # Add grid for better readability
        
        # Adjust layout to prevent overlap
        plt.tight_layout()  
        
        # Save the figure as an image
        plt.savefig(os.path.join(self.out_dir, "CCV_pose_error_distribution.png"), dpi=300)  
        
        # Close the plot to free memory
        plt.close()

    def run_learning_based_segmentation(self):
        # TODO: add in later, for now just read data 
        pass 

    def read_segmentation_predictions(self, seg_filepath):
        for idx, dp in enumerate(self.datapoints): 
            self.datapoints[idx].LBCV_seg_filepath = os.path.join(seg_filepath, os.path.basename(dp.rgb_filepath).replace(".png","_prediction.png"))  # Extract the filename from the path 

    def compute_segmentation_IOU(self): 
        # compute the segmentation accuracy for each datapoint 
        for idx, dp in enumerate(self.datapoints): 
            try: 
                dp.LBCV_seg_filepath 
            except: 
                import pdb; pdb.set_trace() 

            if (dp.seg_png is not None) and (dp.LBCV_seg_filepath is not None): 
                # Read the segmentation mask and ground truth mask
                seg_mask = np.array(self.preprocess_seg_img(seg_img_path=dp.seg_png_filepath, seg_json_path=dp.seg_json_filepath))  # Load the segmentation mask 
                pred_mask = cv2.imread(dp.LBCV_seg_filepath, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
                if pred_mask is None: 
                    print(f"Segmentation prediction not found for datapoint {idx}.") 
                    continue
                self.datapoints[idx].LBCV_seg_IOU = compute_segmentation_IOU(segmentation_mask=seg_mask, ground_truth_mask=pred_mask)  # Compute the IoU score 
            else: 
                self.datapoints[idx].LBCV_seg_IOU = None  # Set to None if either mask is not available 

        # compute mean IOU 
        self.LBCV_mean_IOU = np.mean([dp.LBCV_seg_IOU for dp in self.datapoints if dp.LBCV_seg_IOU is not None]) 
        self.LBCV_std_IOU = np.std([dp.LBCV_seg_IOU for dp in self.datapoints if dp.LBCV_seg_IOU is not None]) 
        print(f"LBCV: Mean segmentation IOU: {self.LBCV_mean_IOU:.2f}") 
        print(f"LBCV: Std segmentation IOU: {self.LBCV_std_IOU:.2f}") 

        # loop through datapoints and compute fraction of successful detections 
        detection_IOU_threshold = 0.5  # Define a threshold for successful detection 
        num_success = 0 
        for dp in self.datapoints: 
            if dp.LBCV_seg_IOU is not None and dp.LBCV_seg_IOU >= detection_IOU_threshold: 
                num_success += 1 
        num_total = len(self.datapoints)
        self.LBCV_detection_fraction = num_success / num_total 
        print(f"LBCV: Fraction of successful detections: {self.LBCV_detection_fraction:.2f} ({num_success}/{num_total})") 

    def image_overlap_error(self, img, img_mask, pred, pred_mask): 

        filter = "mean_threshold" # "min_max", "local", "mean_threshold"

        img = np.array(img) 
        if img.shape[2] == 4:   
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB) 
        img_mask = np.array(img_mask) 
        pred = np.array(pred) 
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY) # convert pred to grayscale 
        _, pred_mask = cv2.threshold(pred, 127, 255, cv2.THRESH_BINARY)  # Threshold to create a binary mask
        pred_mask = np.array(pred_mask) 

        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    

        binary_mask = img_mask.copy() 
        binary_mask[binary_mask > 0] = 255 

        masked_pixels = img[img_mask == 255]

        if filter == "min_max": 
            # Find the minimum and maximum pixel values in the masked region
            min_pixel_value = np.min(masked_pixels)
            max_pixel_value = np.max(masked_pixels)

            # normalize img using range of img_mask 
            img_dot_mask = cv2.bitwise_and(img, img_mask)     
            img_filtered = cv2.normalize(img_dot_mask, None, min_pixel_value, max_pixel_value, cv2.NORM_MINMAX) 
        elif filter == "local":
            # TODO: localized regions normalization 
            pass 
        elif filter == "mean_threshold": 
            img_masked_mean = masked_pixels.mean() 
            # threshold img using img_masked_mean as threshold 
            _, img_filtered = cv2.threshold(img_grey, img_masked_mean, 255, cv2.THRESH_BINARY) 
            img_filtered = np.array(img_filtered)

        intersection = cv2.bitwise_and(img_mask, pred_mask) 
        if len(intersection.shape) == 3:  # In case the image is a color image (unexpected)
            intersection = np.mean(intersection, axis=2)  # Convert to grayscale by averaging the color channels             

        # for each pixel of intersection, compute error between img and pred 
        error = 0 
        for i in range(intersection.shape[0]):
            for j in range(intersection.shape[1]):
                if intersection[i][j] == 255:

                    # error += np.linalg.norm(img_filtered[i][j] - pred[i][j]) 
                    
                    if img_filtered[i][j] == pred[i][j]: 
                        error -= 1 
                    
                    # error += np.sum((img_grey[i][j] - pred[i][j]) ** 2)
        return error 

    def run_optimization_pose_estimation(self, save_image=True): 
        # create directory for saving reprojected marker images 
        dir_save_reprojected = os.path.join(self.out_dir, "reprojected_marker_images") 
        os.makedirs(dir_save_reprojected, exist_ok=True) 

        num_no_corners_detected = 0 
        num_not_4_corners_detected = 0 
        num_corner_on_edge = 0  

        for idx, dp in enumerate(self.datapoints): 
            
            self.datapoints[idx].LBCV_tag_pose = None  
            self.datapoints[idx].LBCV_corners = None  

            # read segmentation image 
            seg_pred = cv2.imread(dp.LBCV_seg_filepath)
            seg_pred_np = np.array(seg_pred)  
            img = cv2.imread(dp.rgb)  

            if len(seg_pred_np.shape) == 3:  # In case the image is a color image (unexpected)
                seg_pred = cv2.cvtColor(seg_pred, cv2.COLOR_BGR2GRAY)
                seg_pred_np = np.array(seg_pred)  
            
            marker_corners_2d = detect_corners(seg_pred_np)  # Detect corners in the segmentation mask
            
            if (marker_corners_2d is None): 
                num_no_corners_detected += 1
                continue 

            # if any marker corner is on the edge of the image, skip this datapoint 
            edge_tol = 3 
            if np.any(marker_corners_2d[:,0] <= edge_tol) or np.any(marker_corners_2d[:,1] <= edge_tol) or np.any(marker_corners_2d[:,0] >= img.shape[1]-edge_tol) or np.any(marker_corners_2d[:,1] >= img.shape[0]-edge_tol):
                num_corner_on_edge += 1
                continue 

            if (marker_corners_2d.shape[0] != 4): 
                num_not_4_corners_detected += 1 
                import pdb; pdb.set_trace() 
                continue 


            # TODO: output image of detected corners on the segmentation mask 

            # loop through all permutations of the corners and compute the overlap error for all, return pose of lowest overlap error 
            for i in range(2): 
                marker_corners_2d = np.flip(marker_corners_2d, axis=0) # flip marker_corners_2d
                for j in range(4): 
                    # roll marker_corners_2d 
                    marker_corners_2d = np.roll(marker_corners_2d, 1, axis=0) 

                    # use PnP algorithm to determine pose of parallelogram 
                    dist_coeffs = None 

                    # Solve PnP to get the rotation and translation vectors
                    try: 
                        success, rotation_vector, translation_vector = cv2.solvePnP(
                            self.marker_corners_3d,  # 3D points in world space
                            np.array(marker_corners_2d, dtype=np.float32),  # Corresponding 2D points in image space
                            self.camera_matrix,      # Camera matrix (intrinsics)
                            dist_coeffs         # Distortion coefficients (can be None if no distortion)
                        )
                    except cv2.error as e:
                        print(f"OpenCV error during solvePnP: {e}")

                    # project marker 
                    # reprojected_marker_image, reprojected_marker_mask, blended_image = marker_reprojection(img, seg_pred_np, self.marker_image, marker_corners_2d, self.marker_corners_3d, rotation_vector, translation_vector, self.camera_matrix, dist_coeffs, 0.8)
                    reprojected_marker_image, reprojected_marker_mask, blended_image = marker_reprojection(img, seg_pred_np, self.marker_image, marker_corners_2d, self.marker_corners_3d, rotation_vector, translation_vector, self.camera_matrix, dist_coeffs, 1.0)
                    
                    reprojection_error = self.image_overlap_error(img, seg_pred_np, reprojected_marker_image, reprojected_marker_mask)  

                    # define solution as minimum reprojection error 
                    if (i == 0 and j == 0) or (reprojection_error < min_reprojection_error): 
                        min_reprojection_error = reprojection_error 
                        rotation_sol = rotation_vector 
                        translation_sol = translation_vector
                        marker_corners_2d_sol = marker_corners_2d 
                        blended_image_sol = blended_image 
                    
            tf_Ccv_Mcv = np.eye(4)  # marker in openCV coordinates wrt openCV camera coordinates
            tf_Ccv_Mcv[:3,:3] = cv2.Rodrigues(rotation_sol)[0]  # Convert rotation vector to rotation matrix 
            tf_Ccv_Mcv[:3,3] = translation_sol.reshape(3)  # Set translation vector 
            tf_W_Mis = self.tf_W_Ccv @ tf_Ccv_Mcv @ self.T_Mcv_Mis 

            self.datapoints[idx].LBCV_tag_pose = tf_W_Mis 
            self.datapoints[idx].LBCV_corners = marker_corners_2d_sol 

            # save blended image to dir_save_reprojected 
            if save_image:
                blended_image_sol = cv2.cvtColor(blended_image_sol, cv2.COLOR_BGR2RGB) 
                # save blended image to dir_save_reprojected
                save_path = os.path.join(dir_save_reprojected, f"reprojected_image_{idx}.png")
                cv2.imwrite(save_path, blended_image_sol)

            # print progress every 10% 
            if len(self.datapoints) < 10 or (idx + 1) % (len(self.datapoints) // 10) == 0:
                print(f"Processed {idx+1}/{len(self.datapoints)} datapoints for LBCV pose estimation.") 

        print(f"Number of datapoints with no corners detected: {num_no_corners_detected}") 
        print(f"Number of datapoints with not 4 corners detected: {num_not_4_corners_detected}") 
        print(f"Number of datapoints with corners on edge: {num_corner_on_edge}") 

    def run_segmentation_pose_estimate(self): 
        # given true segmentation, compute pose as a basline reference for error 
        
        num_no_corners_detected = 0 
        num_not_4_corners_detected = 0 
        num_corner_on_edge = 0 

        for idx, dp in enumerate(self.datapoints):   
            self.datapoints[idx].SEG_tag_pose = None  
            self.datapoints[idx].SEG_corners = None  

            # read segmentation image 
            # seg_pred = cv2.imread(dp.seg_png_filepath) # NOTE: read true segmentation image instead of LBCV prediction
            seg_pred = self.preprocess_seg_img(seg_img_path=dp.seg_png_filepath, seg_json_path=dp.seg_json_filepath)  # Load the segmentation mask
            seg_pred_np = np.array(seg_pred)  
            img = cv2.imread(dp.rgb)  

            if len(seg_pred_np.shape) == 3:  # In case the image is a color image (unexpected)
                seg_pred = cv2.cvtColor(seg_pred, cv2.COLOR_BGR2GRAY)
                seg_pred_np = np.array(seg_pred)  
            
            marker_corners_2d = detect_corners(seg_pred_np)  # Detect corners in the segmentation mask 

            if (marker_corners_2d is None): 
                num_no_corners_detected += 1 
                continue 
            
            # if any marker corner is on the edge of the image, skip this datapoint 
            edge_tol = 10 
            if np.any(marker_corners_2d[:,0] <= edge_tol) or np.any(marker_corners_2d[:,1] <= edge_tol) or np.any(marker_corners_2d[:,0] >= img.shape[1]-edge_tol) or np.any(marker_corners_2d[:,1] >= img.shape[0]-edge_tol):
                num_corner_on_edge += 1 
                continue 

            if (marker_corners_2d.shape[0] != 4): 
                num_not_4_corners_detected += 1 
                continue


            # TODO: output image of detected corners on the segmentation mask 

            # loop through all permutations of the corners and compute the overlap error for all, return pose of lowest overlap error 
            for i in range(2): 
                marker_corners_2d = np.flip(marker_corners_2d, axis=0) # flip marker_corners_2d
                for j in range(4): 
                    # roll marker_corners_2d 
                    marker_corners_2d = np.roll(marker_corners_2d, 1, axis=0) 

                    # use PnP algorithm to determine pose of parallelogram 
                    dist_coeffs = None 

                    # Solve PnP to get the rotation and translation vectors
                    try: 
                        success, rotation_vector, translation_vector = cv2.solvePnP(
                            self.marker_corners_3d,  # 3D points in world space
                            np.array(marker_corners_2d, dtype=np.float32),  # Corresponding 2D points in image space
                            self.camera_matrix,      # Camera matrix (intrinsics)
                            dist_coeffs         # Distortion coefficients (can be None if no distortion)
                        )
                    except cv2.error as e:
                        print(f"OpenCV error during solvePnP: {e}")

                    # project marker 
                    # reprojected_marker_image, reprojected_marker_mask, blended_image = marker_reprojection(img, seg_pred_np, self.marker_image, marker_corners_2d, self.marker_corners_3d, rotation_vector, translation_vector, self.camera_matrix, dist_coeffs, 0.8)
                    reprojected_marker_image, reprojected_marker_mask, blended_image = marker_reprojection(img, seg_pred_np, self.marker_image, marker_corners_2d, self.marker_corners_3d, rotation_vector, translation_vector, self.camera_matrix, dist_coeffs, 1.0)
                    
                    reprojection_error = self.image_overlap_error(img, seg_pred_np, reprojected_marker_image, reprojected_marker_mask)  

                    # define solution as minimum reprojection error 
                    if (i == 0 and j == 0) or (reprojection_error < min_reprojection_error): 
                        min_reprojection_error = reprojection_error 
                        rotation_sol = rotation_vector 
                        translation_sol = translation_vector
                        marker_corners_2d_sol = marker_corners_2d 

            # self.datapoints[idx].SEG_tag_pose = np.eye(4) 
            # self.datapoints[idx].SEG_tag_pose[:3,:3] = cv2.Rodrigues(rotation_sol)[0]  # Convert rotation vector to rotation matrix
            # self.datapoints[idx].SEG_tag_pose[:3,3] = translation_sol.reshape(3)  # Set translation vector 
            
            tf_Ccv_Mcv = np.eye(4)  # marker in openCV coordinates wrt openCV camera coordinates
            tf_Ccv_Mcv[:3,:3] = cv2.Rodrigues(rotation_sol)[0]  # Convert rotation vector to rotation matrix 
            tf_Ccv_Mcv[:3,3] = translation_sol.reshape(3)  # Set translation vector 
            tf_W_Mis = self.tf_W_Ccv @ tf_Ccv_Mcv @ self.T_Mcv_Mis 

            self.datapoints[idx].SEG_tag_pose = tf_W_Mis 
            self.datapoints[idx].SEG_corners = marker_corners_2d_sol 

            # print progress every 10% 
            if len(self.datapoints) < 10 or (idx + 1) % (len(self.datapoints) // 10) == 0:
                print(f"Processed {idx+1}/{len(self.datapoints)} datapoints for LBCV pose estimation.") 

    def compute_LBCV_pose_error(self): 
        print(f"\n")
        for idx, dp in enumerate(self.datapoints): 
            if dp.LBCV_tag_pose is not None: 
                # compute pose error between LBCV and GT tag pose 
                tf_true = dp.tag_pose 
                tf_LBCV = dp.LBCV_tag_pose
                LBCV_tf_error = np.linalg.inv(tf_true) @ tf_LBCV 
                self.datapoints[idx].LBCV_tf_error = LBCV_tf_error  
                # compute error in translation and rotation 
                xyz_error = LBCV_tf_error[:3,3]  # translation error 
                abc_error = R.from_matrix(LBCV_tf_error[:3,:3]).as_euler('xyz', degrees=True)  
                self.datapoints[idx].LBCV_pose_error = np.hstack((xyz_error, abc_error))  # concatenate translation and rotation error 
                self.datapoints[idx].LBCV_corner_error = closest_point_distances(dp.LBCV_corners, dp.outter_corners_image_space)  # compute corner error 
            else: 
                self.datapoints[idx].LBCV_pose_error = None 
                self.datapoints[idx].LBCV_corner_error = None 
        
        # compute mean and std of pose error 
        self.LBCV_pose_error_mean = np.mean([dp.LBCV_pose_error for dp in self.datapoints if dp.LBCV_pose_error is not None], axis=0) 
        self.LBCV_pose_error_std = np.std([dp.LBCV_pose_error for dp in self.datapoints if dp.LBCV_pose_error is not None], axis=0) 
        print(f"Mean pose error: {self.LBCV_pose_error_mean}") 
        print(f"Std pose error: {self.LBCV_pose_error_std}") 

        # compute covariance matrix of pose error
        self.LBCV_pose_error_cov = np.cov(np.array([dp.LBCV_pose_error for dp in self.datapoints if dp.LBCV_pose_error is not None]).T)  # Transpose to get correct shape
        print(f"LBCV: Covariance matrix of pose error: {self.LBCV_pose_error_cov}")

        # compute corners error 
        self.LBCV_corner_errors = np.array([dp.LBCV_corner_error for dp in self.datapoints if dp.LBCV_corner_error is not None]) # shape: (N images, 4 corners)
        self.LBCV_corner_error_MAE = np.mean(np.abs(self.LBCV_corner_errors)) # shape: (1,) 
        self.LBCV_MAE_corner_error = np.mean(self.LBCV_corner_errors, axis=1) # shape: (N images,) 
        print(f"LBCV: corner error MAE: {self.LBCV_corner_error_MAE}") 

        # save violin plots of pose error distributions for each dimension (x,y,z,a,b,c) 
        errors = [dp.LBCV_pose_error for dp in self.datapoints if dp.LBCV_pose_error is not None] 
        if len(errors) == 0: 
            print("No pose errors to plot.")
            return 
        errors = np.array(errors)  # Convert to a NumPy array for easier manipulation 
        fig, axs = plt.subplots(1, 6, figsize=(20, 5))  # Create subplots for each dimension 
        labels = ['x', 'y', 'z', 'a', 'b', 'c']  # Labels for each dimension 
        for i in range(6):  # Iterate over each dimension 
            axs[i].violinplot(errors[:, i], showmeans=True)  # Create a violin plot for the current dimension
            axs[i].set_title(f'Pose Error Distribution - {labels[i]}')  # Set the title for the subplot
            axs[i].set_xlabel(labels[i])  # Set the x-label for the subplot
            axs[i].set_ylabel('Error')  # Set the y-label for the subplot
            axs[i].grid(True)  # Add grid for better readability
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.savefig(os.path.join(self.out_dir, "LBCV_pose_error_distribution.png"), dpi=300)  # Save the figure as an image
        plt.close()

    def compute_segmentation_pose_error(self): 
        print(f"\n")
        for idx, dp in enumerate(self.datapoints): 
            self.datapoints[idx].SEG_tf_error = None 
            self.datapoints[idx].SEG_pose_error = None
            self.datapoints[idx].SEG_corner_error = None 

            if dp.SEG_tag_pose is not None: 
                # compute pose error between segmentation and GT tag pose 
                tf_true = dp.tag_pose 
                tf_seg = dp.SEG_tag_pose
                seg_tf_error = np.linalg.inv(tf_true) @ tf_seg 
                self.datapoints[idx].SEG_tf_error = seg_tf_error  
                # compute error in translation and rotation 
                xyz_error = seg_tf_error[:3,3]
                # translation error
                abc_error = R.from_matrix(seg_tf_error[:3,:3]).as_euler('xyz', degrees=True)
                # rotation error
                self.datapoints[idx].SEG_pose_error = np.hstack((xyz_error, abc_error))  # concatenate translation and rotation error
                self.datapoints[idx].SEG_corner_error = closest_point_distances(dp.SEG_corners, dp.outter_corners_image_space)  # compute corner error

        # compute mean and std of pose error
        self.SEG_pose_errors = np.array([dp.SEG_pose_error for dp in self.datapoints if dp.SEG_pose_error is not None])
        self.SEG_pose_error_MAE = np.mean(np.abs(self.SEG_pose_errors), axis=0)
        self.SEG_pose_error_std = np.std(self.SEG_pose_errors, axis=0)
        print(f"SEG: MAE pose error: {self.SEG_pose_error_MAE}")
        print(f"SEG: Std pose error: {self.SEG_pose_error_std}")

        # compute covariance matrix of pose error
        self.SEG_pose_error_cov = np.cov(np.array(self.SEG_pose_errors).T)  # Transpose to get correct shape
        print(f"SEG: Covariance matrix of pose error: {self.SEG_pose_error_cov}")

        # compute corners error 
        self.SEG_corner_errors = np.array([dp.SEG_corner_error for dp in self.datapoints if dp.SEG_corner_error is not None]) # shape: (N images, 4 corners)
        self.SEG_corner_error_MAE = np.mean(np.abs(self.SEG_corner_errors)) # shape: (1,) 
        self.SEG_MAE_corner_error = np.mean(self.SEG_corner_errors, axis=1) # shape: (N images,) 
        print(f"SEG: MAE corner error: {self.SEG_corner_error_MAE}") 

        # save violin plots of pose error distributions for each dimension (x,y,z,a,b,c)
        errors = self.SEG_pose_errors
        if len(errors) == 0:
            print("No pose errors to plot.")
            return
        errors = np.array(errors)  # Convert to a NumPy array for easier manipulation
        fig, axs = plt.subplots(1, 6, figsize=(20, 5))  # Create subplots for each dimension
        labels = ['x', 'y', 'z', 'a', 'b', 'c']
        # Labels for each dimension
        for i in range(6):  # Iterate over each dimension
            axs[i].violinplot(errors[:, i], showmeans=True)  # Create a violin plot for the current dimension
            axs[i].set_title(f'Pose Error Distribution - {labels[i]}')
            # Set the title for the subplot
            axs[i].set_xlabel(labels[i])  # Set the x-label for the subplot
            axs[i].set_ylabel('Error')  # Set the y-label for the subplot
            axs[i].grid(True)  # Add grid for better readability
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.savefig(os.path.join(self.out_dir, "segmentation_pose_error_distribution.png"), dpi=300)  # Save the figure as an image
        plt.close()  # Close the plot to free up memory

    def compare_pose_estimation_methods(self): 
        """Compare the pose estimation methods by plotting the pose error distributions.""" 
        # create analysis directory 
        analysis_dir = os.path.join(self.out_dir, "analysis") 
        os.makedirs(analysis_dir, exist_ok=True)
        
        # create a bar chart comparing the fraction of successful detections for each method
        detection_fractions = [self.CCV_detection_fraction, self.LBCV_detection_fraction]
        methods = ["Classical CV", "Learning-Based CV"]  # Labels for the methods
        plt.figure(figsize=(12, 6))
        plt.bar(methods, detection_fractions, color=['blue', 'orange'])
        plt.xlabel('Method')
        plt.ylabel('Fraction of Successful Detections')
        plt.title('Comparison of Detection Fractions')
        plt.ylim(0, 1)  # Set y-axis limit to [0, 1]
        plt.grid(axis='y')  # Add grid lines for better readability
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.savefig(os.path.join(analysis_dir, "detection_fraction_comparison.png"), dpi=300)  # Save the figure as an image
        plt.close()  # Close the plot to free up memory
        
        # get non-None pose errors for CCV and LBCV 
        CCV_pose_errors = [dp.CCV_pose_error for dp in self.datapoints if dp.CCV_pose_error is not None] 
        LBCV_pose_errors = [dp.LBCV_pose_error for dp in self.datapoints if dp.LBCV_pose_error is not None] 
        SEG_pose_errors = [dp.SEG_pose_error for dp in self.datapoints if dp.SEG_pose_error is not None] 
        if len(CCV_pose_errors) == 0 or len(LBCV_pose_errors) == 0:
            print("No pose errors to compare.")
            return
        
        # create violin plots of (x,y,z,a,b,c) prediction errors comparing CCV and LBCV side by side 
        fig, axs = plt.subplots(1, 6, figsize=(30, 5))  # Create subplots for each dimension
        labels = ['x', 'y', 'z', 'a', 'b', 'c']  # Labels for each dimension
        for i in range(6):  # Iterate over each dimension
            # Create a violin plot for the current dimension
            axs[i].violinplot([
                                np.array(CCV_pose_errors)[:, i], 
                                np.array(LBCV_pose_errors)[:, i],
                                np.array(SEG_pose_errors)[:, i] if SEG_pose_errors else []  # Add segmentation errors if available
                            ], showmeans=True, showmedians=True)
            axs[i].set_title(f'Pose Error Distribution - {labels[i]}')  # Set the title for the subplot
            axs[i].set_xlabel(labels[i])  # Set the x-label for the subplot
            axs[i].set_ylabel('Error')  # Set the y-label for the subplot
            axs[i].set_xticks([1, 2, 3])  # Set x-ticks to match the number of methods
            axs[i].set_xticklabels(['CCV', 'LBCV', 'SEG'])  # Set x-tick labels to method names
            axs[i].grid(True)  # Add grid for better readability
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.savefig(os.path.join(analysis_dir, "pose_error_distribution_comparison.png"), dpi=300)  # Save the figure as an image
        plt.close()  # Close the plot to free up memory

        # create a bar chart comparing the corner errors for CCV, LBCV, and SEG methods 
        corner_errors = [self.CCV_corner_error_MAE, self.LBCV_corner_error_MAE, self.SEG_corner_error_MAE] 
        plt.figure(figsize=(12, 6))
        plt.bar(['CCV', 'LBCV', 'SEG'], corner_errors, color=['blue', 'orange', 'green'])
        plt.xlabel('Method')
        plt.ylabel('Corner Mean Absolute Error (MAE)') 
        plt.grid() 
        
    def save_prediction_summary_images(self): 
        # save a super image with the following images: 
        # 1. original image 
        # 2. ground truth segmentation mask 
        # 3. CCV pose estimation bounding box 
        # 4. LBCV segmentation prediction mask 
        # 5. LBCV marker reprojection image 

        # create prediction summary image directory 
        prediction_summary_dir = os.path.join(self.out_dir, "prediction_summary")
        os.makedirs(prediction_summary_dir, exist_ok=True) 

        # loop through all datapoints and save the images
        for idx, dp in enumerate(self.datapoints):
            if dp.rgb is None or dp.seg_png_filepath is None or dp.LBCV_seg_filepath is None or dp.LBCV_tag_pose is None: 
                print(f"Skipping datapoint {idx} due to missing data.")
                continue 
            
            # read images 
            rgb_image = cv2.imread(dp.rgb)
            seg_mask = self.preprocess_seg_img(seg_img_path=dp.seg_png_filepath, seg_json_path=dp.seg_json_filepath)
            seg_mask = np.array(seg_mask)  # Load the segmentation mask
            # seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_BGR2GRAY)
            if len(seg_mask.shape) == 3:  # In case the image is a color image (unexpected)
                if seg_mask.shape[2] == 3: # Check if the image has 3 channels
                    seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            seg_mask[seg_mask > 0] = 255
            if dp.CCV_tag_pose is None: 
                # create blank image 
                ccv_bbox_image = np.zeros_like(rgb_image)  # Create a blank image with the same size as the original image 
            else: 
                ccv_bbox_image = cv2.imread(os.path.join(self.out_dir, "classical_CV_bbox", os.path.basename(dp.rgb).replace(".png","_bbox.png")))
            lbcv_seg_pred = cv2.imread(dp.LBCV_seg_filepath)  # Load the LBCV segmentation prediction mask
            if lbcv_seg_pred is None:
                print(f"LBCV segmentation prediction not found for datapoint {idx}.")
                continue 
            lbcv_seg_pred = cv2.cvtColor(lbcv_seg_pred, cv2.COLOR_BGR2GRAY)
            lbcv_seg_pred[lbcv_seg_pred > 0] = 255  # Threshold to create a binary mask
            # read blended image from LBCV pose estimation
            blended_image = cv2.imread(os.path.join(self.out_dir, "reprojected_marker_images", f"reprojected_image_{idx}.png"))
            if blended_image is None:
                print(f"Blended image not found for datapoint {idx}.")
                continue 

            # # combine images into a super image 
            # # create a blank image with the same size as the original image
            # super_image = np.zeros((rgb_image.shape[0], rgb_image.shape[1] * 5, 3), dtype=np.uint8)  # 5 columns for each image
            # # place each image in the super image
            # super_image[:, :rgb_image.shape[1]] = rgb_image
            # super_image[:, rgb_image.shape[1]:rgb_image.shape[1]*2] = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2BGR)
            # super_image[:, rgb_image.shape[1]*2:rgb_image.shape[1]*3] = ccv_bbox_image
            # super_image[:, rgb_image.shape[1]*3:rgb_image.shape[1]*4] = cv2.cvtColor(lbcv_seg_pred, cv2.COLOR_GRAY2BGR)
            # super_image[:, rgb_image.shape[1]*4:rgb_image.shape[1]*5] = blended_image

            # Create a blank image with the size of two rows and three columns
            # 2 rows and 3 columns, so the total width is 3 times the width of the original image
            # and the total height is 2 times the height of the original image.
            super_image = np.zeros((rgb_image.shape[0] * 2, rgb_image.shape[1] * 3, 3), dtype=np.uint8)

            # Place each image in the super image
            super_image[:rgb_image.shape[0], :rgb_image.shape[1]] = rgb_image
            super_image[:rgb_image.shape[0], rgb_image.shape[1]:rgb_image.shape[1]*2] = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2BGR)
            super_image[:rgb_image.shape[0], rgb_image.shape[1]*2:rgb_image.shape[1]*3] = ccv_bbox_image

            super_image[rgb_image.shape[0]:rgb_image.shape[0]*2, :rgb_image.shape[1]] = cv2.cvtColor(lbcv_seg_pred, cv2.COLOR_GRAY2BGR)
            super_image[rgb_image.shape[0]:rgb_image.shape[0]*2, rgb_image.shape[1]:rgb_image.shape[1]*2] = blended_image

            # add text labels to each image in the super image
            cv2.putText(super_image, "Original Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(super_image, "Ground Truth Segmentation", (rgb_image.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(super_image, "CCV Pose Estimation", (rgb_image.shape[1]*2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(super_image, "LBCV Segmentation Prediction", (rgb_image.shape[1]*3 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(super_image, "LBCV Marker Reprojection", (rgb_image.shape[1]*4 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # save the super image to the prediction summary directory
            save_path = os.path.join(prediction_summary_dir, f"prediction_summary_{idx}.png")
            cv2.imwrite(save_path, super_image)

# DATA PROCESSING AND ANALYSIS SCRIPT 

if __name__ == "__main__":
    # DATA PROCESSING 
    sdp = DataProcessor(
        # data_folders = ["C:/Users/NegiA/Desktop/abhay_ws/marker_detection_failure_recovery/segmentation_model/sim_data/markers_20250314-181037"], 
        # out_dir = "C:/Users/NegiA/Desktop/abhay_ws/marker_detection_failure_recovery/segmentation_model/sim_data/markers_20250314-181037/outputs"
    
        data_folders = ["/home/anegi/abhay_ws/marker_detection_failure_recovery/output/sdg_markers_20250324-183440/"],   
        out_dir = "/home/anegi/abhay_ws/marker_detection_failure_recovery/output/sdg_markers_20250324-183440/outputs"
    
    )
    sdp.process_folders() 
    sdp.set_camera_calibration() 
    # sdp.set_marker("C:/Users/NegiA/Desktop/abhay_ws/marker_detection_failure_recovery/synthetic_data_generation/assets/tags/tag36h11_0.png", 0.100)
    sdp.set_marker("/home/anegi/abhay_ws/marker_detection_failure_recovery/synthetic_data_generation/assets/tags/4x4_1000-31.png", 0.100)

    ## INFERENCE 
    # run classical pose estimation 
    print("Running classical marker detection")
    sdp.run_classical_marker_pose_estimation(save_pose=True, save_image=True) 
    sdp.compute_CCV_pose_error() 
    
    # run segmentation model 
    print("Running segmentation pose estimation")
    sdp.run_segmentation_pose_estimate()  
    sdp.compute_segmentation_pose_error() 

    # run prediction models 
    # TODO: add in later, for now just read data 
    # sdp.read_segmentation_predictions("C:/Users/NegiA/Desktop/abhay_ws/marker_detection_failure_recovery/segmentation_model/sim_data/markers_20250314-181037/rgb/predictions_20250315-162709/predictions") 
    print("Reading segmentation predictions")
    sdp.read_segmentation_predictions("/home/anegi/abhay_ws/marker_detection_failure_recovery/segmentation_model/Test Images/sdg_markers_20250325-132238/predictions/") 
    sdp.compute_segmentation_IOU() 
    sdp.run_optimization_pose_estimation()  
    sdp.compute_LBCV_pose_error() 

    ## ANALYSIS 
    sdp.compare_pose_estimation_methods()  

    # scatter plots comparing error vs variables (distance, lighting, etc.) 
    # save data 
    sdp.save_prediction_summary_images() 

    import pdb; pdb.set_trace() 