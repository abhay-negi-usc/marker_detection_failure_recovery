import os 
import matplotlib.pyplot as plt 
import numpy as np 
import json
import cv2 
from PIL import Image 
import random 
import math 
from scipy.spatial.transform import Rotation as R

def project_point_to_image(C,T,P): 
    P_H = np.array([[P[0]],[P[1]],[P[2]],[1]]) 
    T_H = T[:3,:4]  
    uv = C @ T_H @ P_H 
    uv = uv / uv[2] 
    uv = uv[:2] 
    uv = uv.reshape((2)) 
    return uv 

def project_point_list_to_image(C,T,P_list): 
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
        # ret = cv2.aruco.estimatePoseSingleMarkers(corners[i], marker_length, camera_matrix, dist_coeffs)
        # FIXME: cv2.aruco.estimatePoseSingleMarkers is deprecated, use cv::solvePnP instead 
        ret = cv2.solvePnP(marker_frame_corners, corners[i], camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)  # Use the marker frame corners as object points 
        
        # Unpack the results (rotation and translation vectors)
        # rotation_vector, translation_vector = ret[0][0], ret[1][0]
        rotation_vector, translation_vector = ret[1][0], ret[2][0]
        
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

    def read_files(self): 
        # Read the actual data from files and store it
        self.metadata = self._read_json(self.metadata_filepath) if self.metadata_filepath else None
        self.pose = self._read_json(self.pose_filepath) if self.pose_filepath else None
        self.rgb = self._read_rgb(self.rgb_filepath) if self.rgb_filepath else None
        self.seg_png = self._read_segmentation_png(self.seg_png_filepath) if self.seg_png_filepath else None
        self.seg_json = self._read_segmentation_json(self.seg_json_filepath) if self.seg_json_filepath else None 

    def read_pose_data(self): 
        # read pose data from pose json file 
        self.cam_pose = np.array([
                            [1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, -1, 0],
                            [0, 0, 0, 1]
                        ]) # NOTE: cam pose from isaac sim appears to be offset 
        self.tag_pose = np.array(self.pose["tag"]).transpose()  
        self.tag_pose *= np.array([
                            [10,10,10,1],
                            [10,10,10,1],
                            [10,10,10,1],
                            [1,1,1,1]
                        ]) # rescale the tag, FIXME: avoid hardcoding tag scale value 
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

class DataProcessor:

    def __init__(self, data_folders, out_dir):
        self.data_folders = data_folders
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)  
        self.datapoints = []
        self.datapoints_train = []
        self.datapoints_val = []

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
                    aruco_dict=cv2.aruco.DICT_APRILTAG_36h11,
                    marker_length=0.080,  # 80 mm for black edges, 100 mm for white edges 
                )

                # compute transform 
                tf_CCV = np.eye(4)  # Initialize as identity matrix 
                if rotation_vectors is None or translation_vectors is None:
                    tf_CCV = None 
                else: 
                    tf_CCV[:3,:3] = cv2.Rodrigues(rotation_vectors[0])[0]  # Convert rotation vector to rotation matrix
                    tf_CCV[:3,3] = translation_vectors[0].reshape(3)  # Set translation vector 
                    
                # Store the results in the datapoint object
                self.datapoints[idx].CCV_marker_ids = marker_ids
                self.datapoints[idx].CCV_tag_pose = tf_CCV 
                self.datapoints[idx].CCV_corners = corners 
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
                    # TODO: keep count of number of failed detections and print at the end 
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
        print(f"Fraction of successful detections: {self.CCV_detection_fraction:.2f} ({num_success}/{num_total})")

    def compute_CCV_pose_error(self): 
        for idx, dp in enumerate(self.datapoints): 
            if dp.CCV_tag_pose is not None: 
                # compute pose error between CCV and GT tag pose 
                tf_true = dp.tag_pose 
                tf_CCV = dp.CCV_tag_pose
                CCV_tf_error = np.linalg.inv(tf_true) @ tf_CCV 
                self.datapoints[idx].CCV_tf_error = CCV_tf_error  
                # compute error in translation and rotation 
                xyz_error = CCV_tf_error[:3,3]  # translation error 
                abc_error = R.from_matrix(CCV_tf_error[:3,:3]).as_euler('xyz', degrees=True)  
                self.datapoints[idx].CCV_pose_error = np.hstack((xyz_error, abc_error))  # concatenate translation and rotation error 
            else: 
                self.datapoints[idx].CCV_tf_error = None 
                self.datapoints[idx].CCV_pose_error = None 
        
        # compute mean and std of pose error 
        self.CCV_pose_error_mean = np.mean([dp.CCV_pose_error for dp in self.datapoints if dp.CCV_pose_error is not None], axis=0) 
        self.CCV_pose_error_std = np.std([dp.CCV_pose_error for dp in self.datapoints if dp.CCV_pose_error is not None], axis=0) 
        print(f"Mean pose error: {self.CCV_pose_error_mean}") 
        print(f"Std pose error: {self.CCV_pose_error_std}") 

        # save violin plots of pose error distributions for each dimension (x,y,z,a,b,c) 
        errors = [dp.CCV_pose_error for dp in self.datapoints if dp.CCV_pose_error is not None] 
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
            axs[i].set_ylim(-0.5, 0.5)  # Set y-axis limits for better visualization
            axs[i].grid(True)  # Add grid for better readability
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.savefig(os.path.join(self.out_dir, "CCV_pose_error_distribution.png"), dpi=300)  # Save the figure as an image
        plt.close()

if __name__ == "__main__":
    # DATA PROCESSING 
    sdp = DataProcessor(
        data_folders = ["C:/Users/NegiA/Desktop/abhay_ws/marker_detection_failure_recovery/segmentation_model/sim_data/markers_20250314-181037"], 
        out_dir = "C:/Users/NegiA/Desktop/abhay_ws/marker_detection_failure_recovery/segmentation_model/sim_data/markers_20250314-181037/outputs"
    )
    sdp.process_folders() 
    sdp.set_camera_calibration() 

    ## INFERENCE 
    # run classical pose estimation 
    sdp.run_classical_marker_pose_estimation(save_pose=True, save_image=True) 

    # run segmentation model 

    # compute detection accuracy (IOU) 

    # run optimization pose estimation 
    # save data 

    ## ANALYSIS 
    # compute pose errors 
    sdp.compute_CCV_pose_error() 

    # bar chart comparing accuracy 
    # violin plots comparing pose error distributions 
    # scatter plots comparing error vs variables (distance, lighting, etc.) 
    # save data 
