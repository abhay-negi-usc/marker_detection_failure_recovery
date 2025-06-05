import numpy as np 
import pandas as pd 
import os 
import json 
from PIL import Image
import random 
import time 
import cv2 
import math 
import albumentations as A 
from scipy.spatial.transform import Rotation as R 
import matplotlib.pyplot as plt 

from math import gcd, lcm
from random import randint, uniform
from random import random as random_function 
from math import cos, sin, radians 
from perlin_numpy import generate_fractal_noise_2d

def project_point_to_image(C,T,P): 
    P_H = np.array([[P[0]],[P[1]],[P[2]],[1]]) 
    T_H = T[:3,:4]  
    uv = C @ T_H @ P_H 
    if uv[2] != 0: 
        uv = uv / uv[2] # NOTE: check if the alternative case of not dividing when z=0 is valid 
    uv = uv[:2] 
    uv = uv.reshape((2)) 
    return uv 

def project_point_list_to_image(C,T,P_list): 
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
            # check if the point is within the image bounds
            if x < 0 or x >= image.shape[1] or y < 0 or y >= image.shape[0]:
                continue
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


def rotate3d(pic, rot_x, rot_y, rot_z, f_mult = 1.0, fill_color = (0,0,0)):

    height, width = [(2 * i) for i in pic.shape[0:2]]

    pic_exp = np.zeros((height, width, 4), dtype = np.uint8)
    pic_exp[:,:,:3] = fill_color
    pic_exp[pic.shape[0]//2:(height + pic.shape[0])//2,
            pic.shape[1]//2:(width + pic.shape[1])//2, :] = pic

    alpha = radians(rot_x)
    beta = radians(rot_y)
    gamma = radians(rot_z)

    f = (width / 2) * f_mult

    # 2d -> 3d
    proj2d3d = np.asarray([[1, 0, -width / 2],
                           [0, 1, -height / 2],
                           [0, 0, 0],
                           [0, 0, 1]])

    # Rotation matrices
    rx = np.asarray([[1, 0, 0, 0],
                     [0, cos(alpha), -sin(alpha), 0],
                     [0, sin(alpha), cos(alpha), 0],
                     [0, 0, 0, 1]])
    
    ry = np.asarray([[cos(beta), 0, sin(beta), 0],
                     [0, 1, 0, 0],
                     [-sin(beta), 0, cos(beta), 0],
                     [0, 0, 0, 1]])
    
    rz = np.asarray([[cos(gamma), -sin(gamma), 0, 0],
                     [sin(gamma), cos(gamma), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    
    # Translation
    T = np.asarray([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, f],
                    [0, 0, 0, 1]])
    
    # 3d -> 2d
    proj3d2d = np.asarray([[f, 0, width / 2, 0],
                           [0, f, height / 2, 0],
                           [0, 0, 1, 0]])
    
    # Combine all
    transform = proj3d2d @ (T @ ((rx @ ry @ rz) @ proj2d3d))
    pic_exp = cv2.warpPerspective(pic_exp, transform, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=fill_color)

    return pic_exp, transform

def gradient(width, height):

    t_size = max(width, height)
    size = t_size * 2

    grad = np.zeros((size, size))

    for i in range(size):
        grad[i] = (i / size)

    center = grad.shape[0]//2
    mat = cv2.getRotationMatrix2D((center, center), random_function() * 360, 1.0)
    pic = cv2.warpAffine(grad, mat, (size, size))

    # Final crop

    center = grad.shape[0]//2
    pic = pic[center - height//2:center + height//2, center - width//2:center + width//2]

    # Re-range

    pic = (pic - np.min(pic)) / (np.max(pic) - np.min(pic) + 1e-6)

    return pic

def lines(width, height, num_patterns = 3):

    t_size = max(width, height)
    size = t_size * 2

    pic = np.ones((size, size))
    center = pic.shape[0]//2

    for i in range(num_patterns):

        curr = 0

        while curr < size:
            paint = randint(1, max((size - curr)//2, 1))#min(randint(0, 16), size - curr)
            skip = randint(1, max((size - curr - paint)//2, 1))#min(randint(0, 16), size - curr - paint)
            pic[curr:curr + paint] *= uniform(0.0, 2.0)#random()
            curr = curr + paint + skip

        # Rotate

        mat = cv2.getRotationMatrix2D((center, center), random_function() * 360, 1.0)
        pic = cv2.warpAffine(pic, mat, (pic.shape[0], pic.shape[1]))

    # Re-range

    pic = (pic - np.min(pic)) / (np.max(pic) - np.min(pic) + 1e-6)

    # Perspective

    pic = cv2.merge((pic, pic, pic, np.ones(pic.shape))) * 255.0
    pic, _ = rotate3d(pic, randint(-30,30), randint(-30,30), 0)
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY) / 255.0

    # Final crop

    center = pic.shape[0]//2
    pic = pic[center - height//2:center + height//2, center - width//2:center + width//2]

    return pic

def circular(width, height):

    pic = np.zeros((height, width))
    center = (randint(0, height),
              randint(0, width))

    diag = int((width**2 + height**2)**(1/2))
    
    radius = randint(diag//4, diag)
    
    for i in range(height):
        for j in range(width):
            pic[i, j] = max(1 - (((i - center[0])**2 + (j - center[1])**2)**(1/2) / radius), 0)

    pic = (pic - np.min(pic)) / (np.max(pic) - np.min(pic) + 1e-6)
        
    return pic

def perlin(width, height, bins = 0, octaves = 4):

    t_width = lcm(width, 2 ** (octaves - 1))
    t_height = lcm(height, 2 ** (octaves - 1))

    res_x = t_width//gcd(t_width, t_height)
    res_y = t_height//gcd(t_width, t_height)

    # Fractal noise

    pic = generate_fractal_noise_2d((t_height, t_width), 
                                    (res_y, res_x), 
                                    octaves)

    # Re-range

    pic = (pic - np.min(pic)) / (np.max(pic) - np.min(pic) + 1e-6)

    # Threshold

    if bins > 1:
        pic = np.digitize(pic, [(i + 1) / bins for i in range(bins - 1)]) / (bins - 1)
    return pic

def lighting_augmentation(image): 
    # check if image is 0-1 or 0-255, convert to 0-1 
    # final image outputted is 0-255 
    image = np.array(image, dtype=float)
    if image.max() > 1.0:
        # image is 0-255  
        image = image / 255.0 

    height, width = image.shape[:2] 
    
    image_dim3 = image.shape[2] 
    augmented_image = image 
    if np.random.rand() < 0.5: 
        lines_effect = lines(width, height) 
        augmented_image *= np.repeat(lines_effect[:,:,np.newaxis],image_dim3,-1)
    if np.random.rand() < 0.1: 
        perlin_effect = perlin(width, height) 
        augmented_image *= np.repeat(perlin_effect[:,:,np.newaxis],image_dim3,-1)
    if np.random.rand() < 0.5: 
        gradient_effect = gradient(width, height) 
        augmented_image *= np.repeat(gradient_effect[:,:,np.newaxis],image_dim3,-1)  
    if np.random.rand() < 0.2: 
        circular_effect = circular(width, height) 
        augmented_image *= np.repeat(circular_effect[:,:,np.newaxis],image_dim3,-1) 
    
    # augmented_image = np.repeat(lines_effect[:,:,np.newaxis],image_dim3,-1) * np.repeat(perlin_effect[:,:,np.newaxis],image_dim3,-1) * np.repeat(gradient_effect[:,:,np.newaxis],image_dim3,-1) * np.repeat(circular_effect[:,:,np.newaxis],image_dim3,-1) * image 
    if (augmented_image.max() < 0.4) or augmented_image.max() > 1.0: # NOTE: HYPERPARAMETER 
        # renormalize image if too dark 
        pixel_max = np.random.uniform(0.9,1.0)  
        augmented_image = pixel_max * (augmented_image - np.min(augmented_image)) / (np.max(augmented_image) - np.min(augmented_image) + 1e-6) 
    
    augmented_image *= 255 
    
    return augmented_image 

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

        # TODO: self.idx = get_index() # or given as input 

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
        self.tag_pose = np.array(self.pose["tag"]) 
        if self.tag_pose[0,3]==0 and self.tag_pose[1,3]==0 and self.tag_pose[2,3]==0 and self.tag_pose[3,3]==1 and self.tag_pose[3,:3].sum() != 0:  
            self.tag_pose = self.tag_pose.transpose() 
        self.tag_pose *= np.array([
                            [10,10,10,1],
                            [10,10,10,1],
                            [10,10,10,1],
                            [1,1,1,1]
                        ]) # rescale the tag, FIXME: avoid hardcoding tag scale value 
        self.light_pose = self.pose["light"]
        self.tag_xyzabc = np.hstack((np.array(self.tag_pose[:3,3]), R.from_matrix(self.tag_pose[:3,:3]).as_euler("xyz",degrees=True))) # tag position in world frame 

    def compute_keypoints(self, keypoints_tag_frame, camera_matrix): 
        # transformations 
        tf_w_t = self.tag_pose 
        tf_w_c = self.cam_pose 
        tf_c_w = np.linalg.inv(tf_w_c) 
        keypoints_world_frame = [] 
        for kp_t in keypoints_tag_frame: 
            kp_t_homog = np.hstack((kp_t,np.array([1]))).reshape(4,1)
            kp_w_homog = tf_w_t @ kp_t_homog 
            keypoints_world_frame.append(kp_w_homog[:3].reshape(3)) 
        self.keypoints_image_space = project_point_list_to_image(camera_matrix,tf_c_w,keypoints_world_frame) 

        return self.keypoints_image_space 
    
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
        I_incident = 2**light_exposure 
        shininess = 1.0  # Placeholder value 
        self.diffuse_reflection = I_incident * max(np.dot(N, L), 0)

    def preprocess_seg_img(self):
        """
        Preprocesses the segmentation image by resizing and converting it to a binary mask based on tag color.
        """

        seg_img_path = self.seg_png_filepath 
        seg_json_path = self.seg_json_filepath 

        # Validate that the segmentation image file exists
        if not os.path.exists(seg_img_path):
            raise FileNotFoundError(f"Segmentation image file not found: {seg_img_path}")

        # Validate that the JSON file exists
        if not os.path.exists(seg_json_path):
            raise FileNotFoundError(f"Segmentation JSON file not found: {seg_json_path}")

        # Load the segmentation JSON data 
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
        # new_size = (480*2, 270*2)
        # seg_img_resized = seg_img.resize(new_size)
        seg_img_resized = seg_img

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
    
    def get_roi_image(self, seg=None, roi_size=128, padding=5): 
        if seg is None: 
            seg = self.preprocess_seg_img() 

        image_border_size = np.max([np.array(seg).shape[0], np.array(seg).shape[1]]) 

        # get pixel info of seg 
        seg = np.array(seg) 
        seg = cv2.copyMakeBorder(seg, image_border_size, image_border_size, image_border_size, image_border_size, cv2.BORDER_CONSTANT, value=0) 
        tag_pixels = np.argwhere(seg == 255)
        seg_tag_min_x = np.min(tag_pixels[:,1])
        seg_tag_max_x = np.max(tag_pixels[:,1])
        seg_tag_min_y = np.min(tag_pixels[:,0])
        seg_tag_max_y = np.max(tag_pixels[:,0])
        seg_height = seg_tag_max_y - seg_tag_min_y  
        seg_width = seg_tag_max_x - seg_tag_min_x 
        seg_center_x = (seg_tag_min_x + seg_tag_max_x) // 2
        seg_center_y = (seg_tag_min_y + seg_tag_max_y) // 2 

        # get pixel info of rgb 
        rgb = np.array(Image.open(self.rgb_filepath))
        rgb = cv2.copyMakeBorder(rgb, image_border_size, image_border_size, image_border_size, image_border_size, cv2.BORDER_CONSTANT, value=0) 
        rgb_side = max(seg_height, seg_width) + 2*padding 
        rgb_tag_min_x = seg_center_x - rgb_side // 2
        rgb_tag_max_x = seg_center_x + rgb_side // 2
        rgb_tag_min_y = seg_center_y - rgb_side // 2
        rgb_tag_max_y = seg_center_y + rgb_side // 2
        roi_img = rgb[rgb_tag_min_y:rgb_tag_max_y, rgb_tag_min_x:rgb_tag_max_x, :]

        # resize rgb bbox to roi size         
        try: 
            self.roi_img = cv2.resize(roi_img, (roi_size, roi_size))
        except: 
            print("error resizing") 
            import pdb; pdb.set_trace() 

        W = rgb.shape[1] 
        H = rgb.shape[0]
        self.roi_coordinates = np.array([rgb_tag_min_x-W/2, rgb_tag_max_x-W/2, rgb_tag_min_y-H/2, rgb_tag_max_y-H/2]) # FIXME: there is some issue here # image (x,y) coordinates (origin at image center) 

        self.roi_center = np.array([seg_center_x, seg_center_y]) - np.array([image_border_size, image_border_size]) 

        self.W_img = W 
        self.H_img = H 
        self.img_center = np.array([W/2,H/2])

        return self.roi_img, self.roi_coordinates, self.roi_center 
    
    def get_roi_keypoints(self): 

        # check if keypoints and roi have been computed, else return None 
        if not hasattr(self, 'keypoints_image_space') or not hasattr(self, 'roi_img'): 
            return None 
        
        # get keypoints in roi space
        roi_keypoints = []
        for kp in self.keypoints_image_space: 
            s = np.array(self.roi_img.shape[:2]) 
            w = self.roi_coordinates[1] - self.roi_coordinates[0]
            h = self.roi_coordinates[3] - self.roi_coordinates[2]
            m = s / np.array([w, h])     
            # m = s / np.array([self.W_img, self.H_img])     
            kp_roi = m*(kp - self.roi_center) + s/2 
            roi_keypoints.append(kp_roi) 

        self.roi_keypoints = roi_keypoints 

        return self.roi_keypoints  

    def __repr__(self):
        """Custom representation for the datapoint object."""
        # return f"datapoint(metadata_filepath={self.metadata_filepath}, pose_filepath={self.pose_filepath}, rgb_filepath={self.rgb_filepath}, seg_png_filepath={self.seg_png_filepath}, seg_json_filepath={self.seg_json_filepath})"
        description = [
            f"lighting_exposure={self.metadata["light"]["exposure"]:.2f}",
            # f"lighting_color={str(self.metadata["light"]["color"]) }" # FIXME: reduce to two decimal places 
            f"lighting_color=({self.metadata["light"]["color"][0]:.2f},{self.metadata["light"]["color"][1]:.2f},{self.metadata["light"]["color"][2]:.2f})" # FIXME: reduce to two decimal places 
        ]
        return "\n".join(description) 

class DataProcessor:
    def __init__(self, data_folders, out_dir):
        self.data_folders = data_folders
        self.out_dir = out_dir
        self.datapoints = []
        self.datapoints_train = []
        self.datapoints_val = []

        self.set_augmentation_transforms() 
        self.set_camera(camera_name="isaac") 

    def _get_files_in_subfolder(self, folder, file_extension=None):
        """Helper method to get files in a subfolder, with an optional file extension filter."""
        files_list = os.listdir(folder)
        if file_extension:
            files_list = [file for file in files_list if file.endswith(file_extension)]
        # Order files_list by date created
        files_list = sorted(files_list, key=lambda x: os.path.getctime(os.path.join(folder, x)))  # Assumes creation dates are synchronized
        return files_list
    
    def set_marker(self, image_path, num_squares, side_length): 
        self.marker_path = image_path 
        self.marker_image = Image.open(image_path) 
        self.marker_num_squares = num_squares 
        self.marker_side_length = side_length 
        self.keypoints_tag_frame = compute_2D_gridpoints(N=self.marker_num_squares, s=self.marker_side_length) 

    def set_camera(self, camera_name="isaac", camera_matrix=None):  
        # default camera is isaac 
        if camera_name == "isaac": 
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

            self.camera_matrix = np.array([
                [fx,0,cx],
                [0,fy,cy],
                [0,0,1]
            ])
        if camera_matrix is not None: 
            self.camera_matrix = camera_matrix 

    def process_folders(self):
        """Process the folders and create datapoint objects."""
        def get_file_id(filename):
            # Extract numeric ID from file name (e.g., "img_001.png" -> "001")
            return os.path.splitext(filename)[0].split('_')[-1]

        for data_folder in self.data_folders:
            metadata_subfolder = os.path.join(data_folder, "metadata")
            pose_subfolder = os.path.join(data_folder, "pose")
            rgb_subfolder = os.path.join(data_folder, "rgb")
            seg_subfolder = os.path.join(data_folder, "seg")

            metadata_files = self._get_files_in_subfolder(metadata_subfolder, file_extension=".json")
            pose_files     = self._get_files_in_subfolder(pose_subfolder, file_extension=".json")
            rgb_files      = self._get_files_in_subfolder(rgb_subfolder, file_extension=".png")
            seg_png_files  = self._get_files_in_subfolder(seg_subfolder, file_extension=".png")
            seg_json_files = self._get_files_in_subfolder(seg_subfolder, file_extension=".json")

            # Create dictionaries keyed by file index
            metadata_dict = {get_file_id(f): f for f in metadata_files}
            pose_dict     = {get_file_id(f): f for f in pose_files}
            rgb_dict      = {get_file_id(f): f for f in rgb_files}
            seg_png_dict  = {get_file_id(f): f for f in seg_png_files}
            seg_json_dict = {get_file_id(f): f for f in seg_json_files}

            # Only use keys common to all file types
            common_ids = sorted(set(metadata_dict) & set(pose_dict) & set(rgb_dict) & set(seg_png_dict) & set(seg_json_dict))

            if not common_ids:
                print(f"No matching file IDs found in all folders for: {data_folder}")
                continue

            for file_id in common_ids:
                metadata_filepath = os.path.join(metadata_subfolder, metadata_dict[file_id])
                pose_filepath     = os.path.join(pose_subfolder, pose_dict[file_id])
                rgb_filepath      = os.path.join(rgb_subfolder, rgb_dict[file_id])
                seg_png_filepath  = os.path.join(seg_subfolder, seg_png_dict[file_id])
                seg_json_filepath = os.path.join(seg_subfolder, seg_json_dict[file_id])

                # Create and store datapoint
                dp = datapoint(metadata_filepath, pose_filepath, rgb_filepath, seg_png_filepath, seg_json_filepath)
                self.datapoints.append(dp)


    def get_datapoints(self):
        """Return the list of datapoint objects."""
        return self.datapoints
    
    def get_datapoints_filtered(self):
        """Return the list of filtered datapoint objects."""
        return self.datapoints_filtered 
    
    def check_image_okay(self, rgb_img, seg_img, min_tag_area=1000, min_tag_pix_mean=25, max_tag_pix_mean=250): 
        if rgb_img is None: 
            return False 
        seg_img = np.array(seg_img) 
        # compute pixel area of tag segmentation 
        tag_pix_area = np.sum(seg_img == 255) 

        # create list of marker pixels using segmentation 
        marker_pixels = np.argwhere(seg_img == 255)  # Get the indices of pixels where the tag is present 
        # compute contrast of marker pixels using rgb image 
        rgb_img = np.array(rgb_img) 
        if rgb_img.max() <= 1.0: 
            rgb_img *= 255.0 
        marker_rgb_values = rgb_img[marker_pixels[:, 0], marker_pixels[:, 1]]  # Get the RGB values of the marker pixels 
        marker_grey_values = np.mean(marker_rgb_values, axis=1)  # Compute the mean RGB values of the marker pixels 
        # compute contrast as the difference in magnitude of the RGB values of the marker pixels 
        tag_pix_contrast = marker_grey_values.max() - marker_grey_values.min()  
        tag_pix_mean = marker_grey_values.mean()
        if tag_pix_area > min_tag_area and tag_pix_mean > min_tag_pix_mean and tag_pix_mean < max_tag_pix_mean:  # FIXME: hardcoded threshold for tag area and diffuse reflection 
            bool_image_ok = True 
        else:
            bool_image_ok = False
            # if not (tag_pix_mean > min_tag_pix_mean): 
            #     print("tag pix mean too low ") 
            # if not (tag_pix_mean < max_tag_pix_mean):
            #     print("tag pix mean too high")      
        return bool_image_ok

    def filter_datapoints(self, min_tag_area=1000, min_tag_pix_mean=70, max_tag_pix_mean=250): 
        """Compute the diffusion reflectance and only keep datapoints with positive values."""
        self.datapoints_filtered = [] 
        self.datapoints_filtered_out = [] 
        for idx, dp in enumerate(self.datapoints):
            dp.compute_diffusion_reflectance() 
            seg_img = dp.preprocess_seg_img() 
            seg_img = np.array(seg_img) 
            # compute pixel area of tag segmentation 
            dp.tag_pix_area = np.sum(seg_img == 255) 
            self.datapoints[idx].tag_pix_area = np.sum(seg_img == 255) 

            # create list of marker pixels using segmentation 
            marker_pixels = np.argwhere(seg_img == 255)  # Get the indices of pixels where the tag is present 
            # compute contrast of marker pixels using rgb image 
            rgb_img = Image.open(dp.rgb_filepath) 
            rgb_img = np.array(rgb_img) 
            marker_rgb_values = rgb_img[marker_pixels[:, 0], marker_pixels[:, 1]]  # Get the RGB values of the marker pixels 
            marker_grey_values = np.mean(marker_rgb_values, axis=1)  # Compute the mean RGB values of the marker pixels 
            # compute contrast as the difference in magnitude of the RGB values of the marker pixels 

            if marker_grey_values.size > 0: 
                pass 
            else: 
                print(f"empty image at {dp.rgb_filepath}")
                self.datapoints_filtered_out.append(dp) 
                continue 

            dp.tag_pix_contrast = marker_grey_values.max() - marker_grey_values.min()  
            self.datapoints[idx].tag_pix_contrast = marker_grey_values.max() - marker_grey_values.min() 
            self.datapoints[idx].tag_pix_mean = marker_grey_values.mean()
            dp.tag_pix_mean = marker_grey_values.mean() 

            # os.makedirs(os.path.join(self.out_dir, "contrast"), exist_ok=True) 
            # # save contrast image for debugging purposes
            # cv2.imwrite(os.path.join(self.out_dir, "contrast", f"contrast_{idx}_{dp.tag_pix_contrast}_{dp.tag_pix_mean}.png"), rgb_img) 
            if self.check_image_okay(rgb_img, seg_img, min_tag_area=min_tag_area, min_tag_pix_mean=min_tag_pix_mean, max_tag_pix_mean=max_tag_pix_mean): 
                self.datapoints_filtered.append(dp)
            else: 
                self.datapoints_filtered_out.append(dp)

            # if dp.diffuse_reflection > min_diffuse_reflection and dp.tag_pix_area > min_tag_area and dp.tag_pix_mean > min_tag_pix_mean and dp.tag_pix_mean < max_tax_pix_mean:  # FIXME: hardcoded threshold for tag area and diffuse reflection 
            #     self.datapoints_filtered.append(dp) 
            # else: 
            #     self.datapoints_filtered_out.append(dp) 

            if idx % (len(self.datapoints)/10) == 0: 
                print(f"Processed {idx} / {len(self.datapoints)}") 
                
    def split_train_val_data(self, filter=True, frac_train=0.8, num_points_max=-1):
        """Split the datapoints into training and validation datasets."""
        if num_points_max == -1: 
            if filter: 
                num_points = len(self.datapoints_filtered) 
            else: 
                num_points = len(self.datapoints)
        else: 
            if filter: 
                num_points = np.min([num_points_max, len(self.datapoints_filtered)])
            else: 
                num_points = np.min([num_points_max, len(self.datapoints)])

        if filter: 
            self.datapoints_train = random.sample(self.datapoints_filtered, int(frac_train * num_points))
            non_training_datapoints = [dp for dp in self.datapoints_filtered if dp not in self.datapoints_train]
            self.datapoints_val = random.sample(non_training_datapoints, int((1-frac_train) * num_points)) 
        else:
            self.datapoints_train = random.sample(self.datapoints, int(frac_train * num_points)) 
            non_training_datapoints = [dp for dp in self.datapoints if dp not in self.datapoints_train]
            self.datapoints_val = random.sample(non_training_datapoints, int((1-frac_train) * num_points)) 

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
        # new_size = (480*2, 270*2)  # Define the new size
        img = Image.open(img_path)
        # img_resized = img.resize(new_size)
        img_resized = img 
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
        # new_size = (480*2, 270*2)
        # seg_img_resized = seg_img.resize(new_size)
        seg_img_resized = seg_img

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

    def save_preprocessed_images(self, frac_train=0.8, augmentation=True, n_augmentations=0):
        """Loop through train and val datapoints and save preprocessed images and segmentation masks."""
        dir_train_rgb, dir_train_seg, dir_val_rgb, dir_val_seg = self.create_directories()

        if augmentation: 
            transform = A.Compose([
                # A.RandomShadow(shadow_roi=(0,0,1,1), num_shadows_limit=(1,10), shadow_dimension=4, shadow_intensity_range =(0.5, 0.8), p=0.8),  # Apply random shadows to the image
                A.RandomSunFlare(flare_roi=(0,0,1,1), num_flare_circles_range=(10,50), src_radius=100, src_color=(150,150,150), method="physics_based", p=0.8),  # Apply random sun flare to the image, TODO: come back to this, get labels 
                A.GaussNoise(var_limit=(0,0.01), per_channel=True, p=1),  # Add noise to the image 
                A.AdvancedBlur(blur_limit=(5,25), p=0.8),  # Apply blur to the image 
                # A.RandomGamma(gamma_limit=(80, 120), p=0.8),  # Apply gamma correction to the image
                # A.RandomBrightnessContrast(brightness_limit=(-0.25,0.25), contrast_limit=(-0.95,0.95), p=0.8),  # Adjust brightness and contrast
                # A.ISONoise(intensity=(0.1, 0.5), color_shift=(0.01, 0.05), p=0.8),  # Apply ISO noise to the image 
            ]) 

        for i, dp in enumerate(self.datapoints_train): 
            img = self.preprocess_rgb(dp.rgb_filepath) 
            seg = self.preprocess_seg_img(dp.seg_png_filepath, dp.seg_json_filepath)   
            
            img.save(os.path.join(dir_train_rgb, f"img_{i+1}_0.png")) 
            seg.save(os.path.join(dir_train_seg, f"seg_{i+1}.png"))
            
            if augmentation: 
                for j in range(n_augmentations): 
                    augmented_img = transform(image=np.array(img)[:,:,:3])['image']
                    augmented_img = lighting_augmentation(augmented_img) 
                    retry = 0 
                    while not self.check_image_okay(augmented_img, seg, min_tag_area=1000, min_tag_pix_mean=25, max_tag_pix_mean=250): 
                        # print(f"Retry: {retry}")
                        augmented_img = transform(image=np.array(img)[:,:,:3])['image']
                        if retry < 5: 
                            augmented_img = lighting_augmentation(augmented_img) 
                        if retry > 25: 
                            print(f"Exceeded retry limit. Skipping image.")
                            continue 
                        retry += 1
                    # augmented_img.save(os.path.join(dir_train_rgb, f"img_{i}_{j}.png")) 
                    image = Image.fromarray((augmented_img).astype(np.uint8)) 
                    image.save(os.path.join(dir_train_rgb, f"img_{i}_{j}.png")) 

            # print progress 
            if i % (len(self.datapoints_train)/100) == 0: 
                print(f"Processed training data: {i}/{len(self.datapoints_train)}")

        for i, dp in enumerate(self.datapoints_val):
            img = self.preprocess_rgb(dp.rgb_filepath) 
            seg = self.preprocess_seg_img(dp.seg_png_filepath, dp.seg_json_filepath) 

            img.save(os.path.join(dir_val_rgb, f"img_{i+1}_0.png")) 
            seg.save(os.path.join(dir_val_seg, f"seg_{i+1}.png"))

            if augmentation: 
                for j in range(n_augmentations): 
                    augmented_img = transform(image=np.array(img)[:,:,:3])['image']
                    augmented_img = lighting_augmentation(augmented_img) 
                    retry = 0 
                    while not self.check_image_okay(augmented_img, seg, min_tag_area=1000, min_tag_pix_mean=25, max_tag_pix_mean=250): 
                        augmented_img = transform(image=np.array(img)[:,:,:3])['image']
                        if retry < 5: 
                            augmented_img = lighting_augmentation(augmented_img) 
                        if retry > 25:
                            print(f"Exceeded retry limit. Skipping image.")
                            continue  
                        retry += 1 

                    # augmented_img.save(os.path.join(dir_val_rgb, f"img_{i}_{j}.png")) 
                    image = Image.fromarray((augmented_img * 255).astype(np.uint8)) 
                    image.save(os.path.join(dir_val_rgb, f"img_{i}_{j}.png")) 
            # print progress 
            if i % (len(self.datapoints_val)/100) == 0: 
                print(f"Processed training data: {i}/{len(self.datapoints_val)}")

    def set_augmentation_transforms(self): 
        transform = A.Compose([
                # A.RandomShadow(shadow_roi=(0,0,1,1), num_shadows_limit=(1,10), shadow_dimension=4, shadow_intensity_range =(0.5, 0.8), p=0.8),  # Apply random shadows to the image
                A.RandomSunFlare(flare_roi=(0,0,1,1), num_flare_circles_range=(10,50), src_radius=100, src_color=(150,150,150), method="physics_based", p=0.8),  # Apply random sun flare to the image, TODO: come back to this, get labels 
                A.GaussNoise(var_limit=(0,0.001), per_channel=True, p=1),  # Add noise to the image 
                A.AdvancedBlur(blur_limit=(5,51), p=0.8),  # Apply blur to the image 
                # A.RandomGamma(gamma_limit=(80, 120), p=0.8),  # Apply gamma correction to the image
                # A.RandomBrightnessContrast(brightness_limit=(-0.25,0.25), contrast_limit=(-0.95,0.95), p=0.8),  # Adjust brightness and contrast
                # A.ISONoise(intensity=(0.1, 0.5), color_shift=(0.01, 0.05), p=0.8),  # Apply ISO noise to the image 
            ])
        self.albumentations_transform = transform 

    def augment_image(self, image, seg, max_attempts_lighting=5, max_attempts_combined=10):  
        image = np.array(image)[:,:,:3]  
        # check if image is okay 
        if not self.check_image_okay(image, seg, min_tag_area=1000, min_tag_pix_mean=50, max_tag_pix_mean=250): 
            return None  
        # apply albumentations augmentation 
        augmented_image = None 
        for attempt in range(max_attempts_combined): 
            # apply albumentations augmentation 
            augmented_image = self.albumentations_transform(image=image)["image"] 
            # apply lighting augmentation
            if attempt < max_attempts_lighting: 
                augmented_image = lighting_augmentation(augmented_image) 
            if not self.check_image_okay(augmented_image, seg): 
                # print(f"Augmentation attempt {attempt} failed, brightening image.") 
                image = cv2.convertScaleAbs(image, alpha=1, beta=25) 
                continue 
            else: 
                break
        if attempt == max_attempts_combined: 
            print("Failed to augment image after max attempts.")
            return None
        return augmented_image 
    
    def save_train_val_data(self, 
                            save_rgb=True, 
                            save_seg=True, 
                            save_keypoints=True, 
                            save_metadata=True, 
                            num_augmentations=0,
                            save_summary_image=False,
                            save_roi=True, 
                            ):
        
        # Create directories for both training and validation datasets
        self.train_dir = os.path.join(self.out_dir, "train")
        self.val_dir = os.path.join(self.out_dir, "val")
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)

        # Loop over train and val
        for dataset_type in ['train', 'val']:
            dataset_dir = self.train_dir if dataset_type == 'train' else self.val_dir
            datapoints = self.datapoints_train if dataset_type == 'train' else self.datapoints_val

            # Create specific directories for RGB, Segmentation, Keypoints, Metadata, and Summary Images
            if save_rgb: 
                os.makedirs(os.path.join(dataset_dir, "rgb"), exist_ok=True)
            if save_seg:
                os.makedirs(os.path.join(dataset_dir, "seg"), exist_ok=True)
            if save_keypoints:
                os.makedirs(os.path.join(dataset_dir, "keypoints"), exist_ok=True)
            if save_metadata:
                os.makedirs(os.path.join(dataset_dir, "metadata"), exist_ok=True)
            if save_summary_image:
                os.makedirs(os.path.join(dataset_dir, "summary_images"), exist_ok=True)
            if save_roi:
                os.makedirs(os.path.join(dataset_dir, "roi_rgb"), exist_ok=True)
                os.makedirs(os.path.join(dataset_dir, "roi_keypoints"), exist_ok=True) 

            # Process each datapoint in the current dataset
            for i, dp in enumerate(datapoints):
                if save_rgb:
                    rgb_img = Image.open(dp.rgb_filepath)
                    if num_augmentations == 0:
                        rgb_img.save(os.path.join(dataset_dir, "rgb", f"img_{i}.png"))
                    else:
                        # augment image 
                        for j in range(num_augmentations):
                            seg_img = dp.preprocess_seg_img()
                            augmented_img = self.augment_image(rgb_img, seg_img)
                            if augmented_img is None:
                                # print(f"Failed to augment image {i}.")
                                continue
                            else:
                                augmented_img = Image.fromarray((augmented_img).astype(np.uint8))
                                augmented_img.save(os.path.join(dataset_dir, "rgb", f"img_{i}_{j}.png"))

                if save_seg:
                    seg_img = dp.preprocess_seg_img()
                    seg_img.save(os.path.join(dataset_dir, "seg", f"seg_{i}.png"))

                if save_keypoints:
                    keypoints = dp.compute_keypoints(self.keypoints_tag_frame, self.camera_matrix)
                    keypoints_json = {}
                    for i_kp, kp in enumerate(keypoints):
                        keypoints_json[f"keypoints_{i_kp}"] = kp.tolist()
                    with open(os.path.join(dataset_dir, "keypoints", f"keypoints_{i}.json"), 'w') as f:
                        json.dump(keypoints_json, f)

                if save_metadata:
                    metadata = dp.metadata
                    with open(os.path.join(dataset_dir, "metadata", f"metadata_{i}.json"), 'w') as f:
                        json.dump(metadata, f)

                if save_roi:
                    roi_image, roi_coordinates, roi_center = dp.get_roi_image(seg=seg_img)
                    roi_image = Image.fromarray(roi_image)
                    roi_image.save(os.path.join(dataset_dir, "roi_rgb", f"roi_{i}.png")) 
                    roi_keypoints = dp.get_roi_keypoints()
                    if roi_keypoints is not None:
                        roi_keypoints_json = {}
                        for i_kp, kp in enumerate(roi_keypoints):
                            roi_keypoints_json[f"keypoints_{i_kp}"] = kp.tolist()
                        with open(os.path.join(dataset_dir, "roi_keypoints", f"roi_keypoints_{i}.json"), 'w') as f:
                            json.dump(roi_keypoints_json, f)

                if save_summary_image:
                    # Check if images are loaded correctly
                    if rgb_img is None:
                        raise ValueError(f"RGB image at {dp.rgb_filepath} could not be loaded.")
                    if seg_img is None:
                        raise ValueError(f"Segmentation image at {dp.seg_filepath} could not be loaded.")

                    # Convert from BGR (OpenCV default) to RGB (for matplotlib)
                    image_rgb = np.array(rgb_img) 
                    if augmented_img is None:
                        augmented_img = image_rgb
                    augmented_img_rgb = np.array(augmented_img)

                    # Create a new figure for each image
                    plt.figure(figsize=(12, 8))  # Adjust figure size to make space for metadata and the new ROI subplot

                    # Subplot for original RGB image
                    plt.subplot(2, 3, 1)  # 2 rows, 3 columns, 1st subplot
                    plt.imshow(image_rgb)
                    plt.axis('off')  # Hide axes
                    plt.title(f'Original Image {i}')

                    # Subplot for augmented RGB image
                    plt.subplot(2, 3, 2)  # 2 rows, 3 columns, 2nd subplot
                    plt.imshow(augmented_img_rgb)
                    plt.axis('off')  # Hide axes
                    plt.title(f'Augmented Image {i}')

                    # Subplot for segmentation image
                    plt.subplot(2, 3, 3)  # 2 rows, 3 columns, 3rd subplot
                    plt.imshow(seg_img, cmap='viridis')  # Use a colormap for better visualization
                    plt.axis('off')  # Hide axes
                    plt.title(f'Segmentation Image {i}')

                    # Subplot for RGB image - keypoints
                    keypoints_image = overlay_points_on_image(image=np.array(augmented_img_rgb), pixel_points=keypoints, radius=1)
                    plt.subplot(2, 3, 4)  # 2 rows, 3 columns, 4th subplot
                    plt.imshow(keypoints_image)
                    plt.axis('off')  # Hide axes
                    plt.title(f'Keypoints Image {i}')

                    # Subplot for ROI image
                    plt.subplot(2, 3, 5)  # 2 rows, 3 columns, 5th subplot
                    plt.imshow(roi_image)
                    plt.axis('off')  # Hide axes
                    plt.title(f'ROI Image {i}')

                    # Subplot for ROI image with keypoints 
                    roi_keypoints_image = overlay_points_on_image(image=np.array(roi_image), pixel_points=roi_keypoints, radius=1)
                    plt.subplot(2, 3, 6)  # 2 rows, 3 columns, 6th subplot
                    plt.imshow(roi_keypoints_image)
                    plt.axis('off')  # Hide axes
                    plt.title(f'ROI Keypoints Image {i}')
                    
                    # Display metadata as text in a separate area
                    metadata_str = dp.__repr__()

                    # Create a new subplot for metadata
                    plt.text(1.05, 0.5, metadata_str, fontsize=12, ha='left', va='center', transform=plt.gca().transAxes,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=1'))

                    # Adjust layout to avoid overlap and make space for metadata
                    plt.tight_layout()  # Adjust layout
                    plt.subplots_adjust(right=0.8)  # Make space for metadata on the right

                    # Save the image to the summary_images folder
                    save_path = os.path.join(dataset_dir, "summary_images", f"summary_image_{i}.png")
                    plt.savefig(save_path, bbox_inches='tight', dpi=300)  # Save with high resolution
                    plt.close()  # Close the plot to free up memory


                # Print progress every 10%
                if i % (len(datapoints) / 10) == 0:
                    print(f"Processed {dataset_type} data: {i}/{len(datapoints)}") 

data_folders = [
    # "/home/anegi/abhay_ws/marker_detection_failure_recovery/output/sdg_markers_20250324-183440",
    # "/home/anegi/abhay_ws/marker_detection_failure_recovery/output/sdg_markers_20250324-181901",
    # "/home/anegi/abhay_ws/marker_detection_failure_recovery/output/sdg_markers_20250326-143252",
    # "/home/anegi/abhay_ws/marker_detection_failure_recovery/output/sdg_markers_20250323-032914",
    # "/home/anegi/abhay_ws/marker_detection_failure_recovery/output/sdg_markers_20250323-033133",
    # "/home/anegi/abhay_ws/marker_detection_failure_recovery/output/markers_20250314-181037/", # 1K dataset 
    # "/home/anegi/abhay_ws/marker_detection_failure_recovery/output/markers_20250310-005350/" # 72K dataset 

    "/home/anegi/abhay_ws/marker_detection_failure_recovery/output/sdg_markers_20250323-032914/", 
    "/home/anegi/abhay_ws/marker_detection_failure_recovery/output/sdg_markers_20250323-033133/", 
    "/home/anegi/abhay_ws/marker_detection_failure_recovery/output/sdg_markers_20250324-181901/", 
    "/home/anegi/abhay_ws/marker_detection_failure_recovery/output/sdg_markers_20250324-183440/", 
    "/home/anegi/abhay_ws/marker_detection_failure_recovery/output/sdg_markers_20250326-143252/", 
    "/home/anegi/abhay_ws/marker_detection_failure_recovery/output/sdg_markers_20250327-153817/", 

    # "/home/anegi/abhay_ws/marker_detection_failure_recovery/output/markers_20250310-005350/",   
    # "/home/rp/abhay_ws/marker_detection_failure_recovery/test_data/markers_20250314-181037/", # 1K dataset 
    # "/home/rp/abhay_ws/marker_detection_failure_recovery/output/markers_20250320-133701/" 
    # "/home/rp/abhay_ws/marker_detection_failure_recovery/output/markers_20250310-005350/" 
]

# define OUT_DIR based on current date and time 
OUT_DIR = f"/home/anegi/abhay_ws/marker_detection_failure_recovery/segmentation_model/data/data_{time.strftime('%Y%m%d-%H%M%S')}"
# OUT_DIR = f"/home/rp/abhay_ws/marker_detection_failure_recovery/test_data/data_{time.strftime('%Y%m%d-%H%M%S')}"
os.makedirs(OUT_DIR, exist_ok=True) 

# Create an instance of the DataProcessor class
processor = DataProcessor(data_folders, OUT_DIR)
# define marker
processor.set_marker(image_path="./synthetic_data_generation/assets/tags/4x4_1000-31.png", num_squares=6, side_length=0.100) 
# processor.set_marker(image_path="../synthetic_data_generation/assets/tags/tag36h11_0.png", num_squares=10, side_length=0.100) 
# Process the folders to create the datapoint list
processor.process_folders()

# Retrieve and print length of the datapoints before and after filtering 
print(f"Number of datapoints: {len(processor.datapoints)}") 
processor.filter_datapoints() 
print(f"Number of filtered datapoints: {len(processor.datapoints_filtered)}")  
# Retrieve filtered datapoints
datapoints = processor.get_datapoints_filtered()
    
# Split the datapoints into training and validation sets
processor.split_train_val_data(filter=True, frac_train=0.8)
processor.save_train_val_data(num_augmentations=1, save_summary_image=False)  