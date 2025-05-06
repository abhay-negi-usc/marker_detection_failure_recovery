import numpy as np 
import pandas as pd 
import os 
import cv2 
from scipy.spatial.transform import Rotation as R

from real_data_processing.utils import opencv_marker_pose, draw_overlay_square 

dir_data = "./real_data_processing/raw_data/camera_extrinsic_calibration/"

fx, fy, cx, cy =  1363.85, 1365.40, 958.58, 552.25 
dist_coeffs = np.array([0.1693, -0.4755, 0.0018, 0.0023, 0.4114]) * 0 
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]) 
marker_size = 0.0798 

tf_C_M0 = None 
tf_W_M0 = None 

tf_M_M = np.array([
    [-1,0,0,0],
    [0,1,0,0],
    [0,0,-1,0],
    [0,0,0,1]
])

tf_W_C_list = []
for i in [11,12,13,14,15]: 
    image_filename = os.path.join(dir_data, f"m{i}.jpg") 
    image = np.array(cv2.imread(image_filename)) 
    tf_C_M, pose_C_M, corners = opencv_marker_pose(image, K, dist_coeffs, marker_size, show=False)   

    if tf_C_M0 is None:
        tf_C_M0 = tf_C_M 
    
    # tf_M0_Mi_c = np.linalg.inv(tf_C_M0) @ tf_C_M 
    # print(tf_M0_Mi_c)

    pose_filename = os.path.join(dir_data, f"cal_00{i}.csv") 
    pose = pd.read_csv(pose_filename) 
    tf_W_M = np.eye(4) 
    tf_W_M[:3, 3] = pose[['X','Y','Z']].mean()  
    tf_W_M[:3, :3] = R.from_quat(pose[['QX','QY','QZ','QW']].mean()).as_matrix()

    tf_W_M = tf_W_M @ tf_M_M 


    if tf_W_M0 is None: 
        tf_W_M0 = tf_W_M 
    
    # tf_M0_Mi_w = np.linalg.inv(tf_W_M0) @ tf_W_M 
    # print(tf_M0_Mi_w) 

    # tf_dc_dw = np.linalg.inv(tf_M0_Mi_w) @ tf_M0_Mi_c 
    # print(tf_dc_dw)

    tf_W_C = tf_W_M @ np.linalg.inv(tf_C_M) 
    print(tf_W_C) 

    tf_W_C_list.append(tf_W_C)


    # import pdb; pdb.set_trace() 

print("\n\n\n")
tf_W_C_mean = np.mean(np.array(tf_W_C_list), axis=0) 

print(tf_W_C_mean)

for i in [11,12,13,14,15]: 
    image_filename = os.path.join(dir_data, f"m{i}.jpg") 
    image = np.array(cv2.imread(image_filename)) 

    pose_filename = os.path.join(dir_data, f"cal_00{i}.csv") 
    pose = pd.read_csv(pose_filename) 
    tf_W_M = np.eye(4) 
    tf_W_M[:3, 3] = pose[['X','Y','Z']].mean()  
    tf_W_M[:3, :3] = R.from_quat(pose[['QX','QY','QZ','QW']].mean()).as_matrix()


    tf_C_M = np.linalg.inv(tf_W_C_mean) @ tf_W_M 

    # overlay
    image_overlay = draw_overlay_square(image, tf_C_M, marker_size, K) 
    cv2.imshow("overlay", image_overlay) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()