# optitrack data, realsense data --> images (from realsense) with camera frame pose labels (from optitrack) 

# IMPORTS 
import numpy as np 
import pandas as pd 
import os 
import sys 
import matplotlib.pyplot as plt 
from PIL import Image 
import cv2

# MODULE IMPORTS 
sys.path.append(os.curdir) 
from real_data_processing.utils import * 

# CLASS DEFINITIONS
class optitrack_realsense_datapoint(): 
    def __init__(self, timestamp, image, tf_cam_marker):  
        self.timestamp = timestamp
        self.image = image 
        self.tf_cam_marker = tf_cam_marker  # transformation of marker wrt camera 

class optitrack_realsense_data_processor(): 
    def __init__(self, optitrack_csv_file: str=None, realsense_video_file: str=None): 
        self.optitrack_csv_file = optitrack_csv_file
        self.realsense_video_file = realsense_video_file 
        self.optitrack_raw_df, self.optitrack_raw_data  = self.parse_optitrack_csv(self.optitrack_csv_file) 
        self.realsense_raw_data = self.parse_realsense_video(self.realsense_video_file) 

    def parse_optitrack_csv(self, optitrack_csv_file: str):
        """
        Process the optitrack CSV file to extract timestamps and camera poses.
        Returns a list of tuples (timestamp, tf_cam_marker).
        """
        if not os.path.isfile(optitrack_csv_file):
            print(f"Error: Optitrack CSV file {optitrack_csv_file} not found.")
            return []

        df = pd.read_csv(optitrack_csv_file, skiprows=6)
        df.rename(columns={"Time (Seconds)":"TIME", "X":"QX", "Y":"QY", "Z":"QZ", "W":"QW", 
                            "X.1":"X", "Y.1":"Y", "Z.1":"Z"}, inplace=True)  # Remove any leading/trailing whitespace in column names
        optitrack_raw_df = df[["TIME", "X", "Y", "Z", "QW", "QX", "QY", "QZ"]].dropna() 
        optitrack_raw_data = optitrack_raw_df.to_numpy()  # Convert to numpy array for easier processing  
        
        return optitrack_raw_df, optitrack_raw_data 
    
    
    def set_tf_trackmark_fidumark(self, tf_trackmark_fidumark): 
        """
        Set the transformation of fiducial marker frame with respect to the optitrack marker frame 
        """
        self.tf_o_m = tf_trackmark_fidumark 

    def process_optitrack_data(self): 
        if not hasattr(self, 'tf_o_m'):
            print("Error: Transformation of fiducial marker frame not set. Please set it using set_tf_trackmark_fidumark().")
            return None, None, None, None 
        
        tf_w_o = [] 
        tf_w_m = [] 
        tf_mo_mi = [] 
        timestamps = [] 
        tf_mo_w = None 
        tf_c_m = [] 

        for idx, row in enumerate(self.optitrack_raw_data):
            if np.any(np.isnan(row)) or len(row) < 8:
                continue 
            timestamp = row[0]
            timestamps.append(timestamp)  
            xyzquat = row[1:8]  # Extract x, y, z, qw, qx, qy, qz 
            tf_w_o_i = xyzquat_to_tf(xyzquat, input_qw_first=True) # transform of optitrack marker wrt optitrack world frame 
            tf_w_o.append(tf_w_o_i) 
            tf_w_m_i = tf_w_o_i @ self.tf_o_m   
            tf_w_m.append(tf_w_m_i) 

            if tf_mo_w is None: 
                tf_mo_w = np.linalg.inv(tf_w_m_i) 
                tf_mo_mi_i = np.eye(4)
            else: 
                tf_mo_mi_i = tf_mo_w @ tf_w_m_i  

            tf_mo_mi.append(tf_mo_mi_i) 

            tf_c_m_i = self.tf_c_w @ tf_w_m_i 
            tf_c_m.append(tf_c_m_i) 

        self.OPTK_time = np.array(timestamps)
        self.OPTK_tf_w_o = np.array(tf_w_o) 
        self.OPTK_tf_w_m = np.array(tf_w_m)
        self.OPTK_tf_mo_mi = np.array(tf_mo_mi) 
        self.OPTK_tf_c_m = np.array(tf_c_m) 

        print(f"Optitrack detection rate: {len(self.OPTK_time)}/{len(self.optitrack_raw_data)}") 

        return self.OPTK_time, self.OPTK_tf_w_o, self.OPTK_tf_w_m, self.OPTK_tf_mo_mi

    def parse_realsense_video(self, realsense_video_file: str): 
        # save images from the video and extract timestamps 
        self.realsense_frames_dir = self.realsense_video_file.replace('.mp4', '_frames')

        # check if the frames directory already exists and contains images
        already_split = os.path.exists(self.realsense_frames_dir) and len(os.listdir(self.realsense_frames_dir)) > 0

        if not already_split: 
            os.makedirs(self.realsense_frames_dir, exist_ok=True)  # Create directory for frames if it doesn't exist
            self.RLSN_time = split_video_to_frames(realsense_video_file, self.realsense_frames_dir, get_timestamps=True)         
        else:     
            cap = cv2.VideoCapture(realsense_video_file)
            fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames
            timestamps = np.linspace(0, total_frames / fps, total_frames, endpoint=False)
            self.RLSN_time = timestamps  # Store the timestamps for each frame
            cap.release() 

        # get list of frame image paths 
        self.realsense_frames_paths = sorted([
            os.path.join(self.realsense_frames_dir, f) 
            for f in os.listdir(self.realsense_frames_dir) 
            if f.endswith('.png')
        ]) 
        return 

    def set_fiducial_marker_detector(self, camera_matrix, dist_coeffs, aruco_dict, marker_length, tf_optitrack_camera): 
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.aruco_dict = aruco_dict
        self.marker_length = marker_length 
        self.tf_w_c = tf_optitrack_camera  # Transformation of camera wrt optitrack world frame 
        self.tf_c_w = np.linalg.inv(self.tf_w_c) 

    def run_opencv_fiducial_marker_detection(self, save_results=False): 
        # check if frame_path and 
        if len(self.RLSN_time) != len(self.realsense_frames_paths):
            print("Error: Number of frames in video does not match number of timestamps. Please check the video file.")
            return None, None, None, None 
        
        if save_results: 
            output_dir = self.realsense_frames_dir + "_CCV_processed"
            os.makedirs(output_dir, exist_ok=True)

        timestamps = [] 
        tf_c_m = [] 
        tf_w_m = [] 
        tf_mo_mi = [] 
        tf_mo_w = None 

        for idx, frame_path in enumerate(self.realsense_frames_paths): 
            frame = cv2.imread(frame_path) 
            marker_ids, rotation_vectors, translation_vectors, corners = classical_marker_pose_estimation(frame, self.camera_matrix, self.dist_coeffs, self.aruco_dict, self.marker_length, show=False) 

            if save_results: 
                # Draw detected markers on the frame
                frame_markers = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, marker_ids)
                output_frame_path = os.path.join(output_dir, f"CCV_frame_{idx:05d}.png")
                cv2.imwrite(output_frame_path, frame_markers) 

            # TODO: filter based on matching of marker_ids 

            if rotation_vectors is None or translation_vectors is None:
                continue 

            timestamps.append(self.RLSN_time[idx])  # Store the timestamp for this frame 
            tf_Ccv_Mcv = np.eye(4)  # marker in openCV coordinates wrt openCV camera coordinates 
            tf_Ccv_Mcv[:3,:3] = cv2.Rodrigues(rotation_vectors[0])[0] # Convert rotation vector to rotation matrix
            tf_Ccv_Mcv[:3,3] = translation_vectors[0].reshape(3)  # Set translation vector  
            tf_Ccv_Mcv = tf_Ccv_Mcv @ np.array([[0,-1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]]) # FIXME: figure out where this bias comes from 
            # tf_Ccv_Mcv = np.array([[0,-1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]]) @ tf_Ccv_Mcv # FIXME: figure out where this bias comes from 

            tf_w_m_i = self.tf_w_c @ tf_Ccv_Mcv  
            
            if tf_mo_w is None:  # First frame case 
                tf_mo_mi_i = np.eye(4) 
                tf_mo_w = np.linalg.inv(tf_w_m_i) 
            else:
                tf_mo_mi_i = tf_mo_w @ tf_w_m_i 

            tf_c_m.append(tf_Ccv_Mcv) # use opencv convention 
            tf_w_m.append(tf_w_m_i) 
            tf_mo_mi.append(tf_mo_mi_i) 

        self.CCV_time = np.array(timestamps) 
        self.CCV_tf_c_m = np.array(tf_c_m)  
        self.CCV_tf_w_m = np.array(tf_w_m) 
        self.CCV_tf_mo_mi = np.array(tf_mo_mi) 

        print(f"Clasical CV detection rate: {len(self.CCV_time)}/{len(self.RLSN_time)}") 

        return self.CCV_time, self.CCV_tf_c_m, self.CCV_tf_w_m, self.CCV_tf_mo_mi 
    
    def align_timestamps(self): 
        # TODO : Implement timestamp alignment between Optitrack and OpenCV data 
        pass 
    
    def compare_marker_poses(self): 

        self.CCV_xyzabc_mo_mi = np.zeros((len(self.CCV_tf_mo_mi), 6))  
        self.OPTK_xyzabc_mo_mi = np.zeros((len(self.OPTK_tf_mo_mi), 6))   
        self.CCV_xyzabc_w_m = np.zeros((len(self.CCV_tf_w_m), 6))  
        self.OPTK_xyzabc_w_m = np.zeros((len(self.OPTK_tf_w_m), 6))  
        for i, tf in enumerate(self.CCV_tf_mo_mi): 
            self.CCV_xyzabc_mo_mi[i,:] = tf_to_xyzabc(tf) 
            self.CCV_xyzabc_w_m[i,:] = tf_to_xyzabc(self.CCV_tf_w_m[i])  
        for i, tf in enumerate(self.OPTK_tf_mo_mi):
            self.OPTK_xyzabc_mo_mi[i,:] = tf_to_xyzabc(tf) 
            self.OPTK_xyzabc_w_m[i,:] = tf_to_xyzabc(self.OPTK_tf_w_m[i]) 

        # relative pose comparison 
        # # 2x3 subplots of xyzabc vs relative time 
        # plot_labels = ["x", "y", "z", "alpha", "beta", "gamma"] 
        # fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        # fig.suptitle("Relative Pose Comparison: Optitrack vs OpenCV")
        # for i in range(6): 
        #     axs[i//3, i%3].scatter(self.OPTK_time-self.OPTK_time[0], self.OPTK_xyzabc_mo_mi[:,i], label="Optitrack", color='blue') 
        #     axs[i//3, i%3].scatter(self.CCV_time-self.CCV_time[0], self.CCV_xyzabc_mo_mi[:,i], label="OpenCV", color='red') 
        #     axs[i//3, i%3].set_title(plot_labels[i]) 
        #     axs[i//3, i%3].legend()
        #     axs[i//3, i%3].grid()
        # plt.tight_layout()
        # plt.show()

        # absolute pose comparison plots 
        plot_labels = ["x", "y", "z", "alpha", "beta", "gamma"] 
        fig_abs, axs_abs = plt.subplots(2, 3, figsize=(15, 10)) 
        fig_abs.suptitle("Absolute Pose Comparison: Optitrack vs OpenCV")
        for i in range(6):
            axs_abs[i//3, i%3].scatter(self.OPTK_time-self.OPTK_time[0], self.OPTK_xyzabc_w_m[:,i], label="Optitrack", color='blue', s=5)
            axs_abs[i//3, i%3].scatter(self.CCV_time-self.CCV_time[0], self.CCV_xyzabc_w_m[:,i], label="OpenCV", color='red', s=5)
            axs_abs[i//3, i%3].set_title(plot_labels[i])
            axs_abs[i//3, i%3].legend()
            axs_abs[i//3, i%3].grid()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        return 
    

# MAIN SCRIPT 
def main(): 
    trial_idx = 4 
    # fx, fy, cx, cy = 388.505, 388.505, 317.534, 237.229 # from realsense camera 
    # dist_coeffs = np.zeros(5) 
    fx, fy, cx, cy = 722, 698, 310, 272 # from charuco board calibration 
    dist_coeffs = np.array([-1.24113729e-01, 1.64488988e+00, -9.82401198e-03, -5.07274595e-04, -8.23426373e+00]) # from charuco board calibration 
    
    tf_w_CS200 = np.eye(4)
    tf_CS200_mount = np.array([
        [1/np.sqrt(2), 0, -1/np.sqrt(2), 10e-3],
        [0,-1,0,7e-3],
        [-1/np.sqrt(2), 0, -1/np.sqrt(2), 110e-3],  
        [0,0,0,1]
    ])
    tf_mount_camera = np.array([
        [1,0,0,-32.5e-3],
        [0,1,0,-12.5e-3],
        [0,0,1,+14.8e-3],
        [0,0,0,1] 
    ])
    tf_w_c = tf_w_CS200 @ tf_CS200_mount @ tf_mount_camera 

    config = {
        "trial_idx": trial_idx,  
        "optitrack_csv_file": f"./real_data_processing/raw_data/optitrack/optitrack_{trial_idx}.csv",
        "realsense_video_file": f"./real_data_processing/raw_data/realsense/realsense_{trial_idx}.mp4", 
        "tf_trackmark_fidumark": np.eye(4), 
        "camera_intrinsic_matrix": np.array([[fx,0,cx],[0,fy,cy],[0,0,1]]), 
        "camera_dist_coeffs": dist_coeffs,
        "aruco_dict": cv2.aruco.DICT_APRILTAG_36h11, 
        "marker_length": 0.1, 
        "camera_extrinsic_matrix": tf_w_c, # FIXME: placeholder value 
    }
    
    ORDP = optitrack_realsense_data_processor(optitrack_csv_file=config["optitrack_csv_file"], 
                                             realsense_video_file=config["realsense_video_file"]) 
    ORDP.set_tf_trackmark_fidumark(config["tf_trackmark_fidumark"]) 
    ORDP.set_fiducial_marker_detector(
        camera_matrix=config["camera_intrinsic_matrix"], 
        dist_coeffs=config["camera_dist_coeffs"], 
        aruco_dict=config["aruco_dict"], 
        marker_length=config["marker_length"], 
        tf_optitrack_camera=config["camera_extrinsic_matrix"]
    ) 
    ORDP.process_optitrack_data() 
    ORDP.run_opencv_fiducial_marker_detection(save_results=True) 
    ORDP.compare_marker_poses() 

    import pdb; pdb.set_trace()  # Set a breakpoint for debugging 

    return True 

if __name__ == "__main__": 
    main()  
