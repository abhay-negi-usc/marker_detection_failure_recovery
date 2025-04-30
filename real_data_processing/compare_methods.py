import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# MODULE IMPORTS
sys.path.append(os.curdir)
from real_data_processing.utils import *

class datapoint:
    def __init__(self, image_path):
        self.image_path = image_path 
        self.image = None 

    def get_image(self):
        self.image = cv2.imread(self.image_path) 
        return self.image 
    
    def forget_image(self):
        self.image = None 

    def set_time(self, time):   
        self.time = time

    def _set_pose(self, pose):
        if pose is None:
            return None, None, False  
        elif pose.shape == (4, 4): 
            tf = pose 
            xyzabc = tf_to_xyzabc(pose) 
            return tf, xyzabc, True 
        elif pose.shape == (6,):
            xyzabc = pose 
            tf = xyzabc_to_tf(pose) 
            return tf, xyzabc, True 
        else: 
            raise ValueError("Invalid pose shape. Expected (4, 4) or (6).") 

    def set_OPTK_tf_pose(self, pose): 
        self.OPTK_tf, self.OPTK_pose, self.OPTK_detected = self._set_pose(pose) 

    def set_CCV_tf_pose(self, pose): 
        self.CCV_tf, self.CCV_pose, self.CCV_detected = self._set_pose(pose)

        
    def set_LBCV_tf_pose(self, pose): 
        self.LBCV_tf, self.LBCV_pose, self.LBCV_detected = self._set_pose(pose)

    def __repr__(self):
        return f"datapoint(name={self.name}, method={self.method_name})" 
    
class dataset_processor: 
    def __init__(self, config):
        self.config = config
        self._load_parameters()
        self._load_optitrack_data()
        self._load_realsense_data()
        self._set_datapoints() 

    def _load_parameters(self):
        self.optitrack_csv_file = self.config["optitrack_csv_file"]
        self.realsense_video_file = self.config["realsense_video_file"]
        self.t_offset_optk_ccv = self.config["t_offset_optk_ccv"]
        self.tf_trackmark_fidumark = self.config["tf_trackmark_fidumark"]
        self.camera_matrix = self.config["camera_intrinsic_matrix"]
        self.dist_coeffs = self.config["camera_dist_coeffs"]
        self.aruco_dict = self.config["aruco_dict"]
        self.marker_length = self.config["marker_length"]
        self.tf_w_c = self.config["camera_extrinsic_matrix"]
        self.tf_c_w = np.linalg.inv(self.tf_w_c)

    def _load_optitrack_data(self):
        if not os.path.isfile(self.optitrack_csv_file):
            raise FileNotFoundError(f"Optitrack CSV file {self.optitrack_csv_file} not found.")
        df = pd.read_csv(self.optitrack_csv_file)
        df.rename(columns={"Time (Seconds)": "TIME", "QX": "QX", "QY": "QY", "QZ": "QZ", "QW": "QW", "X": "X", "Y": "Y", "Z": "Z"}, inplace=True)
        df = df[["TIME", "X", "Y", "Z", "QW", "QX", "QY", "QZ"]].dropna()
        self.optitrack_data = df.to_numpy()

    def _load_realsense_data(self):
        frames_dir = Path(self.realsense_video_file.replace('.mp4', '_frames'))
        if not frames_dir.exists() or not any(frames_dir.glob('*.png')):
            frames_dir.mkdir(parents=True, exist_ok=True)
            self.RLSN_time = split_video_to_frames(self.realsense_video_file, str(frames_dir), get_timestamps=True)
        else:
            cap = cv2.VideoCapture(self.realsense_video_file)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.RLSN_time = np.linspace(0, total_frames / fps, total_frames, endpoint=False)
            cap.release()

        self.realsense_frames_paths = sorted(frames_dir.glob('*.png'))
        max_frames = self.config.get("max_frames", None)
        if max_frames and len(self.realsense_frames_paths) > max_frames:
            step = len(self.realsense_frames_paths) // max_frames
            self.realsense_frames_paths = self.realsense_frames_paths[::step]
            self.RLSN_time = self.RLSN_time[::step]

    def _set_datapoints(self):
        self.datapoints = []
        for idx, frame_path in enumerate(self.realsense_frames_paths):
            dp = datapoint(frame_path)
            dp.set_time(self.RLSN_time[idx]) 
            self.datapoints.append(dp) 

    def process_optitrack_data(self):
        tf_w_o, tf_w_m, tf_mo_mi, timestamps, tf_c_m = [], [], [], [], []
        tf_mo_w = None

        for row in self.optitrack_data:
            if np.any(np.isnan(row)) or len(row) < 8:
                continue
            timestamp = row[0]
            timestamps.append(timestamp)
            xyzquat = row[1:8]
            tf_w_o_i = xyzquat_to_tf(xyzquat, input_qw_first=True)
            tf_w_o.append(tf_w_o_i)
            tf_w_m_i = tf_w_o_i @ self.tf_trackmark_fidumark
            tf_w_m.append(tf_w_m_i)

            if tf_mo_w is None:
                tf_mo_w = np.linalg.inv(tf_w_m_i)
                tf_mo_mi_i = np.eye(4)
            else:
                tf_mo_mi_i = tf_mo_w @ tf_w_m_i

            tf_mo_mi.append(tf_mo_mi_i)
            tf_c_m_i = self.tf_c_w @ tf_w_m_i
            tf_c_m.append(tf_c_m_i)

        self.OPTK_time = np.array(timestamps) - timestamps[0] + self.t_offset_optk_ccv
        self.OPTK_tf_w_o = np.array(tf_w_o)
        self.OPTK_tf_w_m = np.array(tf_w_m)
        self.OPTK_tf_mo_mi = np.array(tf_mo_mi)
        self.OPTK_tf_c_m = np.array(tf_c_m)

        print(f"Optitrack detection rate: {len(self.OPTK_time)}/{len(self.optitrack_data)}")

        for dp in self.datapoints:
            closest_idx = np.argmin(np.abs(self.OPTK_time - dp.time))
            dp.set_OPTK_tf_pose(self.OPTK_tf_c_m[closest_idx]) 

    def set_marker_detector(self):
        self.detector = cv2.aruco.getPredefinedDictionary(self.aruco_dict)

    def run_opencv_fiducial_marker_detection(self, save_results=False):
        timestamps, tf_c_m, tf_w_m, tf_mo_mi = [], [], [], []
        tf_c_m_all = [] 
        tf_mo_w = None
        output_dir = Path(self.realsense_video_file.replace('.mp4', '_frames_CCV'))
        if save_results:
            output_dir.mkdir(parents=True, exist_ok=True)

        for idx, frame_path in enumerate(self.realsense_frames_paths):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                tf_c_m_all.append(None) 
                continue 

            marker_ids, rvecs, tvecs, corners = marker_pose_estimation_estimatePoseSingleMarkers(
                frame, self.camera_matrix, self.dist_coeffs, self.aruco_dict, self.marker_length, show=False
            )

            if rvecs is None or tvecs is None:
                tf_c_m_all.append(None)
                continue 

            timestamps.append(self.RLSN_time[idx])
            tf_Ccv_Mcv = np.eye(4)
            tf_Ccv_Mcv[:3, :3] = cv2.Rodrigues(rvecs[0])[0]
            tf_Ccv_Mcv[:3, 3] = tvecs[0].reshape(3)

            tf_w_m_i = self.tf_w_c @ tf_Ccv_Mcv
            tf_c_m.append(tf_Ccv_Mcv)
            tf_w_m.append(tf_w_m_i)
            tf_c_m_all.append(tf_Ccv_Mcv) 

            if tf_mo_w is None:
                tf_mo_w = np.linalg.inv(tf_w_m_i)
                tf_mo_mi_i = np.eye(4)
            else:
                tf_mo_mi_i = tf_mo_w @ tf_w_m_i
            tf_mo_mi.append(tf_mo_mi_i)

            if save_results and corners:
                frame_markers = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, marker_ids)
                cv2.imwrite(str(output_dir / f"CCV_frame_{idx:05d}.png"), frame_markers)

        self.CCV_time = np.array(timestamps) - timestamps[0]
        self.CCV_tf_c_m = np.array(tf_c_m)
        self.CCV_tf_w_m = np.array(tf_w_m)
        self.CCV_tf_mo_mi = np.array(tf_mo_mi) 

        for idx, dp in enumerate(self.datapoints):
            dp.set_CCV_tf_pose(tf_c_m_all[idx]) 

    def __repr__(self):
        return f"dataset(name={self.name}, datapoints={self.datapoints})"


# MAIN SCRIPT

def main():
    trial_idx = 5
    calibration = "charuco_415"

    calibration_configs = {
        "realsense_435": (388.505, 388.505, 317.534, 237.229, np.zeros(5)),
        "charuco_435": (722, 698, 310, 272, np.array([-0.124113729, 1.64488988, -0.00982401198, -0.000507274595, -8.23426373])),
        "realsense_415": (1360.49, 1360.49, 957.355, 540.8, np.zeros(5)),
        "charuco_415": (1363.85031, 1365.39541, 958.580898, 552.245426, np.array([0.16925566, -0.47551045, 0.00181413, 0.0022729, 0.4113696]))
    }

    fx, fy, cx, cy, dist_coeffs = calibration_configs[calibration]

    tf_w_CS200 = np.array([
        [1, 0, 0, -1.67e-3],
        [0, -1, 0, 25.365e-3],
        [0, 0, -1, 220.095e-3],
        [0, 0, 0, 1]
    ])

    tf_CS200_mount = np.array([
        [0, 0, -1, 10.0e-3],
        [0, -1, 0, 1.2e-3],
        [-1, 0, 0, 110.0e-3],
        [0, 0, 0, 1]
    ])

    tf_mount_camera = np.array([
        [1, 0, 0, -35e-3],
        [0, 1, 0, -11.5e-3],
        [0, 0, 1, 9.8e-3],
        [0, 0, 0, 1]
    ])

    tf_w_c = tf_w_CS200 @ tf_CS200_mount @ tf_mount_camera

    tf_trackmark_fidumark = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, -20.46e-3],
        [0, 0, 0, 1]
    ])

    config = {
        "trial_idx": trial_idx,
        "optitrack_csv_file": f"./real_data_processing/raw_data/optitrack/optitrack_{trial_idx}.csv",
        "realsense_video_file": f"./real_data_processing/raw_data/realsense/realsense_{trial_idx}.mp4",
        "tf_trackmark_fidumark": tf_trackmark_fidumark,
        "camera_intrinsic_matrix": np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
        "camera_dist_coeffs": dist_coeffs,
        "aruco_dict": cv2.aruco.DICT_APRILTAG_36h11,
        "marker_length": 0.0798,
        "camera_extrinsic_matrix": tf_w_c,
        "t_offset_optk_ccv": 1.9,
        "max_frames": 100 
    }

    processor = dataset_processor(config)
    processor.set_marker_detector()
    processor.process_optitrack_data()
    processor.run_opencv_fiducial_marker_detection(save_results=False) 
    # processor.compare_marker_poses(show=True) 
    # processor.reproject_optitrack_pose_estimates(save_dir=f"./real_data_processing/raw_data/OPTK_reprojections_{trial_idx}", show=False)

    import pdb; pdb.set_trace() 

if __name__ == "__main__":
    main()
