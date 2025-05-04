import numpy as np
import pandas as pd
import cv2
import logging
from pathlib import Path
from typing import Optional, Tuple, List
from real_data_processing.utils import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Constants for config keys
OPTITRACK_CSV_FILE = "optitrack_csv_file"
REALSENSE_VIDEO_FILE = "realsense_video_file"
CAMERA_INTRINSIC_MATRIX = "camera_intrinsic_matrix"
CAMERA_DIST_COEFFS = "camera_dist_coeffs"
CAMERA_EXTRINSIC_MATRIX = "camera_extrinsic_matrix"
ARUCO_DICT = "aruco_dict"
MARKER_LENGTH = "marker_length"
TF_TRACKMARK_FIDUMARK = "tf_trackmark_fidumark"
T_OFFSET_OPTK_CCV = "t_offset_optk_ccv"
MAX_FRAMES = "max_frames"

class DataPoint:
    def __init__(self, image_path: Path):
        self.image_path = image_path
        self.image: Optional[np.ndarray] = None
        self.time: Optional[float] = None

    def get_image(self) -> Optional[np.ndarray]:
        self.image = cv2.imread(str(self.image_path))
        if self.image is None:
            logging.warning(f"Failed to read image: {self.image_path}")
        return self.image

    def forget_image(self):
        self.image = None

    def set_time(self, time: float):
        self.time = time

    def _set_pose(self, pose: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
        if pose is None or not isinstance(pose, np.ndarray):
            return None, None, False
        if pose.shape == (4, 4):
            tf = pose
            xyzabc = tf_to_xyzabc(tf)
            return tf, xyzabc, True
        elif pose.shape == (6,):
            xyzabc = pose
            tf = xyzabc_to_tf(xyzabc)
            return tf, xyzabc, True
        else:
            raise ValueError(f"Invalid pose shape {pose.shape}. Expected (4, 4) or (6,)")

    def set_pose(self, method: str, pose: np.ndarray):
        tf, xyzabc, detected = self._set_pose(pose)
        setattr(self, f"{method}_tf", tf)
        setattr(self, f"{method}_pose", xyzabc)
        setattr(self, f"{method}_detected", detected)

    def __repr__(self):
        return f"DataPoint(path={self.image_path.name}, time={self.time})"
class DatasetProcessor:
    def __init__(self, config: dict):
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
        assert self.camera_matrix.shape == (3, 3), "Camera matrix must be 3x3"
        self.dist_coeffs = self.config["camera_dist_coeffs"]
        self.aruco_dict = self.config["aruco_dict"]
        self.marker_length = self.config["marker_length"]
        self.tf_w_c = self.config["camera_extrinsic_matrix"]
        self.tf_c_w = np.linalg.inv(self.tf_w_c)

    def _load_optitrack_data(self):
        if not Path(self.optitrack_csv_file).exists():
            raise FileNotFoundError(f"Missing OptiTrack CSV: {self.optitrack_csv_file}")
        df = pd.read_csv(self.optitrack_csv_file)
        df = df[["Time (Seconds)", "X", "Y", "Z", "QW", "QX", "QY", "QZ"]].dropna()
        df.rename(columns={"Time (Seconds)": "TIME"}, inplace=True)
        self.optitrack_data = df.to_numpy()

    def _load_realsense_data(self):
        frames_dir = Path(self.realsense_video_file.replace('.mp4', '_frames'))

        if not frames_dir.exists() or not any(frames_dir.glob('*.png')):
            frames_dir.mkdir(parents=True, exist_ok=True)
            self.RLSN_time, self.realsense_frames_paths = split_video_to_frames(
                self.realsense_video_file, str(frames_dir), get_timestamps=True
            )
        else:
            cap = cv2.VideoCapture(self.realsense_video_file)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.RLSN_time = np.linspace(0, total_frames / fps, total_frames, endpoint=False)
            self.realsense_frames_paths = sorted(frames_dir.glob('*.png'))
            cap.release()

        max_frames = self.config.get("max_frames", None)
        if max_frames and len(self.realsense_frames_paths) > max_frames:
            step = len(self.realsense_frames_paths) // max_frames
            self.realsense_frames_paths = self.realsense_frames_paths[::step]
            self.RLSN_time = self.RLSN_time[::step]

    def _set_datapoints(self):
        self.datapoints = []
        for idx, frame_path in enumerate(self.realsense_frames_paths):
            dp = DataPoint(frame_path)
            dp.set_time(self.RLSN_time[idx])
            self.datapoints.append(dp)

    def set_marker_detector(self):
        self.detector = cv2.aruco.getPredefinedDictionary(self.aruco_dict)

    def process_optitrack_data(self):
        tf_w_o, tf_w_m, tf_mo_mi, tf_c_m, timestamps = [], [], [], [], []
        tf_mo_w = None

        for row in self.optitrack_data:
            if np.any(np.isnan(row)) or len(row) < 8:
                continue
            timestamp = row[0]
            timestamps.append(timestamp)
            xyzquat = row[1:8]
            tf_w_o_i = xyzquat_to_tf(xyzquat, input_qw_first=True)
            tf_w_m_i = tf_w_o_i @ self.tf_trackmark_fidumark

            tf_w_o.append(tf_w_o_i)
            tf_w_m.append(tf_w_m_i)
            tf_c_m.append(self.tf_c_w @ tf_w_m_i)

            if tf_mo_w is None:
                tf_mo_w = np.linalg.inv(tf_w_m_i)
                tf_mo_mi.append(np.eye(4))
            else:
                tf_mo_mi.append(tf_mo_w @ tf_w_m_i)

        self.OPTK_time = np.array(timestamps) - timestamps[0] + self.t_offset_optk_ccv
        self.OPTK_tf_w_o = np.array(tf_w_o)
        self.OPTK_tf_w_m = np.array(tf_w_m)
        self.OPTK_tf_mo_mi = np.array(tf_mo_mi)
        self.OPTK_tf_c_m = np.array(tf_c_m)

        logger.info(f"[OptiTrack] Detection rate: {len(self.OPTK_time)}/{len(self.optitrack_data)}")

        # Faster lookup by vectorized abs-min
        optk_times = self.OPTK_time
        for dp in self.datapoints:
            idx = np.argmin(np.abs(optk_times - dp.time))
            dp.set_pose("OPTK", self.OPTK_tf_c_m[idx])

    def run_opencv_fiducial_marker_detection(self, save_results=False):
        tf_c_m_all = []
        tf_w_m, tf_mo_mi, timestamps = [], [], []
        tf_mo_w = None

        output_dir = Path(self.realsense_video_file.replace('.mp4', '_frames_CCV'))
        if save_results:
            output_dir.mkdir(parents=True, exist_ok=True)

        for idx, frame_path in enumerate(self.realsense_frames_paths):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                logger.warning(f"[CCV] Could not read frame {frame_path}")
                tf_c_m_all.append(None)
                continue

            marker_ids, rvecs, tvecs, corners = marker_pose_estimation_estimatePoseSingleMarkers(
                frame, self.camera_matrix, self.dist_coeffs, self.aruco_dict,
                self.marker_length, show=False
            )

            if rvecs is None or tvecs is None:
                tf_c_m_all.append(None)
                continue

            tf_Ccv_Mcv = np.eye(4)
            tf_Ccv_Mcv[:3, :3] = cv2.Rodrigues(rvecs[0])[0]
            tf_Ccv_Mcv[:3, 3] = tvecs[0].reshape(3)

            tf_c_m_all.append(tf_Ccv_Mcv)
            tf_w_m_i = self.tf_w_c @ tf_Ccv_Mcv
            tf_w_m.append(tf_w_m_i)

            if tf_mo_w is None:
                tf_mo_w = np.linalg.inv(tf_w_m_i)
                tf_mo_mi.append(np.eye(4))
            else:
                tf_mo_mi.append(tf_mo_w @ tf_w_m_i)

            timestamps.append(self.RLSN_time[idx])

            if save_results and corners:
                img = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, marker_ids)
                outpath = output_dir / f"CCV_frame_{idx:05d}.png"
                cv2.imwrite(str(outpath), img)

        self.CCV_time = np.array(timestamps) - timestamps[0]
        self.CCV_tf_c_m = np.array([t for t in tf_c_m_all if t is not None])
        self.CCV_tf_w_m = np.array(tf_w_m)
        self.CCV_tf_mo_mi = np.array(tf_mo_mi)

        for idx, dp in enumerate(self.datapoints):
            dp.set_pose("CCV", tf_c_m_all[idx] if idx < len(tf_c_m_all) else None)

    def __repr__(self):
        return f"DatasetProcessor(num_datapoints={len(self.datapoints)})"

    
def main():
    import logging
    logging.basicConfig(level=logging.INFO)

    # === Trial and Calibration Parameters ===
    trial_idx = 5
    calibration_name = "charuco_415"

    calibration_configs = {
        "realsense_435": (388.505, 388.505, 317.534, 237.229, np.zeros(5)),
        "charuco_435": (722, 698, 310, 272, np.array([-0.1241, 1.6449, -0.0098, -0.0005, -8.2343])),
        "realsense_415": (1360.49, 1360.49, 957.355, 540.8, np.zeros(5)),
        "charuco_415": (
            1363.85, 1365.40, 958.58, 552.25,
            np.array([0.1693, -0.4755, 0.0018, 0.0023, 0.4114])
        )
    }

    if calibration_name not in calibration_configs:
        raise ValueError(f"Unknown calibration: {calibration_name}")
    
    fx, fy, cx, cy, dist_coeffs = calibration_configs[calibration_name]

    # === Extrinsic Transforms ===
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

    # === Config ===
    config = {
        "trial_idx": trial_idx,
        OPTITRACK_CSV_FILE: f"./real_data_processing/raw_data/optitrack/optitrack_{trial_idx}.csv",
        REALSENSE_VIDEO_FILE: f"./real_data_processing/raw_data/realsense/realsense_{trial_idx}.mp4",
        TF_TRACKMARK_FIDUMARK: tf_trackmark_fidumark,
        CAMERA_INTRINSIC_MATRIX: np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
        CAMERA_DIST_COEFFS: dist_coeffs,
        ARUCO_DICT: cv2.aruco.DICT_APRILTAG_36h11,
        MARKER_LENGTH: 0.0798,
        CAMERA_EXTRINSIC_MATRIX: tf_w_c,
        T_OFFSET_OPTK_CCV: 1.9,
        MAX_FRAMES: 100
    }

    # === Run Processing ===
    processor = DatasetProcessor(config)
    processor.set_marker_detector()
    processor.process_optitrack_data()
    processor.run_opencv_fiducial_marker_detection(save_results=True)

    logging.info("Data processing complete.")

if __name__ == "__main__":
    main()