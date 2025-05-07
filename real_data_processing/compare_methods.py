# FIXME: 
# delete segmentation mask full output 
# segmenation masks output has wrong channel permutation 


import numpy as np
import pandas as pd
import cv2
import logging
from pathlib import Path
from typing import Optional, Tuple, List
from real_data_processing.utils import *
import PIL 
from PIL import Image
import json
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from keypoints_model.utils import overlay_points_on_image


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
OUT_DIR = "out_dir" 

class DatasetProcessor:
    def __init__(self, config: dict):
        self.config = config
        self._load_parameters()
        self._load_optitrack_data()
        self._load_realsense_data()
        self._set_datapoints()
        self.out_dir = self.config[OUT_DIR] 
        os.makedirs(self.out_dir, exist_ok=True)

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
            # FIXME: revert 
            # step = len(self.realsense_frames_paths) // max_frames
            # self.realsense_frames_paths = self.realsense_frames_paths[::step]
            # self.RLSN_time = self.RLSN_time[::step]
            self.realsense_frames_paths = self.realsense_frames_paths[:max_frames]
            self.RLSN_time = self.RLSN_time[:max_frames]

    def _set_datapoints(self):
        self.datapoints = []
        for idx, frame_path in enumerate(self.realsense_frames_paths):
            dp = DataPoint(frame_path)
            dp.set_time(self.RLSN_time[idx])
            dp.set_camera_matrix(self.camera_matrix)
            dp.set_marker_length(self.marker_length)
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
            # get root directory of the optitrack csv file 
            self.optk_frames_dir = os.path.join(f"./real_data_processing/raw_data/realsense/realsense_{self.config["trial_idx"]}_frames_OPTK")
            dp.save_overlay("OPTK", square_length=self.marker_length, K=self.camera_matrix, output_dir=self.optk_frames_dir)

    def run_opencv_fiducial_marker_detection(self, save_results=False):
        import cv2
        from pathlib import Path
        import logging

        tf_c_m_all = []
        tf_w_m, tf_mo_mi, timestamps = [], [], []
        tf_mo_w = None
        detected = [] 

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

            out_img = frame.copy()
            pose_detected = False

            if rvecs is not None and tvecs is not None:
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

                pose_detected = True
            else:
                tf_c_m_all.append(None)

            timestamps.append(self.RLSN_time[idx])
            detected.append(pose_detected)

            if save_results:
                if pose_detected:
                    out_img = cv2.aruco.drawDetectedMarkers(out_img, corners, marker_ids)
                outpath = output_dir / f"CCV_frame_{idx:05d}.png"
                cv2.imwrite(str(outpath), out_img)

        self.CCV_time = np.array(timestamps) - timestamps[0]
        self.CCV_tf_c_m = np.array([t for t in tf_c_m_all if t is not None])
        self.CCV_tf_w_m = np.array(tf_w_m)
        self.CCV_tf_mo_mi = np.array(tf_mo_mi)
        self.CCV_detected = detected 

        for idx, dp in enumerate(self.datapoints):
            dp.set_pose("CCV", tf_c_m_all[idx] if idx < len(tf_c_m_all) else None)

    def run_learning_based_detection(self, predict_fn, save_results=False, save_segmentation=False):
        from PIL import Image
        import json
        import numpy as np
        import cv2
        from keypoints_model.utils import overlay_points_on_image

        tf_c_m_all = []
        tf_w_m, tf_mo_mi, timestamps = [], [], []
        tf_mo_w = None
        detected = [] 

        output_dir = Path(self.realsense_video_file.replace('.mp4', '_frames_LBCV'))
        if save_results:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "keypoints_overlay").mkdir(exist_ok=True)
            (output_dir / "keypoints_json").mkdir(exist_ok=True)
        if save_segmentation:
            (output_dir / "segmentation_masks").mkdir(parents=True, exist_ok=True)
            (output_dir / "segmentation_masks_full").mkdir(parents=True, exist_ok=True)

        for idx, dp in enumerate(self.datapoints):
            image = dp.get_image()
            if image is None:
                tf_c_m_all.append(None)
                continue

            # Run LBCV predictor (2- or 3-return format)
            result = predict_fn(image)
            if isinstance(result, tuple) and len(result) == 4:
                tf_marker, keypoints, segmentation_mask, segmentation_mask_full = result

            tf_c_m_all.append(tf_marker)

            if tf_marker is None:
                detected.append(False) 
                # Save fallback results
                if save_results:
                    # Raw image
                    overlay_path = output_dir / "keypoints_overlay" / f"LBCV_frame_{idx:05d}.png"
                    Image.fromarray(image).save(str(overlay_path))

                    # Empty keypoints
                    json_path = output_dir / "keypoints_json" / f"LBCV_frame_{idx:05d}.json"
                    with open(json_path, "w") as f:
                        json.dump([], f)

                if save_segmentation:
                    seg_path = output_dir / "segmentation_masks" / f"LBCV_seg_{idx:05d}.png"
                    blank_mask = Image.fromarray(np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8))
                    blank_mask.save(str(seg_path))

                    seg_path = output_dir / "segmentation_masks_full" / f"LBCV_seg_{idx:05d}.png"
                    blank_mask = Image.fromarray(np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8))
                    blank_mask.save(str(seg_path))
                continue
            else: 
                detected.append(True) 

            tf_w_m_i = self.tf_w_c @ tf_marker
            tf_w_m.append(tf_w_m_i)

            if tf_mo_w is None:
                tf_mo_w = np.linalg.inv(tf_w_m_i)
                tf_mo_mi.append(np.eye(4))
            else:
                tf_mo_mi.append(tf_mo_w @ tf_w_m_i)

            timestamps.append(dp.time)
            dp.set_pose("LBCV", tf_marker)

            # --- Save keypoints visualization and JSON ---
            if save_results and keypoints is not None:
                overlay = overlay_points_on_image(image.copy(), keypoints, radius=3)
                overlay_path = output_dir / "keypoints_overlay" / f"LBCV_frame_{idx:05d}.png"
                Image.fromarray(overlay).save(str(overlay_path))

                json_path = output_dir / "keypoints_json" / f"LBCV_frame_{idx:05d}.json"
                with open(json_path, "w") as f:
                    json.dump(keypoints.tolist(), f)

            # --- Save segmentation mask if requested ---
            if save_segmentation and segmentation_mask is not None:
                seg_path = output_dir / "segmentation_masks" / f"LBCV_seg_{idx:05d}.png"
                segmentation_mask.convert("L").save(str(seg_path))

                seg_path = output_dir / "segmentation_masks_full" / f"LBCV_seg_{idx:05d}.png"
                segmentation_mask_full.convert("L").save(str(seg_path))

        self.LBCV_time = np.array(timestamps) - timestamps[0] if timestamps else None
        self.LBCV_tf_c_m = np.array([t for t in tf_c_m_all if t is not None])
        self.LBCV_tf_w_m = np.array(tf_w_m)
        self.LBCV_tf_mo_mi = np.array(tf_mo_mi)
        self.LBCV_detected = detected 

    def compare_detection(self, num_bins=10): 
        # compute the fraction of marker detects for each method 
        ccv_detects = np.sum(self.CCV_detected) / len(self.datapoints)
        lbcv_detects = np.sum(self.LBCV_detected) / len(self.datapoints) 

        # save a bar chart 
        sns.set_theme(style="whitegrid")
        methods = ["CCV", "LBCV"]
        detection_rates = [ccv_detects, lbcv_detects]

        plt.figure(figsize=(8, 6))
        ax = sns.barplot(x=methods, y=detection_rates)

        # Add labels above bars
        for i, rate in enumerate(detection_rates):
            plt.text(i, rate + 0.02, f"{rate:.2f}", ha='center', va='bottom', fontsize=12)

        plt.title("Detection Rates")
        plt.ylabel("Detection Rate")
        plt.ylim(0, 1.05)  # extend limit slightly to make space for label
        plt.savefig(os.path.join(self.out_dir, "detection_rates.png"))
        plt.close()

        logger.info(f"[Detection Rates] CCV: {ccv_detects:.2f}, LBCV: {lbcv_detects:.2f}")

        # plot detection rate vs marker brightness 
        brightness = np.array([dp.get_marker_brightness() for dp in self.datapoints])
        bin_edges = np.linspace(brightness.min(), brightness.max(), num_bins + 1)

        # Prepare counters
        ccv_detections = np.zeros(num_bins)
        lbcv_detections = np.zeros(num_bins)
        total_counts = np.zeros(num_bins)

        # Count detections and totals per brightness bin
        for dp in self.datapoints:
            b = dp.marker_brightness
            bin_idx = np.searchsorted(bin_edges, b, side='right') - 1
            bin_idx = min(max(bin_idx, 0), num_bins - 1)

            total_counts[bin_idx] += 1
            if hasattr(dp, 'CCV_tf') and dp.CCV_tf is not None:
                ccv_detections[bin_idx] += 1
            if hasattr(dp, 'LBCV_tf') and dp.LBCV_tf is not None:
                lbcv_detections[bin_idx] += 1

        # Avoid divide-by-zero
        with np.errstate(divide='ignore', invalid='ignore'):
            ccv_rates = np.where(total_counts > 0, ccv_detections / total_counts, 0)
            lbcv_rates = np.where(total_counts > 0, lbcv_detections / total_counts, 0)

        # Midpoints of bins for plotting
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        plt.figure()
        plt.plot(bin_centers, ccv_rates, label="CCV Detection Rate", marker='o')
        plt.plot(bin_centers, lbcv_rates, label="LBCV Detection Rate", marker='x')
        plt.xlabel("Marker Brightness")
        plt.ylabel("Detection Rate")
        plt.title("Detection Rate vs Marker Brightness")
        plt.ylim(0, 1.05)
        plt.grid(True)
        plt.legend()
        # plt.show()
        plt.savefig(os.path.join(self.out_dir, "marker_brightness_detection_rate.png")) 
        plt.close()

        # plot detection rate vs marker area 
        areas = np.array([dp.get_marker_area() for dp in self.datapoints])
        bin_edges = np.linspace(areas.min(), areas.max(), num_bins + 1)
        # Prepare counters
        ccv_detections = np.zeros(num_bins)
        lbcv_detections = np.zeros(num_bins)
        total_counts = np.zeros(num_bins)
        # Count detections and totals per brightness bin
        for dp in self.datapoints:  
            a = dp.marker_area
            bin_idx = np.searchsorted(bin_edges, a, side='right') - 1
            bin_idx = min(max(bin_idx, 0), num_bins - 1)

            total_counts[bin_idx] += 1
            if hasattr(dp, 'CCV_tf') and dp.CCV_tf is not None:
                ccv_detections[bin_idx] += 1
            if hasattr(dp, 'LBCV_tf') and dp.LBCV_tf is not None:
                lbcv_detections[bin_idx] += 1
        # Avoid divide-by-zero
        with np.errstate(divide='ignore', invalid='ignore'):
            ccv_rates = np.where(total_counts > 0, ccv_detections / total_counts, 0)
            lbcv_rates = np.where(total_counts > 0, lbcv_detections / total_counts, 0)
        # Midpoints of bins for plotting
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.figure()
        plt.plot(bin_centers, ccv_rates, label="CCV Detection Rate", marker='o')
        plt.plot(bin_centers, lbcv_rates, label="LBCV Detection Rate", marker='x')
        plt.xlabel("Marker Area")
        plt.ylabel("Detection Rate")
        plt.title("Detection Rate vs Marker Area")
        plt.ylim(0, 1.05)
        plt.grid(True)
        plt.legend()
        # plt.show()
        plt.savefig(os.path.join(self.out_dir, "marker_area_detection_rate.png")) 

    def compare_pose_estimation(self): 
        # for each datapoint, for each method, compute the error in position and orientation
        self.CCV_translation_errors = [] 
        self.CCV_rotation_errors = [] 
        self.LBCV_translation_errors = [] 
        self.LBCV_rotation_errors = [] 
        
        for dp in self.datapoints:
            CCV_error, LBCV_error  = dp.compute_errors()  
            if CCV_error is not None:
                self.CCV_translation_errors.append(CCV_error["translation"])
                self.CCV_rotation_errors.append(CCV_error["rotation"])
            else:
                self.CCV_translation_errors.append(None)
                self.CCV_rotation_errors.append(None)
            if LBCV_error is not None:
                self.LBCV_translation_errors.append(LBCV_error["translation"])
                self.LBCV_rotation_errors.append(LBCV_error["rotation"])
            else:
                self.LBCV_translation_errors.append(None)
                self.LBCV_rotation_errors.append(None)

        brightness = np.array([dp.get_marker_brightness() for dp in self.datapoints])
        area = np.array([dp.get_marker_area() for dp in self.datapoints])
        
        # plot translation error vs marker brightness 
        plt.figure()
        plt.scatter(brightness, self.CCV_translation_errors, label="CCV Translation Error", alpha=0.5)
        plt.scatter(brightness, self.LBCV_translation_errors, label="LBCV Translation Error", alpha=0.5)
        plt.xlabel("Marker Brightness")
        plt.ylabel("Translation Error (m)")
        plt.title("Translation Error vs Marker Brightness")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.out_dir, "translation_error_vs_brightness.png"))

        # plot translation error vs marker area
        plt.figure()
        plt.scatter(area, self.CCV_translation_errors, label="CCV Translation Error", alpha=0.5)
        plt.scatter(area, self.LBCV_translation_errors, label="LBCV Translation Error", alpha=0.5)
        plt.xlabel("Marker Area")
        plt.ylabel("Translation Error (m)")
        plt.title("Translation Error vs Marker Area")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.out_dir, "translation_error_vs_area.png"))

        # plot rotation error vs marker brightness
        plt.figure()
        plt.scatter(brightness, self.CCV_rotation_errors, label="CCV Rotation Error", alpha=0.5)
        plt.scatter(brightness, self.LBCV_rotation_errors, label="LBCV Rotation Error", alpha=0.5)
        plt.xlabel("Marker Brightness")
        plt.ylabel("Rotation Error (degrees)")
        plt.title("Rotation Error vs Marker Brightness")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.out_dir, "rotation_error_vs_brightness.png"))

        # plot rotation error vs marker area
        plt.figure()
        plt.scatter(area, self.CCV_rotation_errors, label="CCV Rotation Error", alpha=0.5)
        plt.scatter(area, self.LBCV_rotation_errors, label="LBCV Rotation Error", alpha=0.5)
        plt.xlabel("Marker Area")
        plt.ylabel("Rotation Error (degrees)")
        plt.title("Rotation Error vs Marker Area")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.out_dir, "rotation_error_vs_area.png"))


def build_lbcv_predictor(
    seg_model_path: str,
    kp_model_path: str,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_image: 'PIL.Image.Image',
    marker_side_length: float,
    keypoints_tag_frame: np.ndarray,
    method: str = "kp" # "seg" or "kp"
):
    import numpy as np
    import torch
    import cv2
    from torchvision import transforms
    from segmentation_model.model import UNETWithDropout
    from segmentation_model.utils import load_checkpoint as load_seg_ckpt
    from keypoints_model.model import RegressorMobileNetV3
    from keypoints_model.utils import load_checkpoint as load_kp_ckpt
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from keypoints_model.utils import xyzabc_to_tf, rvectvec_to_xyzabc

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    seg_model = UNETWithDropout(in_channels=3, out_channels=1).to(DEVICE)
    load_seg_ckpt(torch.load(seg_model_path, map_location=DEVICE), seg_model)
    seg_model.eval()

    kp_model = RegressorMobileNetV3().to(DEVICE)
    load_kp_ckpt(torch.load(kp_model_path, map_location=DEVICE), kp_model)
    kp_model.eval()

    # tf_W_Ccv = np.array([
    #     [1, 0, 0, 0],
    #     [0, -1, 0, 0],
    #     [0, 0, -1, 0],
    #     [0, 0, 0, 1]
    # ])
    tf_W_Ccv = np.array([
        [-1,0,0,0],
        [0,-1,0,0],
        [0,0,1,0],
        [0,0,0,1],
    ]) # FIXME: don't know why the original tf_W_Ccv is not working 
    # tf_W_Ccv = np.eye(4)

    def compute_roi(seg, rgb):

        padding = 5
        roi_size = 128
        image_border_size = np.max([np.array(seg).shape[0], np.array(seg).shape[1]])

        seg = np.array(seg)
        seg = cv2.copyMakeBorder(seg, image_border_size, image_border_size, image_border_size, image_border_size, cv2.BORDER_CONSTANT, value=0)
        tag_pixels = np.argwhere(seg == 255)
        if tag_pixels.size == 0:
            return None, None

        seg_tag_min_x = np.min(tag_pixels[:, 1])
        seg_tag_max_x = np.max(tag_pixels[:, 1])
        seg_tag_min_y = np.min(tag_pixels[:, 0])
        seg_tag_max_y = np.max(tag_pixels[:, 0])
        seg_height = seg_tag_max_y - seg_tag_min_y
        seg_width = seg_tag_max_x - seg_tag_min_x
        seg_center_x = (seg_tag_min_x + seg_tag_max_x) // 2
        seg_center_y = (seg_tag_min_y + seg_tag_max_y) // 2

        if isinstance(rgb, str):
            rgb = np.array(cv2.imread(rgb))
        if isinstance(rgb, Image.Image):
            rgb = np.array(rgb)
        if isinstance(rgb, np.ndarray):
            rgb = rgb
        rgb = cv2.copyMakeBorder(rgb, image_border_size, image_border_size, image_border_size, image_border_size, cv2.BORDER_CONSTANT, value=0)

        rgb_side = max(seg_height, seg_width) + 2 * padding
        rgb_tag_min_x = seg_center_x - rgb_side // 2
        rgb_tag_max_x = seg_center_x + rgb_side // 2
        rgb_tag_min_y = seg_center_y - rgb_side // 2
        rgb_tag_max_y = seg_center_y + rgb_side // 2
        roi_img = rgb[rgb_tag_min_y:rgb_tag_max_y, rgb_tag_min_x:rgb_tag_max_x, :]
        roi_img = cv2.resize(roi_img, (roi_size, roi_size))
        roi_coordinates = np.array([rgb_tag_min_x, rgb_tag_max_x, rgb_tag_min_y, rgb_tag_max_y]) - image_border_size 

        return roi_img, roi_coordinates

    def predict_pose_from_image(image: np.ndarray):
        seg_size = (640, 480)

        # Resize RGB image to match segmentation input
        resized_rgb = cv2.resize(image, seg_size)  # shape (H, W, 3)

        # Segmentation transform: normalized for model
        seg_transform = A.Compose([
            A.Normalize(max_pixel_value=1.0),
            ToTensorV2()
        ])
        transformed = seg_transform(image=resized_rgb)
        img_tensor = transformed["image"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            seg_mask = torch.sigmoid(seg_model(img_tensor))
            seg_mask = (seg_mask > 0.5).float().cpu()

        seg_mask_img = transforms.ToPILImage()(seg_mask.squeeze(0))  # shape matches resized_rgb

        seg_transform_full = A.Compose([
            A.Normalize(max_pixel_value=1.0),
            ToTensorV2()
        ])
        transformed_full = seg_transform_full(image=image)
        img_tensor_full = transformed_full["image"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            seg_mask_full = torch.sigmoid(seg_model(img_tensor_full))
            seg_mask_full = (seg_mask_full > 0.5).float().cpu()

        seg_mask_img_full = transforms.ToPILImage()(seg_mask_full.squeeze(0))  # shape matches resized_rgb

        # No tag detected
        if np.array(seg_mask_img).max() == 0:
            return None, None, seg_mask_img, seg_mask_img_full

        # FIXME: hardcoding for now 
        method = "kp"

        if method == "seg": 
            # tf_marker = 
            return tf_marker, None, seg_mask_img, seg_mask_img_full 

        if method == "kp":
            # --- Compute ROI using resized RGB and seg ---
            roi_img, roi_coords = compute_roi(seg_mask_img, resized_rgb)
            if roi_img is None:
                return None, None, seg_mask_img, seg_mask_img_full

            # Keypoint transform (no resize)
            kp_transform = A.Compose([ToTensorV2()])
            roi_tensor = kp_transform(image=roi_img)["image"].unsqueeze(0).float().to(DEVICE)

            with torch.no_grad():
                keypoints_roi = kp_model(roi_tensor).cpu().numpy().reshape(-1, 2)

            # Step 1: remap from ROI (128×128) to resized RGB (640×480)
            roi_height, roi_width = roi_img.shape[:2]
            w = roi_coords[1] - roi_coords[0]
            h = roi_coords[3] - roi_coords[2]

            scale_x = w / roi_width
            scale_y = h / roi_height

            origin_x = roi_coords[0]
            origin_y = roi_coords[2]

            keypoints_in_resized_rgb = np.stack([
                keypoints_roi[:, 0] * scale_x + origin_x,
                keypoints_roi[:, 1] * scale_y + origin_y
            ], axis=1)

            # Step 2: remap from resized RGB (640×480) to original image
            H_orig, W_orig = image.shape[:2]
            H_resized, W_resized = resized_rgb.shape[:2]

            scale_x_back = W_orig / W_resized
            scale_y_back = H_orig / H_resized

            keypoints_img = np.stack([
                keypoints_in_resized_rgb[:, 0] * scale_x_back,
                keypoints_in_resized_rgb[:, 1] * scale_y_back
            ], axis=1)

            success, rvec, tvec = cv2.solvePnP(
                objectPoints=keypoints_tag_frame,
                imagePoints=keypoints_img,
                cameraMatrix=camera_matrix,
                distCoeffs=dist_coeffs,
            )
            if not success:
                return None, None, seg_mask_img, seg_mask_img_full 
            
            pose_marker = rvectvec_to_xyzabc(rvec, tvec)
            tf_marker = tf_W_Ccv @ xyzabc_to_tf(pose_marker)

            return tf_marker, keypoints_img, seg_mask_img, seg_mask_img_full 


    return predict_pose_from_image

def main():
    import logging
    logging.basicConfig(level=logging.INFO)

    # === Trial and Calibration Parameters ===
    trial_idx = 6
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

    # # === Extrinsic Transforms ===
    # tf_w_CS200 = np.array([
    #     [1, 0, 0, -1.67e-3],
    #     [0, -1, 0, 25.365e-3],
    #     [0, 0, -1, 220.095e-3],
    #     [0, 0, 0, 1]
    # ])

    # tf_CS200_mount = np.array([
    #     [0, 0, -1, 10.0e-3],
    #     [0, -1, 0, 1.2e-3],
    #     [-1, 0, 0, 110.0e-3],
    #     [0, 0, 0, 1]
    # ])

    # tf_mount_camera = np.array([
    #     [1, 0, 0, -35e-3],
    #     [0, 1, 0, -11.5e-3],
    #     [0, 0, 1, 9.8e-3],
    #     [0, 0, 0, 1]
    # ])

    # tf_camera_camera = np.array([
    #     [-1,0,0,0],
    #     [0,-1,0,0],
    #     [0,0,1,0],
    #     [0,0,0,1]
    # ]) 

    # tf_w_c = tf_w_CS200 @ tf_CS200_mount @ tf_mount_camera @ tf_camera_camera 

    # delta = np.array([[ 0.99940154,  0.00513568,  0.03420804, -0.06748126],
    #    [-0.00538714,  0.99995911,  0.00726272,  0.01338278],
    #    [-0.03416934, -0.00744266,  0.99938834,  0.01271901],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]]) 
    
    tf_w_c = np.array([
        [ 2.75948359e-02,  5.52790892e-04, -9.99538260e-01, -4.83132852e-04],
        [ 3.38597313e-03, -9.99963612e-01, -4.62774725e-04,  2.35655819e-02],
        [-9.99558574e-01, -3.35315861e-03, -2.75932114e-02,  1.59464760e-01],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
    ])  

    # tf_w_c = np.array([
    #     [-0.27116848,  0.07892863, -0.95917653,  0.11814422],
    #     [-0.51339413, -0.85478831,  0.07481102,  0.15727561],
    #     [-0.81413918,  0.51276536,  0.27232667, -0.15179611],
    #     [ 0.        ,  0.        ,  0.        ,  1.        ]
    # ])  

    delta = np.array([[ 0.9984765 ,  0.00725654,  0.05533876,  0.00688403],
       [-0.00747035,  0.99998515,  0.00279132, -0.00463532],
       [-0.05531373, -0.00320403,  0.99851338,  0.00410505],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]) 

    tf_trackmark_fidumark = np.array([
        [-1,0,0,0],
        [0,1,0,0],
        [0,0,-1,0],
        [0,0,0,1]
    ]) @ np.linalg.inv(delta) 

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
        T_OFFSET_OPTK_CCV: 1.15, # increase if the CCV detection is too early
        MAX_FRAMES: 1000,
        OUT_DIR: f"./real_data_processing/results",
    }

    # === Run Processing ===
    processor = DatasetProcessor(config)
    processor.set_marker_detector()
    processor.process_optitrack_data()
    processor.run_opencv_fiducial_marker_detection(save_results=True)

    # --- Load LBCV predictor ---
    from PIL import Image
    from keypoints_model.utils import compute_2D_gridpoints  # adjust import if needed

    # Set marker params
    marker_image_path = "./synthetic_data_generation/assets/tags/tag36h11_0.png"
    marker_image = Image.open(marker_image_path).convert("RGB")
    marker_side_length = 0.0798  # meters
    marker_num_squares = 10
    keypoints_tag_frame = np.array(compute_2D_gridpoints(N=marker_num_squares, s=marker_side_length))

    predict_pose_from_image = build_lbcv_predictor(
        seg_model_path="/home/rp/abhay_ws/marker_detection_failure_recovery/segmentation_model/models/my_checkpoint_20250329.pth.tar",
        kp_model_path="/home/rp/abhay_ws/marker_detection_failure_recovery/keypoints_model/models/my_checkpoint_keypoints_20250401.pth.tar",
        camera_matrix=config["camera_intrinsic_matrix"],
        dist_coeffs=config["camera_dist_coeffs"],
        marker_image=marker_image,
        marker_side_length=marker_side_length,
        keypoints_tag_frame=keypoints_tag_frame,
    )

    processor.run_learning_based_detection(predict_pose_from_image, save_results=False, save_segmentation=False)
    processor.compare_detection(num_bins=10)
    processor.compare_pose_estimation()

    # --- Compare methods ---
    def compare_pose_errors(datapoints, method1="LBCV", method2="OPTK"):
        position_errors = []
        rotation_errors = []

        for dp in datapoints:
            tf1 = getattr(dp, f"{method1}_tf", None)
            tf2 = getattr(dp, f"{method2}_tf", None)
            if tf1 is None or tf2 is None:
                continue

            delta = np.linalg.inv(tf1) @ tf2
            pos_err = np.linalg.norm(delta[:3, 3])
            rot_err = np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1.0, 1.0))

            position_errors.append(pos_err)
            rotation_errors.append(rot_err)

        return np.array(position_errors), np.array(rotation_errors)

    pos_errs, rot_errs = compare_pose_errors(processor.datapoints)
    print(f"\n[Comparison: CCV vs LBCV]")
    print(f"Mean position error: {np.mean(pos_errs):.3f} m")
    print(f"Mean rotation error: {np.degrees(np.mean(rot_errs)):.2f} deg")

    logging.info("Data processing complete.")

    import pdb; pdb.set_trace( )

if __name__ == "__main__":
    main()