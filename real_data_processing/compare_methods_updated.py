import csv
import json
import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from keypoints_model.utils import overlay_points_on_image
from real_data_processing.utils import *

from keypoints_model.utils import compute_2D_gridpoints 

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# === Constants ===
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
POSE_EST_METHOD = "pose_est_method"

class DatasetProcessor:
    def __init__(self, config: dict):
        self.config = config
        self._load_parameters()
        if config["set_CCV_ground_truth"] == False:
            self._load_optitrack_data()
        self._load_realsense_data()
        self._set_datapoints()
        self.out_dir = self.config[OUT_DIR]
        os.makedirs(self.out_dir, exist_ok=True)

    def _load_parameters(self):
        self.optitrack_csv_file = self.config[OPTITRACK_CSV_FILE]
        self.realsense_video_file = self.config[REALSENSE_VIDEO_FILE]
        self.t_offset_optk_ccv = self.config[T_OFFSET_OPTK_CCV]
        self.tf_trackmark_fidumark = self.config[TF_TRACKMARK_FIDUMARK]
        self.camera_matrix = self.config[CAMERA_INTRINSIC_MATRIX]
        assert self.camera_matrix.shape == (3, 3), "Camera matrix must be 3x3"
        self.dist_coeffs = self.config[CAMERA_DIST_COEFFS]
        self.aruco_dict = self.config[ARUCO_DICT]
        self.marker_length = self.config[MARKER_LENGTH]
        self.tf_w_c = self.config[CAMERA_EXTRINSIC_MATRIX]
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
            print("[INFO] Splitting video to frames...")
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
        max_frames = self.config.get(MAX_FRAMES, None)
        if max_frames and len(self.realsense_frames_paths) > max_frames:
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

    def process_optitrack_data(self, save_overlay=False):
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

        optk_times = self.OPTK_time
        for dp in self.datapoints:
            idx = np.argmin(np.abs(optk_times - dp.time))
            dp.set_pose("OPTK", self.OPTK_tf_c_m[idx])

            tf_c_m = self.OPTK_tf_c_m[idx]
            if tf_c_m is None:
                pass 
            else: 
                try:
                    s = self.marker_length * (10/8)  # scale to include border
                    tag_pts_3d = np.array([
                        [+s/2, +s/2, 0], [-s/2, +s/2, 0], [-s/2, -s/2, 0], [+s/2, -s/2, 0]
                    ], dtype=np.float32).reshape(-1, 3)

                    rvec, _ = cv2.Rodrigues(tf_c_m[:3, :3])
                    tvec = tf_c_m[:3, 3].reshape(3, 1)
                    pts_2d, _ = cv2.projectPoints(tag_pts_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
                    pts_2d = pts_2d.squeeze().astype(np.float32)

                    # Get image dimensions
                    image = dp.get_image()
                    if image is None:
                        continue
                    H, W = image.shape[:2]

                    # === Corner visibility checks ===
                    inside_mask = (
                        (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < W) &
                        (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < H)
                    )
                    dp.marker_fully_inframe = np.all(inside_mask)

                    # Original projected polygon
                    marker_poly = pts_2d.astype(np.float32).reshape((-1, 1, 2))

                    # Image boundary polygon (clockwise)
                    image_poly = np.array([
                        [[0, 0]], [[W - 1, 0]], [[W - 1, H - 1]], [[0, H - 1]]
                    ], dtype=np.float32)

                    # Compute intersection polygon (in-frame area)
                    ret, clipped_poly = cv2.intersectConvexConvex(marker_poly, image_poly)
                    inframe_area = ret

                    # Compute total projected area (possibly partially out of frame)
                    full_area = cv2.contourArea(marker_poly)

                    if full_area > 0:
                        dp.marker_inframe_fraction = inframe_area / full_area
                    else:
                        dp.marker_inframe_fraction = 0.0


                    # === Area and brightness statistics ===
                    area = 0.5 * abs(
                        pts_2d[0, 0]*pts_2d[1, 1] + pts_2d[1, 0]*pts_2d[2, 1] +
                        pts_2d[2, 0]*pts_2d[3, 1] + pts_2d[3, 0]*pts_2d[0, 1] -
                        pts_2d[1, 0]*pts_2d[0, 1] - pts_2d[2, 0]*pts_2d[1, 1] -
                        pts_2d[3, 0]*pts_2d[2, 1] - pts_2d[0, 0]*pts_2d[3, 1]
                    )

                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    rgb = image.astype(np.float32).mean(axis=2)
                    del image  # Free memory

                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.fillConvexPoly(mask, pts_2d.astype(np.int32), 255)

                    marker_pixels = rgb[mask == 255]
                    if marker_pixels.size > 0:
                        mean_val = np.mean(marker_pixels)
                        std_val = np.std(marker_pixels)
                    else:
                        mean_val = np.nan
                        std_val = np.nan

                    dp.marker_brightness = mean_val
                    dp.marker_brightness_std = std_val
                    dp.marker_area = area

                except Exception as e:
                    print(f"[WARN] Failed to compute marker stats for frame {idx}: {e}")


            self.optk_frames_dir = os.path.join(
                "./real_data_processing/raw_data/realsense",
                f"realsense_{self.config['trial_idx']}_frames_OPTK"
            )
            if save_overlay: 
                dp.save_overlay("OPTK", square_length=self.marker_length,
                            K=self.camera_matrix, output_dir=self.optk_frames_dir)

    def run_opencv_fiducial_marker_detection(self, save_results=False):
        num_frames = len(self.realsense_frames_paths)
        tf_c_m_all = [None] * num_frames
        CCV_corners = [None] * num_frames
        detected = [False] * num_frames

        output_dir = Path(self.realsense_video_file.replace('.mp4', '_frames_CCV'))

        tf_w_m, tf_mo_mi, timestamps = [], [], []
        tf_mo_w = None

        for idx, frame_path in enumerate(self.realsense_frames_paths):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                logger.warning(f"[CCV] Could not read frame {frame_path}")
                continue

            marker_ids, rvecs, tvecs, corners_tuple = marker_pose_estimation_estimatePoseSingleMarkers(
                frame, self.camera_matrix, self.dist_coeffs, self.aruco_dict,
                self.marker_length, show=False
            )

            del frame # Free memory 

            if corners_tuple is not None:
                corners = np.array([corner.reshape(-1, 2) for corner in corners_tuple])

            if rvecs is not None and tvecs is not None:
                tf = np.eye(4)
                tf[:3, :3] = cv2.Rodrigues(rvecs[0])[0]
                tf[:3, 3] = tvecs[0].reshape(3)
                tf_c_m_all[idx] = tf
                CCV_corners[idx] = corners
                detected[idx] = True

                tf_w_m_i = self.tf_w_c @ tf
                tf_w_m.append(tf_w_m_i)

                if tf_mo_w is None:
                    tf_mo_w = np.linalg.inv(tf_w_m_i)
                    tf_mo_mi.append(np.eye(4))
                else:
                    tf_mo_mi.append(tf_mo_w @ tf_w_m_i)

            dp = self.datapoints[idx]
            if detected[idx]:
                dp.set_pose("CCV", tf_c_m_all[idx])
                dp.set_ccv_corners(corners)
            else:
                dp.set_pose("CCV", None)
                dp.set_ccv_corners(None)

            timestamps.append(self.RLSN_time[idx])
            if save_results:
                out_img = frame.copy()
                if detected[idx]:
                    out_img = cv2.aruco.drawDetectedMarkers(out_img, corners_tuple, marker_ids)
                outpath = output_dir / f"CCV_frame_{idx:05d}.png"
                cv2.imwrite(str(outpath), out_img)
                del out_img # Free memory 

        self.CCV_time = np.array(timestamps) - timestamps[0]
        self.CCV_tf_c_m = tf_c_m_all
        self.CCV_tf_w_m = np.array(tf_w_m)
        self.CCV_tf_mo_mi = np.array(tf_mo_mi)
        self.CCV_detected = detected
        self.CCV_corners = CCV_corners
    def run_learning_based_detection(self, predict_fn, save_results=False, save_segmentation=False):
        num_frames = len(self.datapoints)
        tf_c_m_all = [None] * num_frames
        LBCV_keypoints = [None] * num_frames
        detected = [False] * num_frames

        tf_w_m, tf_mo_mi, timestamps = [], [], []
        tf_mo_w = None

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
                continue

            result = predict_fn(image)
            if isinstance(result, tuple) and len(result) == 4:
                tf_marker, keypoints, segmentation_mask, segmentation_mask_full = result
            else:
                tf_marker, keypoints, segmentation_mask, segmentation_mask_full = None, None, None, None

            tf_c_m_all[idx] = tf_marker
            LBCV_keypoints[idx] = keypoints

            if tf_marker is not None:
                detected[idx] = True
                tf_w_m_i = self.tf_w_c @ tf_marker
                tf_w_m.append(tf_w_m_i)

                if tf_mo_w is None:
                    tf_mo_w = np.linalg.inv(tf_w_m_i)
                    tf_mo_mi.append(np.eye(4))
                else:
                    tf_mo_mi.append(tf_mo_w @ tf_w_m_i)

                timestamps.append(dp.time)

            dp.set_pose("LBCV", tf_marker)
            dp.set_lbcv_keypoints(keypoints)

            if save_results and keypoints is not None:
                overlay = overlay_points_on_image(image.copy(), keypoints, radius=3)
                overlay_path = output_dir / "keypoints_overlay" / f"LBCV_frame_{idx:05d}.png"
                overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                Image.fromarray(overlay).save(str(overlay_path))

                json_path = output_dir / "keypoints_json" / f"LBCV_frame_{idx:05d}.json"
                with open(json_path, "w") as f:
                    json.dump(keypoints.tolist(), f)

            if save_segmentation and segmentation_mask is not None:
                seg_path = output_dir / "segmentation_masks" / f"LBCV_seg_{idx:05d}.png"
                segmentation_mask.convert("L").save(str(seg_path))

                seg_path_full = output_dir / "segmentation_masks_full" / f"LBCV_seg_{idx:05d}.png"
                segmentation_mask_full.convert("L").save(str(seg_path_full))

            # print progress every 5% 
            if num_frames // 20 != 0: 
                if idx % (num_frames // 20) == 0:
                    logger.info(f"[LBCV] Processing frame {idx + 1}/{num_frames}")

        self.LBCV_time = np.array(timestamps) - timestamps[0] if timestamps else None
        self.LBCV_tf_c_m = np.array([t for t in tf_c_m_all if t is not None])
        self.LBCV_tf_w_m = np.array(tf_w_m)
        self.LBCV_tf_mo_mi = np.array(tf_mo_mi)
        self.LBCV_detected = detected
        self.LBCV_keypoints = np.array(LBCV_keypoints, dtype=object)

    def compare_detection(self, num_bins=10):
        ccv_detects = np.sum(self.CCV_detected) / len(self.datapoints)
        lbcv_detects = np.sum(self.LBCV_detected) / len(self.datapoints)

        sns.set_theme(style="whitegrid")
        methods = ["CCV", "LBCV"]
        detection_rates = [ccv_detects, lbcv_detects]

        plt.figure(figsize=(8, 6))
        ax = sns.barplot(x=methods, y=detection_rates)
        for i, rate in enumerate(detection_rates):
            plt.text(i, rate + 0.02, f"{rate:.2f}", ha='center', va='bottom', fontsize=12)

        plt.title("Detection Rates")
        plt.ylabel("Detection Rate")
        plt.ylim(0, 1.05)
        plt.savefig(os.path.join(self.out_dir, "detection_rates.png"))
        plt.close()

        logger.info(f"[Detection Rates] CCV: {ccv_detects:.2f}, LBCV: {lbcv_detects:.2f}")

        brightness = np.array([dp.get_marker_brightness() for dp in self.datapoints])
        areas = np.array([dp.get_marker_area() for dp in self.datapoints])

        def plot_rate_vs_metric(metric, label, filename):
            bin_edges = np.linspace(metric.min(), metric.max(), num_bins + 1)
            ccv_detections = np.zeros(num_bins)
            lbcv_detections = np.zeros(num_bins)
            total_counts = np.zeros(num_bins)

            for dp in self.datapoints:
                val = getattr(dp, label)
                bin_idx = np.searchsorted(bin_edges, val, side='right') - 1
                bin_idx = min(max(bin_idx, 0), num_bins - 1)
                total_counts[bin_idx] += 1
                if hasattr(dp, 'CCV_tf') and dp.CCV_tf is not None:
                    ccv_detections[bin_idx] += 1
                if hasattr(dp, 'LBCV_tf') and dp.LBCV_tf is not None:
                    lbcv_detections[bin_idx] += 1

            with np.errstate(divide='ignore', invalid='ignore'):
                ccv_rates = np.where(total_counts > 0, ccv_detections / total_counts, 0)
                lbcv_rates = np.where(total_counts > 0, lbcv_detections / total_counts, 0)

            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            plt.figure()
            plt.plot(bin_centers, ccv_rates, label="CCV Detection Rate", marker='o')
            plt.plot(bin_centers, lbcv_rates, label="LBCV Detection Rate", marker='x')
            plt.xlabel(label.replace('_', ' ').title())
            plt.ylabel("Detection Rate")
            plt.title(f"Detection Rate vs {label.replace('_', ' ').title()}")
            plt.ylim(0, 1.05)
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(self.out_dir, filename))
            plt.close()

        plot_rate_vs_metric(brightness, 'marker_brightness', "marker_brightness_detection_rate.png")
        plot_rate_vs_metric(areas, 'marker_area', "marker_area_detection_rate.png")

    def compare_pose_estimation(self):
        self.CCV_translation_errors = []
        self.CCV_rotation_errors = []
        self.LBCV_translation_errors = []
        self.LBCV_rotation_errors = []

        for dp in self.datapoints:
            if self.config.get("set_CCV_ground_truth", False):
                dp.set_pose("OPTK", dp.CCV_tf)
            CCV_error, LBCV_error = dp.compute_errors()
            self.CCV_translation_errors.append(CCV_error["translation"] if CCV_error else None)
            self.CCV_rotation_errors.append(CCV_error["rotation"] if CCV_error else None)
            self.LBCV_translation_errors.append(LBCV_error["translation"] if LBCV_error else None)
            self.LBCV_rotation_errors.append(LBCV_error["rotation"] if LBCV_error else None)

        brightness = np.array([dp.get_marker_brightness() for dp in self.datapoints])
        area = np.array([dp.get_marker_area() for dp in self.datapoints])

        def plot_error_vs_metric(metric, errors_c, errors_l, ylabel, title, filename):
            plt.figure()
            plt.scatter(metric, errors_c, label="CCV", alpha=0.5)
            plt.scatter(metric, errors_l, label="LBCV", alpha=0.5)
            plt.xlabel(title)
            plt.ylabel(ylabel)
            plt.title(f"{ylabel} vs {title}")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.out_dir, filename))
            plt.close()

        plot_error_vs_metric(brightness, self.CCV_translation_errors, self.LBCV_translation_errors,
                              "Translation Error (m)", "Marker Brightness", "translation_error_vs_brightness.png")
        plot_error_vs_metric(area, self.CCV_translation_errors, self.LBCV_translation_errors,
                              "Translation Error (m)", "Marker Area", "translation_error_vs_area.png")
        plot_error_vs_metric(brightness, self.CCV_rotation_errors, self.LBCV_rotation_errors,
                              "Rotation Error (degrees)", "Marker Brightness", "rotation_error_vs_brightness.png")
        plot_error_vs_metric(area, self.CCV_rotation_errors, self.LBCV_rotation_errors,
                              "Rotation Error (degrees)", "Marker Area", "rotation_error_vs_area.png")

    def save_summary_csv(self, output_path="marker_pose_summary.csv"):
        marker_pos_velocities = []
        marker_rot_velocities = []

        prev_tf = None
        prev_time = None

        for dp in self.datapoints:
            optk_tf = getattr(dp, 'OPTK_tf', None)
            time = getattr(dp, 'time', None)

            if optk_tf is not None and prev_tf is not None and time is not None and prev_time is not None:
                dt = time - prev_time
                if dt > 0:
                    dp_velocity = np.linalg.norm(optk_tf[:3, 3] - prev_tf[:3, 3]) / dt
                    marker_pos_velocities.append(dp_velocity)
                    R_delta = prev_tf[:3, :3].T @ optk_tf[:3, :3]
                    angle = np.arccos(np.clip((np.trace(R_delta) - 1) / 2, -1.0, 1.0))
                    marker_rot_velocities.append(angle / dt)
                else:
                    marker_pos_velocities.append(None)
                    marker_rot_velocities.append(None)
            else:
                marker_pos_velocities.append(None)
                marker_rot_velocities.append(None)

            prev_tf = optk_tf
            prev_time = time

        rows = []
        for i, dp in enumerate(self.datapoints):
            row = {
                "time": getattr(dp, 'time', None),
                "marker_brightness": getattr(dp, 'marker_brightness', None),
                "marker_area": getattr(dp, 'marker_area', None),
                "marker_pos_vel": marker_pos_velocities[i],
                "marker_rot_vel": marker_rot_velocities[i],
                "marker_inframe_fraction": getattr(dp, 'marker_inframe_fraction', None),
                "marker_fully_inframe": getattr(dp, 'marker_fully_inframe', None),
                "optk_tf": dp.OPTK_tf.flatten().tolist() if dp.OPTK_tf is not None else None,
                "ccv_tf": dp.CCV_tf.flatten().tolist() if dp.CCV_tf is not None else None,
                "ccv_corners": dp.CCV_corners.tolist() if dp.CCV_corners is not None else None,
                "lbcv_tf": dp.LBCV_tf.flatten().tolist() if dp.LBCV_tf is not None else None,
                "lbcv_keypoints": dp.LBCV_keypoints.tolist() if dp.LBCV_keypoints is not None else None,
            }
            rows.append(row)

        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        summary_base = os.path.splitext(output_path)[0]
        np.save(f"{summary_base}.npy", rows)

        with open(f"{summary_base}.json", 'w') as f:
            json.dump(sanitize_for_json(rows), f, indent=2)

    def set_OPTK_to_CCV(self): 
        for dp in self.datapoints:
            dp.set_pose("OPTK", dp.CCV_tf)

def build_lbcv_predictor(
    seg_model_path: str,
    kp_model_path: str,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_image: 'PIL.Image.Image',
    marker_side_length: float,
    keypoints_tag_frame: np.ndarray,
    method: str = "kp_hrnet", # "seg" or "kp_mobilenet" or "kp_hrnet" 
    kp_hrnet_model_path: Optional[str] = None,
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

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    seg_model = UNETWithDropout(in_channels=3, out_channels=1).to(DEVICE)
    load_seg_ckpt(torch.load(seg_model_path, map_location=DEVICE), seg_model)
    seg_model.eval()

    if method == "kp_mobilenet":
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
        if tag_pixels.size < 1000: # min number of pixels to consider a tag, value from training data filtering 
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

    def predict_pose_from_image(image: np.ndarray, method=method):
        from keypoints_model.utils import xyzabc_to_tf, rvectvec_to_xyzabc, rvectvec_to_tf

        tf_marker = None
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
        if np.array(seg_mask_img).max() == 0 and (method == "seg" or method == "kp_mobilenet"):
            return None, None, seg_mask_img, seg_mask_img_full

        if method == "seg": 
            return tf_marker, None, seg_mask_img, seg_mask_img_full 

        if method == "kp_mobilenet": 
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
            
            tf_marker = rvectvec_to_tf(rvec, tvec) @ tf_W_Ccv 

            return tf_marker, keypoints_img, seg_mask_img, seg_mask_img_full 

        if method == "kp_hrnet":
            from hrnet.model import HRNetKeypoint
            from keypoints_model.utils import xyzabc_to_tf, rvectvec_to_xyzabc

            tf_marker = None  

            HRNET_NUM_KEYPOINTS = keypoints_tag_frame.shape[0]
            hrnet_model = HRNetKeypoint(num_keypoints=HRNET_NUM_KEYPOINTS).to(DEVICE)
            hrnet_model.load_state_dict(torch.load(kp_hrnet_model_path, map_location=DEVICE))
            hrnet_model.eval()

            # Prepare input
            hrnet_transform = A.Compose([
                A.Resize(256, 256),
                A.Normalize(max_pixel_value=255.0),
                ToTensorV2()
            ])
            transformed = hrnet_transform(image=image)
            img_tensor = transformed["image"].unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                pred_flat = hrnet_model(img_tensor)[0].cpu().numpy()  # shape: [2K]
                keypoints_normalized = pred_flat.reshape(-1, 2)

            H_img, W_img = image.shape[:2]
            keypoints_img = keypoints_normalized.copy()
            keypoints_img[:, 0] *= W_img
            keypoints_img[:, 1] *= H_img

            success, rvec, tvec = cv2.solvePnP(
                objectPoints=keypoints_tag_frame,
                imagePoints=keypoints_img,
                cameraMatrix=camera_matrix,
                distCoeffs=dist_coeffs,
            )
            if success:
                # pose_marker = rvectvec_to_xyzabc(rvec, tvec)
                # tf_marker = tf_W_Ccv @ xyzabc_to_tf(pose_marker)
                tf_marker = rvectvec_to_tf(rvec, tvec)

            return tf_marker, keypoints_img, None, None

    return predict_pose_from_image

# === Entry Function ===
def run_full_analysis(config, predict_fn=None, summary_path=None):
    processor = DatasetProcessor(config)
    processor.set_marker_detector()
    if config["set_CCV_ground_truth"] == False: 
        processor.process_optitrack_data(save_overlay=False)
    processor.run_opencv_fiducial_marker_detection(save_results=False)
    if config["set_CCV_ground_truth"] == True: 
        processor.set_OPTK_to_CCV() 
    if predict_fn is not None:
        processor.run_learning_based_detection(predict_fn, save_results=False, save_segmentation=False)
    processor.compare_detection()
    processor.compare_pose_estimation()
    if summary_path is None:
        summary_path = os.path.join(config["OUT_DIR"], "marker_pose_summary.csv")
    processor.save_summary_csv(summary_path)
    logger.info(f"Analysis complete. Summary saved to {summary_path}")

if __name__ == "__main__":
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
    tf_w_c = np.array([
        [ 2.75948359e-02,  5.52790892e-04, -9.99538260e-01, -4.83132852e-04],
        [ 3.38597313e-03, -9.99963612e-01, -4.62774725e-04,  2.35655819e-02],
        [-9.99558574e-01, -3.35315861e-03, -2.75932114e-02,  1.59464760e-01],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
    ])  

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

    config = {
        "trial_idx": trial_idx,
        OPTITRACK_CSV_FILE: f"./test_data/optitrack/optitrack_{trial_idx}.csv",
        REALSENSE_VIDEO_FILE: f"./test_data/realsense/realsense_{trial_idx}.mp4",
        "set_CCV_ground_truth":False,

        # OPTITRACK_CSV_FILE: None,
        # REALSENSE_VIDEO_FILE: f"./real_data_processing/raw_data/controlled_tests/bright_test_4.mp4",
        # "set_CCV_ground_truth":True, 

        TF_TRACKMARK_FIDUMARK: tf_trackmark_fidumark,
        CAMERA_INTRINSIC_MATRIX: np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
        CAMERA_DIST_COEFFS: dist_coeffs,
        ARUCO_DICT: cv2.aruco.DICT_APRILTAG_36h11,
        MARKER_LENGTH: 0.0798,
        CAMERA_EXTRINSIC_MATRIX: tf_w_c,
        T_OFFSET_OPTK_CCV: 1.15,
        MAX_FRAMES: 3658, #3658,
        OUT_DIR: f"./real_data_processing/results",
        POSE_EST_METHOD: "kp_mobilenet",  # Options: "seg", "kp_mobilenet", "kp_hrnet"
    }

    # Set marker params
    marker_image_path = "./synthetic_data_generation/assets/tags/tag36h11_0.png"
    marker_image = Image.open(marker_image_path).convert("RGB")
    marker_side_length = 0.0798  # meters
    marker_num_squares = 10
    marker_side_length_with_border = marker_side_length * (marker_num_squares/(marker_num_squares-2))
    keypoints_tag_frame = np.array(compute_2D_gridpoints(N=marker_num_squares, s=marker_side_length_with_border))

    predict_pose_from_image = build_lbcv_predictor(
        seg_model_path="./segmentation_model/models/my_checkpoint_20250329.pth.tar",
        kp_model_path="./keypoints_model/models/my_checkpoint_keypoints_20250401.pth.tar",
        camera_matrix=config["camera_intrinsic_matrix"],
        dist_coeffs=config["camera_dist_coeffs"],
        marker_image=marker_image,
        marker_side_length=MARKER_LENGTH,
        keypoints_tag_frame=keypoints_tag_frame,
        method=config["pose_est_method"],
        kp_hrnet_model_path="./hrnet/checkpoints/hrnet_keypoint_best.pth",
    )

    out_path = os.path.join(config[OUT_DIR], "trial_6.csv")
    # out_path = os.path.join(config[OUT_DIR], "bright_test.csv")
    run_full_analysis(config, predict_pose_from_image, out_path) 