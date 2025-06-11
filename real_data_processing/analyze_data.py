import json
import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt

# data_file = "./real_data_processing/results/marker_pose_summary.json"
# data_file = "./real_data_processing/results/trial_6.json"
data_file = "./real_data_processing/results/trial_6 (copy).json"
# data_file = "./real_data_processing/results/dark_test_3.json"

def tf_to_pose(tf):
    """
    Convert a transformation matrix to a pose (position and orientation).
    """
    position = tf[:3, 3]
    rot = tf[:3, :3]
    euler_angles = R.from_matrix(rot).as_euler('xyz', degrees=True)  # Convert rotation matrix to Euler angles
    pose = np.concatenate((position, euler_angles))
    return pose 

def pose_to_tf(pose):
    """
    Convert a pose (position and orientation) to a transformation matrix.
    """
    position = pose[:3]
    euler_angles = pose[3:6]
    rot = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()  # Convert Euler angles to rotation matrix
    tf = np.eye(4)
    tf[:3, :3] = rot
    tf[:3, 3] = position
    return tf

# read data from json file 
with open(data_file, 'r') as file: 
    data = json.load(file)
    
data = np.concatenate((data[0:1000], data[1150:1700], data[1850:2200], data[2500:3600]), axis=0)
n_datapoints = len(data) 

tf_optk = [None] * n_datapoints
pose_optk = [None] * n_datapoints
tf_CCV = [None] * n_datapoints
pose_CCV = [None] * n_datapoints
tf_LBCV = [None] * n_datapoints
pose_LBCV = [None] * n_datapoints
tf_optk_CCV = [None] * n_datapoints
pose_optk_CCV = [None] * n_datapoints
tf_optk_LBCV = [None] * n_datapoints
pose_optk_LBCV = [None] * n_datapoints
tf_CCV_LBCV = [None] * n_datapoints
pose_CCV_LBCV = [None] * n_datapoints

# apply frame_correction to all tf_optk, tf_CCV, tf_LBCV such that euler angles are centered at 0 
frame_correction = np.array([
    [1,0,0,0],
    [0,-1,0,0],
    [0,0,-1,0],
    [0,0,0,1]
])

for idx, datapoint in enumerate(data): 
    if datapoint['optk_tf'] is None:
        continue
    tf_optk[idx] = np.array(datapoint['optk_tf']).reshape((4, 4)) @ frame_correction
    pose_optk[idx] = tf_to_pose(tf_optk[idx])

    # if tf_optk is None, skip this datapoint
    if datapoint['ccv_tf'] is not None: 
        tf_CCV[idx] = np.array(datapoint['ccv_tf']).reshape((4, 4)) @ frame_correction
        pose_CCV[idx] = tf_to_pose(tf_CCV[idx]) 
        tf_optk_CCV[idx] = np.linalg.inv(tf_optk[idx]) @ tf_CCV[idx]
        pose_optk_CCV[idx] = tf_to_pose(tf_optk_CCV[idx])
    if datapoint['lbcv_tf'] is not None: 
        tf_LBCV[idx] = np.array(datapoint['lbcv_tf']).reshape((4, 4)) @ frame_correction 
        pose_LBCV[idx] = tf_to_pose(tf_LBCV[idx]) 
        tf_optk_LBCV[idx] = np.linalg.inv(tf_optk[idx]) @ tf_LBCV[idx]
        pose_optk_LBCV[idx] = tf_to_pose(tf_optk_LBCV[idx])
    if datapoint['ccv_tf'] is not None and datapoint['lbcv_tf'] is not None: 
        tf_CCV_LBCV[idx] = np.linalg.inv(tf_CCV[idx]) @ tf_LBCV[idx]
        pose_CCV_LBCV[idx] = tf_to_pose(tf_CCV_LBCV[idx])

dimensions = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
pose_data = {
    'optk': pose_optk,
    'CCV': pose_CCV,
    'LBCV': pose_LBCV,
    # 'optk_CCV': pose_optk_CCV,
    # 'optk_LBCV': pose_optk_LBCV,
    # 'CCV_LBCV': pose_CCV_LBCV
}

# Convert all pose lists to arrays with consistent shape, ignoring None
for key in pose_data:
    # pose_data[key] is a list of lists or arrays, some of which may be None
    pose_array = np.array([
        p if p is not None else [np.nan] * 6
        for p in pose_data[key]
    ])
    pose_data[key] = pose_array

fig, axs = plt.subplots(2, 3, figsize=(15, 10))
for i, (key, poses) in enumerate(pose_data.items()):
    poses = np.array(poses)  # shape (T, 6), may contain np.nan

    for j in range(poses.shape[1]):
        ax = axs[j // 3, j % 3]
        ax.scatter(np.arange(len(poses)), poses[:, j], label=key, alpha=0.5, s=5)
    
    ax.set_title(dimensions[j])
    ax.legend()
    ax.grid()

plt.tight_layout()
plt.show()

# --- Compute Velocities from OPTK ---
pose_array = np.array([
    p if p is not None else [np.nan]*6 for p in pose_optk
])  # shape (T, 6)

# Compute finite difference (velocity) between consecutive valid pose entries
trans_vel = np.linalg.norm(np.diff(pose_array[:, :3], axis=0), axis=1)
ang_vel = np.linalg.norm(np.diff(pose_array[:, 3:], axis=0), axis=1)

# Pad to match original length
trans_vel = np.insert(trans_vel, 0, 0.0)
ang_vel = np.insert(ang_vel, 0, 0.0)

# --- Compute Detection Indices and Rates ---

def get_detection_indices(pose_list):
    return [i for i, p in enumerate(pose_list) if p is not None]

ccv_detected = get_detection_indices(pose_CCV)
lbcv_detected = get_detection_indices(pose_LBCV)

ccv_rate = len(ccv_detected) / len(pose_CCV) * 100
lbcv_rate = len(lbcv_detected) / len(pose_LBCV) * 100

# --- Plotting ---

fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

# 1. CCV Detection
for idx in ccv_detected:
    axs[0].axvline(x=idx, color='C0', linewidth=1)
axs[0].set_ylim(0, 1)
axs[0].set_yticks([])
axs[0].set_ylabel("CCV")
axs[0].set_title(f"CCV Detection Timeline ({ccv_rate:.1f}%)")
axs[0].grid(True, axis='x', linestyle='--', alpha=0.5)

# 2. LBCV Detection
for idx in lbcv_detected:
    axs[1].axvline(x=idx, color='C1', linewidth=1)
axs[1].set_ylim(0, 1)
axs[1].set_yticks([])
axs[1].set_ylabel("LBCV")
axs[1].set_title(f"LBCV Detection Timeline ({lbcv_rate:.1f}%)")
axs[1].grid(True, axis='x', linestyle='--', alpha=0.5)

# 3. Translational Velocity
axs[2].plot(trans_vel, color='C2')
axs[2].set_ylabel("Trans. Vel (m/frame)")
axs[2].set_title("Translational Velocity from OPTK")
axs[2].grid(True)

# 4. Angular Velocity
axs[3].plot(ang_vel, color='C3')
axs[3].set_ylabel("Ang. Vel (deg/frame)")
axs[3].set_title("Angular Velocity from OPTK")
axs[3].set_xlabel("Timestep")
axs[3].grid(True)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

# Create masks of valid pose_optk entries (to align with velocity)
# Make sure all arrays are 1D and aligned
valid_mask = ~np.isnan(pose_array[:, 0])
vel_t_all = trans_vel[valid_mask]
vel_r_all = ang_vel[valid_mask]
ccv_detect_all = np.array([p is not None for p in pose_CCV])[valid_mask]
lbcv_detect_all = np.array([p is not None for p in pose_LBCV])[valid_mask]

# Filter out nan values from vel_t and vel_r, apply same mask to detections
mask_t = ~np.isnan(vel_t_all)
vel_t = vel_t_all[mask_t]
ccv_detect_t = ccv_detect_all[mask_t]
lbcv_detect_t = lbcv_detect_all[mask_t]

mask_r = ~np.isnan(vel_r_all)
vel_r = vel_r_all[mask_r]
ccv_detect_r = ccv_detect_all[mask_r]
lbcv_detect_r = lbcv_detect_all[mask_r]

assert vel_t.shape[0] == ccv_detect_t.shape[0]
assert vel_r.shape[0] == ccv_detect_r.shape[0]

# Define bin edges
num_bins = 10
t_bins = np.linspace(vel_t.min(), vel_t.max(), num_bins + 1)
r_bins = np.linspace(vel_r.min(), vel_r.max(), num_bins + 1)

# Compute detection rates
ccv_rate_t, _, _ = binned_statistic(vel_t, ccv_detect_t.astype(float), statistic='mean', bins=t_bins)
lbcv_rate_t, _, _ = binned_statistic(vel_t, lbcv_detect_t.astype(float), statistic='mean', bins=t_bins)

ccv_rate_r, _, _ = binned_statistic(vel_r, ccv_detect_r.astype(float), statistic='mean', bins=r_bins)
lbcv_rate_r, _, _ = binned_statistic(vel_r, lbcv_detect_r.astype(float), statistic='mean', bins=r_bins)


# Use bin centers for plotting
t_centers = 0.5 * (t_bins[:-1] + t_bins[1:])
r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

# --- Plot Detection Rate vs Velocities ---

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Translational velocity
axs[0].plot(t_centers, ccv_rate_t * 100, label='CCV', marker='o')
axs[0].plot(t_centers, lbcv_rate_t * 100, label='LBCV', marker='s')
axs[0].set_xlabel("Translational Velocity (m/frame)")
axs[0].set_ylabel("Detection Rate (%)")
axs[0].set_title("Detection Rate vs Translational Velocity")
axs[0].grid(True)
axs[0].legend()

# Angular velocity
axs[1].plot(r_centers, ccv_rate_r * 100, label='CCV', marker='o')
axs[1].plot(r_centers, lbcv_rate_r * 100, label='LBCV', marker='s')
axs[1].set_xlabel("Angular Velocity (deg/frame)")
axs[1].set_ylabel("Detection Rate (%)")
axs[1].set_title("Detection Rate vs Angular Velocity")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()

import numpy as np

def pose_errors(pose_gt, pose_pred):
    """
    pose_gt and pose_pred: list of [x, y, z, roll, pitch, yaw]
    Returns: [in-plane trans, out-of-plane trans, in-plane rot, out-of-plane rot]
    """
    pose_gt = np.array(pose_gt)
    pose_pred = np.array(pose_pred)
    trans_error = pose_pred[:3] - pose_gt[:3]
    rot_error = pose_pred[3:] - pose_gt[3:]
    
    t_in_plane = np.linalg.norm(trans_error[:2])
    t_out_plane = np.abs(trans_error[2])
    
    r_in_plane = np.linalg.norm(rot_error[:2])
    r_out_plane = np.abs(rot_error[2])
    
    return [t_in_plane, t_out_plane, r_in_plane, r_out_plane]

def tf_errors_plane(tf_gt, tf_pred):
    """
    Compute SE(3)-based pose error between ground truth and predicted transform.
    Returns: [in-plane trans error, out-of-plane trans error, in-plane rot error, out-of-plane rot error]
    """
    tf_rel = np.linalg.inv(tf_gt) @ tf_pred
    trans_error = tf_rel[:3, 3]
    rot_error_euler = R.from_matrix(tf_rel[:3, :3]).as_euler('xyz', degrees=True)

    t_in_plane = np.linalg.norm(trans_error[:2])       # x, y
    t_out_plane = np.abs(trans_error[2])               # z
    r_in_plane = np.linalg.norm(rot_error_euler[:2])   # roll, pitch
    r_out_plane = np.abs(rot_error_euler[2])           # yaw

    return [t_in_plane, t_out_plane, r_in_plane, r_out_plane]

# Gather all errors
errors_CCV = []
errors_LBCV = []
errors_LBCV_CCV_true = []
errors_LBCV_CCV_false = []

for i in range(len(tf_optk)):
    tf_gt = tf_optk[i]
    if tf_gt is None:
        continue

    if tf_CCV[i] is not None:
        err = tf_errors_plane(tf_gt, tf_CCV[i])
        errors_CCV.append(err)

    if tf_LBCV[i] is not None:
        err = tf_errors_plane(tf_gt, tf_LBCV[i])
        errors_LBCV.append(err)

        if tf_CCV[i] is not None:
            errors_LBCV_CCV_true.append(err)
        else:
            errors_LBCV_CCV_false.append(err)

# Convert to arrays
errors_CCV = np.array(errors_CCV)
errors_LBCV = np.array(errors_LBCV)
errors_LBCV_CCV_true = np.array(errors_LBCV_CCV_true)
errors_LBCV_CCV_false = np.array(errors_LBCV_CCV_false)

def print_stats(name, data):
    data = np.array(data)
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] != 4:
        print(f"--- {name} ---")
        print("Insufficient data for statistics.\n")
        return

    print(f"--- {name} ---")
    print(f"In-plane Translation Error (mean ± std): {np.mean(data[:, 0]):.4f} ± {np.std(data[:, 0]):.4f} m")
    print(f"Out-of-plane Translation Error (mean ± std): {np.mean(data[:, 1]):.4f} ± {np.std(data[:, 1]):.4f} m")
    print(f"In-plane Rotation Error (mean ± std): {np.mean(data[:, 2]):.4f} ± {np.std(data[:, 2]):.4f} deg")
    print(f"Out-of-plane Rotation Error (mean ± std): {np.mean(data[:, 3]):.4f} ± {np.std(data[:, 3]):.4f} deg\n")


print_stats("CCV", errors_CCV)
print_stats("LBCV (all)", errors_LBCV)
print_stats("LBCV (when CCV detects)", errors_LBCV_CCV_true)
print_stats("LBCV (when CCV fails)", errors_LBCV_CCV_false)

import json
import numpy as np
import pandas as pd

# Initialize detection flags
n = len(data)
ccv_detected = np.zeros(n, dtype=bool)
lbcv_detected = np.zeros(n, dtype=bool)

# Parse detections
for i, dp in enumerate(data):
    ccv_detected[i] = dp['ccv_tf'] is not None
    lbcv_detected[i] = dp['lbcv_tf'] is not None

# Create 2x2 detection count matrix
detection_matrix = np.zeros((2, 2), dtype=int)
for ccv, lbcv in zip(ccv_detected, lbcv_detected):
    detection_matrix[int(ccv), int(lbcv)] += 1

# Normalize to get detection rates (as percentages)
detection_rate_matrix = detection_matrix / n * 100

# Format as DataFrame for display
df = pd.DataFrame(
    detection_rate_matrix,
    index=["CCV: No", "CCV: Yes"],
    columns=["LBCV: No", "LBCV: Yes"]
)

print(f"detection_rate_matrix:") 
print(df)