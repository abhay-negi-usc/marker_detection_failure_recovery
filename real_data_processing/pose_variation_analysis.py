import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import json
from scipy.stats import binned_statistic

# Load JSON data
data_file = "./real_data_processing/results/trial_6.json"
with open(data_file, 'r') as file:
    data = json.load(file)

# Apply selection FIRST
data = np.concatenate((data[0:1000], data[1150:1700], data[1850:2200], data[2500:3600]), axis=0)

# Now define n and valid_mask
n = len(data)
exclude_indices = set()
valid_mask = np.array([i not in exclude_indices for i in range(n)])

# Initialize storage
tf_optk, tf_CCV, tf_LBCV = [None]*n, [None]*n, [None]*n
marker_fully_inframe = [None]*n
marker_area = [dp.get("marker_area", 0) for dp in data]
marker_inframe_fraction = [dp.get("marker_inframe_fraction", 0.0) for dp in data]

def tf_to_pose(tf):
    pos = tf[:3, 3]
    rot = tf[:3, :3]
    eul = R.from_matrix(rot).as_euler('xyz', degrees=True)
    return np.concatenate((pos, eul))

def compute_errors(mask, tf_gt_list, tf_pred_list):
    errors = []
    for m, tf_gt, tf_pred in zip(mask, tf_gt_list, tf_pred_list):
        if m and tf_gt is not None and tf_pred is not None:
            tf_rel = np.linalg.inv(tf_gt) @ tf_pred
            t = tf_rel[:3, 3] *1e3 # mm
            r = R.from_matrix(tf_rel[:3, :3]).as_euler("xyz", degrees=True)
            errors.append([
                np.linalg.norm(t[:2]), # in-plane translation error
                abs(t[2]), # out-of-plane translation error
                abs(r[2]), # in-plane rotation error 
                np.linalg.norm(r[:2]) # out-of-plane rotation error 
            ])
    return np.array(errors)

# Parse JSON entries
for i, dp in enumerate(data):
    marker_fully_inframe[i] = bool(dp.get("marker_fully_inframe", False))
    if dp['optk_tf'] is not None:
        tf_optk[i] = np.array(dp['optk_tf']).reshape(4, 4)
    if dp['ccv_tf'] is not None:
        tf_CCV[i] = np.array(dp['ccv_tf']).reshape(4, 4)
    if dp['lbcv_tf'] is not None:
        tf_LBCV[i] = np.array(dp['lbcv_tf']).reshape(4, 4)

# Convert to numpy arrays and apply valid_mask
tf_optk_valid = [tf for i, tf in enumerate(tf_optk) if valid_mask[i]]
tf_CCV_valid = [tf for i, tf in enumerate(tf_CCV) if valid_mask[i]]
tf_LBCV_valid = [tf for i, tf in enumerate(tf_LBCV) if valid_mask[i]]

marker_fully_inframe = np.array(marker_fully_inframe)[valid_mask]

# Total number of valid samples
n_valid = len(tf_optk)


# Masks
ccv_detected = np.array([x is not None for x in tf_CCV])
lbcv_detected = np.array([x is not None for x in tf_LBCV])
mask_full = np.array(marker_fully_inframe, dtype=bool)
mask_partial = ~mask_full


# Summary categories and masks
conditions = {
    "OpenCV": ccv_detected,
    "OpenCV and Fully In Frame": ccv_detected & mask_full,
    "OpenCV and Partially In Frame": ccv_detected & mask_partial,
    
    "Our Method and OpenCV Success": lbcv_detected & ccv_detected,
    
    "Our Method and OpenCV Fail (Fully In Frame)": lbcv_detected & ~ccv_detected & mask_full,
    "Our Method and OpenCV Fail (Partially In Frame)": lbcv_detected & ~ccv_detected & mask_partial,
    "Our Method and OpenCV Fail (Any)": lbcv_detected & ~ccv_detected,
    
    "Our Method (Any) and Fully In Frame": lbcv_detected & mask_full,
    "Our Method (Any) and Partially In Frame": lbcv_detected & mask_partial,
    "Our Method (Any) and Any Visibility": lbcv_detected,
}

# Evaluate
summary = []
for name, mask in conditions.items():
    # pred_list = tf_CCV if "OpenCV" in name else tf_LBCV
    if name.startswith("OpenCV"):
        pred_list = tf_CCV
    else:
        pred_list = tf_LBCV
    valid_condition_mask = mask[valid_mask]
    errors = compute_errors(mask[valid_mask], tf_optk_valid, pred_list)    
    row = {
        "Method": name,
        "Fraction of Cases": len(errors) / n
    }
    if len(errors) > 0:
        row["In-Plane Trans. Error (mm)"] = f"{np.mean(errors[:, 0]):.2f} ± {np.std(errors[:, 0]):.2f}"
        row["Out-of-Plane Trans. Error (mm)"] = f"{np.mean(errors[:, 1]):.2f} ± {np.std(errors[:, 1]):.2f}"
        row["In-Plane Rot. Error (deg)"] = f"{np.mean(errors[:, 2]):.2f} ± {np.std(errors[:, 2]):.2f}"
        row["Out-of-Plane Rot. Error (deg)"] = f"{np.mean(errors[:, 3]):.2f} ± {np.std(errors[:, 3]):.2f}"
    else:
        row["In-Plane Trans. Error (mm)"] = "-"
        row["Out-of-Plane Trans. Error (mm)"] = "-"
        row["In-Plane Rot. Error (deg)"] = "-"
        row["Out-of-Plane Rot. Error (deg)"] = "-"
    summary.append(row)

df_summary = pd.DataFrame(summary)


print(df_summary.to_string(index=False))

import matplotlib.pyplot as plt

# --- Compute errors (already done): errors_LBCV_full has shape (N, 4)
# Columns: [in-plane trans, out-of-plane trans, in-plane rot, out-of-plane rot]

# Rebuild filtered aligned arrays for both LBCV and CCV
z_vals = []
yaw_vals = []
lbcv_detect_vals = []
ccv_detect_vals = []
lbcv_pos_errors = []
lbcv_rot_errors = []
ccv_pos_errors = []
ccv_rot_errors = []

n_datapoints = len(tf_optk) 
pose_optk = [tf_to_pose(tf) if tf is not None else None for tf in tf_optk]

for i in range(n_datapoints):
    if not mask_full[i]:
        continue
    if pose_optk[i] is None or tf_optk[i] is None:
        continue
    z = pose_optk[i][2]
    yaw = abs(pose_optk[i][5])
    if np.isnan(z) or np.isnan(yaw):
        continue

    tf_gt = tf_optk[i]

    # LBCV
    if tf_LBCV[i] is not None:
        tf_pred = tf_LBCV[i]
        tf_rel = np.linalg.inv(tf_gt) @ tf_pred
        t = tf_rel[:3, 3]
        R_rel = tf_rel[:3, :3]
        angle = np.linalg.norm(R.from_matrix(R_rel).as_rotvec(degrees=False))
        lbcv_detect_vals.append(1.0)
        lbcv_pos_errors.append(np.linalg.norm(t)* 1e3)   # mm
        lbcv_rot_errors.append(np.degrees(angle))  # deg
    else:
        lbcv_detect_vals.append(0.0)

    # CCV
    if tf_CCV[i] is not None:
        tf_pred = tf_CCV[i]
        tf_rel = np.linalg.inv(tf_gt) @ tf_pred
        t = tf_rel[:3, 3]
        R_rel = tf_rel[:3, :3]
        angle = np.linalg.norm(R.from_matrix(R_rel).as_rotvec(degrees=False))
        ccv_detect_vals.append(1.0)
        ccv_pos_errors.append(np.linalg.norm(t)* 1e3)   # mm
        ccv_rot_errors.append(np.degrees(angle))  # deg
    else:
        ccv_detect_vals.append(0.0)

    z_vals.append(z)
    yaw_vals.append(yaw)

# Convert to arrays
z_vals = np.array(z_vals)
yaw_vals = np.array(yaw_vals)
lbcv_detect_vals = np.array(lbcv_detect_vals)
ccv_detect_vals = np.array(ccv_detect_vals)
lbcv_pos_errors = np.array(lbcv_pos_errors)
lbcv_rot_errors = np.array(lbcv_rot_errors)
ccv_pos_errors = np.array(ccv_pos_errors)
ccv_rot_errors = np.array(ccv_rot_errors)

# Align all error/detection arrays with z/yaw
mask_lbcv = lbcv_detect_vals == 1
mask_ccv = ccv_detect_vals == 1

def binned_curve(x, y, bins=10):
    bin_means, bin_edges, _ = binned_statistic(x, y, statistic='mean', bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, bin_means

# Binned curves
bins = 10
z_centers, lbcv_rate_z = binned_curve(z_vals, lbcv_detect_vals, bins)
_, ccv_rate_z = binned_curve(z_vals, ccv_detect_vals, bins)
_, lbcv_pos_z = binned_curve(z_vals[mask_lbcv], lbcv_pos_errors, bins)
_, ccv_pos_z = binned_curve(z_vals[mask_ccv], ccv_pos_errors, bins)
_, lbcv_rot_z = binned_curve(z_vals[mask_lbcv], lbcv_rot_errors, bins)
_, ccv_rot_z = binned_curve(z_vals[mask_ccv], ccv_rot_errors, bins)

yaw_centers, lbcv_rate_yaw = binned_curve(yaw_vals, lbcv_detect_vals, bins)
_, ccv_rate_yaw = binned_curve(yaw_vals, ccv_detect_vals, bins)
_, lbcv_pos_yaw = binned_curve(yaw_vals[mask_lbcv], lbcv_pos_errors, bins)
_, ccv_pos_yaw = binned_curve(yaw_vals[mask_ccv], ccv_pos_errors, bins)
_, lbcv_rot_yaw = binned_curve(yaw_vals[mask_lbcv], lbcv_rot_errors, bins)
_, ccv_rot_yaw = binned_curve(yaw_vals[mask_ccv], ccv_rot_errors, bins)

# --- Create partial in-frame masks and arrays ---
mask_partial = np.array(marker_fully_inframe) == False
z_vals_partial = []
yaw_vals_partial = []
lbcv_detect_partial = []
lbcv_pos_partial = []
lbcv_rot_partial = []

for i in range(n_datapoints):
    if not mask_partial[i]:
        continue
    if tf_optk[i] is None:
        continue

    tf_gt = tf_optk[i]
    z = tf_gt[2, 3]
    yaw = abs(R.from_matrix(tf_gt[:3, :3]).as_euler("xyz", degrees=True)[2])
    if np.isnan(z) or np.isnan(yaw):
        continue

    z_vals_partial.append(z)
    yaw_vals_partial.append(yaw)

    if tf_LBCV[i] is not None:
        tf_pred = tf_LBCV[i]
        tf_rel = np.linalg.inv(tf_gt) @ tf_pred
        t = tf_rel[:3, 3]
        R_rel = tf_rel[:3, :3]
        angle = np.linalg.norm(R.from_matrix(R_rel).as_rotvec())
        lbcv_detect_partial.append(1.0)
        lbcv_pos_partial.append(np.linalg.norm(t)*1e3)   # mm
        lbcv_rot_partial.append(np.degrees(angle))
    else:
        lbcv_detect_partial.append(0.0)

# --- Convert to arrays ---
z_vals_partial = np.array(z_vals_partial)
yaw_vals_partial = np.array(yaw_vals_partial)
lbcv_detect_partial = np.array(lbcv_detect_partial)
lbcv_pos_partial = np.array(lbcv_pos_partial)
lbcv_rot_partial = np.array(lbcv_rot_partial)

# --- Compute binned curves ---
bins = 10
z_centers_partial, lbcv_rate_z_partial = binned_curve(z_vals_partial, lbcv_detect_partial, bins)
_, lbcv_pos_z_partial = binned_curve(z_vals_partial[lbcv_detect_partial == 1], lbcv_pos_partial, bins)
_, lbcv_rot_z_partial = binned_curve(z_vals_partial[lbcv_detect_partial == 1], lbcv_rot_partial, bins)

yaw_centers_partial, lbcv_rate_yaw_partial = binned_curve(yaw_vals_partial, lbcv_detect_partial, bins)
_, lbcv_pos_yaw_partial = binned_curve(yaw_vals_partial[lbcv_detect_partial == 1], lbcv_pos_partial, bins)
_, lbcv_rot_yaw_partial = binned_curve(yaw_vals_partial[lbcv_detect_partial == 1], lbcv_rot_partial, bins)

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Set Seaborn theme and font scale for publication-quality figures
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.8)

def plot_metric(ax, x, y_list, labels, colors, markers, title, xlabel, ylabel):
    for y, label, color, marker in zip(y_list, labels, colors, markers):
        sns.lineplot(x=x, y=y, ax=ax, label=label, color=color, marker=marker)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)

# === FULLY IN FRAME ===
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# Colors and labels
color_ccv = "#126782"
color_lbcv = "#fb8500"
labels = ["OpenCV", "Our Method"]
colors = [color_ccv, color_lbcv]
markers = ['s', 'o']

plot_metric(axs[0, 0], z_centers, [ccv_rate_z * 100, lbcv_rate_z * 100], labels,
            colors, markers, "Detection Rate vs Z", "Out-of-Plane Distance (m)", "Detection Rate (%)")
plot_metric(axs[0, 1], z_centers, [ccv_pos_z, lbcv_pos_z], labels,
            colors, markers, "Position Error vs Z", "Out-of-Plane Distance (m)", "Translation Error (mm)")
plot_metric(axs[0, 2], z_centers, [ccv_rot_z, lbcv_rot_z], labels,
            colors, markers, "Rotation Error vs Z", "Out-of-Plane Distance (m)", "Rotation Error (deg)")
plot_metric(axs[1, 0], yaw_centers, [ccv_rate_yaw * 100, lbcv_rate_yaw * 100], labels,
            colors, markers, "Detection Rate vs Out-of-Plane Rotation", "Out-of-Plane Rotation (deg)", "Detection Rate (%)")
plot_metric(axs[1, 1], yaw_centers, [ccv_pos_yaw, lbcv_pos_yaw], labels,
            colors, markers, "Position Error vs Out-of-Plane Rotation", "Out-of-Plane Rotation (deg)", "Translation Error (mm)")
plot_metric(axs[1, 2], yaw_centers, [ccv_rot_yaw, lbcv_rot_yaw], labels,
            colors, markers, "Rotation Error vs Out-of-Plane Rotation", "Out-of-Plane Rotation (deg)", "Rotation Error (deg)")

plt.tight_layout()
plt.savefig("./real_data_processing/results/in_frame_performance_vs_pose.png")
plt.show()

# === NOT FULLY IN FRAME ===
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
plt.suptitle("Marker Not Fully in View", fontsize=20)

# Only LBCV is evaluated in this case
plot_metric(axs[0, 0], z_centers_partial, [lbcv_rate_z_partial * 100], ["Our Method"],
            [color_lbcv], ['o'], "Detection Rate vs Z", "Out-of-Plane Distance (m)", "Detection Rate (%)")
plot_metric(axs[0, 1], z_centers_partial, [lbcv_pos_z_partial], ["Our Method"],
            [color_lbcv], ['o'], "Position Error vs Z", "Out-of-Plane Distance (m)", "Translation Error (mm)")
plot_metric(axs[0, 2], z_centers_partial, [lbcv_rot_z_partial], ["Our Method"],
            [color_lbcv], ['o'], "Rotation Error vs Z", "Out-of-Plane Distance (m)", "Rotation Error (deg)")
plot_metric(axs[1, 0], yaw_centers_partial, [lbcv_rate_yaw_partial * 100], ["Our Method"],
            [color_lbcv], ['o'], "Detection Rate vs Yaw", "Out-of-Plane Rotation (deg)", "Detection Rate (%)")
plot_metric(axs[1, 1], yaw_centers_partial, [lbcv_pos_yaw_partial], ["Our Method"],
            [color_lbcv], ['o'], "Position Error vs Yaw", "Out-of-Plane Rotation (deg)", "Translation Error (mm)")
plot_metric(axs[1, 2], yaw_centers_partial, [lbcv_rot_yaw_partial], ["Our Method"],
            [color_lbcv], ['o'], "Rotation Error vs Yaw", "Out-of-Plane Rotation (deg)", "Rotation Error (deg)")

plt.tight_layout()
plt.savefig("./real_data_processing/results/partial_view_performance_vs_pose.png")
plt.show()

