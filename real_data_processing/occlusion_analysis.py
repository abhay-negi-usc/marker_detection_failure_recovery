import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binned_statistic
import json 
from scipy.spatial.transform import Rotation as R


# Seaborn style for publication-quality figures
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.8)

# Load data
with open("./real_data_processing/results/trial_6.json", "r") as f:
    data = json.load(f)

# Filter and truncate
data = np.concatenate((data[0:1000], data[1150:1700], data[1850:2200], data[2500:3600]), axis=0)
n = len(data)

# Extract relevant fields
marker_area = np.array([dp.get("marker_area", 0) for dp in data])
marker_fraction = np.array([dp.get("marker_inframe_fraction", 0.0) for dp in data])
tf_optk = [np.array(dp["optk_tf"]).reshape(4, 4) if dp["optk_tf"] is not None else None for dp in data]
tf_CCV = [np.array(dp["ccv_tf"]).reshape(4, 4) if dp["ccv_tf"] is not None else None for dp in data]
tf_LBCV = [np.array(dp["lbcv_tf"]).reshape(4, 4) if dp["lbcv_tf"] is not None else None for dp in data]

# === Metric extraction ===
def get_error(tf_gt, tf_pred):
    tf_rel = np.linalg.inv(tf_gt) @ tf_pred
    t = tf_rel[:3, 3]
    R_rel = tf_rel[:3, :3]
    pos_err = np.linalg.norm(t) * 1e3  # mm
    rot_err = np.linalg.norm(R.from_matrix(R_rel).as_rotvec()) * 180 / np.pi  # deg
    return pos_err, rot_err

# Initialize lists
ccv_detected = []
ccv_pos_err = []
ccv_rot_err = []

lbcv_detected = []
lbcv_pos_err = []
lbcv_rot_err = []

for tf_gt, tf1, tf2 in zip(tf_optk, tf_CCV, tf_LBCV):
    # CCV
    if tf_gt is not None:
        if tf1 is not None:
            p_err, r_err = get_error(tf_gt, tf1)
            ccv_detected.append(1.0)
            ccv_pos_err.append(p_err)
            ccv_rot_err.append(r_err)
        else:
            ccv_detected.append(0.0)
            ccv_pos_err.append(np.nan)
            ccv_rot_err.append(np.nan)

        # LBCV
        if tf2 is not None:
            p_err, r_err = get_error(tf_gt, tf2)
            lbcv_detected.append(1.0)
            lbcv_pos_err.append(p_err)
            lbcv_rot_err.append(r_err)
        else:
            lbcv_detected.append(0.0)
            lbcv_pos_err.append(np.nan)
            lbcv_rot_err.append(np.nan)
    else:
        ccv_detected.append(np.nan)
        lbcv_detected.append(np.nan)
        ccv_pos_err.append(np.nan)
        lbcv_pos_err.append(np.nan)
        ccv_rot_err.append(np.nan)
        lbcv_rot_err.append(np.nan)

# Convert to np arrays
ccv_detected = np.array(ccv_detected)
ccv_pos_err = np.array(ccv_pos_err)
ccv_rot_err = np.array(ccv_rot_err)

lbcv_detected = np.array(lbcv_detected)
lbcv_pos_err = np.array(lbcv_pos_err)
lbcv_rot_err = np.array(lbcv_rot_err)

# === Binning helper ===
def binned_stats(x, y, bins=10):
    bin_means, bin_edges, _ = binned_statistic(x, y, statistic="mean", bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, bin_means

# === Marker Area Metrics ===
bins = 10
marker_pixels = marker_area

ccv_detect_pixel_centers, ccv_detect_pixel = binned_stats(marker_pixels, ccv_detected, bins)
lbcv_detect_pixel_centers, lbcv_detect_pixel = binned_stats(marker_pixels, lbcv_detected, bins)

_, ccv_pos_pixel = binned_stats(marker_pixels[ccv_detected == 1], ccv_pos_err[ccv_detected == 1], bins)
_, lbcv_pos_pixel = binned_stats(marker_pixels[lbcv_detected == 1], lbcv_pos_err[lbcv_detected == 1], bins)

_, ccv_rot_pixel = binned_stats(marker_pixels[ccv_detected == 1], ccv_rot_err[ccv_detected == 1], bins)
_, lbcv_rot_pixel = binned_stats(marker_pixels[lbcv_detected == 1], lbcv_rot_err[lbcv_detected == 1], bins)

# === Marker Fraction Metrics ===
marker_fraction = np.array(marker_fraction)

ccv_detect_fraction_centers, ccv_detect_fraction = binned_stats(marker_fraction, ccv_detected, bins)
lbcv_detect_fraction_centers, lbcv_detect_fraction = binned_stats(marker_fraction, lbcv_detected, bins)

_, ccv_pos_fraction = binned_stats(marker_fraction[ccv_detected == 1], ccv_pos_err[ccv_detected == 1], bins)
_, lbcv_pos_fraction = binned_stats(marker_fraction[lbcv_detected == 1], lbcv_pos_err[lbcv_detected == 1], bins)

_, ccv_rot_fraction = binned_stats(marker_fraction[ccv_detected == 1], ccv_rot_err[ccv_detected == 1], bins)
_, lbcv_rot_fraction = binned_stats(marker_fraction[lbcv_detected == 1], lbcv_rot_err[lbcv_detected == 1], bins)

# === Plotting ===

color_ccv = "#126782"
color_lbcv = "#fb8500"
# === Combined 2x3 Performance Plot ===
fig, axs = plt.subplots(2, 3, figsize=(20, 12))
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.8)

# Marker Pixels
# sns.lineplot(x=ccv_detect_pixel_centers/1e3, y=ccv_detect_pixel * 100,
#              ax=axs[0, 0], label="OpenCV", marker='s')
sns.lineplot(x=lbcv_detect_pixel_centers/1e3, y=lbcv_detect_pixel * 100,
             ax=axs[0, 0], label="Our Method", marker='o', color=color_lbcv)
axs[0, 0].set_title("Detection Rate vs Marker Pixels")
axs[0, 0].set_xlabel("Marker Area (1K pixels)")
axs[0, 0].set_ylabel("Detection Rate (%)")

# sns.lineplot(x=ccv_detect_pixel_centers/1e3, y=ccv_pos_pixel,
#              ax=axs[0, 1], label="OpenCV", marker='s')
sns.lineplot(x=lbcv_detect_pixel_centers/1e3, y=lbcv_pos_pixel,
             ax=axs[0, 1], label="Our Method", marker='o', color=color_lbcv)
axs[0, 1].set_title("Position Error vs Marker Pixels")
axs[0, 1].set_xlabel("Marker Area (1K pixels)")
axs[0, 1].set_ylabel("Translation Error (mm)")

# sns.lineplot(x=ccv_detect_pixel_centers/1e3, y=ccv_rot_pixel,
#              ax=axs[0, 2], label="OpenCV", marker='s')
sns.lineplot(x=lbcv_detect_pixel_centers/1e3, y=lbcv_rot_pixel,
             ax=axs[0, 2], label="Our Method", marker='o', color=color_lbcv)
axs[0, 2].set_title("Rotation Error vs Marker Pixels")
axs[0, 2].set_xlabel("Marker Area (1K pixels)")
axs[0, 2].set_ylabel("Rotation Error (deg)")

# Marker Fraction
# sns.lineplot(x=ccv_detect_fraction_centers, y=ccv_detect_fraction * 100,
            #  ax=axs[1, 0], label="OpenCV", marker='s')
sns.lineplot(x=lbcv_detect_fraction_centers, y=lbcv_detect_fraction * 100,
             ax=axs[1, 0], label="Our Method", marker='o', color=color_lbcv)
axs[1, 0].set_title("Detection Rate vs Fraction of Marker Visible")
axs[1, 0].set_xlabel("Fraction of Marker Visible")
axs[1, 0].set_ylabel("Detection Rate (%)")

# sns.lineplot(x=ccv_detect_fraction_centers, y=ccv_pos_fraction,
            #  ax=axs[1, 1], label="OpenCV", marker='s')
sns.lineplot(x=lbcv_detect_fraction_centers, y=lbcv_pos_fraction,
             ax=axs[1, 1], label="Our Method", marker='o', color=color_lbcv)
axs[1, 1].set_title("Position Error vs Fraction of Marker Visible")
axs[1, 1].set_xlabel("Fraction of Marker Visible")
axs[1, 1].set_ylabel("Translation Error (mm)")

# sns.lineplot(x=ccv_detect_fraction_centers, y=ccv_rot_fraction,
#              ax=axs[1, 2], label="OpenCV", marker='s')
sns.lineplot(x=lbcv_detect_fraction_centers, y=lbcv_rot_fraction,
             ax=axs[1, 2], label="Our Method", marker='o', color=color_lbcv)
axs[1, 2].set_title("Rotation Error vs Fraction of Marker Visible")
axs[1, 2].set_xlabel("Fraction of Marker Visible")
axs[1, 2].set_ylabel("Rotation Error (deg)")

# Final layout
for ax in axs.flatten():
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig("./real_data_processing/results/performance_vs_marker_metrics_combined.png")
plt.show()


# Extract relevant fields
marker_pos_vel = np.array([dp.get("marker_pos_vel", 0.0) for dp in data])
marker_rot_vel = np.array([dp.get("marker_rot_vel", 0.0) for dp in data])
tf_optk = [np.array(dp["optk_tf"]).reshape(4, 4) if dp["optk_tf"] is not None else None for dp in data]
tf_CCV = [np.array(dp["ccv_tf"]).reshape(4, 4) if dp["ccv_tf"] is not None else None for dp in data]
tf_LBCV = [np.array(dp["lbcv_tf"]).reshape(4, 4) if dp["lbcv_tf"] is not None else None for dp in data]

# Compute errors
def get_error(tf_gt, tf_pred):
    tf_rel = np.linalg.inv(tf_gt) @ tf_pred
    t = tf_rel[:3, 3]
    R_rel = tf_rel[:3, :3]
    pos_err = np.linalg.norm(t) * 1e3  # mm
    rot_err = np.linalg.norm(R.from_matrix(R_rel).as_rotvec()) * 180 / np.pi  # deg
    return pos_err, rot_err

ccv_detected, ccv_pos_err, ccv_rot_err = [], [], []
lbcv_detected, lbcv_pos_err, lbcv_rot_err = [], [], []

for tf_gt, tf1, tf2 in zip(tf_optk, tf_CCV, tf_LBCV):
    if tf_gt is not None:
        if tf1 is not None:
            p_err, r_err = get_error(tf_gt, tf1)
            ccv_detected.append(1.0)
            ccv_pos_err.append(p_err)
            ccv_rot_err.append(r_err)
        else:
            ccv_detected.append(0.0)
            ccv_pos_err.append(np.nan)
            ccv_rot_err.append(np.nan)
        if tf2 is not None:
            p_err, r_err = get_error(tf_gt, tf2)
            lbcv_detected.append(1.0)
            lbcv_pos_err.append(p_err)
            lbcv_rot_err.append(r_err)
        else:
            lbcv_detected.append(0.0)
            lbcv_pos_err.append(np.nan)
            lbcv_rot_err.append(np.nan)
    else:
        ccv_detected.append(np.nan)
        ccv_pos_err.append(np.nan)
        ccv_rot_err.append(np.nan)
        lbcv_detected.append(np.nan)
        lbcv_pos_err.append(np.nan)
        lbcv_rot_err.append(np.nan)

# Convert to arrays
ccv_detected = np.array(ccv_detected)
ccv_pos_err = np.array(ccv_pos_err)
ccv_rot_err = np.array(ccv_rot_err)
lbcv_detected = np.array(lbcv_detected)
lbcv_pos_err = np.array(lbcv_pos_err)
lbcv_rot_err = np.array(lbcv_rot_err)

# Convert to clean float arrays, replacing invalid entries with np.nan
marker_pos_vel = np.array([dp.get("marker_pos_vel", np.nan) for dp in data], dtype=float)
marker_rot_vel = np.array([dp.get("marker_rot_vel", np.nan) for dp in data], dtype=float)

# Convert detection metrics to arrays if not already
ccv_detected = np.array(ccv_detected, dtype=float)
lbcv_detected = np.array(lbcv_detected, dtype=float)


# Helper for binned stats
def binned_stats(x, y, bins=10):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Only keep entries where both x and y are finite
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    bin_means, bin_edges, _ = binned_statistic(x, y, statistic="mean", bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, bin_means



# Compute binned metrics
bins = 10
posvel_centers, ccv_detect_vel = binned_stats(marker_pos_vel, ccv_detected, bins)
_, ccv_pos_vel = binned_stats(marker_pos_vel[ccv_detected == 1], ccv_pos_err[ccv_detected == 1], bins)
_, ccv_rot_vel = binned_stats(marker_pos_vel[ccv_detected == 1], ccv_rot_err[ccv_detected == 1], bins)

_, lbcv_detect_vel = binned_stats(marker_pos_vel, lbcv_detected, bins)
_, lbcv_pos_vel = binned_stats(marker_pos_vel[lbcv_detected == 1], lbcv_pos_err[lbcv_detected == 1], bins)
_, lbcv_rot_vel = binned_stats(marker_pos_vel[lbcv_detected == 1], lbcv_rot_err[lbcv_detected == 1], bins)

rotvel_centers, ccv_detect_rvel = binned_stats(marker_rot_vel, ccv_detected, bins)
_, ccv_pos_rvel = binned_stats(marker_rot_vel[ccv_detected == 1], ccv_pos_err[ccv_detected == 1], bins)
_, ccv_rot_rvel = binned_stats(marker_rot_vel[ccv_detected == 1], ccv_rot_err[ccv_detected == 1], bins)

_, lbcv_detect_rvel = binned_stats(marker_rot_vel, lbcv_detected, bins)
_, lbcv_pos_rvel = binned_stats(marker_rot_vel[lbcv_detected == 1], lbcv_pos_err[lbcv_detected == 1], bins)
_, lbcv_rot_rvel = binned_stats(marker_rot_vel[lbcv_detected == 1], lbcv_rot_err[lbcv_detected == 1], bins)

# Show output in figure
import seaborn as sns
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.8)

fig, axs = plt.subplots(2, 3, figsize=(18, 10))
color_ccv = "#126782"
color_lbcv = "#fb8500"

sns.lineplot(x=posvel_centers, y=ccv_detect_vel * 100, ax=axs[0, 0], label="OpenCV", color=color_ccv, marker="s")
sns.lineplot(x=posvel_centers, y=lbcv_detect_vel * 100, ax=axs[0, 0], label="Our Method", color=color_lbcv, marker="o")
axs[0, 0].set_title("Detection Rate vs Positional Velocity")
axs[0, 0].set_xlabel("Positional Velocity (m/s)")
axs[0, 0].set_ylabel("Detection Rate (%)")

sns.lineplot(x=posvel_centers, y=ccv_pos_vel, ax=axs[0, 1], label="OpenCV", color=color_ccv, marker="s")
sns.lineplot(x=posvel_centers, y=lbcv_pos_vel, ax=axs[0, 1], label="Our Method", color=color_lbcv, marker="o")
axs[0, 1].set_title("Position Error vs Positional Velocity")
axs[0, 1].set_xlabel("Positional Velocity (m/s)")
axs[0, 1].set_ylabel("Translation Error (mm)")

sns.lineplot(x=posvel_centers, y=ccv_rot_vel, ax=axs[0, 2], label="OpenCV", color=color_ccv, marker="s")
sns.lineplot(x=posvel_centers, y=lbcv_rot_vel, ax=axs[0, 2], label="Our Method", color=color_lbcv, marker="o")
axs[0, 2].set_title("Rotation Error vs Positional Velocity")
axs[0, 2].set_xlabel("Positional Velocity (m/s)")
axs[0, 2].set_ylabel("Rotation Error (deg)")

sns.lineplot(x=rotvel_centers, y=ccv_detect_rvel * 100, ax=axs[1, 0], label="OpenCV", color=color_ccv, marker="s")
sns.lineplot(x=rotvel_centers, y=lbcv_detect_rvel * 100, ax=axs[1, 0], label="Our Method", color=color_lbcv, marker="o")
axs[1, 0].set_title("Detection Rate vs Rotational Velocity")
axs[1, 0].set_xlabel("Rotational Velocity (deg/s)")
axs[1, 0].set_ylabel("Detection Rate (%)")

sns.lineplot(x=rotvel_centers, y=ccv_pos_rvel, ax=axs[1, 1], label="OpenCV", color=color_ccv, marker="s")
sns.lineplot(x=rotvel_centers, y=lbcv_pos_rvel, ax=axs[1, 1], label="Our Method", color=color_lbcv, marker="o")
axs[1, 1].set_title("Position Error vs Rotational Velocity")
axs[1, 1].set_xlabel("Rotational Velocity (deg/s)")
axs[1, 1].set_ylabel("Translation Error (mm)")

sns.lineplot(x=rotvel_centers, y=ccv_rot_rvel, ax=axs[1, 2], label="OpenCV", color=color_ccv, marker="s")
sns.lineplot(x=rotvel_centers, y=lbcv_rot_rvel, ax=axs[1, 2], label="Our Method", color=color_lbcv, marker="o")
axs[1, 2].set_title("Rotation Error vs Rotational Velocity")
axs[1, 2].set_xlabel("Rotational Velocity (deg/s)")
axs[1, 2].set_ylabel("Rotation Error (deg)")

plt.tight_layout()
plt.savefig("./real_data_processing/results/performance_vs_velocity.png")
plt.show()
