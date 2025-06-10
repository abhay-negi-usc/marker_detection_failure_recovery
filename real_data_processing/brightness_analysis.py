import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binned_statistic
import json
from scipy.spatial.transform import Rotation as R

# Seaborn style
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.8)

color_ccv = "#126782"
color_lbcv = "#fb8500"
labels = ["OpenCV", "Our Method"]
colors = [color_ccv, color_lbcv]
markers = ['s', 'o']
bins = 25

def load_and_process(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    tf_optk = [np.array(dp["optk_tf"]).reshape(4, 4) if dp["optk_tf"] is not None else None for dp in data]
    tf_CCV = [np.array(dp["ccv_tf"]).reshape(4, 4) if dp["ccv_tf"] is not None else None for dp in data]
    tf_LBCV = [np.array(dp["lbcv_tf"]).reshape(4, 4) if dp["lbcv_tf"] is not None else None for dp in data]

    tf_optk_clean = [tf for tf in tf_optk if tf is not None]
    tf_true = np.mean(tf_optk_clean, axis=0)

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
        ccv_detected.append(1.0 if tf_gt is not None else 0.0)
        if tf1 is not None:
            p_err, r_err = get_error(tf_true, tf1)
            ccv_pos_err.append(p_err)
            ccv_rot_err.append(r_err)
        else:
            ccv_pos_err.append(np.nan)
            ccv_rot_err.append(np.nan)

        if tf2 is not None:
            p_err, r_err = get_error(tf_true, tf2)
            lbcv_detected.append(1.0)
            lbcv_pos_err.append(p_err)
            lbcv_rot_err.append(r_err)
        else:
            lbcv_detected.append(0.0)
            lbcv_pos_err.append(np.nan)
            lbcv_rot_err.append(np.nan)

    def binned_stats(x, y, bins=10):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        bin_means, bin_edges, _ = binned_statistic(x, y, statistic="mean", bins=bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return bin_centers, bin_means

    # === Normalize brightness to [0, 1] ===
    marker_brightness = np.array([dp.get("marker_brightness", np.nan) for dp in data], dtype=float)
    marker_brightness = pd.Series(marker_brightness).interpolate().to_numpy()
    marker_brightness = marker_brightness / 255.0  # Normalize to [0, 1]

    ccv_detected = np.array(ccv_detected)
    ccv_pos_err = np.array(ccv_pos_err)
    ccv_rot_err = np.array(ccv_rot_err)
    lbcv_detected = np.array(lbcv_detected)
    lbcv_pos_err = np.array(lbcv_pos_err)
    lbcv_rot_err = np.array(lbcv_rot_err)

    brightness_centers, ccv_detect = binned_stats(marker_brightness, ccv_detected, bins)
    _, lbcv_detect = binned_stats(marker_brightness, lbcv_detected, bins)

    _, ccv_pos = binned_stats(marker_brightness[ccv_detected == 1], ccv_pos_err[ccv_detected == 1], bins)
    _, lbcv_pos = binned_stats(marker_brightness[lbcv_detected == 1], lbcv_pos_err[lbcv_detected == 1], bins)

    _, ccv_rot = binned_stats(marker_brightness[ccv_detected == 1], ccv_rot_err[ccv_detected == 1], bins)
    _, lbcv_rot = binned_stats(marker_brightness[lbcv_detected == 1], lbcv_rot_err[lbcv_detected == 1], bins)

    return brightness_centers, (ccv_detect, lbcv_detect), (ccv_pos, lbcv_pos), (ccv_rot, lbcv_rot)

# Load both experiments
dark_brightness, dark_detect, dark_pos, dark_rot = load_and_process("./real_data_processing/results/dark_test.json")
bright_brightness, bright_detect, bright_pos, bright_rot = load_and_process("./real_data_processing/results/bright_test.json")

# === Plot ===
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

def plot_metric(ax, x, y_list, labels, colors, markers, ylabel):
    for y, label, color, marker in zip(y_list, labels, colors, markers):
        sns.lineplot(x=x, y=y, ax=ax, label=label, color=color, marker=marker)
    ax.set_xlabel("Marker Brightness (Normalized)")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()

# === Row 0: Low Lighting ===
plot_metric(axs[0, 0], dark_brightness, [dark_detect[0]*100, dark_detect[1]*100], labels, colors, markers, "Detection Rate (%)")
plot_metric(axs[0, 1], dark_brightness, [dark_pos[0], dark_pos[1]], labels, colors, markers, "Translation Error (mm)")
plot_metric(axs[0, 2], dark_brightness, [dark_rot[0], dark_rot[1]], labels, colors, markers, "Rotation Error (deg)")

# === Row 1: Glare ===
plot_metric(axs[1, 0], bright_brightness, [bright_detect[0]*100, bright_detect[1]*100], labels, colors, markers, "Detection Rate (%)")
plot_metric(axs[1, 1], bright_brightness, [bright_pos[0], bright_pos[1]], labels, colors, markers, "Translation Error (mm)")
plot_metric(axs[1, 2], bright_brightness, [bright_rot[0], bright_rot[1]], labels, colors, markers, "Rotation Error (deg)")

# === Row titles ===
fig.text(0.55, 0.97, "Low-Lighting Experiment", ha="center", va="top", fontsize=18, weight="bold")
fig.text(0.55, 0.48, "Glare Experiment", ha="center", va="top", fontsize=18, weight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitles
plt.savefig("./real_data_processing/results/performance_vs_brightness_combined.png")
plt.show()
