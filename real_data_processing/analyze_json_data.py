import numpy as np 
import pandas as pd 
import json 
import matplotlib.pyplot as plt
import seaborn as sns
from real_data_processing.utils import tf_to_pose

# read json file 
json_path = "./real_data_processing/results/trial_6.json"
with open(json_path, 'r') as file:
    data = json.load(file)

# create lists of poses of each method 
methods = ['optk','ccv','lbcv','hcv'] 
transforms = {method: [] for method in methods} 
poses = {method: [] for method in methods} 
transform_errors = {method: [] for method in methods}
pose_errors = {method: [] for method in methods}
hcv_residuals = [] 
# iterate through the data and extract poses for each method
for idx, datapoint in enumerate(data): 
    for method in methods:
        col_name = f"{method}" 
        tf_flat = datapoint[f"{method}_tf"]
        if tf_flat is None or len(tf_flat) != 16:
            transforms[method].append(None)  
            poses[method].append(None)
            if method == 'hcv':
                hcv_residuals.append(None)
                transform_errors[method].append(None)
                pose_errors[method].append(None)
            continue
        tf = np.array(tf_flat).reshape(4, 4) 
        if method == 'lbcv': 
            tf = np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]) @ tf @ np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]) 
        if method == 'hcv': 
            tf = tf @ np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])  
            if np.linalg.norm(tf[:3,3]) > 1.0: 
                transforms[method].append(None)  
                poses[method].append(None)
                hcv_residuals.append(None)
                transform_errors[method].append(None)
                pose_errors[method].append(None)
                continue
            else: 
                hcv_residuals.append(datapoint['hcv_residual'])
        # reframe to make euler angles centered around zero 
        tf = tf @ np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        pose = tf_to_pose(tf)  
        transforms[method].append(tf)  
        poses[method].append(pose)

        tf_true = transforms['optk'][idx]  # assuming 'optk' is the ground truth
        if tf_true is not None:
            tf_err = np.linalg.inv(tf_true) @ tf 
            pose_err = tf_to_pose(tf_err) 
            transform_errors[method].append(tf_err)
            pose_errors[method].append(pose_err)
        else:
            transform_errors[method].append(None)
            pose_errors[method].append(None)


# figure of 2x3 subplots for each pose dimension, with each method coplotted 
fig, axs = plt.subplots(2, 3, figsize=(15, 10)) 
pose_labels = ['x', 'y', 'z', 'roll', 'pitch', 'yaw'] 
for i, label in enumerate(pose_labels):
    ax = axs[i // 3, i % 3]
    for method in methods:
        if transforms[method] is not None:
            pose_values = [pose[i] if pose is not None else None for pose in poses[method]]
            ax.plot(pose_values, label=method)
    ax.set_title(label)
    ax.set_xlabel('Data Point Index')
    ax.set_ylabel(label)
    ax.legend()
plt.tight_layout()
# plt.savefig('./real_data_processing/results/pose_analysis.png')
plt.show()

# plot histogram of pose errors for each method
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

for i, label in enumerate(pose_labels):
    ax = axs[i // 3, i % 3]

    for method in methods:
        if pose_errors[method] is not None:
            # Replace None with np.nan, then filter out
            pose_err_values = [
                pose[i] if (pose is not None and pose[i] is not None) else np.nan
                for pose in pose_errors[method]
            ]
            pose_err_values = np.array(pose_err_values)
            pose_err_values = pose_err_values[~np.isnan(pose_err_values)]

            if len(pose_err_values) > 0:
                # Seaborn histplot with KDE
                sns.histplot(
                    pose_err_values,
                    bins=20,
                    kde=True,
                    stat="count",
                    alpha=0.5,
                    label=method,
                    ax=ax
                )

    ax.set_title(f'Error Distribution: {label}')
    ax.set_xlabel(label)
    ax.set_ylabel('Frequency')
    ax.legend()

plt.tight_layout()
# plt.savefig('./real_data_processing/results/pose_error_histograms.png')
plt.show()
# print out the mean and std of each pose error dimension for each method
for method in methods:
    print(f"Method: {method}")
    for i, label in enumerate(pose_labels):
        pose_err_values = [
            pose[i] if (pose is not None and pose[i] is not None) else np.nan
            for pose in pose_errors[method]
        ]
        pose_err_values = np.array(pose_err_values)
        pose_err_values = pose_err_values[~np.isnan(pose_err_values)]
        
        if len(pose_err_values) > 0:
            mean = np.mean(pose_err_values)
            MAE = np.mean(np.abs(pose_err_values))
            std = np.std(pose_err_values)
            print(f"  {label}: Mean = {mean:.4f}, MAE = {MAE:.4f}, Std = {std:.4f}")
        else:
            print(f"  {label}: No valid data")


# plot 2x3 hcv absolute pose errors vs residual values 
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
for i, label in enumerate(pose_labels): 
    ax = axs[i // 3, i % 3]
    hcv_errors = [pose[i] if pose is not None else np.nan for pose in pose_errors['hcv']]
    hcv_residuals_filtered = [res for res in hcv_residuals if res is not None]

    # Filter out NaN values
    hcv_errors = np.array(hcv_errors)
    hcv_errors = hcv_errors[~np.isnan(hcv_errors)]

    if len(hcv_errors) > 0 and len(hcv_residuals_filtered) > 0:
        ax.scatter(hcv_residuals_filtered, np.abs(hcv_errors), label='HCV Errors', alpha=0.5)
        ax.set_title(f'HCV {label} Error vs Residual')
        ax.set_xlabel('Residual Value')
        ax.set_ylabel(label)
        ax.legend()
plt.tight_layout()
# plt.savefig('./real_data_processing/results/hcv_error_vs_residuals.png')
plt.show()

# bin the error data and compute mean and std for each bin and plot the binned data
def bin_data(x, y, num_bins=10):
    """Bin the data and compute mean and std for each bin."""
    bins = np.linspace(np.min(x), np.max(x), num_bins + 1)
    bin_means = []
    bin_stds = []
    bin_centers = []

    for i in range(num_bins):
        bin_mask = (x >= bins[i]) & (x < bins[i + 1])
        if np.any(bin_mask):
            bin_means.append(np.mean(y[bin_mask]))
            bin_stds.append(np.std(y[bin_mask]))
            bin_centers.append((bins[i] + bins[i + 1]) / 2)

    return np.array(bin_centers), np.array(bin_means), np.array(bin_stds)
# plot binned data for each method
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
for i, label in enumerate(pose_labels): 
    ax = axs[i // 3, i % 3]
    for method in ['hcv']:
        if pose_errors[method] is not None:
            pose_values = [pose[i] if pose is not None else np.nan for pose in pose_errors[method]]
            hcv_residuals_filtered = [res for res in hcv_residuals if res is not None]

            # Filter out NaN values
            pose_values = np.array(pose_values)
            pose_values = pose_values[~np.isnan(pose_values)]

            if len(pose_values) > 0 and len(hcv_residuals_filtered) > 0:
                bin_centers, bin_means, bin_stds = bin_data(np.array(hcv_residuals_filtered), pose_values)
                ax.errorbar(bin_centers, bin_means, yerr=bin_stds, label=method, fmt='o', capsize=5)

    ax.set_title(f'Binned {label} Error vs Residual')
    ax.set_xlabel('Residual Value')
    ax.set_ylabel(label)
    ax.legend()
plt.tight_layout()
# plt.savefig('./real_data_processing/results/binned_pose_error_vs_residuals.png')
plt.show()