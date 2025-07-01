import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os 
import yaml 
from scipy.stats import gaussian_kde

class Plotter():
    def __init__(self, config, global_ranges):
        self.config = config
        self.output_dir = config.get("output_path", "./ablations/analysis/plots")
        os.makedirs(self.output_dir, exist_ok=True)
        self.df_data = pd.read_csv(self.config["results_path"])
        # self.global_ranges = global_ranges
        self.global_ranges = {
            'x': 0.10,
            'y': 0.10,
            'z': 0.10,
            'a': 30.0,
            'b': 30.0,
            'c': 30.0
        }

    @staticmethod
    def calculate_global_max_ranges(dataframes, plot_LBCV=True, plot_HCV=False):
        """Calculate the global max range for each pose variable across all ablations."""
        pose_variables = ['x', 'y', 'z', 'a', 'b', 'c']
        max_ranges = {var: 0 for var in pose_variables}

        for df_data in dataframes:
            for var in pose_variables:
                max_ranges[var] = max(
                    max_ranges[var],
                    df_data[f'pose_error_CCV_{var}'].abs().max(),
                    df_data[f'pose_error_LBCV_{var}'].abs().max() if plot_LBCV else 0,
                    df_data[f'pose_error_HCV_{var}'].abs().max() if plot_HCV else 0
                )

        # Ensure x and y share the same range, and a and b share the same range
        max_ranges['x'] = max_ranges['y'] = max(max_ranges['x'], max_ranges['y'])
        max_ranges['a'] = max_ranges['b'] = max(max_ranges['a'], max_ranges['b'])

        return max_ranges

    def detection_plot(self, ablation_variable, n_bins=10):    
        ablation_variable_pretty = ablation_variable.replace("_", " ").title()
        bin_name = f'{ablation_variable}_bin'

        # Create bins and labels from value ranges
        bins = pd.cut(self.df_data[ablation_variable], bins=n_bins)
        self.df_data[bin_name] = bins

        # Grouped detection rates
        grouped = self.df_data.groupby(bin_name).agg({
            'detected_CCV': 'mean',
            'detected_LBCV': 'mean'
        }).reset_index().rename(columns={
            'detected_CCV': 'CCV',
            'detected_LBCV': 'LBCV'
        })

        # Convert Interval index to strings for x-axis labeling
        grouped[bin_name] = grouped[bin_name].astype(str)

        # Melt for seaborn plotting
        detection_rate_melted = grouped.melt(id_vars=bin_name, value_vars=['CCV', 'LBCV'], 
                                            var_name='Method', value_name='Detection Rate')

        # --- Barplot (binned) with actual bin ranges as x-axis labels ---
        plt.figure(figsize=(10, 6))
        sns.barplot(x=bin_name, y='Detection Rate', hue='Method', data=detection_rate_melted)
        plt.xlabel(f'{ablation_variable_pretty} (range)')
        plt.ylabel('Detection Rate')
        plt.title(f'Detection Rate vs {ablation_variable_pretty}')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{ablation_variable}_detection_rate_binned.png"))
        plt.close()


        # --- Scatter plot (raw) + Moving Mean ---
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=ablation_variable, y='detected_CCV', data=self.df_data, label='CCV', alpha=0.5)
        sns.scatterplot(x=ablation_variable, y='detected_LBCV', data=self.df_data, label='LBCV', alpha=0.5)

        # Sort by ablation variable for consistent rolling
        df_sorted = self.df_data.sort_values(by=ablation_variable)

        # Rolling mean for smoother trend
        window = max(5, len(df_sorted) // 20)  # dynamic window size
        mean_ccv = df_sorted[ablation_variable].rolling(window).mean()
        roll_ccv = df_sorted['detected_CCV'].rolling(window).mean()
        roll_lbcv = df_sorted['detected_LBCV'].rolling(window).mean()

        plt.plot(mean_ccv, roll_ccv, label='CCV (Moving Mean)', linewidth=2)
        plt.plot(mean_ccv, roll_lbcv, label='LBCV (Moving Mean)', linewidth=2)

        plt.xlabel(ablation_variable_pretty)
        plt.ylabel('Detection Rate')
        plt.title(f'Detection Rate vs {ablation_variable_pretty}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{ablation_variable}_detection_rate_scatter.png"))
        plt.close()


    def IOU_plot(self, ablation_variable, n_bins=10):    
        ablation_variable_pretty = ablation_variable.replace("_", " ").title()
        bin_name = f'{ablation_variable}_bin'
        self.df_data[bin_name] = pd.cut(self.df_data[ablation_variable], bins=n_bins, labels=False)

        # Grouped detection rates
        grouped = self.df_data.groupby(bin_name).agg({
            # 'detected_CCV': 'mean',
            'LBCV_IOU': 'mean'
        }).reset_index().rename(columns={
            # 'detected_CCV': 'CCV',
            'LBCV_IOU': 'LBCV'
        })

        # Melt for seaborn plotting
        detection_rate_melted = grouped.melt(id_vars=bin_name, value_vars=['LBCV'], var_name='Method', value_name='LBCV_IOU')

        # --- Barplot (binned) ---
        plt.figure(figsize=(10, 6))
        sns.barplot(x=bin_name, y='LBCV_IOU', hue='Method', data=detection_rate_melted)
        plt.xlabel(f'{ablation_variable_pretty} (binned)')
        plt.ylabel('IOU')
        plt.title(f'IOU vs {ablation_variable_pretty}')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{ablation_variable}_IOU_binned.png"))
        plt.close()

        # --- Scatter plot (raw) + Moving Mean ---
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=ablation_variable, y='LBCV_IOU', data=self.df_data, label='LBCV', alpha=0.5)

        df_sorted = self.df_data.sort_values(by=ablation_variable)
        window = max(5, len(df_sorted) // 20)
        mean_var = df_sorted[ablation_variable].rolling(window).mean()
        roll_iou = df_sorted['LBCV_IOU'].rolling(window).mean()
        plt.plot(mean_var, roll_iou, label='LBCV IOU (Moving Mean)', linewidth=2)

        plt.xlabel(ablation_variable_pretty)
        plt.ylabel('IOU')
        plt.title(f'IOU vs {ablation_variable_pretty}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{ablation_variable}_IOU_scatter.png"))
        plt.close()

    def error_plot(self, ablation_variable):
        ablation_variable_pretty = ablation_variable.replace("_", " ").title()
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.flatten()
        error_types = ['x', 'y', 'z', 'a', 'b', 'c']

        for i, err_type in enumerate(error_types):
            col_CCV = f'pose_error_CCV_{err_type}'
            col_LBCV = f'pose_error_LBCV_{err_type}'

            # Plot raw scatter for CCV
            sns.scatterplot(x=ablation_variable, y=col_CCV, data=self.df_data, ax=axs[i], label='CCV', alpha=0.5)

            # Plot raw scatter for LBCV if enabled
            if self.config.get("plot_LBCV", True):
                sns.scatterplot(x=ablation_variable, y=col_LBCV, data=self.df_data, ax=axs[i], label='LBCV', alpha=0.5)

            # Compute moving mean
            df_sorted = self.df_data.sort_values(by=ablation_variable)
            window = max(5, len(df_sorted) // 50)
            x_rolling = df_sorted[ablation_variable].rolling(window).mean()
            roll_ccv = df_sorted[col_CCV].rolling(window).mean()

            # Plot moving mean for CCV
            axs[i].plot(x_rolling, roll_ccv, label='CCV (Moving Mean)', linewidth=2)

            # Plot moving mean for LBCV if enabled
            if self.config.get("plot_LBCV", True):
                roll_lbcv = df_sorted[col_LBCV].rolling(window).mean()
                axs[i].plot(x_rolling, roll_lbcv, label='LBCV (Moving Mean)', linewidth=2)

            # Plot HCV if enabled
            if self.config.get("plot_HCV", False):
                col_HCV = f'pose_error_HCV_{err_type}'
                sns.scatterplot(x=ablation_variable, y=col_HCV, data=self.df_data, ax=axs[i], label='HCV', alpha=0.5)
                roll_hcv = df_sorted[col_HCV].rolling(window).mean()
                axs[i].plot(x_rolling, roll_hcv, label='HCV (Moving Mean)', linewidth=2)

            # Labels and title
            err_type_name = {'x': 'X', 'y': 'Y', 'z': 'Z', 'a': 'Pitch', 'b': 'Yaw', 'c': 'Roll'}[err_type]
            axs[i].set_xlabel(ablation_variable_pretty, fontsize=14)
            ylabel = f'{err_type_name.title()} Error (m)' if err_type in ['x', 'y', 'z'] else f'{err_type_name.title()} Error (deg)'
            axs[i].set_ylabel(ylabel, fontsize=14)
            axs[i].set_title(f'{err_type_name.title()} Error vs {ablation_variable_pretty}', fontsize=16)
            axs[i].legend(fontsize=12)

            axs[i].axhline(0, color='black', linestyle='--', linewidth=1)
            axs[i].grid(True)

            # Use global ranges for y-axis
            axs[i].set_ylim(-self.global_ranges[err_type], self.global_ranges[err_type])

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{ablation_variable}_error_scatter.png"))
        plt.close()

    def err_mean_std_plot(self, ablation_variable):
        ablation_variable_pretty = ablation_variable.replace("_", " ").title()
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.flatten()
        error_types = ['x', 'y', 'z', 'a', 'b', 'c']

        for i, err_type in enumerate(error_types):
            ax = axs[i]
            err_type_name = {'x': 'X', 'y': 'Y', 'z': 'Z', 'a': 'Pitch', 'b': 'Yaw', 'c': 'Roll'}[err_type]

            col_CCV = f'pose_error_CCV_{err_type}'
            col_LBCV = f'pose_error_LBCV_{err_type}'

            # Group by ablation_variable
            grouped_CCV = self.df_data.groupby(ablation_variable, observed=False)[col_CCV].agg(['mean', 'std']).reset_index()
            ax.plot(grouped_CCV[ablation_variable], grouped_CCV['mean'], label='CCV Mean', linewidth=2)
            ax.fill_between(grouped_CCV[ablation_variable],
                            grouped_CCV['mean'] - grouped_CCV['std'],
                            grouped_CCV['mean'] + grouped_CCV['std'],
                            alpha=0.3, label='CCV ±1 Std')

            if self.config.get("plot_LBCV", True):
                grouped_LBCV = self.df_data.groupby(ablation_variable, observed=False)[col_LBCV].agg(['mean', 'std']).reset_index()
                ax.plot(grouped_LBCV[ablation_variable], grouped_LBCV['mean'], label='LBCV Mean', linewidth=2)
                ax.fill_between(grouped_LBCV[ablation_variable],
                                grouped_LBCV['mean'] - grouped_LBCV['std'],
                                grouped_LBCV['mean'] + grouped_LBCV['std'],
                                alpha=0.3, label='LBCV ±1 Std')

            if self.config.get("plot_HCV", False):
                col_HCV = f'pose_error_HCV_{err_type}'
                grouped_HCV = self.df_data.groupby(ablation_variable, observed=False)[col_HCV].agg(['mean', 'std']).reset_index()
                ax.plot(grouped_HCV[ablation_variable], grouped_HCV['mean'], label='HCV Mean', linewidth=2)
                ax.fill_between(grouped_HCV[ablation_variable],
                                grouped_HCV['mean'] - grouped_HCV['std'],
                                grouped_HCV['mean'] + grouped_HCV['std'],
                                alpha=0.3, label='HCV ±1 Std')

            ax.set_xlabel(ablation_variable_pretty, fontsize=14)
            ylabel = f'{err_type_name.title()} Error (m)' if err_type in ['x', 'y', 'z'] else f'{err_type_name.title()} Error (deg)'
            ax.set_ylabel(ylabel, fontsize=14)
            ax.set_title(f'{err_type_name.title()} Error vs {ablation_variable_pretty}', fontsize=16)
            ax.legend(fontsize=12)

            ax.axhline(0, color='black', linestyle='--', linewidth=1)
            ax.grid(True)

            # Dynamically set scale based on global ranges
            ax.set_ylim(-self.global_ranges[err_type], self.global_ranges[err_type])

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{ablation_variable}_error_mean_std.png"))
        plt.close()

    def error_ridge_plots(self, ablation_variable, n_ablation_bins=10, plot_CCV=True):

        ablation_variable_pretty = ablation_variable.replace("_", " ").title()
        bin_name = f"{ablation_variable}_bin"

        # Bin the ablation variable
        self.df_data[bin_name] = pd.cut(self.df_data[ablation_variable], bins=n_ablation_bins)

        # Error types
        error_types = ['x', 'y', 'z', 'a', 'b', 'c']

        # Methods (both shown if requested)
        methods = []
        if plot_CCV:
            methods.append('CCV')
        if self.config.get("plot_HCV", False):
            methods.append('HCV')
        if self.config.get("plot_LBCV", False):
            methods.append('LBCV')

        colors = {'CCV': 'blue', 'LBCV': 'orange', 'HCV': 'green'}

        for err_type in error_types:
            plt.figure(figsize=(10, 6))
            bin_labels = self.df_data[bin_name].cat.categories
            n_bins = len(bin_labels)

            # Determine global x-axis range (across all bins and methods)
            all_vals = []
            for method in methods:
                all_vals.append(self.df_data[f'pose_error_{method}_{err_type}'].dropna())
            all_errors = pd.concat(all_vals)
            max_abs = max(abs(all_errors.min()), abs(all_errors.max()))
            x_vals = np.linspace(-max_abs, max_abs, 500)

            for i, bin_label in enumerate(bin_labels):  # Normal order
                y_shift = i

                bin_data = self.df_data[self.df_data[bin_name] == bin_label]

                for method in methods:
                    col = f'pose_error_{method}_{err_type}'
                    error_vals = bin_data[col].dropna().values

                    if len(error_vals) < 1:
                        continue

                    unique_vals = np.unique(error_vals)

                    # If not enough unique values, plot vertical mean line
                    if len(unique_vals) < 2:
                        mean_val = np.mean(error_vals)
                        plt.plot([mean_val, mean_val], [y_shift, y_shift + 1], color=colors[method], linestyle='-', linewidth=1, label=method if i == 0 else "")
                        continue

                    try:
                        kde = gaussian_kde(error_vals)
                        y_vals = kde(x_vals)

                        mean_val = np.mean(error_vals)
                        plt.plot([mean_val, mean_val], [y_shift, y_shift + 1], color=colors[method], linestyle='-', linewidth=2, label=method if i == 0 else "")

                        y_vals = y_vals / y_vals.max()  # normalize
                        plt.fill_between(x_vals, y_shift, y_vals + y_shift, alpha=0.5, color=colors[method], label=method if i == 0 else "")
                        plt.plot(x_vals, y_vals + y_shift, color=colors[method], linewidth=1)
                    except np.linalg.LinAlgError:
                        mean_val = np.mean(error_vals)
                        plt.plot([mean_val, mean_val], [y_shift, y_shift + 1], color=colors[method], linestyle='-', linewidth=1, label=method if i == 0 else "")
                        continue

                # Bin label on left
                plt.text(-max_abs * 1.05, y_shift + 0.2, str(bin_label), va='center', ha='right', fontsize=8)

            # Axis formatting
            err_type_name = {'x': 'X', 'y': 'Y', 'z': 'Z', 'a': 'Pitch', 'b': 'Yaw', 'c': 'Roll'}[err_type]
            xlabel = f"{err_type_name} Error (m)" if err_type in ['x', 'y', 'z'] else f"{err_type_name} Error (deg)"
            plt.xlabel(xlabel)
            plt.yticks([])
            plt.title(f"{err_type_name} Error Distribution by {ablation_variable_pretty}")
            # plt.axvline(0, color='black', linestyle='--', linewidth=1)
            plt.xlim(-max_abs, max_abs)
            plt.legend(loc='upper right')
            plt.tight_layout()

            save_path = os.path.join(self.output_dir, f"{ablation_variable}_ridgeplot_combined_{err_type}.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()


    def output_labeled_images(self, ablation_variable): 
        # Create directory for labeled images
        images_dir = self.config["results_path"].replace("results/results.csv", "rgb/")
        dir_labeled = os.path.join(self.output_dir, "labeled_images")
        os.makedirs(dir_labeled, exist_ok=True)

        # Iterate through the DataFrame and save labeled images
        for idx, row in self.df_data.iterrows():
            image_path = row['image_path']
            ablation_value = row[ablation_variable]
            plt.figure(figsize=(10, 6))
            img = plt.imread(image_path)
            plt.imshow(img)
            plt.title(f'{ablation_variable.replace("_", " ").title()}: {float(ablation_value):.4f}')
            plt.axis('off')
            output_path = os.path.join(dir_labeled, f"labeled_{idx}.png")
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()

    def find_worst_performing(self, ablation_variable):
        # find indices where LBCV detection is false 
        indices_undetected = self.df_data[self.df_data['detected_LBCV'] == 0].index
        images_dir = self.config["results_path"].replace("results/results.csv", f"rgb/") 
        dir_undetected = os.path.join(self.output_dir, "LBCV_undetected") 
        dir_lowest_iou = os.path.join(self.output_dir, "LBCV_lowest_IOU") 
        dir_highest_pose_err = os.path.join(self.output_dir, "LBCV_highest_pose_error")
        os.makedirs(dir_undetected, exist_ok=True)
        os.makedirs(dir_lowest_iou, exist_ok=True)
        os.makedirs(dir_highest_pose_err, exist_ok=True)
        for idx in indices_undetected: 
            image_path = self.df_data.loc[idx, 'image_path']  
            # add text of the ablation variable and its value to the image 
            ablation_value = self.df_data.loc[idx, ablation_variable]
            plt.figure(figsize=(10, 6))
            img = plt.imread(image_path)
            # add text showing the ablation variable and its value, and the IOU value, and the MAE translation error and MAE rotation error 
            plt.text(10, 20, f'{ablation_variable.replace("_", " ").title()}: {ablation_value}', color='white', fontsize=14, bbox=dict(facecolor='black', alpha=0.5))
            plt.text(10, 40, f'IOU: {self.df_data.loc[idx, "LBCV_IOU"]:.2f}', color='white', fontsize=14, bbox=dict(facecolor='black', alpha=0.5))
            plt.text(10, 60, f'MAE Translation: {self.df_data.loc[idx, "pose_error_LBCV_x"]:.2f}, {self.df_data.loc[idx, "pose_error_LBCV_y"]:.2f}, {self.df_data.loc[idx, "pose_error_LBCV_z"]:.2f}', color='white', fontsize=14, bbox=dict(facecolor='black', alpha=0.5))
            plt.text(10, 80, f'MAE Rotation: {self.df_data.loc[idx, "pose_error_LBCV_a"]:.2f}, {self.df_data.loc[idx, "pose_error_LBCV_b"]:.2f}, {self.df_data.loc[idx, "pose_error_LBCV_c"]:.2f}', color='white', fontsize=14, bbox=dict(facecolor='black', alpha=0.5))
            # save the image with the text
            plt.title(f'LBCV Undetected - {ablation_variable.replace("_", " ").title()}: {float(ablation_value):.4f}', fontsize=16)
            plt.tight_layout()
            plt.imshow(img)
            # Hide axes 
            plt.axis('off')
            output_path = os.path.join(dir_undetected, f"LBCV_undetected_{idx}.png")
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()

        # find indices of lowest IOU values
        indices_lowest_iou = self.df_data.nsmallest(10, 'LBCV_IOU').index
        for idx in indices_lowest_iou: 
            image_path = self.df_data.loc[idx, 'image_path']  
            # add text of the ablation variable and its value to the image 
            ablation_value = self.df_data.loc[idx, ablation_variable]
            plt.figure(figsize=(10, 6))
            img = plt.imread(image_path)
            plt.imshow(img)
            plt.title(f'{ablation_variable.replace("_", " ").title()}: {float(ablation_value):.4f}')
            plt.axis('off')
            output_path = os.path.join(dir_lowest_iou, f"LBCV_lowest_IOU_{idx}.png")
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()

        # find indices of highest pose error values
        normalized_pose_error = self.df_data[['pose_error_LBCV_x', 'pose_error_LBCV_y', 'pose_error_LBCV_z',
                                              'pose_error_LBCV_a', 'pose_error_LBCV_b', 'pose_error_LBCV_c']].abs().max(axis=1)
        indices_highest_pose_error = normalized_pose_error.nlargest(10).index
        for idx in indices_highest_pose_error:
            image_path = self.df_data.loc[idx, 'image_path']  
            # add text of the ablation variable and its value to the image 
            ablation_value = self.df_data.loc[idx, ablation_variable]
            plt.figure(figsize=(10, 6))
            img = plt.imread(image_path)
            plt.imshow(img)
            plt.text(10, 20, f'{ablation_variable.replace("_", " ").title()}: {ablation_value}', color='white', fontsize=14, bbox=dict(facecolor='black', alpha=0.5))
            plt.text(10, 40, f'MAE Translation: {self.df_data.loc[idx, "pose_error_LBCV_x"]:.2f}, {self.df_data.loc[idx, "pose_error_LBCV_y"]:.2f}, {self.df_data.loc[idx, "pose_error_LBCV_z"]:.2f}', color='white', fontsize=14, bbox=dict(facecolor='black', alpha=0.5))
            plt.text(10, 60, f'MAE Rotation: {self.df_data.loc[idx, "pose_error_LBCV_a"]:.2f}, {self.df_data.loc[idx, "pose_error_LBCV_b"]:.2f}, {self.df_data.loc[idx, "pose_error_LBCV_c"]:.2f}', color='white', fontsize=14, bbox=dict(facecolor='black', alpha=0.5))
            # save the image with the text
            plt.title(f'LBCV Highest Pose Error - {ablation_variable.replace("_", " ").title()}: {float(ablation_value):.4f}', fontsize=16)
            plt.tight_layout()
            plt.imshow(img)
            # Hide axes 
            plt.axis('off')
            output_path = os.path.join(dir_highest_pose_err, f"LBCV_highest_pose_error_{idx}.png")
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()

    def find_best_performing(self, ablation_variable):
        # find indices where LBCV detection is true 
        indices_detected = self.df_data[self.df_data['detected_LBCV'] == 1].index
        images_dir = self.config["results_path"].replace("results/results.csv", f"rgb/") 
        dir_detected = os.path.join(self.output_dir, "LBCV_detected") 
        os.makedirs(dir_detected, exist_ok=True)
        for idx in indices_detected: 
            image_path = self.df_data.loc[idx, 'image_path']  
            # add text of the ablation variable and its value to the image 
            ablation_value = self.df_data.loc[idx, ablation_variable]
            plt.figure(figsize=(10, 6))
            img = plt.imread(image_path)
            plt.imshow(img)
            plt.text(10, 20, f'{ablation_variable.replace("_", " ").title()}: {ablation_value}', color='white', fontsize=14, bbox=dict(facecolor='black', alpha=0.5))
            plt.text(10, 40, f'IOU: {self.df_data.loc[idx, "LBCV_IOU"]:.2f}', color='white', fontsize=14, bbox=dict(facecolor='black', alpha=0.5))
            plt.text(10, 60, f'MAE Translation: {self.df_data.loc[idx, "pose_error_LBCV_x"]:.2f}, {self.df_data.loc[idx, "pose_error_LBCV_y"]:.2f}, {self.df_data.loc[idx, "pose_error_LBCV_z"]:.2f}', color='white', fontsize=14, bbox=dict(facecolor='black', alpha=0.5))
            plt.text(10, 80, f'MAE Rotation: {self.df_data.loc[idx, "pose_error_LBCV_a"]:.2f}, {self.df_data.loc[idx, "pose_error_LBCV_b"]:.2f}, {self.df_data.loc[idx, "pose_error_LBCV_c"]:.2f}', color='white', fontsize=14, bbox=dict(facecolor='black', alpha=0.5))
            # save the image with the text
            plt.title(f'LBCV Detected - {ablation_variable.replace("_", " ").title()}: {float(ablation_value):.4f}', fontsize=16)
            plt.tight_layout()
            plt.imshow(img)
            # Hide axes 
            plt.axis('off')
            output_path = os.path.join(dir_detected, f"LBCV_detected_{idx}.png")
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()     

if __name__ == "__main__":
    # ablations = ["truncation_blank_background","distance_blank_background","skew_blank_background","underexposure_blank_background"]
    ablations = ["distance_blank_background","skew_blank_background","underexposure_blank_background"]
    data_yaml_path = "./ablations/data/data_description.yaml"
    with open(data_yaml_path, 'r') as f:
        data_description = yaml.safe_load(f)

    # Load all dataframes to calculate global ranges
    dataframes = []
    for ablation in ablations:
        data_path = data_description[ablation]["data_path"]
        results_path = os.path.join(data_path, "results/results.csv")
        dataframes.append(pd.read_csv(results_path))

    global_ranges = Plotter.calculate_global_max_ranges(dataframes) # FIXME: doesn't appear to correctly compute or apply global ranges 

    for ablation in ablations:
        data_path = data_description[ablation]["data_path"]
        ablation_variable = data_description[ablation]["ablation_variable"]

        config = {
            "results_path": os.path.join(data_path, "results/results.csv"),
            "output_path": os.path.join(data_path, "results/plots"),
            "plot_HCV": False,
            "plot_LBCV": True,
        }

        plotter_instance = Plotter(config, global_ranges)
        plotter_instance.detection_plot(ablation_variable, n_bins=10)
        plotter_instance.IOU_plot(ablation_variable, n_bins=10) 
        plotter_instance.error_plot(ablation_variable) 
        plotter_instance.err_mean_std_plot(ablation_variable) 
        plotter_instance.error_ridge_plots(ablation_variable, plot_CCV=True)
        plotter_instance.find_worst_performing(ablation_variable) 
        # plotter_instance.find_best_performing(ablation_variable) 
        # plotter_instance.output_labeled_images(ablation_variable)

