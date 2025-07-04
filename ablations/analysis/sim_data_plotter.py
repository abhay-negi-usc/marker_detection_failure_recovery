import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os 
import yaml 
from scipy.stats import gaussian_kde

# Define a mapping for renaming ablation variables
ABALATION_VARIABLE_PRETTY_NAMES = {
    "distance_to_camera": "Distance to Camera (m)",
    "skew": "Pitch Angle to Camera (deg)",
    "truncation": "Fraction of Marker Pixels Visible",
    "underexposure": "Ambient Light Intensity",
    "glare": "Glare Cone Angle (deg)",
}

class Plotter():
    def __init__(self, config, global_ranges):
        self.config = config
        self.output_dir = config.get("output_path", "./ablations/analysis/plots")
        os.makedirs(self.output_dir, exist_ok=True)
        self.df_data = pd.read_csv(self.config["results_path"])
        # self.global_ranges = global_ranges
        self.global_ranges = {
            "x": 0.01,
            "y": 0.01,
            "z": 0.02,
            "a": 15.0,  # Pitch error in degrees
            "b": 15.0,  # Yaw error in degrees
            "c": 30.0   # Roll error in degrees
        }

        # Determine which methods to plot based on ablation name
        ablation_name = self.config.get("ablation_name", "").lower()
        if "truncation" in ablation_name:
            self.config["plot_CCV"] = False
            self.config["plot_LBCV"] = True
            self.config["plot_HCV"] = False 
        else:
            self.config["plot_CCV"] = True
            self.config["plot_LBCV"] = False
            self.config["plot_HCV"] = True

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

    def detection_plot(self, ablation_variable, coplot_blank_background=False, save_central=False):    
        ablation_variable_pretty = ABALATION_VARIABLE_PRETTY_NAMES.get(ablation_variable, ablation_variable.replace("_", " ").title())

        # Define aggregation dictionary and filter out None values
        aggregation_dict = {
            'detected_CCV': 'mean' if self.config.get("plot_CCV", False) else None,
            'detected_LBCV': 'mean' if self.config.get("plot_LBCV", False) else None,
            'detected_HCV': 'mean' if self.config.get("plot_HCV", False) else None
        }
        aggregation_dict = {key: value for key, value in aggregation_dict.items() if value is not None}

        # Group data by ablation variable and calculate detection rates
        grouped_data = self.df_data.groupby(ablation_variable).agg(aggregation_dict).reset_index()

        # --- Scatter plot (detection rates) + Line Connecting Points ---
        plt.figure(figsize=(10, 6))
        if self.config.get("plot_CCV", False):
            plt.plot(grouped_data[ablation_variable], grouped_data['detected_CCV'], label='CCV', linewidth=2)
            plt.scatter(grouped_data[ablation_variable], grouped_data['detected_CCV'], label="_nolegend_")
        if self.config.get("plot_LBCV", False):
            plt.plot(grouped_data[ablation_variable], grouped_data['detected_LBCV'], label='LBCV', linewidth=2)
            plt.scatter(grouped_data[ablation_variable], grouped_data['detected_LBCV'], label="_nolegend_")
        if self.config.get("plot_HCV", False):
            plt.plot(grouped_data[ablation_variable], grouped_data['detected_HCV'], label='HCV', linewidth=2)
            plt.scatter(grouped_data[ablation_variable], grouped_data['detected_HCV'], label="_nolegend_")

        # If coplot_blank_background is enabled, plot blank background data
        if coplot_blank_background:
            self.df_data_blank = pd.read_csv(self.config["results_path"].replace("multi_background", "blank_background"))
            grouped_data_blank = self.df_data_blank.groupby(ablation_variable).agg(aggregation_dict).reset_index()

            if self.config.get("plot_CCV", False):
                plt.plot(grouped_data_blank[ablation_variable], grouped_data_blank['detected_CCV'], label='CCV Blank', linestyle='--', linewidth=2)
                plt.scatter(grouped_data_blank[ablation_variable], grouped_data_blank['detected_CCV'], alpha=0.5, label="_nolegend_")
            if self.config.get("plot_LBCV", False):
                plt.plot(grouped_data_blank[ablation_variable], grouped_data_blank['detected_LBCV'], label='LBCV Blank', linestyle='--', linewidth=2)
                plt.scatter(grouped_data_blank[ablation_variable], grouped_data_blank['detected_LBCV'], alpha=0.5, label="_nolegend_")
            if self.config.get("plot_HCV", False):
                plt.plot(grouped_data_blank[ablation_variable], grouped_data_blank['detected_HCV'], label='HCV Blank', linestyle='--', linewidth=2)
                plt.scatter(grouped_data_blank[ablation_variable], grouped_data_blank['detected_HCV'], alpha=0.5, label="_nolegend_")

        plt.xlabel(ablation_variable_pretty, fontsize=16)
        plt.ylabel('Detection Rate', fontsize=16)
        plt.title(f'Detection Rate vs {ablation_variable_pretty}', fontsize=18)
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.tight_layout()

        # Save to central output path if save_central is True
        if save_central:
            central_output_path = self.config.get("central_output_path", self.output_dir)
            os.makedirs(central_output_path, exist_ok=True)
            save_path = os.path.join(central_output_path, f"{ablation_variable}_detection_rate_scatter.png")
        else:
            save_path = os.path.join(self.output_dir, f"{ablation_variable}_detection_rate_scatter.png")
        plt.savefig(save_path)
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

            # Remove scaling to mm for translation errors
            # (No multiplication by 1000 here)

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

            # Labels and title
            err_type_name = {'x': 'X', 'y': 'Y', 'z': 'Z', 'a': 'Pitch', 'b': 'Yaw', 'c': 'Roll'}[err_type]
            if err_type in ['x', 'y', 'z']:
                ylabel = f'{err_type_name.title()} Error (m)'
            else:
                ylabel = f'{err_type_name.title()} Error (deg)'
            axs[i].set_xlabel(ablation_variable_pretty, fontsize=14)
            axs[i].set_ylabel(ylabel, fontsize=14)
            axs[i].set_title(f'{err_type_name.title()} Error vs {ablation_variable_pretty}', fontsize=16)
            axs[i].legend(fontsize=12)

            axs[i].axhline(0, color='black', linestyle='--', linewidth=1)
            axs[i].grid(True)

            # Use global ranges for y-axis
            if err_type in ['x', 'y', 'z']:
                axs[i].set_ylim(-self.global_ranges[err_type], self.global_ranges[err_type])
            else:
                axs[i].set_ylim(-self.global_ranges[err_type], self.global_ranges[err_type])

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{ablation_variable}_error_scatter.png"))
        plt.close()

    def err_mean_std_plot(self, ablation_variable, coplot_blank_background=False, save_central=False):
        ablation_variable_pretty = ABALATION_VARIABLE_PRETTY_NAMES.get(ablation_variable, ablation_variable.replace("_", " ").title())
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.flatten()
        error_types = ['x', 'y', 'z', 'a', 'b', 'c']

        for i, err_type in enumerate(error_types):
            ax = axs[i]
            err_type_name = {'x': 'X', 'y': 'Y', 'z': 'Z', 'a': 'Pitch', 'b': 'Yaw', 'c': 'Roll'}[err_type]

            col_CCV = f'pose_error_CCV_{err_type}'
            col_HCV = f'pose_error_HCV_{err_type}'
            col_LBCV = f'pose_error_LBCV_{err_type}'

            # Scale translation errors to mm
            if err_type in ['x', 'y', 'z']:
                self.df_data[col_CCV] *= 1000
                self.df_data[col_HCV] *= 1000
                self.df_data[col_LBCV] *= 1000

            # Group by ablation_variable
            grouped_CCV = self.df_data.groupby(ablation_variable, observed=False)[col_CCV].agg(['mean', 'std']).reset_index()
            ax.plot(grouped_CCV[ablation_variable], grouped_CCV['mean'], label='CCV Mean', linewidth=2)
            ax.fill_between(grouped_CCV[ablation_variable],
                            grouped_CCV['mean'] - grouped_CCV['std'],
                            grouped_CCV['mean'] + grouped_CCV['std'],
                            alpha=0.3, label='CCV ±1 Std')
            
            if self.config.get("plot_HCV", False):
                grouped_HCV = self.df_data.groupby(ablation_variable, observed=False)[col_HCV].agg(['mean', 'std']).reset_index()
                ax.plot(grouped_HCV[ablation_variable], grouped_HCV['mean'], label='HCV Mean', linewidth=2)
                ax.fill_between(grouped_HCV[ablation_variable],
                                grouped_HCV['mean'] - grouped_HCV['std'],
                                grouped_HCV['mean'] + grouped_HCV['std'],
                                alpha=0.3, label='HCV ±1 Std')
                
            if self.config.get("plot_LBCV", False): 
                grouped_LBCV = self.df_data.groupby(ablation_variable, observed=False)[col_LBCV].agg(['mean', 'std']).reset_index()
                ax.plot(grouped_LBCV[ablation_variable], grouped_LBCV['mean'], label='LBCV Mean', linewidth=2)
                ax.fill_between(grouped_LBCV[ablation_variable],
                                grouped_LBCV['mean'] - grouped_LBCV['std'],
                                grouped_LBCV['mean'] + grouped_LBCV['std'],
                                alpha=0.3, label='LBCV ±1 Std')
                

            # Labels and title
            if err_type in ['x', 'y', 'z']:
                ylabel = f'{err_type_name.title()} Error (mm)'
            else:
                ylabel = f'{err_type_name.title()} Error (deg)'
            ax.set_xlabel(ablation_variable_pretty, fontsize=18)
            ax.set_ylabel(ylabel, fontsize=18)

            # Dynamically set scale based on global ranges or mean ± 1 std
            if err_type in ['x', 'y', 'z']:
                if ablation_variable == "fraction_marker_visible": 
                    if err_type == 'x' or err_type == 'y': 
                        ax.set_ylim(-0.030 * 1000, 0.030 * 1000)  # Set range to ±0.5 m converted to mm
                    elif err_type == 'z':
                        ax.set_ylim(-0.05 * 1000, 0.05 * 1000)  # Set range to ±0.5 m converted to mm
                elif (ablation_variable == "distance_to_camera") and err_type == 'z':
                    ax.set_ylim(-0.05 * 1000, 0.05 * 1000)  # Set range to ±0.5 m converted to mm
                else:
                    ax.set_ylim(-self.global_ranges[err_type] * 1000, self.global_ranges[err_type] * 1000)
            else:
                ax.set_ylim(-self.global_ranges[err_type], self.global_ranges[err_type])
                if (ablation_variable == "fraction_marker_visible") and (err_type == 'a' or err_type == 'b'):
                    ax.set_ylim(-30, 30)

            ax.tick_params(axis='both', labelsize=14)
            if i == 0:
                ax.legend(fontsize=14)
            ax.axhline(0, color='black', linestyle='--', linewidth=1)
            ax.grid(True)

        fig.suptitle(f'Pose Estimation Error vs {ablation_variable_pretty}', fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(self.output_dir, f"{ablation_variable}_error_mean_std.png")
        plt.savefig(save_path)
        if save_central:
            central_output_path = self.config.get("central_output_path", self.output_dir)
            os.makedirs(central_output_path, exist_ok=True)
            save_path = os.path.join(central_output_path, f"{ablation_variable}_error_mean_std.png")
            plt.savefig(save_path)
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

    def print_detection_rates(self):
        """
        Compute and print the detection rate for each method (CCV, LBCV, HCV).
        """
        methods = []
        if self.config.get("plot_CCV", False):
            methods.append("CCV")
        if self.config.get("plot_LBCV", False):
            methods.append("LBCV")
        if self.config.get("plot_HCV", False):
            methods.append("HCV")

        print("Detection Rates:")
        for method in methods:
            col_name = f'detected_{method}'
            if col_name in self.df_data.columns:
                detection_rate = self.df_data[col_name].mean()
                print(f"{method}: {detection_rate:.2%}")

    def save_summary_table(self, ablation_variable):
        """
        Save a summary table to CSV with the following columns:
        method (CCV, LBCV with CCV success, LBCV with CCV fail),
        detection rate, X MAE +/- std dev (mm), Y MAE +/- std dev (mm),
        Z MAE +/- std dev (mm), Pitch MAE +/- std dev (deg),
        Yaw MAE +/- std dev (deg), Roll MAE +/- std dev (deg).
        """
        methods = ["CCV", "LBCV with CCV success", "LBCV with CCV fail"]
        summary_data = []

        for method in methods:
            if method == "CCV":
                mask = self.df_data['detected_CCV'] == 1
            elif method == "LBCV with CCV success":
                mask = (self.df_data['detected_CCV'] == 1) & (self.df_data['detected_LBCV'] == 1)
            elif method == "LBCV with CCV fail":
                mask = (self.df_data['detected_CCV'] == 0) & (self.df_data['detected_LBCV'] == 1)

            filtered_data = self.df_data[mask]
            detection_rate = mask.mean()

            # Calculate MAE and standard deviation for each error type
            error_stats = {}
            if method == "CCV":
                method_type = "CCV"
            else:
                method_type = "LBCV"
            for err_type in ['x', 'y', 'z', 'a', 'b', 'c']:
                col_name = f'pose_error_{method_type}_{err_type}'
                mae = filtered_data[col_name].abs().mean()
                std_dev = filtered_data[col_name].abs().std()
                unit = "mm" if err_type in ['x', 'y', 'z'] else "deg"
                error_stats[err_type] = f"{mae * 1000:.2f} ± {std_dev * 1000:.2f}" if unit == "mm" else f"{mae:.2f} ± {std_dev:.2f}"

            summary_data.append([
                method,
                f"{detection_rate:.2%}",
                error_stats['x'],
                error_stats['y'],
                error_stats['z'],
                error_stats['a'],
                error_stats['b'],
                error_stats['c']
            ])

        # Create a DataFrame for the summary table
        summary_df = pd.DataFrame(summary_data, columns=[
            "Method",
            "Detection Rate",
            "X MAE ± Std Dev (mm)",
            "Y MAE ± Std Dev (mm)",
            "Z MAE ± Std Dev (mm)",
            "Pitch MAE ± Std Dev (deg)",
            "Yaw MAE ± Std Dev (deg)",
            "Roll MAE ± Std Dev (deg)"
        ])

        # Save the summary table to a CSV file
        output_path = os.path.join(self.output_dir, f"{ablation_variable}_summary_table.csv")
        summary_df.to_csv(output_path, index=False)
        print(f"Summary table saved to {output_path}")

if __name__ == "__main__":
    # ablations = ["distance_multi_background", "skew_multi_background", "truncation_multi_background", "underexposure_multi_background", "glare_corner_multi_background"] 
    # ablations = ["distance_blank_background", "skew_blank_background", "truncation_blank_background", "underexposure_blank_background", "glare_corner_blank_background"] 
    ablations = ["truncation_multi_background_v2"]
    data_yaml_path = "./ablations/data_description.yaml"
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
            "plot_HCV": True,
            "plot_LBCV": True,
            "central_output_path": "./ablations/analysis/plots",
            "ablation_name": ablation
        }

        plotter_instance = Plotter(config, global_ranges)
        # plotter_instance.save_summary_table(ablation_variable)
        plotter_instance.detection_plot(ablation_variable, coplot_blank_background=False, save_central=True)
        # plotter_instance.IOU_plot(ablation_variable)
        # plotter_instance.error_plot(ablation_variable)
        # plotter_instance.error_ridge_plots(ablation_variable, n_ablation_bins=10, plot_CCV=True)
        plotter_instance.err_mean_std_plot(ablation_variable, coplot_blank_background=False, save_central=True)
        # plotter_instance.output_labeled_images(ablation_variable)
        # plotter_instance.find_worst_performing(ablation_variable)
        # plotter_instance.find_best_performing(ablation_variable)

