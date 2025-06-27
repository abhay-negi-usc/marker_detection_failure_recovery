import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os 
import yaml 
import re 

class Plotter(): 
    def __init__(self, config):
        self.config = config 
        self.output_dir = config.get("output_path", "./ablations/analysis/plots")
        os.makedirs(self.output_dir, exist_ok=True)
        self.df_data = pd.read_csv(self.config["results_path"])  

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

        # Calculate shared scales for x/y and a/b plots
        max_abs_xy = max(
            self.df_data[f'pose_error_CCV_x'].abs().max(),
            self.df_data[f'pose_error_LBCV_x'].abs().max() if self.config.get("plot_LBCV", True) else 0,
            self.df_data[f'pose_error_CCV_y'].abs().max(),
            self.df_data[f'pose_error_LBCV_y'].abs().max() if self.config.get("plot_LBCV", True) else 0
        )
        max_abs_ab = max(
            self.df_data[f'pose_error_CCV_a'].abs().max(),
            self.df_data[f'pose_error_LBCV_a'].abs().max() if self.config.get("plot_LBCV", True) else 0,
            self.df_data[f'pose_error_CCV_b'].abs().max(),
            self.df_data[f'pose_error_LBCV_b'].abs().max() if self.config.get("plot_LBCV", True) else 0
        )

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
            err_type_name = err_type
            if err_type == "a": 
                err_type_name = "Pitch"
            elif err_type == "b": 
                err_type_name = "Yaw"
            elif err_type == "c":
                err_type_name = "Roll"

            axs[i].set_xlabel(ablation_variable_pretty)
            ylabel = f'{err_type_name.upper()} Error (m)' if err_type_name in ['x', 'y', 'z'] else f'{err_type_name.upper()} Error (deg)'
            axs[i].set_ylabel(ylabel)
            axs[i].set_title(f'{err_type_name.upper()} Error vs {ablation_variable_pretty}')
            axs[i].legend()

            # Center y-axis at zero and add grid
            axs[i].axhline(0, color='black', linestyle='--', linewidth=1)
            axs[i].grid(True)

            # Dynamically set scale based on shared scales for x/y and a/b
            if err_type in ['x', 'y']:
                axs[i].set_ylim(-max_abs_xy, max_abs_xy)
            elif err_type in ['a', 'b']:
                axs[i].set_ylim(-max_abs_ab, max_abs_ab)
            else:
                max_abs_value = max(
                    self.df_data[col_CCV].abs().max(),
                    self.df_data[col_LBCV].abs().max() if self.config.get("plot_LBCV", True) else 0,
                    self.df_data[col_HCV].abs().max() if self.config.get("plot_HCV", False) else 0
                )
                axs[i].set_ylim(-max_abs_value, max_abs_value)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{ablation_variable}_error_scatter.png"))
        plt.close()

    def error_ridge_plots(self, ablation_variable, n_bins=10, plot_CCV=False): 
        ablation_variable_pretty = ablation_variable.replace("_", " ").title()
        bin_name = f"{ablation_variable}_bin"
        
        # Bin the ablation variable
        self.df_data[bin_name] = pd.cut(self.df_data[ablation_variable], bins=n_bins)

        # Error dimensions to plot
        error_types = ['x', 'y', 'z', 'a', 'b', 'c']
        
        for err_type in error_types:
            plt.figure(figsize=(10, 6))
            df_plot = pd.DataFrame()

            # Add CCV data if enabled
            if plot_CCV:
                df_plot = pd.concat([
                    df_plot,
                    pd.DataFrame({
                        'Error': self.df_data[f'pose_error_CCV_{err_type}'],
                        'Method': 'CCV',
                        'AblationBin': self.df_data[bin_name].astype(str)
                    })
                ])

            # Add HCV data if enabled
            if self.config.get("plot_HCV", False):
                df_plot = pd.concat([
                    df_plot,
                    pd.DataFrame({
                        'Error': self.df_data[f'pose_error_HCV_{err_type}'],
                        'Method': 'HCV',
                        'AblationBin': self.df_data[bin_name].astype(str)
                    })
                ])

            # Remove rows with NaN values in 'Error' or 'AblationBin'
            df_plot = df_plot.dropna(subset=['Error', 'AblationBin']) 

            # Extract and sort AblationBin values by lower bound
            unique_bins = df_plot['AblationBin'].dropna().unique()
            row_order = sorted(unique_bins, key=lambda s: float(s.strip('()[]').split(',')[0]), reverse=True)

            # Create FacetGrid ridge plot
            g = sns.FacetGrid(
                df_plot, 
                row='AblationBin', 
                hue='Method', 
                aspect=8, 
                height=0.8, 
                palette='muted',
                row_order=row_order,
            )

            g.map(sns.kdeplot, 'Error', bw_adjust=0.7, fill=True, alpha=0.6, linewidth=1.5)
            g.map(sns.kdeplot, 'Error', bw_adjust=0.7, color='k', linewidth=0.5)
            g.map(plt.axhline, y=0, lw=1, clip_on=False)

            # Add labels for ablation variable values
            # Use the exact row_order (in correct order) for axis labeling
            for ax, bin_label in zip(g.axes.flat, row_order):
                ax.text(0, 0.02, f"{bin_label}", transform=ax.transAxes, ha="center", va="bottom", fontsize=10)

            # Format
            g.set_titles("")
            g.set(yticks=[], ylabel="")
            g.despine(bottom=True, left=True)

            # Common x-axis label
            err_type_name = {'x': 'X', 'y': 'Y', 'z': 'Z', 'a': 'Pitch', 'b': 'Yaw', 'c': 'Roll'}[err_type]
            xlabel = f"{err_type_name} Error (m)" if err_type in ['x', 'y', 'z'] else f"{err_type_name} Error (deg)"
            g.set_xlabels(xlabel)

            # Center all ridge plots horizontally at 0
            g.set(xlim=(-max(abs(df_plot['Error'].min()), abs(df_plot['Error'].max())), 
                        max(abs(df_plot['Error'].min()), abs(df_plot['Error'].max()))))

            plt.subplots_adjust(hspace=-0.8)
            plt.suptitle(f"{err_type_name} Error Distribution by {ablation_variable_pretty}", y=1.02)
            plt.tight_layout()

            save_path = os.path.join(self.output_dir, f"{ablation_variable}_ridgeplot_{err_type}.png")
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
        os.makedirs(dir_undetected, exist_ok=True)
        os.makedirs(dir_lowest_iou, exist_ok=True)
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

    # underexposure: exp_sdg_20250617-211257 
    # truncation, fixed background: exp_sdg_20250618-102429 
    # distance: exp_sdg_20250618-125218
    # skew: exp_sdg_20250618-144529 

    # get ablation data path 
    ablation = "skew_background" # distance_background_v3, skew_background, truncation_background, underexposure_background, 
    data_yaml_path = "./ablations/data/data_description.yaml" 
    with open(data_yaml_path, 'r') as f:
        data_description = yaml.safe_load(f) 
    data_path = data_description[ablation]["data_path"] 
    ablation_variable = data_description[ablation]["ablation_variable"] 

    config = {
        "results_path": os.path.join(data_path, "results/results.csv"),
        "output_path": os.path.join(data_path, "results/plots"),
        "plot_HCV": True, 
        "plot_LBCV": False,
    } 

    plotter_instance = Plotter(config) 
    plotter_instance.detection_plot(ablation_variable, n_bins=10)
    plotter_instance.IOU_plot(ablation_variable, n_bins=10) 
    plotter_instance.error_plot(ablation_variable) 
    plotter_instance.error_ridge_plots(ablation_variable, plot_CCV=False)
    # plotter_instance.find_worst_performing(ablation_variable) 
    # plotter_instance.find_best_performing(ablation_variable) 
    # plotter_instance.output_labeled_images(ablation_variable)

