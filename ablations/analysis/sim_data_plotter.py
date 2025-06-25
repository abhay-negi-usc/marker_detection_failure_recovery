import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os 
import json 
import yaml 

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

        for i, err_type in enumerate(error_types):
            col_CCV = f'pose_error_CCV_{err_type}'
            col_LBCV = f'pose_error_LBCV_{err_type}'

            # Plot raw scatter
            sns.scatterplot(x=ablation_variable, y=col_CCV, data=self.df_data, ax=axs[i], label='CCV', alpha=0.5)
            sns.scatterplot(x=ablation_variable, y=col_LBCV, data=self.df_data, ax=axs[i], label='LBCV', alpha=0.5)

            # Compute moving mean
            df_sorted = self.df_data.sort_values(by=ablation_variable)
            window = max(5, len(df_sorted) // 50)
            x_rolling = df_sorted[ablation_variable].rolling(window).mean()
            roll_ccv = df_sorted[col_CCV].rolling(window).mean()
            roll_lbcv = df_sorted[col_LBCV].rolling(window).mean()

            # Plot moving means
            axs[i].plot(x_rolling, roll_ccv, label='CCV (Moving Mean)', linewidth=2)
            axs[i].plot(x_rolling, roll_lbcv, label='LBCV (Moving Mean)', linewidth=2)

            # Labels and title
            axs[i].set_xlabel(ablation_variable_pretty)
            ylabel = f'{err_type.upper()} Error (m)' if err_type in ['x', 'y', 'z'] else f'{err_type.upper()} Error (deg)'
            axs[i].set_ylabel(ylabel)
            axs[i].set_title(f'{err_type.upper()} Error vs {ablation_variable_pretty}')
            axs[i].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{ablation_variable}_error_scatter.png"))
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
        images_dir = self.config["results_path"].replace("results/results.csv",f"rgb/") 
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
            plt.text(10, 20, f'{ablation_variable.replace("_", " ").title()}: {ablation_value}', color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))
            plt.text(10, 40, f'IOU: {self.df_data.loc[idx, "LBCV_IOU"]:.2f}', color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))
            plt.text(10, 60, f'MAE Translation: {self.df_data.loc[idx, "pose_error_LBCV_x"]:.2f}, {self.df_data.loc[idx, "pose_error_LBCV_y"]:.2f}, {self.df_data.loc[idx, "pose_error_LBCV_z"]:.2f}', color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))
            plt.text(10, 80, f'MAE Rotation: {self.df_data.loc[idx, "pose_error_LBCV_a"]:.2f}, {self.df_data.loc[idx, "pose_error_LBCV_b"]:.2f}, {self.df_data.loc[idx, "pose_error_LBCV_c"]:.2f}', color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))
            # save the image with the text
            plt.title(f'LBCV Undetected - {ablation_variable.replace("_", " ").title()}: {float(ablation_value):.4f}')
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


if __name__ == "__main__":

    # underexposure: exp_sdg_20250617-211257 
    # truncation, fixed background: exp_sdg_20250618-102429 
    # distance: exp_sdg_20250618-125218
    # skew: exp_sdg_20250618-144529 

    # get ablation data path 
    ablation = "background" 
    data_yaml_path = "./ablations/data/data_description.yaml" 
    with open(data_yaml_path, 'r') as f:
        data_description = yaml.safe_load(f) 
    data_path = data_description[ablation]["data_path"] 

    config = {
        "results_path": os.path.join(data_path, "results/results.csv"),
        "output_path": os.path.join(data_path, "results/plots"),
    } 

    ablation_variable = "background_id" # ambient_light_intensity, fraction_marker_visible, distance, skew 

    plotter_instance = Plotter(config) 
    plotter_instance.detection_plot(ablation_variable, n_bins=10)
    plotter_instance.IOU_plot(ablation_variable, n_bins=10) 
    plotter_instance.error_plot(ablation_variable) 
    plotter_instance.find_worst_performing(ablation_variable) 
    plotter_instance.output_labeled_images(ablation_variable) 
