import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os 
import yaml 
from scipy.stats import gaussian_kde

# Define a mapping for renaming ablation variables
ABLATION_VARIABLE_PRETTY_NAMES = {
    "distance_to_camera": "Distance to Camera (m)",
    "distance": "Distance to Camera (m)",
    "skew": "Pitch Angle to Camera (deg)",
    "truncation": "Fraction of Marker Pixels Visible",
    "underexposure": "Ambient Light Intensity",
    "glare": "Glare Cone Angle (deg)",
    "fraction_saturated_low_pixels": "Fraction of Pixels Saturated Low",
    "fraction_saturated_high_pixels": "Fraction of Pixels Saturated High", 
}
ABLATION_VARIABLE_TITLE_NAMES = {
    "distance_to_camera": "Distance",
    "distance": "Distance",
    "skew": "Pitch Angle",
    "truncation": "Fraction Visible",
    "underexposure": "Underexposure",
    "mean_marker_pixel_brightness": "Underexposure",
    # "glare": "Glare Cone Angle",
    "fraction_saturated_low_pixels": "Shadowing",
    "fraction_saturated_high_pixels": "Glare", 
}

bar_colors = {
    # 'CCV': '#FF8888',
    # 'HCV': '#FF595E',
    # 'LBCV': '#3594cc',
    # 'PBCV': '#05af6b'
    'CCV': "#58c2ff",
    'HCV': '#FF595E',
    'LBCV': "#f59b14",
    'PBCV': '#f59b14'
}

class Plotter():
    def __init__(self, config, global_ranges):
        self.config = config
        self.output_dir = config.get("output_path", "./ablations/analysis/plots")
        os.makedirs(self.output_dir, exist_ok=True)
        self.df_data = pd.read_csv(self.config["results_path"])
        # self.global_ranges = global_ranges
        self.global_ranges = {
            "x": 5,
            "y": 5,
            "z": 10,
            "a": 5.0,  # Pitch error in degrees
            "b": 10.0,  # Yaw error in degrees
            "c": 10.0   # Roll error in degrees
        }

        # Determine which methods to plot based on ablation name
        ablation_name = self.config.get("ablation_name", "").lower()
        if "truncation" in ablation_name:
            self.config["plot_CCV"] = False
            self.config["plot_LBCV"] = False #True
            self.config["plot_HCV"] = False
            self.config["plot_PBCV"] = True 
        else:
            self.config["plot_CCV"] = True
            self.config["plot_LBCV"] = False #True
            self.config["plot_HCV"] = False
            self.config["plot_PBCV"] = True

        if config["ablation_variable_min"] != "None":
            # Filter the dataframe based on the ablation variable range
            ablation_variable = config["ablation_variable"]
            min_val = config["ablation_variable_min"]
            max_val = config["ablation_variable_max"]
            self.df_data = self.df_data[(self.df_data[ablation_variable] >= min_val) & 
                                        (self.df_data[ablation_variable] <= max_val)]

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

    def detection_plot(self, ablation_variable, coplot_blank_background=False, save_central=False, n_bins=10):

        # Pretty/title names to match detection_and_IOU_plot
        ablation_variable_pretty = ABLATION_VARIABLE_PRETTY_NAMES.get(
            ablation_variable, ablation_variable.replace("_", " ").title()
        )
        ablation_variable_title = ABLATION_VARIABLE_TITLE_NAMES.get(
            ablation_variable, ablation_variable.replace("_", " ").title()
        )

        # --- Aggregations (filter Nones) ---
        aggregation_dict = {
            'detected_CCV': 'mean' if self.config.get("plot_CCV", False) else None,
            'detected_LBCV': 'mean' if self.config.get("plot_LBCV", False) else None,
            'detected_HCV': 'mean' if self.config.get("plot_HCV", False) else None,
            'detected_PBCV': 'mean' if self.config.get("plot_PBCV", False) else None,
        }
        aggregation_dict = {k: v for k, v in aggregation_dict.items() if v is not None}

        # --- Binning (match style) ---
        bin_name = f'{ablation_variable}_bin'
        bins = pd.cut(self.df_data[ablation_variable], bins=n_bins, retbins=True)
        self.df_data[bin_name] = bins[0]
        bin_edges = bins[1]

        bin_labels = [f'{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}' for i in range(len(bin_edges) - 1)]
        bin_label_mapping = {i: bin_labels[i] for i in range(len(bin_labels))}
        self.df_data[f'{bin_name}_numeric'] = self.df_data[bin_name].cat.codes

        # --- Grouped detection rates ---
        grouped_data = self.df_data.groupby(f'{bin_name}_numeric').agg(aggregation_dict).reset_index()
        grouped_data['bin_label'] = grouped_data[f'{bin_name}_numeric'].map(bin_label_mapping)

        # --- Methods & legend mapping (fix PBCV label) ---
        methods_to_plot = []
        if self.config.get("plot_CCV", False): methods_to_plot.append('detected_CCV')
        if self.config.get("plot_LBCV", False): methods_to_plot.append('detected_LBCV')
        if self.config.get("plot_HCV", False): methods_to_plot.append('detected_HCV')
        if self.config.get("plot_PBCV", False): methods_to_plot.append('detected_PBCV')

        method_mapping = {
            'detected_CCV': 'CCV',
            'detected_LBCV': 'LBCV',
            'detected_HCV': 'HCV',
            'detected_PBCV': 'PBCV',
        }

        # Melt for seaborn; lock bin order
        melted_data = grouped_data.melt(
            id_vars=['bin_label'],
            value_vars=methods_to_plot,
            var_name='Method',
            value_name='Detection_Rate'
        )
        melted_data['Method'] = melted_data['Method'].map(method_mapping)
        melted_data['bin_label'] = pd.Categorical(melted_data['bin_label'], categories=bin_labels, ordered=True)

        # --- Figure & styling to match detection_and_IOU_plot ---
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        # Use the same palette/alpha if available
        palette = globals().get('bar_colors', None)

        sns.barplot(
            x='bin_label',
            y='Detection_Rate',
            hue='Method',
            data=melted_data,
            ax=ax,
            palette=palette,
            alpha=0.95,
            order=bin_labels,
        )

        ax.set_xlabel(f'{ablation_variable_pretty}', fontsize=24)
        ax.set_ylabel('Detection Rate', fontsize=24)
        ax.set_title(f'Detection Rate vs {ablation_variable_title}', fontsize=28)
        ax.tick_params(axis='x', rotation=45, labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.set_ylim(0, 1)
        leg = ax.legend(fontsize=16)
        if leg is not None:
            leg.set_title(None)
        ax.grid(True, alpha=0.3)

        # --- Optional co-plot blank background, styled consistently ---
        if coplot_blank_background:
            # Load and bin blank background data using the same edges
            df_blank = pd.read_csv(self.config["results_path"].replace("multi_background", "blank_background"))
            df_blank[bin_name] = pd.cut(df_blank[ablation_variable], bins=bin_edges)
            df_blank[f'{bin_name}_numeric'] = df_blank[bin_name].cat.codes

            grouped_blank = df_blank.groupby(f'{bin_name}_numeric').agg(aggregation_dict).reset_index()
            grouped_blank['bin_label'] = grouped_blank[f'{bin_name}_numeric'].map(bin_label_mapping)

            blank_melt = grouped_blank.melt(
                id_vars=['bin_label'],
                value_vars=methods_to_plot,
                var_name='Method',
                value_name='Detection_Rate_Blank'
            )
            blank_melt['Method'] = blank_melt['Method'].map(method_mapping)
            blank_melt['bin_label'] = pd.Categorical(blank_melt['bin_label'], categories=bin_labels, ordered=True)

            # Overlay as a dashed line to keep visual consistency and avoid bar stacking confusion
            # (One line per method)
            for method_name, dfm in blank_melt.groupby('Method'):
                dfm_sorted = dfm.sort_values('bin_label')
                ax.plot(
                    dfm_sorted['bin_label'],
                    dfm_sorted['Detection_Rate_Blank'],
                    linestyle='--',
                    marker='o',
                    linewidth=2,
                    label=f'{method_name} (Blank)'
                )
            # Refresh legend to include overlay
            leg = ax.legend(fontsize=16)
            if leg is not None:
                leg.set_title(None)

        plt.tight_layout()
        plt.grid(True, alpha=0.3)

        # --- Save (dpi/bbox to match) ---
        save_path = os.path.join(self.output_dir, f"{ablation_variable}_detection_rate_bar.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if save_central:
            central_output_path = self.config.get("central_output_path", self.output_dir)
            os.makedirs(central_output_path, exist_ok=True)
            central_save_path = os.path.join(central_output_path, f"{ablation_variable}_detection_rate_bar.png")
            plt.savefig(central_save_path, dpi=300, bbox_inches='tight')

        plt.close(fig)

        
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

    def detection_and_IOU_plot(self, ablation_variable, n_bins=10, save_central=False, IOU_threshold=0.5):
        """
        Create a 1x2 figure:
        - Left: true positive rate (for CCV, LBCV, PBCV using IOU > IOU_threshold)
        - Right: mean IOU per method
        """
        ablation_variable_pretty = ABLATION_VARIABLE_PRETTY_NAMES.get(ablation_variable, ablation_variable.replace("_", " ").title())
        ablation_variable_title = ABLATION_VARIABLE_TITLE_NAMES.get(ablation_variable, ablation_variable.replace("_", " ").title())

        # --- Prepare bins ---
        bin_name = f'{ablation_variable}_bin'
        bins = pd.cut(self.df_data[ablation_variable], bins=n_bins, retbins=True)
        self.df_data[bin_name] = bins[0]
        bin_edges = bins[1]

        bin_labels = []
        for i in range(len(bin_edges) - 1):
            bin_labels.append(f'{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}')

        bin_label_mapping = {i: bin_labels[i] for i in range(len(bin_labels))}
        self.df_data[f'{bin_name}_numeric'] = self.df_data[bin_name].cat.codes

        # --- True positive filtering ---
        self.df_data['tp_CCV'] = self.df_data['detected_CCV'] if self.config.get("plot_CCV", False) else False
        if self.config.get("plot_LBCV", False):
            self.df_data['tp_LBCV'] = (self.df_data['detected_LBCV'] == True) & (self.df_data['LBCV_IOU'] > IOU_threshold)
        if self.config.get("plot_PBCV", False):
            self.df_data['tp_PBCV'] = (self.df_data['detected_PBCV'] == True) & (self.df_data['PBCV_IOU'] > IOU_threshold)
        aggregation_dict = {
            'tp_CCV': 'mean' if self.config.get("plot_CCV", False) else None,
            'tp_LBCV': 'mean' if self.config.get("plot_LBCV", False) else None,
            'tp_PBCV': 'mean' if self.config.get("plot_PBCV", False) else None
        }
        aggregation_dict = {k: v for k, v in aggregation_dict.items() if v is not None}



        grouped_data = self.df_data.groupby(f'{bin_name}_numeric').agg(aggregation_dict).reset_index()
        grouped_data['bin_label'] = grouped_data[f'{bin_name}_numeric'].map(bin_label_mapping)

        # Prepare methods to plot
        methods_to_plot = []
        if self.config.get("plot_CCV", False):
            methods_to_plot.append('tp_CCV')
        if self.config.get("plot_LBCV", False):
            methods_to_plot.append('tp_LBCV')
        if self.config.get("plot_PBCV", False):
            methods_to_plot.append('tp_PBCV')

        method_mapping = {
            'tp_CCV': 'CCV',
            'tp_LBCV': 'LBCV',
            'tp_PBCV': 'PBCV'
        }

        melted_data = grouped_data.melt(id_vars=['bin_label'], value_vars=methods_to_plot, 
                                 var_name='Method', value_name='True_Positive_Rate')
        melted_data['Method'] = melted_data['Method'].map(method_mapping)


        # --- IOU plot aggregation ---
        iou_agg_dict = {}
        if 'LBCV_IOU' in self.df_data.columns and self.config.get("plot_LBCV", False):
            iou_agg_dict['LBCV_IOU'] = 'mean'
        if 'PBCV_IOU' in self.df_data.columns and self.config.get("plot_PBCV", False):
            iou_agg_dict['PBCV_IOU'] = 'mean'

        grouped_iou = self.df_data.groupby(f'{bin_name}_numeric').agg(iou_agg_dict).reset_index()
        grouped_iou['bin_label'] = grouped_iou[f'{bin_name}_numeric'].map(bin_label_mapping)

        # Melt for seaborn
        iou_melted = grouped_iou.melt(id_vars='bin_label', value_name='Mean_IOU', var_name='Method')
        iou_melted['Method'] = iou_melted['Method'].map({
            'LBCV_IOU': 'LBCV',
            'PBCV_IOU': 'PBCV'
        })

        # --- Create figure ---
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(f'Detection Estimation Performance vs {ablation_variable_title}', fontsize=24)

        # --- Detection bar plot ---
        # sns.barplot(x='bin_label', y='Detection_Rate', hue='Method', data=melted_data, ax=axes[0])
        sns.barplot(
            x='bin_label',
            y='True_Positive_Rate',
            hue='Method',
            data=melted_data,
            ax=axes[0],
            palette=bar_colors,
            alpha=0.9
        )
        axes[0].set_ylabel('True Positive Rate', fontsize=24)

        axes[0].set_xlabel(f'{ablation_variable_pretty}', fontsize=24)
        axes[0].set_ylabel('Detection Rate', fontsize=24)
        # axes[0].set_title(f'Detection Rate vs {ablation_variable_title}', fontsize=28)
        axes[0].tick_params(axis='x', rotation=45, labelsize=16)
        axes[0].tick_params(axis='y', labelsize=16)
        axes[0].legend(fontsize=16)
        axes[0].grid(True, alpha=0.3)

        # --- IOU bar plot ---
        # sns.barplot(x='bin_label', y='Mean_IOU', hue='Method', data=iou_melted, ax=axes[1])
        sns.barplot(
            x='bin_label',
            y='Mean_IOU',
            hue='Method',
            data=iou_melted,
            ax=axes[1],
            palette=bar_colors,
            alpha=0.9
        )
        axes[1].set_xlabel(f'{ablation_variable_pretty}', fontsize=24)
        axes[1].set_ylabel('Mean Segmentation IOU', fontsize=24)
        # axes[1].set_title(f'Mean Segmentation IOU vs {ablation_variable_title}', fontsize=28)
        axes[1].tick_params(axis='x', rotation=45, labelsize=16)
        axes[1].tick_params(axis='y', labelsize=16)
        axes[1].set_ylim(0, 1)
        # Adjust legend on second plot
        legend_iou = axes[1].legend(fontsize=16)
        legend_iou.set_title(None)  # Remove legend title
        axes[1].grid(True, alpha=0.3)


        plt.tight_layout()

        # --- Save figure ---
        save_path = os.path.join(self.output_dir, f"{ablation_variable}_detection_and_IOU.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if save_central:
            central_output_path = self.config.get("central_output_path", self.output_dir)
            os.makedirs(central_output_path, exist_ok=True)
            save_path = os.path.join(central_output_path, f"{ablation_variable}_detection_and_IOU.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close()

        # --- Save IOU plot as separate figure ---
        fig_iou, ax_iou = plt.subplots(1, 1, figsize=(10, 6))
        # sns.barplot(x='bin_label', y='Mean_IOU', hue='Method', data=iou_melted, ax=ax_iou)
        sns.barplot(
            x='bin_label',
            y='Mean_IOU',
            hue='Method',
            data=iou_melted,
            ax=ax_iou,
            palette=bar_colors,
            alpha=0.9
        )
        ax_iou.set_xlabel(f'{ablation_variable_pretty}', fontsize=24)
        ax_iou.set_ylabel('Mean Segmentation IOU', fontsize=24)
        ax_iou.set_title(f'Mean Segmentation IOU vs {ablation_variable_title}', fontsize=28)
        ax_iou.tick_params(axis='x', rotation=45, labelsize=16)
        ax_iou.tick_params(axis='y', labelsize=16)
        ax_iou.set_ylim(0, 1)
        ax_iou.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save separate IOU plot
        iou_save_path = os.path.join(self.output_dir, f"{ablation_variable}_IOU_only.png")
        plt.savefig(iou_save_path, dpi=300, bbox_inches='tight')
        
        if save_central:
            central_output_path = self.config.get("central_output_path", self.output_dir)
            os.makedirs(central_output_path, exist_ok=True)
            iou_save_path = os.path.join(central_output_path, f"{ablation_variable}_IOU_only.png")
            plt.savefig(iou_save_path, dpi=300, bbox_inches='tight')
        
        plt.close()

    def error_plot(self, ablation_variable):
        ablation_variable_pretty = ablation_variable.replace("_", " ").title()
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.flatten()
        error_types = ['x', 'y', 'z', 'a', 'b', 'c']

        for i, err_type in enumerate(error_types):
            col_CCV = f'pose_error_CCV_{err_type}'
            col_LBCV = f'pose_error_LBCV_{err_type}'
            col_PBCV = f'pose_error_PBCV_{err_type}'


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
        ablation_variable_pretty = ABLATION_VARIABLE_PRETTY_NAMES.get(ablation_variable, ablation_variable.replace("_", " ").title())
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.flatten()
        error_types = ['x', 'y', 'z', 'a', 'b', 'c']

        for i, err_type in enumerate(error_types):
            ax = axs[i]
            err_type_name = {'x': 'X', 'y': 'Y', 'z': 'Z', 'a': 'Pitch', 'b': 'Yaw', 'c': 'Roll'}[err_type]

            col_CCV = f'pose_error_CCV_{err_type}'
            col_HCV = f'pose_error_HCV_{err_type}'
            col_LBCV = f'pose_error_LBCV_{err_type}'
            col_PBCV = f'pose_error_PBCV_{err_type}'

            # Scale translation errors to mm
            if err_type in ['x', 'y', 'z']:
                self.df_data[col_CCV] *= 1000
                self.df_data[col_HCV] *= 1000
                self.df_data[col_LBCV] *= 1000
                self.df_data[col_PBCV] *= 1000

            # Group by ablation_variable
            grouped_CCV = self.df_data.groupby(ablation_variable, observed=False)[col_CCV].agg(['mean', 'std']).reset_index()
            ax.plot(grouped_CCV[ablation_variable], grouped_CCV['mean'], label='CCV Mean', linewidth=3)
            ax.fill_between(grouped_CCV[ablation_variable],
                            grouped_CCV['mean'] - grouped_CCV['std'],
                            grouped_CCV['mean'] + grouped_CCV['std'],
                            alpha=0.3, label='CCV ±1 Std')
            
            if self.config.get("plot_HCV", False):
                grouped_HCV = self.df_data.groupby(ablation_variable, observed=False)[col_HCV].agg(['mean', 'std']).reset_index()
                ax.plot(grouped_HCV[ablation_variable], grouped_HCV['mean'], label='HCV Mean', linewidth=3)
                ax.fill_between(grouped_HCV[ablation_variable],
                                grouped_HCV['mean'] - grouped_HCV['std'],
                                grouped_HCV['mean'] + grouped_HCV['std'],
                                alpha=0.3, label='HCV ±1 Std')
                
            if self.config.get("plot_LBCV", False): 
                grouped_LBCV = self.df_data.groupby(ablation_variable, observed=False)[col_LBCV].agg(['mean', 'std']).reset_index()
                ax.plot(grouped_LBCV[ablation_variable], grouped_LBCV['mean'], label='LBCV Mean', linewidth=3)
                ax.fill_between(grouped_LBCV[ablation_variable],
                                grouped_LBCV['mean'] - grouped_LBCV['std'],
                                grouped_LBCV['mean'] + grouped_LBCV['std'],
                                alpha=0.3, label='LBCV ±1 Std')
                
            if self.config.get("plot_PBCV", False): 
                grouped_PBCV = self.df_data.groupby(ablation_variable, observed=False)[col_PBCV].agg(['mean', 'std']).reset_index()
                ax.plot(grouped_PBCV[ablation_variable], grouped_PBCV['mean'], label='PBCV Mean', linewidth=3)
                ax.fill_between(grouped_PBCV[ablation_variable],
                                grouped_PBCV['mean'] - grouped_PBCV['std'],
                                grouped_PBCV['mean'] + grouped_PBCV['std'],
                                alpha=0.3, label='PBCV ±1 Std')

            # Labels and title
            if err_type in ['x', 'y', 'z']:
                ylabel = f'{err_type_name.title()} Error (mm)'
            else:
                ylabel = f'{err_type_name.title()} Error (deg)'
            ax.set_xlabel(ablation_variable_pretty, fontsize=18)
            ax.set_ylabel(ylabel, fontsize=18)

            ax.set_ylim(-self.global_ranges[err_type], self.global_ranges[err_type])
            # # Dynamically set scale based on global ranges or mean ± 1 std
            # if err_type in ['x', 'y', 'z']:
            #     if ablation_variable == "fraction_marker_visible": 
            #         if err_type == 'x' or err_type == 'y': 
            #             ax.set_ylim(-0.030 * 1000, 0.030 * 1000)  # Set range to ±0.5 m converted to mm
            #         elif err_type == 'z':
            #             ax.set_ylim(-0.05 * 1000, 0.05 * 1000)  # Set range to ±0.5 m converted to mm
            #     elif (ablation_variable == "distance_to_camera") and err_type == 'z':
            #         ax.set_ylim(-0.05 * 1000, 0.05 * 1000)  # Set range to ±0.5 m converted to mm
            #     else:
            #         ax.set_ylim(-self.global_ranges[err_type] * 1000, self.global_ranges[err_type] * 1000)
            # else:
            #     ax.set_ylim(-self.global_ranges[err_type], self.global_ranges[err_type])
            #     if (ablation_variable == "fraction_marker_visible") and (err_type == 'a' or err_type == 'b'):
            #         ax.set_ylim(-30, 30)
            

            ax.tick_params(axis='both', labelsize=14)
            if i == 0:
                ax.legend(fontsize=14)
            ax.axhline(0, color='black', linestyle='--', linewidth=1)
            ax.grid(True)

        fig.suptitle(f'Pose Estimation Performance vs {ablation_variable_pretty}', fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plot_filename = f"{ablation_variable}_error_mean_std.png" 
        if self.config.get("plot_HCV", False):
            plot_filename = f"{ablation_variable}_error_mean_std_HCV.png"
        if self.config.get("plot_PBCV", False):
            plot_filename = f"{ablation_variable}_error_mean_std_PBCV.png"
        save_path = os.path.join(self.output_dir, plot_filename)
        plt.savefig(save_path)
        if save_central:
            central_output_path = self.config.get("central_output_path", self.output_dir)
            os.makedirs(central_output_path, exist_ok=True)
            save_path = os.path.join(central_output_path, plot_filename)
            plt.savefig(save_path)
        plt.close()

    def err_moving_mean_std_plot(self, ablation_variable, window_size=20, save_central=False, ccv_no_moving_mean=False, ccv_window_size=None, center=True, min_periods=None, win_type=None):
        """
        Plot mean error curves computed using a moving window, with std dev as shaded area.
        Args:
            ablation_variable (str): Ablation variable to plot against.
            window_size (int): Size of the moving window (number of samples) for all methods except CCV.
            save_central (bool): Save also to central output path if True.
            ccv_no_moving_mean (bool): If True, don't apply moving mean to CCV, plot raw data instead.
            ccv_window_size (int): Optional different window size for CCV. If None, uses window_size.
            center (bool): Whether to center the rolling window. Default True.
            min_periods (int): Minimum number of observations in window required to have a value. Default None.
            win_type (str): Window type for rolling calculation. Default None (simple rolling mean).
        """
        ablation_variable_pretty = ABLATION_VARIABLE_PRETTY_NAMES.get(ablation_variable, ablation_variable.replace("_", " ").title())
        ablation_variable_title = ABLATION_VARIABLE_TITLE_NAMES.get(ablation_variable, ablation_variable.replace("_", " ").title())
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Pose Estimation Error vs {ablation_variable_title}', fontsize=24, y=0.98)
        axs = axs.flatten()
        error_types = ['x', 'y', 'z', 'a', 'b', 'c']

        for i, err_type in enumerate(error_types):
            ax = axs[i]
            err_type_name = {'x': 'X', 'y': 'Y', 'z': 'Z', 'a': 'Pitch', 'b': 'Yaw', 'c': 'Roll'}[err_type]

            col_CCV = f'pose_error_CCV_{err_type}'
            col_HCV = f'pose_error_HCV_{err_type}'
            col_LBCV = f'pose_error_LBCV_{err_type}'
            col_PBCV = f'pose_error_PBCV_{err_type}'

            # Scale translations to mm
            df_sorted = self.df_data.sort_values(by=ablation_variable).copy()
            # if err_type in ['x', 'y', 'z']:
            #     df_sorted[col_CCV] *= 1000
            #     df_sorted[col_HCV] *= 1000
            #     df_sorted[col_LBCV] *= 1000
            #     df_sorted[col_PBCV] *= 1000

            x_vals = df_sorted[ablation_variable]

            # Helper function to plot for each method
            def plot_method(col_name, label_name, color=None, use_moving_mean=True, custom_window_size=None):
                if col_name in df_sorted.columns:
                    method_key = col_name.split('_')[2]  # "CCV", "LBCV", etc.
                    detection_col = f'detected_{method_key}'
                    if detection_col in df_sorted.columns:
                        method_df = df_sorted[df_sorted[detection_col] == True].copy()
                    else:
                        method_df = df_sorted.copy()

                    if method_df.empty:
                        return  # nothing to plot

                    if use_moving_mean:
                        effective_window_size = custom_window_size if custom_window_size is not None else window_size
                        roll_mean = method_df[col_name].rolling(
                            window=effective_window_size,
                            center=center,
                            min_periods=min_periods,
                            win_type=win_type
                        ).mean()
                        roll_std = method_df[col_name].rolling(
                            window=effective_window_size,
                            center=center,
                            min_periods=min_periods,
                            win_type=win_type
                        ).std()

                        ax.plot(method_df[ablation_variable], roll_mean, label=f'{label_name} (Moving Mean)', linewidth=2, color=color, alpha=1.0)
                        ax.fill_between(method_df[ablation_variable], roll_mean - roll_std, roll_mean + roll_std, alpha=0.5, color=color)
                    else:
                        ax.plot(method_df[ablation_variable], method_df[col_name], label=f'{label_name} (Raw)', linewidth=1, color=color, alpha=1.0)


            if self.config.get("plot_CCV", False):
                plot_method(col_CCV, "CCV", color=bar_colors['CCV'], use_moving_mean=not ccv_no_moving_mean, custom_window_size=ccv_window_size)
            if self.config.get("plot_HCV", False):
                plot_method(col_HCV, "HCV", color=bar_colors['HCV'], use_moving_mean=True, custom_window_size=None)
            if self.config.get("plot_LBCV", False):
                plot_method(col_LBCV, "LBCV", color=bar_colors['LBCV'], use_moving_mean=True, custom_window_size=None)
            if self.config.get("plot_PBCV", False):
                # plot_method(col_PBCV, "PBCV", color=bar_colors['PBCV'], use_moving_mean=True, custom_window_size=None)
                plot_method(col_PBCV, "LBCV", color=bar_colors['PBCV'], use_moving_mean=True, custom_window_size=None)

            # Labeling
            if err_type in ['x', 'y', 'z']:
                ylabel = f'{err_type_name.title()} Error (mm)'
            else:
                ylabel = f'{err_type_name.title()} Error (deg)'

            ax.set_xlabel(ablation_variable_pretty, fontsize=24)
            ax.set_ylabel(ylabel, fontsize=24)
            # ax.set_title(f'{err_type_name} Error vs {ablation_variable_title}', fontsize=24)

            ax.axhline(0, color='black', linestyle='--', linewidth=1)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=24)

            if i == 0:
                ax.legend(fontsize=24, loc='best')

            # Determine observed min/max from plotted shaded areas
            observed_min = float('inf')
            observed_max = float('-inf')

            for method_col in [col_CCV, col_HCV, col_LBCV, col_PBCV]:
                method_key = method_col.split('_')[2]
                detection_col = f'detected_{method_key}'
                if method_col in df_sorted.columns and self.config.get(f"plot_{method_key}", False):
                    method_df = df_sorted[df_sorted[detection_col] == True].copy()
                    if method_df.empty:
                        continue
                    window = ccv_window_size if method_col == col_CCV and not ccv_no_moving_mean else window_size
                    roll_mean = method_df[method_col].rolling(
                        window=window, center=center, min_periods=min_periods, win_type=win_type
                    ).mean()
                    roll_std = method_df[method_col].rolling(
                        window=window, center=center, min_periods=min_periods, win_type=win_type
                    ).std()
                    lower = (roll_mean - roll_std).min()
                    upper = (roll_mean + roll_std).max()
                    observed_min = min(observed_min, lower)
                    observed_max = max(observed_max, upper)

            # Use global_ranges if all data is inside the range, else fall back to observed range
            global_limit = self.global_ranges[err_type]
            if observed_min >= -global_limit and observed_max <= global_limit:
                ax.set_ylim(-global_limit, global_limit)
            else:
                margin = 0.05 * (observed_max - observed_min) if observed_max > observed_min else 1e-3
                ax.set_ylim(observed_min - margin, observed_max + margin)



            # # Dynamically set scale based on global ranges or mean ± 1 std
            # if err_type in ['x', 'y', 'z']:
            #     if ablation_variable == "fraction_marker_visible": 
            #         if err_type == 'x' or err_type == 'y': 
            #             ax.set_ylim(-0.030 * 1000, 0.030 * 1000)  # Set range to ±0.5 m converted to mm
            #         elif err_type == 'z':
            #             ax.set_ylim(-0.05 * 1000, 0.05 * 1000)  # Set range to ±0.5 m converted to mm
            #     elif (ablation_variable == "distance_to_camera") and err_type == 'z':
            #         ax.set_ylim(-0.05 * 1000, 0.05 * 1000)  # Set range to ±0.5 m converted to mm
            #     else:
            #         ax.set_ylim(-self.global_ranges[err_type] * 1000, self.global_ranges[err_type] * 1000)
            # else:
            #     ax.set_ylim(-self.global_ranges[err_type], self.global_ranges[err_type])
            #     if (ablation_variable == "fraction_marker_visible") and (err_type == 'a' or err_type == 'b'):
            #         ax.set_ylim(-30, 30)

        plt.tight_layout()
        plt.grid(True, alpha=0.3)

        plot_filename = f"{ablation_variable}_error_moving_mean_std.png"
        if self.config.get("plot_HCV", False):
            plot_filename = f"{ablation_variable}_error_moving_mean_std_HCV.png"
        if self.config.get("plot_PBCV", False):
            plot_filename = f"{ablation_variable}_error_moving_mean_std_PBCV.png"

        save_path = os.path.join(self.output_dir, plot_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if save_central:
            central_output_path = self.config.get("central_output_path", self.output_dir)
            os.makedirs(central_output_path, exist_ok=True)
            save_path_central = os.path.join(central_output_path, plot_filename)
            plt.savefig(save_path_central, dpi=300, bbox_inches='tight')
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
        if self.config.get("plot_PBCV", False):
            methods.append('PBCV')

        colors = {'CCV': 'blue', 'LBCV': 'orange', 'HCV': 'green', 'PBCV': 'red'}

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
            plt.title(f"{ablation_variable.replace('_', ' ').title()}: {float(ablation_value):.4f}, Harris Score: {float(row['harris_corner_response_score']):.0f}")
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
        if self.config.get("plot_PBCV", False):
            methods.append("PBCV")

        print("Detection Rates:")
        for method in methods:
            col_name = f'detected_{method}'
            if col_name in self.df_data.columns:
                detection_rate = self.df_data[col_name].mean()
                print(f"{method}: {detection_rate:.2%}")

    def save_summary_table(self, ablation_variable):
        """
        Save a summary table to CSV with the following columns:
        method (e.g., CCV, LBCV with CCV success, PBCV with CCV fail, ...),
        detection rate, true positive detection rate,
        MAE ± std dev (mm or deg) for x, y, z, a, b, c.
        """
        methods = [
            "CCV",
            "LBCV with CCV success", "LBCV with CCV fail", "LBCV",
            "HCV with CCV success", "HCV with CCV fail", "HCV",
            "PBCV with CCV success", "PBCV with CCV fail", "PBCV"
        ]
        summary_data = []

        for method in methods:
            if method == "CCV":
                mask = self.df_data['detected_CCV'] == 1
            elif method == "LBCV with CCV success":
                mask = (self.df_data['detected_CCV'] == 1) & (self.df_data['detected_LBCV'] == 1)
            elif method == "LBCV with CCV fail":
                mask = (self.df_data['detected_CCV'] == 0) & (self.df_data['detected_LBCV'] == 1)
            elif method == "LBCV":
                mask = self.df_data['detected_LBCV'] == 1
            elif method == "HCV with CCV success":
                mask = (self.df_data['detected_CCV'] == 1) & (self.df_data['detected_HCV'] == 1)
            elif method == "HCV with CCV fail":
                mask = (self.df_data['detected_CCV'] == 0) & (self.df_data['detected_HCV'] == 1)
            elif method == "HCV":
                mask = self.df_data['detected_HCV'] == 1
            elif method == "PBCV with CCV success":
                mask = (self.df_data['detected_PBCV'] == 1) & (self.df_data['detected_CCV'] == 1)
            elif method == "PBCV with CCV fail":
                mask = (self.df_data['detected_PBCV'] == 1) & (self.df_data['detected_CCV'] == 0)
            elif method == "PBCV":
                mask = self.df_data['detected_PBCV'] == 1

            filtered_data = self.df_data[mask]
            detection_rate = mask.mean()

            # Determine method type
            if "LBCV" in method:
                method_type = "LBCV"
            elif "HCV" in method:
                method_type = "HCV"
            elif "PBCV" in method:
                method_type = "PBCV"
            else:
                method_type = "CCV"

            # Compute TP detection rate
            if method_type == "LBCV":
                tp_mask = filtered_data['LBCV_IOU'] > 0.5
            elif method_type == "PBCV":
                tp_mask = filtered_data['PBCV_IOU'] > 0.5
            elif method_type == "CCV":
                tp_mask = filtered_data['detected_CCV'] == 1
            else:
                tp_mask = pd.Series(False, index=filtered_data.index)  # fallback

            tp_detection_rate = tp_mask.mean()

            # Compute MAE and Std Dev for errors
            error_stats = {}
            for err_type in ['x', 'y', 'z', 'a', 'b', 'c']:
                col_name = f'pose_error_{method_type}_{err_type}'
                mae = filtered_data[col_name].abs().mean()
                std_dev = filtered_data[col_name].abs().std()
                unit = "mm" if err_type in ['x', 'y', 'z'] else "deg"
                error_stats[err_type] = (
                    f"{mae * 1000:.2f} ± {std_dev * 1000:.2f}" if unit == "mm" else f"{mae:.2f} ± {std_dev:.2f}"
                )

            summary_data.append([
                method,
                f"{detection_rate:.2%}",
                f"{tp_detection_rate:.2%}",
                error_stats['x'],
                error_stats['y'],
                error_stats['z'],
                error_stats['a'],
                error_stats['b'],
                error_stats['c']
            ])

        # Build final DataFrame
        summary_df = pd.DataFrame(summary_data, columns=[
            "Method",
            "Detection Rate",
            "True Positive Detection Rate",
            "X MAE ± Std Dev (mm)",
            "Y MAE ± Std Dev (mm)",
            "Z MAE ± Std Dev (mm)",
            "Pitch MAE ± Std Dev (deg)",
            "Yaw MAE ± Std Dev (deg)",
            "Roll MAE ± Std Dev (deg)"
        ])

        # Save
        output_path = os.path.join(self.output_dir, f"{ablation_variable}_summary_table.csv")
        summary_df.to_csv(output_path, index=False)
        print(f"Summary table saved to {output_path}")


    def combined_plot(self, ablation_variable, n_bins=10, coplot_blank_background=False, save_central=False):
        """
        Create a combined plot with 2x4 subplots:
        - Top left: detection plot
        - Bottom left: IOU plot
        - Right 2x3: error mean/std plots for x, y, z, a, b, c
        """
        ablation_variable_pretty = ABLATION_VARIABLE_PRETTY_NAMES.get(ablation_variable, ablation_variable.replace("_", " ").title())
        
        # Create figure with 2x4 subplots
        fig = plt.figure(figsize=(24, 12))
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
        
        # Top left: Detection plot
        ax_detection = fig.add_subplot(gs[0, 0])
        
        # Prepare detection rate data (similar to detection_plot method)
        aggregation_dict = {
            'detected_CCV': 'mean' if self.config.get("plot_CCV", False) else None,
            'detected_LBCV': 'mean' if self.config.get("plot_LBCV", False) else None,
            'detected_HCV': 'mean' if self.config.get("plot_HCV", False) else None
        }
        aggregation_dict = {key: value for key, value in aggregation_dict.items() if value is not None}
        
        # Create bins for the ablation variable
        bin_name = f'{ablation_variable}_bin'
        bins = pd.cut(self.df_data[ablation_variable], bins=n_bins, retbins=True)
        self.df_data[bin_name] = bins[0]
        bin_edges = bins[1]
        
        # Create bin labels
        bin_labels = []
        for i in range(len(bin_edges) - 1):
            bin_labels.append(f'{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}')
        
        bin_label_mapping = {i: bin_labels[i] for i in range(len(bin_labels))}
        self.df_data[f'{bin_name}_numeric'] = self.df_data[bin_name].cat.codes
        
        # Group data by bins and calculate detection rates
        grouped_data = self.df_data.groupby(f'{bin_name}_numeric').agg(aggregation_dict).reset_index()
        grouped_data['bin_label'] = grouped_data[f'{bin_name}_numeric'].map(bin_label_mapping)
        
        # Prepare methods to plot
        methods_to_plot = []
        if self.config.get("plot_CCV", False):
            methods_to_plot.append('detected_CCV')
        if self.config.get("plot_LBCV", False):
            methods_to_plot.append('detected_LBCV')
        if self.config.get("plot_HCV", False):
            methods_to_plot.append('detected_HCV')
        
        # Melt data for plotting
        melted_data = grouped_data.melt(id_vars=['bin_label'], value_vars=methods_to_plot, 
                                       var_name='Method', value_name='Detection_Rate')
        
        method_mapping = {
            'detected_CCV': 'CCV',
            'detected_LBCV': 'LBCV', 
            'detected_HCV': 'HCV'
        }
        melted_data['Method'] = melted_data['Method'].map(method_mapping)
        
        # Plot detection rates
        sns.barplot(x='bin_label', y='Detection_Rate', hue='Method', data=melted_data, ax=ax_detection)
        ax_detection.set_xlabel(f'{ablation_variable_pretty}', fontsize=12)
        ax_detection.set_ylabel('Detection Rate', fontsize=12)
        ax_detection.set_title(f'Detection Rate vs {ablation_variable_pretty}', fontsize=14)
        ax_detection.tick_params(axis='x', rotation=45)
        ax_detection.legend(fontsize=10)
        ax_detection.grid(True, alpha=0.3)
        
        # Bottom left: IOU plot
        ax_iou = fig.add_subplot(gs[1, 0])
        
        # Prepare IOU data (similar to IOU_plot method)
        self.df_data[f'{ablation_variable}_iou_bin'] = pd.cut(self.df_data[ablation_variable], bins=n_bins, labels=False)
        
        grouped_iou = self.df_data.groupby(f'{ablation_variable}_iou_bin').agg({
            'LBCV_IOU': 'mean'
        }).reset_index()
        
        # Plot IOU
        sns.barplot(x=f'{ablation_variable}_iou_bin', y='LBCV_IOU', data=grouped_iou, ax=ax_iou)
        ax_iou.set_xlabel(f'{ablation_variable_pretty} (binned)', fontsize=12)
        ax_iou.set_ylabel('IOU', fontsize=12)
        ax_iou.set_title(f'IOU vs {ablation_variable_pretty}', fontsize=14)
        ax_iou.tick_params(axis='x', rotation=45)
        ax_iou.set_ylim(0, 1)  # Set y-axis limits to [0,1]
        ax_iou.grid(True, alpha=0.3)
        
        # Right 2x3: Error plots (similar to err_mean_std_plot method)
        error_types = ['x', 'y', 'z', 'a', 'b', 'c']
        
        for i, err_type in enumerate(error_types):
            row = i // 3
            col = (i % 3) + 1  # Start from column 1 (columns 0 are for detection and IOU)
            ax = fig.add_subplot(gs[row, col])
            
            err_type_name = {'x': 'X', 'y': 'Y', 'z': 'Z', 'a': 'Pitch', 'b': 'Yaw', 'c': 'Roll'}[err_type]
            
            col_CCV = f'pose_error_CCV_{err_type}'
            col_HCV = f'pose_error_HCV_{err_type}'
            col_LBCV = f'pose_error_LBCV_{err_type}'
            col_PBCV = f'pose_error_PBCV_{err_type}'

            # Scale translation errors to mm
            if err_type in ['x', 'y', 'z']:
                df_temp = self.df_data.copy()
                df_temp[col_CCV] *= 1000
                df_temp[col_HCV] *= 1000
                df_temp[col_LBCV] *= 1000
            else:
                df_temp = self.df_data.copy()
            
            # Plot CCV if enabled
            if self.config.get("plot_CCV", False):
                grouped_CCV = df_temp.groupby(ablation_variable, observed=False)[col_CCV].agg(['mean', 'std']).reset_index()
                ax.plot(grouped_CCV[ablation_variable], grouped_CCV['mean'], label='CCV Mean', linewidth=2)
                ax.fill_between(grouped_CCV[ablation_variable],
                               grouped_CCV['mean'] - grouped_CCV['std'],
                               grouped_CCV['mean'] + grouped_CCV['std'],
                               alpha=0.3, label='CCV ±1 Std')
            
            # Plot HCV if enabled
            if self.config.get("plot_HCV", False):
                grouped_HCV = df_temp.groupby(ablation_variable, observed=False)[col_HCV].agg(['mean', 'std']).reset_index()
                ax.plot(grouped_HCV[ablation_variable], grouped_HCV['mean'], label='HCV Mean', linewidth=2)
                ax.fill_between(grouped_HCV[ablation_variable],
                               grouped_HCV['mean'] - grouped_HCV['std'],
                               grouped_HCV['mean'] + grouped_HCV['std'],
                               alpha=0.3, label='HCV ±1 Std')
            
            # Plot LBCV if enabled
            if self.config.get("plot_LBCV", False):
                grouped_LBCV = df_temp.groupby(ablation_variable, observed=False)[col_LBCV].agg(['mean', 'std']).reset_index()
                ax.plot(grouped_LBCV[ablation_variable], grouped_LBCV['mean'], label='LBCV Mean', linewidth=2)
                ax.fill_between(grouped_LBCV[ablation_variable],
                               grouped_LBCV['mean'] - grouped_LBCV['std'],
                               grouped_LBCV['mean'] + grouped_LBCV['std'],
                               alpha=0.3, label='LBCV ±1 Std')
                
            if self.config.get("plot_PBCV", False):
                grouped_PBCV = df_temp.groupby(ablation_variable, observed=False)[col_PBCV].agg(['mean', 'std']).reset_index()
                ax.plot(grouped_PBCV[ablation_variable], grouped_PBCV['mean'], label='PBCV Mean', linewidth=2)
                ax.fill_between(grouped_PBCV[ablation_variable],
                            grouped_PBCV['mean'] - grouped_PBCV['std'],
                            grouped_PBCV['mean'] + grouped_PBCV['std'],
                            alpha=0.3, label='PBCV ±1 Std')

                
            # Labels and formatting
            if err_type in ['x', 'y', 'z']:
                ylabel = f'{err_type_name.title()} Error (mm)'
            else:
                ylabel = f'{err_type_name.title()} Error (deg)'
            
            ax.set_xlabel(ablation_variable_pretty, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(f'{err_type_name} Error', fontsize=14)
            
            # Set y-axis limits based on global ranges
            if err_type in ['x', 'y', 'z']:
                if ablation_variable == "fraction_marker_visible": 
                    if err_type == 'x' or err_type == 'y': 
                        ax.set_ylim(-0.030 * 1000, 0.030 * 1000)
                    elif err_type == 'z':
                        ax.set_ylim(-0.05 * 1000, 0.05 * 1000)
                elif (ablation_variable == "distance_to_camera") and err_type == 'z':
                    ax.set_ylim(-0.05 * 1000, 0.05 * 1000)
                else:
                    ax.set_ylim(-self.global_ranges[err_type] * 1000, self.global_ranges[err_type] * 1000)
            else:
                ax.set_ylim(-self.global_ranges[err_type], self.global_ranges[err_type])
                if (ablation_variable == "fraction_marker_visible") and (err_type == 'a' or err_type == 'b'):
                    ax.set_ylim(-30, 30)
            
            ax.tick_params(axis='both', labelsize=10)
            if i == 0:  # Only show legend on first error plot
                ax.legend(fontsize=8)
            ax.axhline(0, color='black', linestyle='--', linewidth=1)
            ax.grid(True, alpha=0.3)
        
        # Overall title
        fig.suptitle(f'Combined Analysis: {ablation_variable_pretty}', fontsize=16, y=0.95)
        
        # Save the plot
        save_path = os.path.join(self.output_dir, f"{ablation_variable}_combined_plot.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if save_central:
            central_output_path = self.config.get("central_output_path", self.output_dir)
            os.makedirs(central_output_path, exist_ok=True)
            save_path = os.path.join(central_output_path, f"{ablation_variable}_combined_plot.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.close()
    
    def detection_score_plot(self):
        error_types = ['x', 'y', 'z', 'a', 'b', 'c']
        score_types = {
            'harris_corner_response_score': 'Harris Corner Response',
            'keypoint_residual_score': 'Keypoint Residual',
            'image_similarity_score': 'Image Similarity',
        }

        for score_col, score_label in score_types.items():
            fig, axs = plt.subplots(2, 3, figsize=(18, 10))
            axs = axs.flatten()

            for i, err_type in enumerate(error_types):
                col_name = f'pose_error_PBCV_{err_type}'
                if col_name not in self.df_data.columns or score_col not in self.df_data.columns:
                    continue

                ax = axs[i]
                sns.scatterplot(
                    x=self.df_data[score_col],
                    y=self.df_data[col_name],
                    ax=ax,
                    alpha=0.5
                )
                ax.set_xlabel(score_label, fontsize=14)
                if err_type in ['x', 'y', 'z']:
                    ax.set_ylabel(f'{err_type.upper()} Error (m)', fontsize=14)
                else:
                    ax.set_ylabel(f'{err_type.upper()} Error (deg)', fontsize=14)
                ax.set_title(f'{err_type.upper()} Error vs {score_label}', fontsize=16)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = os.path.join(self.output_dir, f"PBCV_error_vs_{score_col}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        for score_col, score_label in score_types.items():
            fig, axs = plt.subplots(2, 3, figsize=(18, 10))
            axs = axs.flatten()

            for i, err_type in enumerate(error_types):
                col_name = f'pose_error_PBCV_{err_type}'
                if col_name not in self.df_data.columns or score_col not in self.df_data.columns:
                    continue

                ax = axs[i]
                sns.scatterplot(
                    x=self.df_data[score_col],
                    y=self.df_data[col_name],
                    ax=ax,
                    alpha=0.5
                )
                ax.set_xlabel(score_label, fontsize=14)
                if err_type in ['x', 'y', 'z']:
                    ax.set_ylabel(f'{err_type.upper()} Error (m)', fontsize=14)
                    ax.set_ylim(-self.global_ranges[err_type], self.global_ranges[err_type])
                else:
                    ax.set_ylabel(f'{err_type.upper()} Error (deg)', fontsize=14)
                    ax.set_ylim(-self.global_ranges[err_type], self.global_ranges[err_type])
                ax.set_title(f'{err_type.upper()} Error vs {score_label}', fontsize=16)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = os.path.join(self.output_dir, f"PBCV_error_vs_{score_col}_withrangelimits.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        # plot filtered data 
        df_filtered = self.df_data[self.df_data['detected_PBCV'] == True]
        for score_col, score_label in score_types.items():
            fig, axs = plt.subplots(2, 3, figsize=(18, 10))
            axs = axs.flatten()

            for i, err_type in enumerate(error_types):
                col_name = f'pose_error_PBCV_{err_type}'
                if col_name not in df_filtered.columns or score_col not in df_filtered.columns:
                    continue

                ax = axs[i]
                sns.scatterplot(
                    x=df_filtered[score_col],
                    y=df_filtered[col_name],
                    ax=ax,
                    alpha=0.5
                )
                ax.set_xlabel(score_label, fontsize=14)
                if err_type in ['x', 'y', 'z']:
                    ax.set_ylabel(f'{err_type.upper()} Error (m)', fontsize=14)
                    ax.set_ylim(-self.global_ranges[err_type], self.global_ranges[err_type])
                else:
                    ax.set_ylabel(f'{err_type.upper()} Error (deg)', fontsize=14)
                    ax.set_ylim(-self.global_ranges[err_type], self.global_ranges[err_type])
                ax.set_title(f'{err_type.upper()} Error vs {score_label}', fontsize=16)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = os.path.join(self.output_dir, f"PBCV_error_vs_{score_col}_filtered.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def refilter_PBCV_detection(self, harris_corner_response_thresh=0.001, num_valid_proj_points_thresh=2, tf_PBCV_tz_thresh=10, image_similarity_thresh=20_000):
        """
        Refilter the data based on specified thresholds.
        """
        len_pre_filter = len(self.df_data)
        # self.df_data = self.df_data[
        #     (self.df_data['harris_corner_response_score'] >= harris_corner_response_thresh) &
        #     (self.df_data['num_valid_projection_points'] >= num_valid_proj_points_thresh) &
        #     (self.df_data['tf_PBCV_tz'] <= tf_PBCV_tz_thresh) &
        #     (self.df_data['image_similarity_score'] >= image_similarity_thresh)
        # ]
        self.df_data['detected_PBCV'] = (
            (self.df_data['detected_LBCV'] == True) & 
            (self.df_data['harris_corner_response_score'] >= harris_corner_response_thresh) &
            (self.df_data['num_valid_proj_points'] >= num_valid_proj_points_thresh) &
            (self.df_data['tf_PBCV_tz'] <= tf_PBCV_tz_thresh) &
            (self.df_data['image_similarity_score'] >= image_similarity_thresh)
        )
        print(f"Refiltered data from {len_pre_filter} to {len(self.df_data)} entries based on thresholds.")

    def combined_detection_iou_and_error_plot(
        self,
        ablation_variable,
        n_bins=10,
        IOU_threshold=0.5,
        window_size=20,
        save_central=False,
        ccv_no_moving_mean=False,
        ccv_window_size=None,
        center=True,
        min_periods=None,
        win_type=None,
    ):
        import matplotlib.gridspec as gridspec

        # Pretty names
        ablation_variable_pretty = ABLATION_VARIABLE_PRETTY_NAMES.get(
            ablation_variable, ablation_variable.replace("_", " ").title()
        )
        ablation_variable_title = ABLATION_VARIABLE_TITLE_NAMES.get(
            ablation_variable, ablation_variable.replace("_", " ").title()
        )

        # -----------------------------
        # Binning (shared across plots)
        # -----------------------------
        bin_name = f'{ablation_variable}_bin'
        bins = pd.cut(self.df_data[ablation_variable], bins=n_bins, retbins=True)
        self.df_data[bin_name] = bins[0]
        bin_edges = bins[1]

        bin_labels = [f'{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}' for i in range(len(bin_edges) - 1)]
        bin_label_mapping = {i: bin_labels[i] for i in range(len(bin_labels))}
        self.df_data[f'{bin_name}_numeric'] = self.df_data[bin_name].cat.codes

        # -----------------------------
        # Detection Rate (bars, top-left)
        # -----------------------------
        det_agg = {
            'detected_CCV': 'mean' if self.config.get("plot_CCV", False) else None,
            'detected_LBCV': 'mean' if self.config.get("plot_LBCV", False) else None,
            'detected_PBCV': 'mean' if self.config.get("plot_PBCV", False) else None,
        }
        det_agg = {k: v for k, v in det_agg.items() if v is not None}

        grouped_det = self.df_data.groupby(f'{bin_name}_numeric').agg(det_agg).reset_index()
        grouped_det['bin_label'] = grouped_det[f'{bin_name}_numeric'].map(bin_label_mapping)

        det_methods = []
        if self.config.get("plot_CCV", False): det_methods.append('detected_CCV')
        if self.config.get("plot_LBCV", False): det_methods.append('detected_LBCV')
        if self.config.get("plot_PBCV", False): det_methods.append('detected_PBCV')

        det_map = {
            'detected_CCV': 'CCV',
            'detected_LBCV': 'LBCV',
            'detected_PBCV': 'PBCV',
        }
        det_melt = grouped_det.melt(
            id_vars=['bin_label'],
            value_vars=det_methods,
            var_name='Method',
            value_name='Detection_Rate'
        )
        det_melt['Method'] = det_melt['Method'].map(det_map)
        det_melt['bin_label'] = pd.Categorical(det_melt['bin_label'], categories=bin_labels, ordered=True)

        # -----------------------------
        # IOU (bars, bottom-left)
        # -----------------------------
        iou_agg = {}
        if 'LBCV_IOU' in self.df_data.columns and self.config.get("plot_LBCV", False):
            iou_agg['LBCV_IOU'] = 'mean'
        if 'PBCV_IOU' in self.df_data.columns and self.config.get("plot_PBCV", False):
            iou_agg['PBCV_IOU'] = 'mean'

        grouped_iou = self.df_data.groupby(f'{bin_name}_numeric').agg(iou_agg).reset_index()
        grouped_iou['bin_label'] = grouped_iou[f'{bin_name}_numeric'].map(bin_label_mapping)

        iou_melt = grouped_iou.melt(
            id_vars='bin_label',
            value_name='Mean_IOU',
            var_name='Method'
        )
        iou_melt['Method'] = iou_melt['Method'].map({'LBCV_IOU': 'LBCV', 'PBCV_IOU': 'PBCV'})
        iou_melt['bin_label'] = pd.Categorical(iou_melt['bin_label'], categories=bin_labels, ordered=True)

        # -----------------------------
        # Figure layout: 2x4 via nested GridSpec
        # -----------------------------
        fig = plt.figure(figsize=(28, 10))
        gs = gridspec.GridSpec(
            2, 4, figure=fig, width_ratios=[1.2, 1, 1, 1], wspace=0.35, hspace=0.55
        )
        fig.suptitle(f'Detection and Pose Estimation vs {ablation_variable_title}', fontsize=28, y=0.98)

        # Axes: top-left (detection rate)
        ax_det = fig.add_subplot(gs[0, 0])

        # Axes: bottom-left (IOU)
        ax_iou = fig.add_subplot(gs[1, 0])

        # Right nested 2x3 grid for moving mean/std
        gs_right = gs[:, 1:].subgridspec(2, 3, wspace=0.35, hspace=0.55)
        axes_err = [fig.add_subplot(gs_right[r, c]) for r in range(2) for c in range(3)]

        # Shared palette
        palette = globals().get('bar_colors', None)

        # --- Font size settings ---
        LABEL_FONT_SIZE = 20
        TICK_FONT_SIZE = 16
        YLABEL_PAD_ERROR = 4  # moves y-axis label closer for error subplots

        # ---- Plot detection rate (top-left)
        sns.barplot(
            x='bin_label', y='Detection_Rate', hue='Method',
            data=det_melt, ax=ax_det, palette=palette, alpha=0.9, order=bin_labels
        )
        ax_det.set_title('Detection Rate', fontsize=LABEL_FONT_SIZE, pad=10)
        ax_det.set_xlabel(f'{ablation_variable_pretty}', fontsize=LABEL_FONT_SIZE)
        ax_det.set_ylabel('Detection Rate', fontsize=LABEL_FONT_SIZE, labelpad=YLABEL_PAD_ERROR)
        ax_det.set_ylim(0, 1.05)
        ax_det.tick_params(axis='x', rotation=30, labelsize=TICK_FONT_SIZE)
        ax_det.tick_params(axis='y', labelsize=TICK_FONT_SIZE)
        leg = ax_det.legend(fontsize=TICK_FONT_SIZE)
        if leg is not None: leg.set_title(None)
        ax_det.grid(True, alpha=0.3)

        # ---- Plot IOU (bottom-left)
        if not iou_melt.empty:
            sns.barplot(
                x='bin_label', y='Mean_IOU', hue='Method',
                data=iou_melt, ax=ax_iou, palette=palette, alpha=0.9, order=bin_labels
            )
            ax_iou.set_title('Mean Segmentation IOU', fontsize=LABEL_FONT_SIZE, pad=10)
            ax_iou.set_xlabel(f'{ablation_variable_pretty}', fontsize=LABEL_FONT_SIZE)
            ax_iou.set_ylabel('Mean Segmentation IOU', fontsize=LABEL_FONT_SIZE, labelpad=YLABEL_PAD_ERROR)
            ax_iou.set_ylim(0, 1.05)
            ax_iou.tick_params(axis='x', rotation=30, labelsize=TICK_FONT_SIZE)
            ax_iou.tick_params(axis='y', labelsize=TICK_FONT_SIZE)
            leg2 = ax_iou.legend(fontsize=TICK_FONT_SIZE)
            if leg2 is not None: leg2.set_title(None)
            ax_iou.grid(True, alpha=0.3)
        else:
            ax_iou.text(0.5, 0.5, 'No IOU data', ha='center', va='center', fontsize=LABEL_FONT_SIZE)
            ax_iou.axis('off')

        # -----------------------------
        # Moving mean ± std (right 2x3)
        # -----------------------------
        error_types = ['x', 'y', 'z', 'a', 'b', 'c']
        err_labels = {'x': 'X', 'y': 'Y', 'z': 'Z', 'a': 'Pitch', 'b': 'Yaw', 'c': 'Roll'}
        # Pre-sort once
        df_sorted = self.df_data.sort_values(by=ablation_variable).copy()

        def plot_method(ax, df, col_name, label_name, color, use_mmean=True, custom_win=None):
            if col_name not in df.columns:
                return False
            method_key = col_name.split('_')[2]  # CCV/LBCV/PBCV/HCV
            det_col = f'detected_{method_key}'
            if det_col in df.columns:
                mdf = df[df[det_col] == True].copy()
            else:
                mdf = df.copy()
            if mdf.empty:
                return False

            if use_mmean:
                eff_w = custom_win if custom_win is not None else window_size
                roll_mean = mdf[col_name].rolling(
                    window=eff_w, center=center, min_periods=min_periods, win_type=win_type
                ).mean()
                roll_std = mdf[col_name].rolling(
                    window=eff_w, center=center, min_periods=min_periods, win_type=win_type
                ).std()
                ax.plot(mdf[ablation_variable], roll_mean, label=f'{label_name}', linewidth=2, color=color, alpha=1.0)
                ax.fill_between(mdf[ablation_variable], roll_mean - roll_std, roll_mean + roll_std, alpha=0.25, color=color)
            else:
                ax.plot(mdf[ablation_variable], mdf[col_name], label=f'{label_name} (Raw)', linewidth=1, color=color, alpha=1.0)
            return True

        for i, err_type in enumerate(error_types):
            ax = axes_err[i]
            err_name = err_labels[err_type]
            col_CCV = f'pose_error_CCV_{err_type}'
            col_HCV = f'pose_error_HCV_{err_type}'
            col_LBCV = f'pose_error_LBCV_{err_type}'
            col_PBCV = f'pose_error_PBCV_{err_type}'

            # plot each enabled method
            any_plotted = False
            if self.config.get("plot_CCV", False):
                any_plotted |= plot_method(ax, df_sorted, col_CCV, 'CCV', bar_colors['CCV'],
                                        use_mmean=not ccv_no_moving_mean, custom_win=ccv_window_size)
            if self.config.get("plot_HCV", False) and col_HCV in df_sorted.columns:
                any_plotted |= plot_method(ax, df_sorted, col_HCV, 'HCV', bar_colors['HCV'])
            if self.config.get("plot_LBCV", False):
                any_plotted |= plot_method(ax, df_sorted, col_LBCV, 'LBCV', bar_colors['LBCV'])
            if self.config.get("plot_PBCV", False):
                any_plotted |= plot_method(ax, df_sorted, col_PBCV, 'PBCV', bar_colors['PBCV'])
            
            if err_type in ['x', 'y', 'z']:
                ylabel = f'{err_name} Error (mm)'
            else:
                ylabel = f'{err_name} Error (deg)'

            ax.set_title(f'{err_name} Error', fontsize=LABEL_FONT_SIZE, pad=10)
            ax.set_xlabel(ablation_variable_pretty, fontsize=LABEL_FONT_SIZE)
            ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE, labelpad=YLABEL_PAD_ERROR)
            ax.axhline(0, color='black', linestyle='--', linewidth=1)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)


            # Compute observed range from rolling mean ± std (if available)
            observed_min, observed_max = float('inf'), float('-inf')
            for method_col, enabled, key in [
                (col_CCV, self.config.get("plot_CCV", False), 'CCV'),
                (col_HCV, self.config.get("plot_HCV", False), 'HCV'),
                (col_LBCV, self.config.get("plot_LBCV", False), 'LBCV'),
                (col_PBCV, self.config.get("plot_PBCV", False), 'PBCV'),
            ]:
                if not enabled or method_col not in df_sorted.columns:
                    continue
                det_col = f'detected_{key}'
                mdf = df_sorted[df_sorted[det_col] == True].copy() if det_col in df_sorted.columns else df_sorted.copy()
                if mdf.empty:
                    continue
                win = (ccv_window_size if (method_col == col_CCV and not ccv_no_moving_mean) else window_size)
                rmean = mdf[method_col].rolling(window=win, center=center, min_periods=min_periods, win_type=win_type).mean()
                rstd  = mdf[method_col].rolling(window=win, center=center, min_periods=min_periods, win_type=win_type).std()
                lower = (rmean - rstd).min()
                upper = (rmean + rstd).max()
                if pd.notna(lower): observed_min = min(observed_min, lower)
                if pd.notna(upper): observed_max = max(observed_max, upper)

            # Apply global ranges when they contain the data, else fallback to observed
            gl = self.global_ranges[err_type]
            if observed_min >= -gl and observed_max <= gl:
                ax.set_ylim(-gl, gl)
            else:
                if np.isfinite(observed_min) and np.isfinite(observed_max) and observed_max > observed_min:
                    margin = 0.05 * (observed_max - observed_min)
                    ax.set_ylim(observed_min - margin, observed_max + margin)

            # Only first error subplot shows legend
            if i == 0 and any_plotted:
                ax.legend(fontsize=TICK_FONT_SIZE, loc='best')

        # -----------------------------
        # Save
        # -----------------------------
        plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.94])
        fname = f"{ablation_variable}_combined_detection_iou_error.png"
        save_path = os.path.join(self.output_dir, fname)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)

        if save_central:
            central_output_path = self.config.get("central_output_path", self.output_dir)
            os.makedirs(central_output_path, exist_ok=True)
            central_save_path = os.path.join(central_output_path, fname)
            plt.savefig(central_save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)

        plt.close(fig)

    def create_ablation_variable_images_figure(self, ablation_variable, N, save_central=False, figsize=(15, 8)):
        """
        Create a figure with N images showing ablation variable values ranging from min to max.
        
        The function automatically determines the ordering:
        - Increasing order (left to right): distance, skew, glare, shadow
        - Decreasing order (left to right): truncation, underexposure
        
        Args:
            ablation_variable (str): The ablation variable to visualize
            N (int): Number of images to display
            save_central (bool): Whether to save to central output path
            figsize (tuple): Figure size (width, height)
        
        The ablation variable name is displayed vertically on the left side of the figure,
        and each subplot shows the actual value as its title.
        """
        # Get pretty name for the ablation variable
        ablation_variable_pretty = ABLATION_VARIABLE_PRETTY_NAMES.get(
            ablation_variable, ablation_variable.replace("_", " ").title()
        )
        
        # Determine ordering based on ablation variable
        decreasing_order_vars = ['truncation', 'underexposure', 'fraction_marker_visible', 'mean_marker_pixel_brightness']
        use_decreasing_order = any(var in ablation_variable.lower() for var in decreasing_order_vars)
        
        # Get min and max values for the ablation variable
        min_val = self.df_data[ablation_variable].min()
        max_val = self.df_data[ablation_variable].max()
        
        # Create N equally spaced values
        if use_decreasing_order:
            target_values = np.linspace(max_val, min_val, N)
        else:
            target_values = np.linspace(min_val, max_val, N)
        
        # Create figure and subplots
        fig, axes = plt.subplots(1, N, figsize=figsize)
        if N == 1:
            axes = [axes]
            
        # # Add vertical text for ablation variable name on the left
        # fig.text(0.02, 0.5, ablation_variable_pretty, rotation=90, 
        #         fontsize=16, fontweight='bold', ha='center', va='center')
        
        # Determine if this is a truncation variable for special cropping
        is_truncation = 'truncation' in ablation_variable.lower() or 'fraction_marker_visible' in ablation_variable.lower()
        
        # For each target value, find the closest image
        for i, target_val in enumerate(target_values):
            # Find the row with value closest to target
            closest_idx = (self.df_data[ablation_variable] - target_val).abs().idxmin()
            closest_row = self.df_data.loc[closest_idx]
            
            # Load and display the image
            image_path = closest_row['image_path']
            actual_value = closest_row[ablation_variable]
            
            try:
                img = plt.imread(image_path)
                
                # Crop the image based on the ablation variable type
                height, width = img.shape[:2]
                crop_size = 300
                
                if is_truncation:
                    # For truncation: crop to rightmost 300x300 pixels (vertically centered)
                    center_y = height // 2
                    start_y = max(0, center_y - crop_size // 2)
                    end_y = min(height, start_y + crop_size)
                    start_x = max(0, width - crop_size)
                    end_x = width
                else:
                    # For other variables: crop to center 300x300 pixels
                    center_y = height // 2
                    center_x = width // 2
                    start_y = max(0, center_y - crop_size // 2)
                    end_y = min(height, start_y + crop_size)
                    start_x = max(0, center_x - crop_size // 2)
                    end_x = min(width, start_x + crop_size)
                
                # Adjust if image is smaller than crop size
                if end_y - start_y < crop_size:
                    start_y = 0
                    end_y = height
                if end_x - start_x < crop_size:
                    start_x = 0
                    end_x = width
                
                cropped_img = img[start_y:end_y, start_x:end_x]
                axes[i].imshow(cropped_img)
                axes[i].axis('off')
                
            except Exception as e:
                # If image can't be loaded, show placeholder
                axes[i].text(0.5, 0.5, f'Image not found\n{actual_value:.2f}', 
                           ha='center', va='center', transform=axes[i].transAxes,
                           fontsize=10, bbox=dict(facecolor='lightgray', alpha=0.8))
                axes[i].set_xlim(0, 1)
                axes[i].set_ylim(0, 1)
                axes[i].axis('off')
        
        # Adjust layout to fit tightly to the N images
        plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.02, hspace=0.02)
        
        # Save the figure
        save_filename = f"{ablation_variable}_progression_{N}_images.png"
        save_path = os.path.join(self.output_dir, "sample_images", save_filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if save_central:
            central_output_path = self.config.get("central_output_path", self.output_dir, "sample_images")
            os.makedirs(central_output_path, exist_ok=True)
            central_save_path = os.path.join(central_output_path, save_filename)
            plt.savefig(central_save_path, dpi=300, bbox_inches='tight')
        
        plt.close(fig)
        print(f"Saved ablation variable progression figure: {save_path}")

if __name__ == "__main__":

    # ablations = ["distance_20250712","skew_20250712","truncation_20250712","underexposure_20250712","glare_20250712","shadow_20250712"] 
    # ablations = ["underexposure_20250712","glare_20250712","truncation_20250712"] 
    ablations = ["underexposure_20250712","glare_20250712","truncation_20250712","shadow_20250712"] 
    # ablations = ["shadow_20250712"]
    # ablations = ["glare_20250712"]
    
    data_yaml_path = "./ablations/real_exp_data_description.yaml" 
    with open(data_yaml_path, 'r') as f:
        data_description = yaml.safe_load(f)

    # Load all dataframes to calculate global ranges
    dataframes = []
    for ablation in ablations:
        data_path = data_description[ablation]["data_path"]
        results_path = os.path.join(data_path, "results/results.csv")
        df = pd.read_csv(results_path) 
        # filtering 
        if "underexposure" in ablation: 
            # filter out all brightness values above 30
            df = df[df["mean_marker_pixel_brightness"] <= 30]
        if "truncation" in ablation: 
            df = df[df["fraction_marker_visible"] >= 0.33]
        dataframes.append(df)

    global_ranges = Plotter.calculate_global_max_ranges(dataframes) # FIXME: doesn't appear to correctly compute or apply global ranges 

    for ablation in ablations:
        data_path = data_description[ablation]["data_path"]
        ablation_variable = data_description[ablation]["ablation_variable"]
        window_size = data_description[ablation]["window_size"] 
        ablation_variable_min = data_description[ablation]["ablation_variable_min"]
        ablation_variable_max = data_description[ablation]["ablation_variable_max"]

        config = {
            "results_path": os.path.join(data_path, "results/results.csv"),
            # "output_path": os.path.join(data_path, "results/plots"),
            "output_path": os.path.join(data_path, "results/plots_JSR"),
            "plot_LBCV": True,
            "plot_HCV": False,
            "plot_PBCV": True,
            # "central_output_path": "./ablations/analysis/real_exp/plots",
            "central_output_path": "./ablations/analysis/real_exp/plots_JSR",
            "ablation_name": ablation, 
            "ablation_variable": ablation_variable,
            "ablation_variable_min": ablation_variable_min,
            "ablation_variable_max": ablation_variable_max, 
        }

        plotter_instance = Plotter(config, global_ranges)

        # plotter_instance.refilter_PBCV_detection(harris_corner_response_thresh = 0.0, num_valid_proj_points_thresh = 0, tf_PBCV_tz_thresh = 10, image_similarity_thresh = 20_000)
        # plotter_instance.save_summary_table(ablation_variable)
        # plotter_instance.detection_plot(ablation_variable, coplot_blank_background=False, save_central=True, n_bins=10)
        # # plotter_instance.IOU_plot(ablation_variable)
        # plotter_instance.detection_and_IOU_plot(ablation_variable, n_bins=10, save_central=True, IOU_threshold=0.50)
        # # plotter_instance.error_plot(ablation_variable)
        # # plotter_instance.error_ridge_plots(ablation_variable, n_ablation_bins=10, plot_CCV=True)
        # plotter_instance.err_mean_std_plot(ablation_variable, coplot_blank_background=False, save_central=True)
        # if "glare" not in ablation:  # FIXME: make this a parameter in yaml file 
        #     plotter_instance.err_moving_mean_std_plot(ablation_variable, window_size=window_size, save_central=True, ccv_no_moving_mean=False, ccv_window_size=30, min_periods=1)
        # else:             
        #     plotter_instance.err_moving_mean_std_plot(ablation_variable, window_size=window_size, save_central=True, ccv_no_moving_mean=False, ccv_window_size=150, min_periods=1)
        # plotter_instance.detection_score_plot()
        # # plotter_instance.output_labeled_images(ablation_variable)
        # # plotter_instance.find_worst_performing(ablation_variable)
        # # plotter_instance.find_best_performing(ablation_variable)
        # # plotter_instance.combined_plot(ablation_variable, n_bins=10, coplot_blank_background=False, save_central=True)
        # plotter_instance.combined_detection_iou_and_error_plot(
        #     ablation_variable, n_bins=5, save_central=True, IOU_threshold=0.50,
        #     window_size=window_size, ccv_no_moving_mean=False, ccv_window_size=30, center=True, min_periods=1
        # )
        plotter_instance.create_ablation_variable_images_figure(ablation_variable, N=8, save_central=False)
        

