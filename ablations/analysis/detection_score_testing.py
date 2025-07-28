import numpy as np 
import pandas as pd
import yaml 
import os 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# --- Config ---

# ablations = ["underexposure_20250712","glare_20250712","shadow_20250712"]
ablations = ["distance_20250712","skew_20250712","truncation_20250712","underexposure_20250712","glare_20250712","shadow_20250712"]
out_dir = "./ablations/analysis/plots/"
os.makedirs(out_dir, exist_ok=True)

colors = sns.color_palette("husl", len(ablations))
pose_err_cols = ['pose_error_PBCV_x', 'pose_error_PBCV_y', 'pose_error_PBCV_z', 'pose_error_PBCV_a', 'pose_error_PBCV_b', 'pose_error_PBCV_c']
score_columns = {
    # "harris_corner_response_score": "Harris Corner Response Score",
    # "keypoint_residual_score": "Keypoint Residual Score",
    # "num_valid_proj_points": "Num Valid Projected Points",
    "image_similarity_score": "Image Similarity Score"
}
label_rename_dict = {
    "distance_20250712 (PBCV)": "Distance (PBCV)",
    "skew_20250712 (PBCV)": "Skew (PBCV)",
    "truncation_20250712 (PBCV)": "Truncation (PBCV)",
    "underexposure_20250712 (PBCV)": "Underexposure (PBCV)",
    "glare_20250712 (PBCV)": "Glare (PBCV)",
    "shadow_20250712 (PBCV)": "Shadow (PBCV)",
    "distance_20250712 (LBCV)": "Distance (LBCV)",
    "skew_20250712 (LBCV)": "Skew (LBCV)",
    "truncation_20250712 (LBCV)": "Truncation (LBCV)",
    "underexposure_20250712 (LBCV)": "Underexposure (LBCV)",
    "glare_20250712 (LBCV)": "Glare (LBCV)",
    "shadow_20250712 (LBCV)": "Shadow (LBCV)",
}

# --- Load YAML ---
data_yaml_path = "./ablations/real_exp_data_description.yaml" 
with open(data_yaml_path, 'r') as f:
    data_description = yaml.safe_load(f)

# --- Load DataFrames ---
df_ablations = []
for ablation in ablations:
    data_path = data_description[ablation]["data_path"]
    results_path = os.path.join(data_path, "results/results.csv")
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        # orig_len = len(df)
        # df = df[df['detected_LBCV'] == True]
        # df = df[df['image_similarity_score'] > 20_000]
        # df = df[df['num_valid_proj_points'] > 2]
        # df = df[df['tf_PBCV_tz'] < 10]
        # df = df[df['harris_corner_response_score'] > 0.002]
        ablation_variable = data_description[ablation].get("ablation_variable")
        if data_description[ablation].get("ablation_variable_min") != 'None': 
            df = df[df[ablation_variable] >= data_description[ablation]["ablation_variable_min"]]
        if data_description[ablation].get("ablation_variable_max") != 'None': 
            df = df[df[ablation_variable] <= data_description[ablation]["ablation_variable_max"]]
        df_ablations.append(df)
        # print(f"\n{ablation}")
        # print(df[pose_err_cols].abs().max())
        # print(f"Detection rate: {len(df) / orig_len * 100:.2f}%")

# --- Plotting Function ---
def plot_mae_vs_score_threshold(
    df_ablation, ablation_label, ablation_color, score_key, score_display_name, save_prefix, out_dir
):
    score_vals = df_ablation[score_key]
    thresholds = np.linspace(score_vals.min(), score_vals.max(), 50)
    
    MAE_pose_errors = []
    for threshold in thresholds:
        df_filtered = df_ablation[df_ablation[score_key] >= threshold]
        if df_filtered.empty:
            MAE_pose_errors.append([np.nan] * 6)
        else:
            MAE_pose_errors.append(df_filtered[pose_err_cols].abs().max())

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axs = axs.flatten()
    fig.suptitle(f'MAE of Pose Errors vs {score_display_name} Thresholds - {ablation_label}', fontsize=16, y=0.98)
    
    for i, err_type in enumerate(['x', 'y', 'z', 'a', 'b', 'c']):
        ax = axs[i]
        mae_values = [mae[i] if not np.isnan(mae[i]) else None for mae in MAE_pose_errors]
        valid_thresholds = [thresh for thresh, mae in zip(thresholds, mae_values) if mae is not None]
        valid_mae_values = [mae for mae in mae_values if mae is not None]

        if valid_mae_values:
            ax.plot(valid_thresholds, valid_mae_values, marker='o', color=ablation_color, linewidth=2, markersize=4)

        ax.set_xlabel(f'{score_display_name} Threshold', fontsize=12)
        ax.set_ylabel(f'{err_type.upper()} Error ({"m" if err_type in "xyz" else "deg"})', fontsize=12)
        ax.set_title(f'{err_type.upper()} Error', fontsize=14)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"{save_prefix}_{ablation_label}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved plot: {save_path}")

# --- Plotting Function for Detection Rate ---
def plot_pbcv_detection_rate_vs_threshold(df_ablations, ablations, score_key, score_display_name, out_dir):
    """
    Plot PBCV detection rate vs threshold value for each ablation.
    """
    plt.figure(figsize=(12, 8))
    
    for ablation_idx, ablation in enumerate(ablations):
        df_ablation = df_ablations[ablation_idx]
        if df_ablation.empty:
            print(f"No data for ablation: {ablation}")
            continue
            
        # Check if score column exists
        if score_key not in df_ablation.columns:
            print(f"Score column '{score_key}' not found in {ablation}")
            continue
            
        # Get score values and create threshold range
        score_vals = df_ablation[score_key].dropna()
        if len(score_vals) == 0:
            print(f"No valid score values for {score_key} in {ablation}")
            continue
            
        thresholds = np.linspace(score_vals.min(), score_vals.max(), 50)
        detection_rates = []
        
        for threshold in thresholds:
            df_filtered = df_ablation[df_ablation[score_key] >= threshold]
            if len(df_filtered) == 0:
                detection_rates.append(0)
            else:
                # Calculate PBCV detection rate
                pbcv_detected = df_filtered['detected_PBCV'].sum()
                total_samples = len(df_filtered)
                detection_rate = (pbcv_detected / total_samples) * 100
                detection_rates.append(detection_rate)
        
        # Get ablation label and color
        ablation_label = data_description[ablation].get("label", ablation)
        color = colors[ablation_idx]
        
        # Plot the detection rate curve
        plt.plot(thresholds, detection_rates, marker='o', color=color, 
                linewidth=2, markersize=4, label=ablation_label, alpha=0.8)
    
    plt.xlabel(f'{score_display_name} Threshold', fontsize=14)
    plt.ylabel('PBCV Detection Rate (%)', fontsize=14)
    plt.title(f'PBCV Detection Rate vs {score_display_name} Threshold', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    save_filename = f"PBCV_Detection_Rate_vs_{score_key.upper()}_Threshold_{ablations[0].split('_')[-1]}.png"
    save_path = os.path.join(out_dir, save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved plot: {save_path}")

# --- Refiltering Function ---
def refilter_and_recompute_detected_pbcv(df, 
                                        harris_corner_threshold=None,
                                        keypoint_residual_threshold=None, 
                                        num_valid_proj_points_threshold=None,
                                        image_similarity_threshold=None,
                                        detected_lbcv_filter=True):
    """
    Refilter the dataframe and recompute detected_PBCV based on score thresholds.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with detection data
    harris_corner_threshold : float, optional
        Minimum threshold for harris_corner_response_score
    keypoint_residual_threshold : float, optional
        Minimum threshold for keypoint_residual_score  
    num_valid_proj_points_threshold : int, optional
        Minimum threshold for num_valid_proj_points
    image_similarity_threshold : float, optional
        Minimum threshold for image_similarity_score
    detected_lbcv_filter : bool, default=True
        Whether to filter by detected_LBCV == True
        
    Returns:
    --------
    pandas.DataFrame
        Filtered dataframe with recomputed detected_PBCV column
    """
    
    # Make a copy to avoid modifying original data
    df_filtered = df.copy()
    
    # Apply detected_LBCV filter first if requested
    if detected_lbcv_filter and 'detected_LBCV' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['detected_LBCV'] == True]
    
    # Apply score threshold filters
    if harris_corner_threshold is not None and 'harris_corner_response_score' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['harris_corner_response_score'] >= harris_corner_threshold]
        
    if keypoint_residual_threshold is not None and 'keypoint_residual_score' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['keypoint_residual_score'] >= keypoint_residual_threshold]
        
    if num_valid_proj_points_threshold is not None and 'num_valid_proj_points' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['num_valid_proj_points'] >= num_valid_proj_points_threshold]
        
    if image_similarity_threshold is not None and 'image_similarity_score' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['image_similarity_score'] >= image_similarity_threshold]
    
    # Recompute detected_PBCV based on filtering
    # If no rows remain after filtering, detected_PBCV should be False for all original rows
    if len(df_filtered) == 0:
        df_result = df.copy()
        df_result['detected_PBCV'] = False
    else:
        # Create a new detected_PBCV column - True for rows that pass all filters, False otherwise
        df_result = df.copy()
        df_result['detected_PBCV'] = df_result.index.isin(df_filtered.index)
    
    return df_result

# --- New ROC Plotting Function ---
def plot_roc_curves_by_score(df_ablations, ablations, score_columns, data_description, colors, out_dir, iou_threshold=0.5):
    """
    Create ROC curves for each score type with all ablations plotted together.
    True Positive: IOU > 0.5 and PBCV == true
    False Positive: IOU < 0.5 and PBCV == true
    """
    
    for score_key, score_display_name in score_columns.items():
        plt.figure(figsize=(10, 8))
        
        for ablation_idx, ablation in enumerate(ablations):
            df_ablation = df_ablations[ablation_idx]
            
            if df_ablation.empty:
                print(f"No data for ablation: {ablation}")
                continue
                
            # Check if required columns exist
            required_cols = [score_key, 'detected_PBCV']
            
            # Check for IOU column (could be named differently)
            iou_col = None
            possible_iou_cols = ['LBCV_IOU', 'IOU', 'iou', 'PBCV_IOU']
            for col in possible_iou_cols:
                if col in df_ablation.columns:
                    iou_col = col
                    break
            
            if iou_col is None:
                print(f"IOU column not found in {ablation}. Skipping.")
                continue
                
            required_cols.append(iou_col)
            
            missing_cols = [col for col in required_cols if col not in df_ablation.columns]
            if missing_cols:
                print(f"Missing columns {missing_cols} in {ablation}. Skipping.")
                continue
            
            # Filter data where PBCV detection was attempted (detected_PBCV == True)
            df_pbcv_detected = df_ablation[df_ablation['detected_PBCV'] == True].copy()
            
            if len(df_pbcv_detected) == 0:
                print(f"No PBCV detections in {ablation}. Skipping.")
                continue
            
            # Define ground truth labels based on IOU threshold
            # True Positive: IOU > 0.5 (good detection)
            # False Positive: IOU <= 0.5 (poor detection)
            df_pbcv_detected['ground_truth'] = (df_pbcv_detected[iou_col] > iou_threshold).astype(int)

            # Check if we have both positive and negative samples
            if len(df_pbcv_detected['ground_truth'].unique()) < 2:
                print(f"Only one class present in {ablation} for {score_key}. Skipping ROC curve.")
                continue
            
            # Get scores (higher scores should indicate better detection)
            scores = df_pbcv_detected[score_key].values
            labels = df_pbcv_detected['ground_truth'].values
            
            # Remove NaN values
            valid_mask = ~np.isnan(scores)
            scores = scores[valid_mask]
            labels = labels[valid_mask]
            
            if len(scores) == 0:
                print(f"No valid scores for {score_key} in {ablation}. Skipping.")
                continue
            
            # Compute ROC curve
            try:
                fpr, tpr, thresholds = roc_curve(labels, scores)
                roc_auc = auc(fpr, tpr)
                
                # Get ablation label and color
                ablation_label = data_description[ablation].get("label", ablation)
                color = colors[ablation_idx]
                
                # Plot ROC curve
                plt.plot(fpr, tpr, color=color, linewidth=2, 
                        label=f'{ablation_label} (AUC = {roc_auc:.2f})', alpha=0.8)
                
            except Exception as e:
                print(f"Error computing ROC curve for {ablation}, {score_key}: {e}")
                continue
        
        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
        
        # Formatting
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title(f'ROC Curves for {score_display_name}\n(TP: IOU > {iou_threshold:.2f}, FP: IOU ≤ {iou_threshold:.2f})', fontsize=16)        
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        save_filename = f"ROC_Curves_{score_key.upper()}_{ablations[0].split('_')[-1]}.png"
        save_path = os.path.join(out_dir, save_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved ROC curve plot: {save_path}")

# --- Function to Compute and Display AUC Table ---
def compute_auc_table(df_ablations, ablations, score_columns, data_description, out_dir, iou_threshold=0.5):
    """
    Compute AUC values for each score type and ablation, and display in a table format.
    Returns a DataFrame with AUC values for easy analysis.
    """
    
    # Initialize results dictionary
    auc_results = {}
    
    # Initialize the table with ablation names as rows
    ablation_labels = []
    for ablation in ablations:
        ablation_label = data_description[ablation].get("label", ablation)
        ablation_labels.append(ablation_label)
    
    # For each score type, compute AUC values
    for score_key, score_display_name in score_columns.items():
        auc_values = []
        
        for ablation_idx, ablation in enumerate(ablations):
            df_ablation = df_ablations[ablation_idx]
            
            if df_ablation.empty:
                auc_values.append(np.nan)
                continue
                
            # Check if required columns exist
            required_cols = [score_key, 'detected_PBCV']
            
            # Check for IOU column
            iou_col = None
            possible_iou_cols = ['LBCV_IOU', 'IOU', 'iou', 'PBCV_IOU']
            for col in possible_iou_cols:
                if col in df_ablation.columns:
                    iou_col = col
                    break
            
            if iou_col is None:
                auc_values.append(np.nan)
                continue
                
            required_cols.append(iou_col)
            
            missing_cols = [col for col in required_cols if col not in df_ablation.columns]
            if missing_cols:
                auc_values.append(np.nan)
                continue
            
            # Filter data where PBCV detection was attempted
            df_pbcv_detected = df_ablation[df_ablation['detected_PBCV'] == True].copy()
            
            if len(df_pbcv_detected) == 0:
                auc_values.append(np.nan)
                continue
            
            # Define ground truth labels based on IOU threshold
            df_pbcv_detected['ground_truth'] = (df_pbcv_detected[iou_col] > iou_threshold).astype(int)            
        
            # Check if we have both positive and negative samples
            if len(df_pbcv_detected['ground_truth'].unique()) < 2:
                auc_values.append(np.nan)
                continue
            
            # Get scores and labels
            scores = df_pbcv_detected[score_key].values
            labels = df_pbcv_detected['ground_truth'].values
            
            # Remove NaN values
            valid_mask = ~np.isnan(scores)
            scores = scores[valid_mask]
            labels = labels[valid_mask]
            
            if len(scores) == 0:
                auc_values.append(np.nan)
                continue
            
            # Compute AUC
            try:
                fpr, tpr, thresholds = roc_curve(labels, scores)
                roc_auc = auc(fpr, tpr)
                auc_values.append(roc_auc)
            except Exception as e:
                print(f"Error computing AUC for {ablation}, {score_key}: {e}")
                auc_values.append(np.nan)
        
        auc_results[score_display_name] = auc_values
    
    # Create DataFrame
    auc_df = pd.DataFrame(auc_results, index=ablation_labels)
    
    # Display the table
    print("\n" + "="*80)
    print("AUC VALUES TABLE")
    print("="*80)
    print("True Positive: IOU > {:.2f}, False Positive: IOU ≤ {:.2f}".format(iou_threshold, iou_threshold))
    print("-"*80)
    
    # Format the DataFrame for nice display
    pd.set_option('display.float_format', '{:.3f}'.format)
    print(auc_df)
    
    # Reset display options
    pd.reset_option('display.float_format')
    
    # Save to CSV
    csv_filename = f"AUC_Values_Table_{ablations[0].split('_')[-1]}.csv"
    csv_path = os.path.join(out_dir, csv_filename)
    auc_df.to_csv(csv_path)
    print(f"\nAUC table saved to: {csv_path}")
    
    # Display summary statistics
    print("\n" + "-"*80)
    print("SUMMARY STATISTICS")
    print("-"*80)
    print("Mean AUC per Score Type:")
    mean_auc = auc_df.mean()
    for score_name, mean_val in mean_auc.items():
        print(f"  {score_name}: {mean_val:.3f}")
    
    print("\nMean AUC per Ablation:")
    mean_auc_ablation = auc_df.mean(axis=1)
    for ablation_name, mean_val in mean_auc_ablation.items():
        print(f"  {ablation_name}: {mean_val:.3f}")
    
    print("\nBest performing combinations:")
    # Find the best score type for each ablation
    for ablation_name in auc_df.index:
        best_score = auc_df.loc[ablation_name].idxmax()
        best_value = auc_df.loc[ablation_name].max()
        if not np.isnan(best_value):
            print(f"  {ablation_name}: {best_score} (AUC = {best_value:.3f})")
    
    return auc_df

from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_precision_recall_curves_by_score(df_ablations, ablations, score_columns, data_description, colors, out_dir, iou_threshold=0.5, coplot_lbcv=False):
    """
    Plot Precision-Recall curves for each score type across all ablations.
    If coplot_lbcv=True, plot LBCV curves alongside PBCV curves.
    """
    for score_key, score_display_name in score_columns.items():
        plt.figure(figsize=(10, 8))

        for ablation_idx, ablation in enumerate(ablations):
            df_ablation = df_ablations[ablation_idx]
            if df_ablation.empty:
                print(f"No data for ablation: {ablation}")
                continue

            ablation_label = data_description[ablation].get("label", ablation)
            color = colors[ablation_idx]

            for method in ["PBCV", "LBCV"] if coplot_lbcv else ["PBCV"]:
                detect_col = f"detected_{method}"
                iou_col = f"{method}_IOU"

                if detect_col not in df_ablation.columns or score_key not in df_ablation.columns or iou_col not in df_ablation.columns:
                    continue

                df_detected = df_ablation[df_ablation[detect_col] == True].copy()
                if df_detected.empty:
                    continue

                df_detected['ground_truth'] = (df_detected[iou_col] > iou_threshold).astype(int)
                if len(df_detected['ground_truth'].unique()) < 2:
                    continue

                scores = df_detected[score_key].values
                labels = df_detected['ground_truth'].values

                valid_mask = ~np.isnan(scores)
                scores = scores[valid_mask]
                labels = labels[valid_mask]
                if len(scores) == 0:
                    continue

                try:
                    precision, recall, _ = precision_recall_curve(labels, scores)
                    ap_score = average_precision_score(labels, scores)
                    line_style = '-' if method == "PBCV" else '--'

                    plt.plot(recall, precision, color=color, linestyle=line_style, linewidth=2,
                             label=f'{ablation_label} ({method}) (AP = {ap_score:.2f})', alpha=0.85)
                except Exception as e:
                    print(f"Error computing PR curve for {ablation} ({method}), {score_key}: {e}")
                    continue

        # Formatting
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title(f'Precision-Recall Curves for {score_display_name}\n(TP: IOU > {iou_threshold:.2f}, FP: IOU ≤ {iou_threshold:.2f})', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="lower left", fontsize=12)
        plt.tight_layout()

        # Save the plot
        lbcv_suffix = "_withLBCV" if coplot_lbcv else ""
        save_filename = f"PR_Curves_{score_key.upper()}_{ablations[0].split('_')[-1]}{lbcv_suffix}.png"
        save_path = os.path.join(out_dir, save_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved PR curve plot: {save_path}")

def plot_pbcv_iou_vs_score(df_ablations, ablations, score_columns, data_description, colors, out_dir):
    """
    Plot PBCV IOU vs detection score for each score type across all ablations.
    Shows the relationship between detection confidence scores and actual detection quality (IOU).
    """
    for score_key, score_display_name in score_columns.items():
        plt.figure(figsize=(12, 8))
        
        for ablation_idx, ablation in enumerate(ablations):
            df_ablation = df_ablations[ablation_idx]
            
            if df_ablation.empty:
                print(f"No data for ablation: {ablation}")
                continue
                
            # Check if required columns exist
            iou_col = 'PBCV_IOU'
            required_cols = [score_key, 'detected_PBCV', iou_col]
            
            # Check for alternative IOU column names if PBCV_IOU doesn't exist
            if iou_col not in df_ablation.columns:
                possible_iou_cols = ['LBCV_IOU', 'IOU', 'iou']
                for col in possible_iou_cols:
                    if col in df_ablation.columns:
                        iou_col = col
                        break
                        
            if iou_col not in df_ablation.columns:
                print(f"IOU column not found in {ablation}. Skipping.")
                continue
                
            missing_cols = [col for col in required_cols if col not in df_ablation.columns]
            if missing_cols:
                print(f"Missing columns {missing_cols} in {ablation}. Skipping.")
                continue
            
            # Filter data where PBCV detection was attempted (detected_PBCV == True)
            df_pbcv_detected = df_ablation[df_ablation['detected_PBCV'] == True].copy()
            
            if len(df_pbcv_detected) == 0:
                print(f"No PBCV detections in {ablation}. Skipping.")
                continue
            
            # Get scores and IOU values
            scores = df_pbcv_detected[score_key].values
            iou_values = df_pbcv_detected[iou_col].values
            
            # Remove NaN values
            valid_mask = ~np.isnan(scores) & ~np.isnan(iou_values)
            scores = scores[valid_mask]
            iou_values = iou_values[valid_mask]
            
            if len(scores) == 0:
                print(f"No valid score-IOU pairs for {score_key} in {ablation}. Skipping.")
                continue
            
            # Get ablation label and color
            ablation_label = data_description[ablation].get("label", ablation)
            color = colors[ablation_idx]
            
            # Create scatter plot
            plt.scatter(scores, iou_values, color=color, alpha=0.6, s=20, 
                       label=ablation_label, edgecolors='none')
            
            # Add trend line (optional - can be commented out if too cluttered)
            try:
                # Fit a simple polynomial trend line
                z = np.polyfit(scores, iou_values, 1)
                p = np.poly1d(z)
                sorted_scores = np.sort(scores)
                plt.plot(sorted_scores, p(sorted_scores), color=color, 
                        linestyle='--', linewidth=1.5, alpha=0.8)
            except:
                pass  # Skip trend line if fitting fails
        
        # Add horizontal line at IOU = 0.5 (common threshold for "good" detection)
        plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                   label='IOU = 0.5 (Good Detection Threshold)')
        
        # Formatting
        plt.xlabel(f'{score_display_name}', fontsize=14)
        plt.ylabel('PBCV IOU', fontsize=14)
        plt.title(f'PBCV IOU vs {score_display_name}', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12, loc='best')
        plt.tight_layout()
        
        # Set reasonable axis limits
        plt.ylim(-0.05, 1.05)  # IOU ranges from 0 to 1
        
        # Save the plot
        save_filename = f"PBCV_IOU_vs_{score_key.upper()}_{ablations[0].split('_')[-1]}.png"
        save_path = os.path.join(out_dir, save_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved PBCV IOU vs Score plot: {save_path}")

def plot_ROC_PR_curves_by_pose_error_thresholds(
    df_ablations,
    ablations,
    score_columns,
    data_description,
    colors,
    out_dir,
    pose_error_thresholds={"x": 0.02, "y": 0.02, "z": 0.02, "a": 5.0, "b": 5.0, "c": 5.0},
    coplot_LBCV=False
):
    """
    Plot ROC and Precision-Recall curves using absolute pose error thresholds to define TP/FP.
    Optionally co-plot LBCV if coplot_LBCV is True. Always plots a datapoint even if only positives or only negatives.
    """
    for score_key, score_display_name in score_columns.items():
        fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
        fig_pr, ax_pr = plt.subplots(figsize=(10, 8))

        for ablation_idx, ablation in enumerate(ablations):
            df_ablation = df_ablations[ablation_idx]
            if df_ablation.empty or score_key not in df_ablation.columns:
                continue

            for method in ['PBCV', 'LBCV'] if coplot_LBCV else ['PBCV']:
                detect_col = f'detected_{method}'
                prefix = f'pose_error_{method}_'

                if detect_col not in df_ablation.columns:
                    continue

                df_detected = df_ablation[df_ablation[detect_col] == True].copy()
                if df_detected.empty:
                    continue

                err_mask = np.ones(len(df_detected), dtype=bool)
                for key in ['x', 'y', 'z', 'a', 'b', 'c']:
                    col = prefix + key
                    if col not in df_detected.columns:
                        print(f"Missing column {col} in {ablation}, skipping...")
                        err_mask[:] = False
                        break
                    threshold = pose_error_thresholds.get(key, np.inf)
                    err_mask &= np.abs(df_detected[col]) < threshold

                df_detected['ground_truth'] = err_mask.astype(int)

                scores = df_detected[score_key].values
                labels = df_detected['ground_truth'].values

                valid_mask = ~np.isnan(scores)
                scores = scores[valid_mask]
                labels = labels[valid_mask]

                if len(scores) == 0:
                    continue

                ablation_label = data_description[ablation].get("label", ablation)
                color = colors[ablation_idx]
                linestyle = '--' if method == 'LBCV' else '-'
                alpha = 0.6 if method == 'LBCV' else 0.9

                try:
                    if len(np.unique(labels)) < 2:
                        print(f"Only one class in {ablation} for {score_key} ({method}). Adding placeholder.")

                        if np.all(labels == 1):
                            # All TP
                            ax_roc.plot([0.0], [1.0], marker='o', color=color, label=f'{ablation_label} - {method} (AUC = 1.00)', linestyle=linestyle, alpha=alpha)
                            ax_pr.plot([1.0], [1.0], marker='o', color=color, label=f'{ablation_label} - {method} (AP = 1.00)', linestyle=linestyle, alpha=alpha)
                        else:
                            # All TN or all FP
                            ax_roc.plot([0.0], [0.0], marker='o', color=color, label=f'{ablation_label} - {method} (AUC = 0.00)', linestyle=linestyle, alpha=alpha)
                            ax_pr.plot([0.0], [0.0], marker='o', color=color, label=f'{ablation_label} - {method} (AP = 0.00)', linestyle=linestyle, alpha=alpha)
                        continue

                    # Compute and plot full curves
                    fpr, tpr, _ = roc_curve(labels, scores)
                    roc_auc = auc(fpr, tpr)

                    precision, recall, _ = precision_recall_curve(labels, scores)
                    pr_auc = average_precision_score(labels, scores)

                    ax_roc.plot(fpr, tpr, linestyle=linestyle, color=color, linewidth=2,
                                label=f'{ablation_label} - {method} (AUC = {roc_auc:.2f})', alpha=alpha)
                    ax_pr.plot(recall, precision, linestyle=linestyle, color=color, linewidth=2,
                               label=f'{ablation_label} - {method} (AP = {pr_auc:.2f})', alpha=alpha)

                except Exception as e:
                    print(f"Error in {ablation}, {score_key} ({method}): {e}")
                    continue

        # Finalize ROC plot
        ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate', fontsize=14)
        ax_roc.set_ylabel('True Positive Rate', fontsize=14)
        ax_roc.set_title(f'ROC Curves for {score_display_name}\n(TP: Pose error < thresholds)', fontsize=16)
        ax_roc.legend(loc="lower right", fontsize=12)
        ax_roc.grid(True, alpha=0.3)
        fig_roc.tight_layout()
        roc_path = os.path.join(out_dir, f"ROC_Curves_{score_key.upper()}_PoseError_{ablations[0].split('_')[-1]}.png")
        fig_roc.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close(fig_roc)
        print(f"Saved ROC plot: {roc_path}")

        # Finalize PR plot
        ax_pr.set_xlim([0.0, 1.0])
        ax_pr.set_ylim([0.0, 1.05])
        ax_pr.set_xlabel('Recall', fontsize=14)
        ax_pr.set_ylabel('Precision', fontsize=14)
        ax_pr.set_title(f'Precision-Recall Curves for {score_display_name}\n(TP: Pose error < thresholds)', fontsize=16)
        ax_pr.legend(loc="lower left", fontsize=12)
        ax_pr.grid(True, alpha=0.3)
        fig_pr.tight_layout()
        pr_path = os.path.join(out_dir, f"PR_Curves_{score_key.upper()}_PoseError_{ablations[0].split('_')[-1]}.png")
        fig_pr.savefig(pr_path, dpi=300, bbox_inches='tight')
        plt.close(fig_pr)
        print(f"Saved PR plot: {pr_path}")


def plot_ROC_PR_curves_by_IOU(
    df_ablations,
    ablations,
    score_columns,
    data_description,
    colors,
    out_dir,
    iou_threshold=0.5,
    coplot_LBCV=False
):
    """
    Plot ROC and Precision-Recall curves using IOU > threshold to define TP/FP.
    Optionally co-plot LBCV if coplot_LBCV is True. Always plots a datapoint even if only positives or only negatives.
    """
    for score_key, score_display_name in score_columns.items():
        fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
        fig_pr, ax_pr = plt.subplots(figsize=(10, 8))

        for ablation_idx, ablation in enumerate(ablations):
            df_ablation = df_ablations[ablation_idx]
            if df_ablation.empty or score_key not in df_ablation.columns:
                continue

            for method in ['PBCV', 'LBCV'] if coplot_LBCV else ['PBCV']:
                detect_col = f'detected_{method}'
                iou_col = f'{method}_IOU'

                if detect_col not in df_ablation.columns or iou_col not in df_ablation.columns:
                    print(f"Missing {detect_col} or {iou_col} in {ablation}. Skipping.")
                    continue

                df_detected = df_ablation[df_ablation[detect_col] == True].copy()
                if df_detected.empty:
                    continue

                df_detected['ground_truth'] = (df_detected[iou_col] > iou_threshold).astype(int)

                scores = df_detected[score_key].values
                labels = df_detected['ground_truth'].values
                valid_mask = ~np.isnan(scores)
                scores = scores[valid_mask]
                labels = labels[valid_mask]

                if len(scores) == 0:
                    continue

                ablation_label = data_description[ablation].get("label", ablation)
                color = colors[ablation_idx]
                linestyle = '--' if method == 'LBCV' else '-'
                alpha = 0.6 if method == 'LBCV' else 0.9

                try:
                    if len(np.unique(labels)) < 2:
                        # Only positives or only negatives
                        print(f"Only one class in {ablation} for {score_key} ({method}). Adding placeholder.")

                        if np.all(labels == 1):
                            # All TP: perfect classifier
                            ax_roc.plot([0.0], [1.0], marker='o', color=color, label=f'{ablation_label} - {method} (AUC = 1.00)', linestyle=linestyle, alpha=alpha)
                            ax_pr.plot([1.0], [1.0], marker='o', color=color, label=f'{ablation_label} - {method} (AP = 1.00)', linestyle=linestyle, alpha=alpha)
                        else:
                            # All TN or all FP: meaningless classifier
                            ax_roc.plot([0.0], [0.0], marker='o', color=color, label=f'{ablation_label} - {method} (AUC = 0.00)', linestyle=linestyle, alpha=alpha)
                            ax_pr.plot([0.0], [0.0], marker='o', color=color, label=f'{ablation_label} - {method} (AP = 0.00)', linestyle=linestyle, alpha=alpha)
                        continue

                    # Valid ROC/PR computation
                    fpr, tpr, _ = roc_curve(labels, scores)
                    roc_auc = auc(fpr, tpr)

                    precision, recall, _ = precision_recall_curve(labels, scores)
                    pr_auc = average_precision_score(labels, scores)

                    ax_roc.plot(fpr, tpr, linestyle=linestyle, color=color, linewidth=2,
                                label=f'{ablation_label} - {method} (AUC = {roc_auc:.2f})', alpha=alpha)
                    ax_pr.plot(recall, precision, linestyle=linestyle, color=color, linewidth=2,
                               label=f'{ablation_label} - {method} (AP = {pr_auc:.2f})', alpha=alpha)

                except Exception as e:
                    print(f"Error in {ablation}, {score_key} ({method}): {e}")
                    continue

        # Finalize ROC plot
        ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate', fontsize=14)
        ax_roc.set_ylabel('True Positive Rate', fontsize=14)
        ax_roc.set_title(f'ROC Curves for {score_display_name}\n(TP: IOU > {iou_threshold})', fontsize=16)
        ax_roc.legend(loc="lower right", fontsize=12)
        ax_roc.grid(True, alpha=0.3)
        fig_roc.tight_layout()
        roc_path = os.path.join(out_dir, f"ROC_Curves_{score_key.upper()}_IOU_{ablations[0].split('_')[-1]}.png")
        fig_roc.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close(fig_roc)
        print(f"Saved ROC plot: {roc_path}")

        # Finalize PR plot
        ax_pr.set_xlim([0.0, 1.0])
        ax_pr.set_ylim([0.0, 1.05])
        ax_pr.set_xlabel('Recall', fontsize=14)
        ax_pr.set_ylabel('Precision', fontsize=14)
        ax_pr.set_title(f'Precision-Recall Curves for {score_display_name}\n(TP: IOU > {iou_threshold})', fontsize=16)
        ax_pr.legend(loc="lower left", fontsize=12)
        ax_pr.grid(True, alpha=0.3)
        fig_pr.tight_layout()
        pr_path = os.path.join(out_dir, f"PR_Curves_{score_key.upper()}_IOU_{ablations[0].split('_')[-1]}.png")
        fig_pr.savefig(pr_path, dpi=300, bbox_inches='tight')
        plt.close(fig_pr)
        print(f"Saved PR plot: {pr_path}")

def plot_max_pose_error_vs_score_thresholds(
    df_ablations,
    ablations,
    score_key,
    data_description,
    colors,
    out_dir,
    methods=['PBCV', 'LBCV'],
    num_thresholds=50
):
    """
    Create 2x3 subplot of max pose error vs score threshold for each pose component (x,y,z,a,b,c).
    Co-plots LBCV and PBCV across all ablations. Automatically computes score threshold range and infers pose error keys.

    Parameters:
    - df_ablations: list of DataFrames corresponding to ablation sets
    - ablations: list of ablation names
    - score_key: str, e.g., 'keypoint_residual_score'
    - data_description: dict mapping ablation names to labels
    - colors: list of colors (one per ablation)
    - out_dir: output directory
    - methods: list like ['PBCV', 'LBCV']
    - num_thresholds: number of score thresholds to evaluate
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import re

    # --- Detect pose components (x, y, z, a, b, c) present across methods ---
    pose_components = ['x', 'y', 'z', 'a', 'b', 'c']  # default expected
    found_components = set()

    sample_df = df_ablations[0]
    for method in methods:
        pattern = re.compile(f"pose_error_{method}_(\\w+)")
        for col in sample_df.columns:
            match = pattern.match(col)
            if match:
                found_components.add(match.group(1))

    pose_components = sorted(found_components)
    if not pose_components:
        print("No pose error columns found in dataset. Skipping plot.")
        return

    # --- Automatically determine score threshold range ---
    all_scores = []
    for df_ablation in df_ablations:
        for method in methods:
            detect_col = f'detected_{method}'
            if detect_col in df_ablation.columns and score_key in df_ablation.columns:
                scores = df_ablation[df_ablation[detect_col] == True][score_key].dropna().values
                all_scores.append(scores)

    if not all_scores:
        print(f"No valid scores found for {score_key}. Skipping plot.")
        return

    all_scores_concat = np.concatenate(all_scores)
    score_min, score_max = np.nanmin(all_scores_concat), np.nanmax(all_scores_concat)
    score_thresholds = np.linspace(score_min, score_max, num_thresholds)

    # --- Create plot ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, component in enumerate(pose_components):
        if idx >= 6:
            break  # 2x3 layout

        ax = axes[idx]
        ax.set_title(f'Max {component} Pose Error vs {score_key}', fontsize=14)
        ax.set_xlabel(f'{score_key} Threshold', fontsize=12)
        ax.set_ylabel(f'Max |{component}| (m or deg)', fontsize=12)
        ax.grid(True, alpha=0.3)

        for ablation_idx, ablation in enumerate(ablations):
            df_ablation = df_ablations[ablation_idx]
            if df_ablation.empty or score_key not in df_ablation.columns:
                continue

            for method in methods:
                detect_col = f'detected_{method}'
                pose_col = f'pose_error_{method}_{component}'

                if detect_col not in df_ablation.columns or pose_col not in df_ablation.columns:
                    continue

                df_detected = df_ablation[df_ablation[detect_col] == True].copy()
                if df_detected.empty:
                    continue

                ablation_label = data_description[ablation].get("label", ablation)
                linestyle = '-' if method == 'PBCV' else '--'
                alpha = 0.9 if method == 'PBCV' else 0.6

                max_errors = []
                for thresh in score_thresholds:
                    subset = df_detected[df_detected[score_key] >= thresh]
                    if subset.empty or pose_col not in subset.columns:
                        max_errors.append(np.nan)
                    else:
                        max_val = np.abs(subset[pose_col]).max()
                        max_errors.append(max_val)

                ax.plot(score_thresholds, max_errors,
                        label=f'{ablation_label} - {method}',
                        color=colors[ablation_idx],
                        linestyle=linestyle,
                        linewidth=2,
                        alpha=alpha)

    # Global legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=12)

    fig.suptitle(f'Max Pose Error vs Score Threshold: {score_key}', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(out_dir, f"MaxPoseError_vs_{score_key.upper()}_Thresholds.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {save_path}")

# Re-defining the updated function after kernel reset

def plot_dual_precision_recall_figure_by_IOU_and_PoseError(
    df_ablations,
    ablations,
    score_columns,
    data_description,
    colors,
    out_dir,
    iou_threshold=0.5,
    pose_error_thresholds={"x": 0.025, "y": 0.025, "z": 0.050, "a": 15.0, "b": 15.0, "c": 30.0},
    coplot_LBCV=True,
    label_rename_dict=None
):
    from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    label_map = {"x": "X", "y": "Y", "z": "Z", "a": "Roll", "b": "Yaw", "c": "Pitch"}

    for score_key, score_display_name in score_columns.items():
        pr_curves_iou = []
        pr_curves_pose = []

        for ablation_idx, ablation in enumerate(ablations):
            df_ablation = df_ablations[ablation_idx]
            if df_ablation.empty:
                continue

            for method in ['PBCV', 'LBCV'] if coplot_LBCV else ['PBCV']:
                # Override score key for method
                if method == "LBCV":
                    method_score_key = "LBCV_mean_mask_score"
                elif method == "PBCV":
                    method_score_key = "image_similarity_score"
                else:
                    continue

                if method_score_key not in df_ablation.columns:
                    continue

                detect_col = f'detected_{method}'
                iou_col = f'{method}_IOU'
                prefix = f'pose_error_{method}_'

                if detect_col not in df_ablation.columns or iou_col not in df_ablation.columns:
                    continue

                ablation_label = data_description[ablation].get("label", ablation)
                color = colors[ablation_idx]
                linestyle = '-' if method == 'PBCV' else '--'

                label_base = f"{ablation_label} ({method})"
                if label_rename_dict and label_base in label_rename_dict:
                    label_base = label_rename_dict[label_base]

                # --- IOU-based PR ---
                df_detected_iou = df_ablation[df_ablation[detect_col] == True].copy()
                if df_detected_iou.empty:
                    continue
                df_detected_iou['ground_truth'] = (df_detected_iou[iou_col] > iou_threshold).astype(int)

                scores = df_detected_iou[method_score_key].values
                labels = df_detected_iou['ground_truth'].values
                valid_mask = ~np.isnan(scores)
                scores = scores[valid_mask]
                labels = labels[valid_mask]

                if len(np.unique(labels)) > 1:
                    precision, recall, _ = precision_recall_curve(labels, scores)
                    ap_score = average_precision_score(labels, scores)
                    fpr, tpr, _ = roc_curve(labels, scores)
                    auc_score = auc(fpr, tpr)
                    label = f"{label_base} (AP={ap_score:.2f}, AUC={auc_score:.2f})"
                else:
                    ap_score = 1.0 if np.all(labels == 1) else 0.0
                    auc_score = ap_score
                    label = f"{label_base} (AP={ap_score:.2f}, AUC={auc_score:.2f})"
                    recall, precision = [1.0], [ap_score]
                pr_curves_iou.append((label, recall, precision, color, linestyle))

                # --- Pose error-based PR ---
                df_detected_pose = df_ablation[df_ablation[detect_col] == True].copy()
                if df_detected_pose.empty:
                    continue

                err_mask = np.ones(len(df_detected_pose), dtype=bool)
                for key, threshold in pose_error_thresholds.items():
                    col = prefix + key
                    if col not in df_detected_pose.columns:
                        err_mask[:] = False
                        break
                    err_mask &= np.abs(df_detected_pose[col]) < threshold
                df_detected_pose['ground_truth'] = err_mask.astype(int)

                scores = df_detected_pose[method_score_key].values
                labels = df_detected_pose['ground_truth'].values
                valid_mask = ~np.isnan(scores)
                scores = scores[valid_mask]
                labels = labels[valid_mask]

                if len(np.unique(labels)) > 1:
                    precision, recall, _ = precision_recall_curve(labels, scores)
                    ap_score = average_precision_score(labels, scores)
                    fpr, tpr, _ = roc_curve(labels, scores)
                    auc_score = auc(fpr, tpr)
                    label = f"{label_base} (AP={ap_score:.2f}, AUC={auc_score:.2f})"
                else:
                    ap_score = 1.0 if np.all(labels == 1) else 0.0
                    auc_score = ap_score
                    label = f"{label_base} (AP={ap_score:.2f}, AUC={auc_score:.2f})"
                    recall, precision = [1.0], [ap_score]
                pr_curves_pose.append((label, recall, precision, color, linestyle))

        fig, axs = plt.subplots(1, 2, figsize=(26, 10), sharey=False)
        fig.suptitle(f"Precision-recall curves for {score_display_name}", fontsize=24)
        ax_iou, ax_pose = axs

        for (label, recall, precision, color, linestyle) in pr_curves_iou:
            ax_iou.plot(recall, precision, label=label, linewidth=2, linestyle=linestyle, color=color)
        ax_iou.set_title(f"True positive: IOU > {iou_threshold}", fontsize=20)
        ax_iou.set_xlabel("Recall", fontsize=20)
        ax_iou.set_ylabel("Precision", fontsize=20)
        ax_iou.tick_params(axis='both', labelsize=20)
        ax_iou.grid(True, alpha=0.3)

        thresh_parts = []
        for k, v in pose_error_thresholds.items():
            name = label_map[k]
            unit = " mm" if k in {"x", "y", "z"} else "°"
            val = int(round(v * 1000)) if unit.strip() == "mm" else int(round(v))
            thresh_parts.append(f"{name}<{val}{unit}")
        pose_thresh_str = ", ".join(thresh_parts)

        for (label, recall, precision, color, linestyle) in pr_curves_pose:
            ax_pose.plot(recall, precision, label=label, linewidth=2, linestyle=linestyle, color=color)
        ax_pose.set_title(f"True positive: Pose error thresholds\n({pose_thresh_str})", fontsize=20)
        ax_pose.set_xlabel("Recall", fontsize=20)
        ax_pose.set_ylabel("Precision", fontsize=20)
        ax_pose.tick_params(axis='both', labelsize=20)
        ax_pose.grid(True, alpha=0.3)

        # Add compact 2-column legends inside plot
        ax_iou.legend(loc='best', fontsize=14, ncol=2, frameon=True)
        ax_pose.legend(loc='best', fontsize=14, ncol=2, frameon=True)

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        save_path = os.path.join(out_dir, f"Dual_PR_Curves_{score_key.replace(' ', '_')}.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved dual PR curve figure: {save_path}")

        # Save individual plots as well
        save_individual_PR_curves(
            pr_curves_iou=pr_curves_iou,
            pr_curves_pose=pr_curves_pose,
            score_key=score_key,
            score_display_name=score_display_name,
            iou_threshold=iou_threshold,
            pose_error_thresholds=pose_error_thresholds,
            out_dir=out_dir
        )



import os
import numpy as np
import matplotlib.pyplot as plt

def save_individual_PR_curves(pr_curves_iou, pr_curves_pose, score_key, score_display_name, iou_threshold, pose_error_thresholds, out_dir):
    label_map = {"x": "X", "y": "Y", "z": "Z", "a": "Roll", "b": "Yaw", "c": "Pitch"}
    os.makedirs(out_dir, exist_ok=True)

    # --- IOU-based PR Curve ---
    fig_iou, ax_iou = plt.subplots(figsize=(12, 8))
    for (label, recall, precision, color, linestyle) in pr_curves_iou:
        ax_iou.plot(recall, precision, label=label, linewidth=2, linestyle=linestyle, color=color)
    ax_iou.set_title(f"PR Curve (IOU > {iou_threshold}) for {score_display_name}", fontsize=16)
    ax_iou.set_xlabel("Recall", fontsize=14)
    ax_iou.set_ylabel("Precision", fontsize=14)
    ax_iou.grid(True, alpha=0.3)
    if pr_curves_iou:
        ax_iou.legend(loc="lower left", fontsize=10, ncol=2)
    fig_iou.tight_layout()
    save_path_iou = os.path.join(out_dir, f"PR_Curve_IOU_{score_key.replace(' ', '_')}.png")
    fig_iou.savefig(save_path_iou, dpi=300, bbox_inches='tight')
    plt.close(fig_iou)

    # --- PoseError-based PR Curve ---
    fig_pose, ax_pose = plt.subplots(figsize=(12, 8))
    for (label, recall, precision, color, linestyle) in pr_curves_pose:
        # ax_pose.plot(recall, precision, label=label, linewidth=2, linestyle=linestyle, color=color)
        # Only show AP (remove AUC from label)
        label_ap_only = label.split(",")[0] + ")"
        ax_pose.plot(recall, precision, label=label_ap_only, linewidth=2, linestyle=linestyle, color=color)

    thresh_parts = []
    for k, v in pose_error_thresholds.items():
        name = label_map[k]
        unit = " mm" if k in {"x", "y", "z"} else "°"
        val = int(round(v * 1000)) if unit.strip() == "mm" else int(round(v))
        thresh_parts.append(f"{name}<{val}{unit}")
    pose_thresh_str = ", ".join(thresh_parts)

    ax_pose.set_title(f"Precision-Recall Curve for Mean Mask Score (LBCV) and Image Similarity Score (PBCV) \nTrue Positive: ({pose_thresh_str}) ", fontsize=16)
    ax_pose.set_xlabel("Recall", fontsize=14)
    ax_pose.set_ylabel("Precision", fontsize=14)
    ax_pose.grid(True, alpha=0.3)
    if pr_curves_pose:
        ax_pose.legend(loc="lower left", fontsize=10, ncol=2)
    fig_pose.tight_layout()
    save_path_pose = os.path.join(out_dir, f"PR_Curve_PoseError_{score_key.replace(' ', '_')}.png")
    fig_pose.savefig(save_path_pose, dpi=300, bbox_inches='tight')
    plt.close(fig_pose)

    return save_path_iou, save_path_pose

import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import numpy as np

def plot_ROC_and_PR_side_by_side_for_methods(
    df_ablations,
    ablations,
    data_description,
    colors,
    out_dir,
    iou_threshold=0.5,
    pose_error_thresholds={"x": 0.025, "y": 0.025, "z": 0.050, "a": 15.0, "b": 15.0, "c": 30.0},
    coplot_LBCV=True,
    label_rename_dict=None
):
    """
    For each ablation and method (PBCV, LBCV), compute and plot:
    - ROC Curve (left) with AUC in legend
    - Precision-Recall Curve (right) with AP in legend
    Ground truth is defined by pose error thresholds.
    """
    label_map = {"x": "X", "y": "Y", "z": "Z", "a": "Roll", "b": "Yaw", "c": "Pitch"}

    for ablation_idx, ablation in enumerate(ablations):
        df_ablation = df_ablations[ablation_idx]
        if df_ablation.empty:
            continue

        ablation_label = data_description[ablation].get("label", ablation)
        color = colors[ablation_idx]

        for method in ['PBCV', 'LBCV'] if coplot_LBCV else ['PBCV']:
            if method == "LBCV":
                detect_col = 'detected_LBCV'
                score_col = 'LBCV_mean_mask_score'
            else:
                detect_col = 'detected_PBCV'
                score_col = 'image_similarity_score'

            prefix = f'pose_error_{method}_'
            if detect_col not in df_ablation.columns or score_col not in df_ablation.columns:
                continue

            df_detected = df_ablation[df_ablation[detect_col] == True].copy()
            if df_detected.empty:
                continue

            # Compute ground truth
            err_mask = np.ones(len(df_detected), dtype=bool)
            for key, threshold in pose_error_thresholds.items():
                col = prefix + key
                if col not in df_detected.columns:
                    err_mask[:] = False
                    break
                err_mask &= np.abs(df_detected[col]) < threshold
            df_detected['ground_truth'] = err_mask.astype(int)

            scores = df_detected[score_col].values
            labels = df_detected['ground_truth'].values
            valid_mask = ~np.isnan(scores)
            scores = scores[valid_mask]
            labels = labels[valid_mask]

            if len(np.unique(labels)) < 2:
                print(f"Only one class in {ablation} ({method}). Skipping.")
                continue

            # ROC & PR
            fpr, tpr, _ = roc_curve(labels, scores)
            roc_auc = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(labels, scores)
            ap_score = average_precision_score(labels, scores)

            linestyle = '-' if method == 'PBCV' else '--'
            alpha = 0.9 if method == 'PBCV' else 0.6

            method_label = f"{ablation_label} ({method})"
            if label_rename_dict and method_label in label_rename_dict:
                method_label = label_rename_dict[method_label]

            # Plotting
            fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(18, 8))
            fig.suptitle(f"ROC & PR Curves for {method_label}", fontsize=22)

            ax_roc.plot(fpr, tpr, linestyle=linestyle, color=color, linewidth=2,
                        label=f"{method_label} (AUC = {roc_auc:.2f})", alpha=alpha)
            ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label="Random")
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel("False Positive Rate", fontsize=14)
            ax_roc.set_ylabel("True Positive Rate", fontsize=14)
            ax_roc.set_title("ROC Curve", fontsize=16)
            ax_roc.grid(True, alpha=0.3)
            ax_roc.legend(loc="lower right", fontsize=12)

            ax_pr.plot(recall, precision, linestyle=linestyle, color=color, linewidth=2,
                       label=f"{method_label} (AP = {ap_score:.2f})", alpha=alpha)
            ax_pr.set_xlim([0.0, 1.0])
            ax_pr.set_ylim([0.0, 1.05])
            ax_pr.set_xlabel("Recall", fontsize=14)
            ax_pr.set_ylabel("Precision", fontsize=14)

            thresh_parts = []
            for k, v in pose_error_thresholds.items():
                name = label_map[k]
                unit = " mm" if k in {"x", "y", "z"} else "°"
                val = int(round(v * 1000)) if unit.strip() == "mm" else int(round(v))
                thresh_parts.append(f"{name}<{val}{unit}")
            pose_thresh_str = ", ".join(thresh_parts)

            ax_pr.set_title(f"PR Curve\n(TP: {pose_thresh_str})", fontsize=16)
            ax_pr.grid(True, alpha=0.3)
            ax_pr.legend(loc="lower left", fontsize=12)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            os.makedirs(out_dir, exist_ok=True)
            save_name = f"ROC_PR_{method}_{ablation.replace('/', '_')}.png"
            save_path = os.path.join(out_dir, save_name)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved ROC+PR plot: {save_path}")


if __name__ == "__main__":
    # # --- Generate Plots ---
    # for ablation_idx, ablation in enumerate(ablations):
    #     df_ablation = df_ablations[ablation_idx]
    #     if df_ablation.empty:
    #         print(f"No data for ablation: {ablation}")
    #         continue

    #     ablation_label = data_description[ablation].get("label", ablation)
    #     ablation_color = colors[ablation_idx]

    #     for score_key, score_display_name in score_columns.items():
    #         save_prefix = f"MAE_vs_{score_key.upper()}_Thresholds"
    #         plot_mae_vs_score_threshold(
    #             df_ablation, ablation_label, ablation_color, score_key, score_display_name, save_prefix, out_dir
    #         )

    # # --- Generate ROC Plots ---
    # print("\nGenerating ROC curve plots...")
    # plot_roc_curves_by_score(df_ablations, ablations, score_columns, data_description, colors, out_dir)
    
    # # --- Generate AUC Table ---
    # print("\nGenerating AUC values table...")
    # auc_table = compute_auc_table(df_ablations, ablations, score_columns, data_description, out_dir)

    # # --- Compute and Display AUC Table ---
    # print("\nComputing and displaying AUC values table...")
    # compute_auc_table(df_ablations, ablations, score_columns, data_description, out_dir)


    # # --- Generate Detection Rate Plots ---
    # for score_key, score_display_name in score_columns.items():
    #     plot_pbcv_detection_rate_vs_threshold(df_ablations, ablations, score_key, score_display_name, out_dir)

    # # --- Generate PBCV Detection Rate Plots ---
    # print("\nGenerating PBCV Detection Rate plots...")
    # for score_key, score_display_name in score_columns.items():
    #     plot_pbcv_detection_rate_vs_threshold(df_ablations, ablations, score_key, score_display_name, out_dir)

    # print("\nGenerating Precision-Recall curve plots...")
    # plot_precision_recall_curves_by_score(
    #     df_ablations, ablations, score_columns, data_description, colors, out_dir,
    #     iou_threshold=0.5,
    #     coplot_lbcv=True
    # )
    
    # print("\nGenerating PBCV IOU vs Score plots...")
    # plot_pbcv_iou_vs_score(df_ablations, ablations, score_columns, data_description, colors, out_dir)

    print("\nGenerating ROC curves based on pose error thresholds...")
    pose_err_thresh = {"x": 0.025, "y": 0.025, "z": 0.050, "a": 15.0, "b": 15.0, "c": 30.0}
    plot_ROC_PR_curves_by_pose_error_thresholds(
        df_ablations, ablations, score_columns, data_description, colors, out_dir, pose_error_thresholds=pose_err_thresh, coplot_LBCV=True
    )

    print("\nGenerating ROC curves based on IOU...")
    plot_ROC_PR_curves_by_IOU(
        df_ablations, ablations, score_columns, data_description, colors, out_dir, iou_threshold=0.75, coplot_LBCV=True
    )

    print("\nGenerating pose error plots")
    for score_key in score_columns:
        plot_max_pose_error_vs_score_thresholds(
            df_ablations=df_ablations,
            ablations=ablations,
            score_key=score_key,
            data_description=data_description,
            colors=colors,
            out_dir=out_dir,
            num_thresholds=100
        )

    plot_dual_precision_recall_figure_by_IOU_and_PoseError(
        df_ablations=df_ablations,
        ablations=ablations,
        score_columns=score_columns,
        data_description=data_description,
        colors=colors,
        out_dir=out_dir,
        iou_threshold=0.50,
        pose_error_thresholds={"x": 0.010, "y": 0.010, "z": 0.050, "a": 10, "b": 15.0, "c": 15.0},
        coplot_LBCV=True,
        label_rename_dict=label_rename_dict
    )

    # plot_ROC_and_PR_side_by_side_for_methods(
    #     df_ablations=df_ablations,
    #     ablations=ablations,
    #     data_description=data_description,
    #     colors=colors,
    #     out_dir=out_dir,
    #     iou_threshold=0.50,
    #     pose_error_thresholds={"x": 0.010, "y": 0.010, "z": 0.050, "a": 10, "b": 15.0, "c": 15.0},
    #     coplot_LBCV=True,
    #     label_rename_dict=label_rename_dict
    # )