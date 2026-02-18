import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os, glob
import numpy as np
from scipy import stats


def visualize(task):
    pattern = f'/gpfs/work1/0/prjs1425/shark/OS_seg/OS_000*/*/{task}/*.nii.gz.csv'
    matching_paths = glob.glob(pattern)

    # exclude

    print(f"Found {len(matching_paths)} matching files:")
    all_data = []
    for path in matching_paths:
        df = pd.read_csv(path)

        # check if df has more than one row
        if df.shape[0] > 1:
            print(f"File {path} has {len(df)} rows. Here are the first few rows:")
            print(df.head())

        all_data.append(df)

        

    # Combine all DataFrames
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\nCombined DataFrame shape: {combined_df.shape}")

    # Save the combined CSV
    output_path = f'/gpfs/work1/0/prjs1425/shark/preprocessing/dice_analysis/combined_{task}.csv'
    combined_df.to_csv(output_path, index=False)

    # plot
    # Set up the style
    sns.set(style="whitegrid")
    plt.figure(figsize=(15, 10))

    # Create a melted DataFrame for easier plotting with seaborn
    metric_columns = ['dice', 'jaccard', 'precision', 'recall', 'fpr', 'fnr', 'vs', 'hd', 'msd', 'hd95']
    # Define appropriate y-axis limits for each metric if needed
    y_limits = {
        'dice': [0, 1],
        'jaccard': [0, 1],
        'precision': [0, 1],
        'recall': [0, 1],
        'fpr': [0, 0.1],  # False positive rate should be low
        'fnr': [0, 1],
        'vs': [-1, 1],    # Volume similarity
        'hd': [0, 50],    # Hausdorff distance
        'msd': [0, 10],   # Mean surface distance
        'hd95': [0, 30]   # 95% Hausdorff distance
    }

    # Create individual box plots
    fig, axes = plt.subplots(4, 3, figsize=(18, 15))
    axes = axes.flatten()

    for i, metric in enumerate(metric_columns):
        if metric in combined_df.columns:
            ax = axes[i]

            # Create boxplot
            boxplot = ax.boxplot(combined_df[metric].dropna(), patch_artist=True)

            # Customize boxplot colors
            boxplot['boxes'][0].set_facecolor('lightblue')
            boxplot['medians'][0].set_color('red')
            boxplot['medians'][0].set_linewidth(2)
            boxplot['whiskers'][0].set_color('black')
            boxplot['whiskers'][1].set_color('black')
            boxplot['caps'][0].set_color('black')
            boxplot['caps'][1].set_color('black')
            boxplot['fliers'][0].set_markerfacecolor('red')
            boxplot['fliers'][0].set_markeredgecolor('red')
            boxplot['fliers'][0].set_alpha(0.6)

            # Set appropriate y-axis limits
            if metric in y_limits:
                ax.set_ylim(y_limits[metric])

            # Add statistics
            data = combined_df[metric].dropna()
            mean_val = data.mean()
            median_val = data.median()

            # Add mean line and statistics text
            ax.axhline(mean_val, color='green', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')

            stats_text = f'N: {len(data)}\nMean: {mean_val:.2f}\nMedian: {median_val:.2f}'
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                    ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

            ax.set_title(f'{metric.upper()} Distribution', fontweight='bold', fontsize=12)
            ax.set_ylabel('Value', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend()

        else:
            axes[i].set_visible(False)

    # Hide empty subplots
    for j in range(len(metric_columns), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    if task == 'dice_0':
        plt.suptitle('Segmentation Metrics - V0 - V2', fontsize=12, fontweight='bold', y=1.02)
    if task == 'dice':
        plt.suptitle('Segmentation Metrics - V1 - V2', fontsize=12, fontweight='bold', y=1.02)

    plt.savefig(f'/gpfs/work1/0/prjs1425/shark/preprocessing/dice_analysis/{task}.png', 
        bbox_inches='tight', dpi=300, facecolor='white')
    plt.show()

    return combined_df



def create_comparison_table(df_0, df_1):
    """
    Create a comparison table with mean ± std for both DataFrames and statistical tests
    """
    # Define the metrics to compare
    metric_columns = ['dice', 'jaccard', 'precision', 'recall', 'fpr', 'fnr', 'vs', 'hd', 'msd', 'hd95']
    
    # Initialize lists to store data
    comparison_data = []
    
    for metric in metric_columns:
        if metric in df_0.columns and metric in df_1.columns:
            # Get data for both groups
            data_0 = df_0[metric].dropna()
            data_1 = df_1[metric].dropna()
            
            # Calculate mean and std for df_0 (dice_0)
            mean_0 = data_0.mean()
            std_0 = data_0.std()
            n_0 = len(data_0)
            
            # Calculate mean and std for df_1 (dice)
            mean_1 = data_1.mean()
            std_1 = data_1.std()
            n_1 = len(data_1)
            
            # Create formatted strings with 3 decimal places for means and stds
            result_0 = f"{mean_0:.2f} ± {std_0:.2f} (n={n_0})"
            result_1 = f"{mean_1:.2f} ± {std_1:.2f} (n={n_1})"
            
            # Calculate difference
            diff = mean_1 - mean_0
            diff_pct = (diff / mean_0 * 100) if mean_0 != 0 else np.nan
            
            # Perform statistical test
            # Check if we have enough data
            if n_0 > 1 and n_1 > 1:
                # For most metrics, use t-test or Mann-Whitney U test
                # Check normality assumption
                _, p_normal_0 = stats.shapiro(data_0) if n_0 <= 5000 else (None, 0.05)  # Shapiro works up to 5000 samples
                _, p_normal_1 = stats.shapiro(data_1) if n_1 <= 5000 else (None, 0.05)
                
                # If both groups are normally distributed, use t-test
                if p_normal_0 > 0.05 and p_normal_1 > 0.05:
                    # Check equal variance
                    _, p_var = stats.levene(data_0, data_1)
                    if p_var > 0.05:
                        # Equal variance t-test
                        _, p_value = stats.ttest_ind(data_0, data_1, equal_var=True)
                        test_used = "t-test (equal var)"
                    else:
                        # Welch's t-test (unequal variance)
                        _, p_value = stats.ttest_ind(data_0, data_1, equal_var=False)
                        test_used = "t-test (unequal var)"
                else:
                    # Use non-parametric Mann-Whitney U test
                    _, p_value = stats.mannwhitneyu(data_0, data_1, alternative='two-sided')
                    test_used = "Mann-Whitney U"
                
                # Format p-value
                if p_value < 0.001:
                    p_str = "p < 0.001"
                elif p_value < 0.01:
                    p_str = f"p = {p_value:.2f}"
                elif p_value < 0.05:
                    p_str = f"p = {p_value:.2f}"
                else:
                    p_str = f"p = {p_value:.2f}"
                
                # Add significance stars
                significance = ""
                if p_value < 0.001:
                    significance = "***"
                elif p_value < 0.01:
                    significance = "**"
                elif p_value < 0.05:
                    significance = "*"
                
                p_value_str = f"{p_str}{significance} ({test_used})"
                
            else:
                p_value_str = "Insufficient data"
            
            # Add to comparison data
            comparison_data.append({
                'Metric': metric.upper(),
                'V0-V2': result_0,
                'V1-V2': result_1,
                'Difference (V1-V2 - V0-V2)': f"{diff:.2f} ({diff_pct:.1f}%)" if not np.isnan(diff_pct) else f"{diff:.2f}",
                'P-value': p_value_str
            })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    comparison_csv_path = '/gpfs/work1/0/prjs1425/shark/preprocessing/dice_analysis/metrics_comparison_with_pvalues.csv'
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"\nComparison table with p-values saved to: {comparison_csv_path}")
    
    # Display the table
    print("\n" + "="*120)
    print("METRICS COMPARISON: V0-V2 vs V1-V2 (with statistical tests)")
    print("="*120)
    print(comparison_df.to_string(index=False))
    print("="*120)
    
    # Add significance key
    print("\nSignificance levels: *p < 0.05, **p < 0.01, ***p < 0.001")
    
    # Create a summary table for significant results only
    print("\n" + "="*80)
    print("SUMMARY OF SIGNIFICANT DIFFERENCES")
    print("="*80)
    
    significant_results = []
    for _, row in comparison_df.iterrows():
        p_str = row['P-value']
        if 'p < 0.001' in p_str or ('p =' in p_str and any(marker in p_str for marker in ['*', '**', '***'])):
            metric = row['Metric']
            v0_val = row['V0-V2'].split(' (n=')[0]
            v1_val = row['V1-V2'].split(' (n=')[0]
            diff_info = row['Difference (V1-V2 - V0-V2)']
            
            # Extract just the p-value number for sorting
            if 'p < 0.001' in p_str:
                p_num = 0.0009
            else:
                # Extract p-value from string like "p = 0.023*"
                p_part = p_str.split('p = ')[1].split()[0]
                p_part = p_part.replace("*","")
                p_num = float(p_part)
            
            significant_results.append({
                'Metric': metric,
                'V0-V2': v0_val,
                'V1-V2': v1_val,
                'Difference': diff_info,
                'P-value': p_str,
                'p_num': p_num
            })
    
    if significant_results:
        # Sort by p-value (most significant first)
        significant_results.sort(key=lambda x: x['p_num'])
        
        # Create and display summary DataFrame
        summary_data = []
        for result in significant_results:
            summary_data.append({
                'Metric': result['Metric'],
                'V0-V2': result['V0-V2'],
                'V1-V2': result['V1-V2'],
                'Difference': result['Difference'],
                'P-value': result['P-value']
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
    else:
        print("No significant differences found at p < 0.05 level.")
    
    print("="*80)
    
    return comparison_df


if __name__ == "__main__":
    df_0 = visualize('dice_0')
    df_1 = visualize('dice')

    # df_0['id'] = df_0['filename'].apply(lambda x: x.split('/')[6] if 'lkeb' in x else x.split('/')[7])
    # unique_ids_0, counts = np.unique(df_0.id.values.tolist(), return_counts=True)
    # print(len(unique_ids_0))

    # df_1['id'] = df_1['filename'].apply(lambda x: x.split('/')[6] if 'lkeb' in x else x.split('/')[7])
    # unique_ids_1, counts = np.unique(df_1.id.values.tolist(), return_counts=True)
    # unique_ids_1[counts != 1]
    # print(len(unique_ids_1))

    # diff = set(unique_ids_1) - set(unique_ids_0)
    # print(diff)
     
    # Create comparison table with p-values
    comparison_df = create_comparison_table(df_0, df_1)