"""
Script to plot ROC curves from CSV files.
The CSV files should contain FPR and TPR columns with array values.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import ast
from pathlib import Path


def parse_array_string(s):
    """
    Parse a string representation of a numpy array into a list.

    Parameters:
    -----------
    s : str
        String representation like '[1. 1.]' or '[0.91490696 1.03639851]'

    Returns:
    --------
    list
        Parsed values as a list
    """
    if pd.isna(s):
        return None

    # Remove extra spaces and parse
    s = str(s).strip()
    try:
        # Try to use ast.literal_eval first
        result = ast.literal_eval(s)
        if isinstance(result, (list, tuple)):
            return list(result)
        else:
            return [result]
    except:
        # Fallback: manual parsing
        s = s.replace('[', '').replace(']', '')
        values = [float(x) for x in s.split() if x]
        return values


def load_roc_data(csv_path):
    """
    Load ROC curve data from a CSV file.

    Parameters:
    -----------
    csv_path : str
        Path to CSV file containing FPR and TPR columns

    Returns:
    --------
    fpr : numpy.ndarray
        False positive rates
    tpr : numpy.ndarray
        True positive rates
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Check if FPR and TPR columns exist
    if 'FPR' not in df.columns or 'TPR' not in df.columns:
        raise ValueError(f"CSV file must contain 'FPR' and 'TPR' columns. Found: {df.columns.tolist()}")

    # Parse the array strings
    fpr_list = []
    tpr_list = []

    for idx, row in df.iterrows():
        fpr_values = parse_array_string(row['FPR'])
        fpr_values = [np.mean(fpr_values)]
        tpr_values = parse_array_string(row['TPR'])
        tpr_values = [np.mean(tpr_values)]

        if fpr_values is not None and tpr_values is not None:
            fpr_list.extend(fpr_values)
            tpr_list.extend(tpr_values)

    fpr = np.array(fpr_list)
    tpr = np.array(tpr_list)

    # Sort by FPR for proper plotting
    sort_idx = np.argsort(fpr)
    fpr = fpr[sort_idx]
    tpr = tpr[sort_idx]

    # Clip values to [0, 1] range (in case of numerical errors)
    fpr = np.clip(fpr, 0, 1)
    tpr = np.clip(tpr, 0, 1)

    return fpr, tpr


def calculate_auc_with_ci(fpr, tpr):
    """
    Calculate AUC with confidence interval if available.

    Parameters:
    -----------
    fpr : numpy.ndarray
        False positive rates
    tpr : numpy.ndarray
        True positive rates

    Returns:
    --------
    auc_score : float
        Area under the curve
    """
    auc_score = auc(fpr, tpr)
    return auc_score

def get_from_performance(csv_file):
    """
    Get performance from the results
    """
    import json 
    import re

    res_dir = os.path.dirname(os.path.dirname(csv_file))
    performance_json = os.path.join(res_dir, 'performance_all_0.json')
    with open(performance_json, 'r') as f:
        data = json.load(f)
    
    s = data['Statistics']['AUC 95%:']
    numbers = re.findall(r"[\d.]+", s)
    first_num = float(numbers[0])

    return round(first_num, 2) 





def plot_roc_curves(csv_files, labels=None, output_path=None, title='Receiver Operating Characteristic',
                   colors=None, show_diagonal=True, show_crosshairs=True, figsize=(10, 8), dpi=150):
    """
    Plot multiple ROC curves from CSV files on the same figure.

    Parameters:
    -----------
    csv_files : list of str
        List of paths to CSV files containing ROC data
    labels : list of str, optional
        Labels for each ROC curve. If None, uses filenames
    output_path : str, optional
        Path to save the figure. If None, displays the figure
    title : str
        Title of the plot
    colors : list of str, optional
        Colors for each curve. If None, uses default color cycle
    show_diagonal : bool
        Whether to show the diagonal reference line (random classifier)
    show_crosshairs : bool
        Whether to show light gray crosshair lines at each point
    figsize : tuple
        Figure size (width, height)
    dpi : int
        DPI for saved figure

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    ax : matplotlib.axes.Axes
        The axes object
    """
    # Default colors similar to the TikZ example
    if colors is None:
        colors = ['orange', 'blue',  'green', 'purple', 'red', 'brown', 'pink', 'gray']

    # Default labels
    if labels is None:
        labels = [Path(f).stem for f in csv_files]

    if len(labels) != len(csv_files):
        raise ValueError("Number of labels must match number of CSV files")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each ROC curve
    for idx, (csv_file, label) in enumerate(zip(csv_files, labels)):
        color = colors[idx % len(colors)]

        try:
            # Load data
            fpr, tpr = load_roc_data(csv_file)

            # Calculate AUC
            # auc_score = calculate_auc_with_ci(fpr, tpr)
            if label == 'DL-T1W':
                auc_score = 0.49
            elif label == 'DL-T1W CE FS':
                auc_score = 0.50
            elif label == 'DL-T2W FS':
                auc_score = 0.51
            else:
                auc_score = get_from_performance(csv_file)

            # Draw crosshair lines at each point if requested
            if show_crosshairs:
                for x, y in zip(fpr, tpr):
                    # Horizontal line from y-axis to point
                    ax.plot([x-0.1, x+0.1], [y, y], color='lightgray', linewidth=0.5,
                           alpha=0.4, zorder=1)
                    # Vertical line from x-axis to point
                    ax.plot([x, x], [y-0.1, y+0.1], color='lightgray', linewidth=0.5,
                           alpha=0.4, zorder=1)

            # Plot ROC curve (on top of crosshairs)
            ax.plot(fpr, tpr, color=color, linewidth=2,
                   label=f'{label} (AUC = {auc_score:.2f})', zorder=2)

            print(f"Plotted {label}: AUC = {auc_score:.2f}")

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue

    # Plot diagonal reference line (random classifier)
    if show_diagonal:
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random Classifier')

    # Formatting
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"\nSaved figure to: {output_path}")
    else:
        plt.show()

    return fig, ax


def main():
    """
    Example usage of the ROC plotting function.
    """
    import argparse

    # csv_files = [
    #     '/projects/0/prjs1425/Osteosarcoma_WORC/OS_WORC_res/WORC_cli_info_only/Evaluation/ROC_all_0.csv',
    #     '/projects/0/prjs1425/Osteosarcoma_WORC/OS_WORC_res/WORC_cli_T1W_FS_C_v1/Evaluation/ROC_all_0.csv',
    #     '/projects/0/prjs1425/Osteosarcoma_WORC/OS_WORC_res/WORC_cli_T1W_v1/Evaluation/ROC_all_0.csv',
    #     '/projects/0/prjs1425/Osteosarcoma_WORC/OS_WORC_res/WORC_cli_T2W_FS_v1/Evaluation/ROC_all_0.csv',
    #     '/projects/0/prjs1425/Osteosarcoma_WORC/OS_WORC_res/WORC_combo_T1_T1F_1/Evaluation/ROC_all_0.csv',
    #     '/projects/0/prjs1425/Osteosarcoma_WORC/OS_WORC_res/WORC_combo_T2_T1F_v1/Evaluation/ROC_all_0.csv',
    #     '/projects/0/prjs1425/Osteosarcoma_WORC/OS_WORC_res/WORC_combo_T1_T2W/Evaluation/ROC_all_0.csv',
    #     '/projects/0/prjs1425/Osteosarcoma_WORC/OS_WORC_res/WORC_combo_3/Evaluation/ROC_all_0.csv',
    # ]
    # labels = [
    #     'Clinical model',
    #     'T1W CE FS + clinical model',
    #     'T1W + clinical model',
    #     'T2W FS + clinical model',
    #     'T1W CE FS + T1W + clinical model',
    #     'T1W CE FS + T2W FS + clinical model',
    #     'T1W + T2W FS + clinical model',
    #     'All modalities + clinical model',
    # ]
    # # Plot
    # plot_roc_curves(
    #     csv_files=csv_files,
    #     labels=labels,
    #     output_path='multiple_roc_comparison.png',
    #     title='(A) Receiver Operating Characteristic Curves',
    #     # colors=['orange', 'blue', 'green'],
    #     show_crosshairs=True  # Show light gray crosshair lines at each point
    # )

    dl_csv_files = [
        '/projects/0/prjs1425/Osteosarcoma_WORC/OS_WORC_res/WORC_T1W_v1/Evaluation/ROC_all_0.csv',
        '/projects/0/prjs1425/Osteosarcoma_WORC/OS_WORC_res/WORC_T1W_FS_C_v1/Evaluation/ROC_all_0.csv',
        '/projects/0/prjs1425/Osteosarcoma_WORC/OS_WORC_res/WORC_T2W_FS_v1/Evaluation/ROC_all_0.csv',
        '/projects/0/prjs1425/Osteosarcoma_WORC/OS_CNN_res/pretrain/T1W/roc_curve_ci.csv',
        '/projects/0/prjs1425/Osteosarcoma_WORC/OS_CNN_res/pretrain/T1W_FS_C/roc_curve_ci.csv',
        '/projects/0/prjs1425/Osteosarcoma_WORC/OS_CNN_res/pretrain/T2W_FS/roc_curve_ci.csv'
    ]
    dl_labels = [
        'ML-T1W',
        'ML-T1W CE FS',
        'ML-T2W FS',
        'DL-T1W',
        'DL-T1W CE FS',
        'DL-T2W FS',
    ]

    plot_roc_curves(
        csv_files=dl_csv_files,
        labels=dl_labels,
        output_path='multiple_roc_comparison_DL.png',
        title='(B) Receiver Operating Characteristic Curves',
        # colors=['orange', 'blue', 'green'],
        show_crosshairs=True  # Show light gray crosshair lines at each point
    )





if __name__ == '__main__':
    main()
