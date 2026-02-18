import pandas as pd
import numpy as np
from pathlib import Path


def load_and_merge_metadata(current_csv_path, source_csv_path, output_path=None):
    """
    Merge metadata from source CSV to current CSV with included images.

    Parameters:
    -----------
    current_csv_path : str
        Path to the current CSV file with included images
    source_csv_path : str
        Path to the source metadata CSV file
    output_path : str, optional
        Path to save the merged CSV file

    Returns:
    --------
    merged_df : pd.DataFrame
        DataFrame with merged metadata
    """
    # Load CSV files
    print("Loading CSV files...")
    current_df = pd.read_csv(current_csv_path)
    source_df = pd.read_csv(source_csv_path)

    print(f"Current CSV shape: {current_df.shape}")
    print(f"Source CSV shape: {source_df.shape}")

    # Select only the columns we need from source
    metadata_columns = [
        'Subject', 'Experiment', 'Scan',
        'Manufacturer', 'slice_thickness',
        'Repetition Time (ms)', 'Echo Time (ms)',
        'Inversion Time (ms)', 'Tesla'
    ]

    # Check if all required columns exist in source
    missing_cols = [col for col in metadata_columns if col not in source_df.columns]
    if missing_cols:
        print(f"Warning: Missing columns in source CSV: {missing_cols}")

    available_cols = [col for col in metadata_columns if col in source_df.columns]
    source_subset = source_df[available_cols].copy()

    # Rename 'scan' to 'Scan' in current_df if needed for consistency
    if 'scan' in current_df.columns:
        current_df = current_df.rename(columns={'scan': 'Scan'})

    # Rename 'session' to 'Experiment' in current_df to match source
    if 'session' in current_df.columns:
        current_df = current_df.rename(columns={'session': 'Experiment'})

    # Merge on Subject, Experiment, and Scan
    print("\nMerging dataframes...")
    merged_df = current_df.merge(
        source_subset,
        on=['Subject', 'Experiment', 'Scan'],
        how='left'
    )

    print(f"Merged CSV shape: {merged_df.shape}")
    print(f"\nMerge statistics:")
    print(f"  - Total rows in merged: {len(merged_df)}")
    print(f"  - Rows with metadata: {merged_df['Manufacturer'].notna().sum()}")
    print(f"  - Rows without metadata: {merged_df['Manufacturer'].isna().sum()}")

    # Save merged CSV if output path is provided
    if output_path:
        merged_df.to_csv(output_path, index=False)
        print(f"\nMerged CSV saved to: {output_path}")

    return merged_df


def create_summary_table(merged_df):
    """
    Create summary table with overview of included MRI sequences and acquisition parameters.

    Parameters:
    -----------
    merged_df : pd.DataFrame
        Merged dataframe with metadata

    Returns:
    --------
    summary_dict : dict
        Dictionary containing summary statistics
    """
    print("\n" + "="*80)
    print("OVERVIEW OF INCLUDED MRI SEQUENCES AND ACQUISITION PARAMETERS")
    print("="*80)

    summary_dict = {}

    # Filter only included images
    included_df = merged_df[merged_df['included'] == 'yes'].copy()

    # 1. Number of images and subjects per modality
    print("\n1. NUMBER OF IMAGES AND SUBJECTS PER MODALITY")
    print("-" * 80)

    modality_summary = included_df.groupby('modality').agg({
        'image_name': 'count',
        'Subject': 'nunique'
    }).rename(columns={
        'image_name': 'Number of Images',
        'Subject': 'Number of Subjects'
    })

    print(modality_summary)
    summary_dict['modality_summary'] = modality_summary

    # 2. Distribution of metadata features
    metadata_features = [
        'Manufacturer', 'slice_thickness',
        'Repetition Time (ms)', 'Echo Time (ms)',
        'Inversion Time (ms)', 'Tesla'
    ]

    print("\n2. DISTRIBUTION OF ACQUISITION PARAMETERS")
    print("-" * 80)

    for feature in metadata_features:
        if feature not in included_df.columns:
            print(f"\n{feature}: Column not found")
            continue

        print(f"\n{feature}:")

        # For categorical features
        if feature == 'Manufacturer' or included_df[feature].dtype == 'object':
            value_counts = included_df[feature].value_counts(dropna=False)
            print(value_counts)
            summary_dict[feature] = value_counts

        # For numerical features
        else:
            # Remove NaN values for statistics
            valid_values = included_df[feature].dropna()

            if len(valid_values) > 0:
                stats = {
                    'count': len(valid_values),
                    'mean': valid_values.mean(),
                    'std': valid_values.std(),
                    'min': valid_values.min(),
                    '25%': valid_values.quantile(0.25),
                    '50%': valid_values.quantile(0.50),
                    '75%': valid_values.quantile(0.75),
                    'max': valid_values.max(),
                    'missing': included_df[feature].isna().sum()
                }

                stats_df = pd.DataFrame(stats, index=[feature])
                print(stats_df.T)
                summary_dict[f"{feature}_stats"] = stats_df

                # Also show unique values distribution if not too many
                unique_values = valid_values.nunique()
                if unique_values <= 20:
                    print(f"\nValue distribution:")
                    value_counts = included_df[feature].value_counts(dropna=False)
                    print(value_counts)
                    summary_dict[f"{feature}_distribution"] = value_counts
            else:
                print("No valid values found")

    # 3. Cross-tabulation: Modality vs Manufacturer
    print("\n3. MODALITY VS MANUFACTURER")
    print("-" * 80)
    crosstab = pd.crosstab(
        included_df['modality'],
        included_df['Manufacturer'],
        margins=True
    )
    print(crosstab)
    summary_dict['modality_manufacturer_crosstab'] = crosstab

    # 4. Summary by modality
    print("\n4. ACQUISITION PARAMETERS BY MODALITY")
    print("-" * 80)

    numerical_features = [
        'slice_thickness', 'Repetition Time (ms)',
        'Echo Time (ms)', 'Inversion Time (ms)', 'Tesla'
    ]

    for feature in numerical_features:
        if feature in included_df.columns:
            print(f"\n{feature} by modality:")
            modality_stats = included_df.groupby('modality')[feature].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(2)
            print(modality_stats)
            summary_dict[f"{feature}_by_modality"] = modality_stats

    return summary_dict


def save_summary_to_file(summary_dict, output_path):
    """
    Save summary statistics to a text file.

    Parameters:
    -----------
    summary_dict : dict
        Dictionary containing summary statistics
    output_path : str
        Path to save the summary file
    """
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("OVERVIEW OF INCLUDED MRI SEQUENCES AND ACQUISITION PARAMETERS\n")
        f.write("="*80 + "\n\n")

        for key, value in summary_dict.items():
            f.write(f"\n{key}:\n")
            f.write("-" * 80 + "\n")
            if isinstance(value, pd.DataFrame):
                f.write(value.to_string())
            elif isinstance(value, pd.Series):
                f.write(value.to_string())
            else:
                f.write(str(value))
            f.write("\n\n")

    print(f"\nSummary saved to: {output_path}")


def main():
    """
    Main function to run the metadata merge and summary generation.
    """
    # Define file paths (modify these as needed)
    current_csv_path = "path/to/current.csv"
    source_csv_path = "path/to/source.csv"
    output_merged_path = "path/to/merged_output.csv"
    output_summary_path = "path/to/summary_report.txt"

    # Load and merge metadata
    merged_df = load_and_merge_metadata(
        current_csv_path,
        source_csv_path,
        output_path=output_merged_path
    )

    # Create summary table
    summary_dict = create_summary_table(merged_df)

    # Save summary to file
    save_summary_to_file(summary_dict, output_summary_path)

    return merged_df, summary_dict


if __name__ == "__main__":
    # Example usage with actual file paths
    # Update these paths to your actual CSV files

    current_csv = "current.csv"  # Replace with your current CSV path
    source_csv = "source.csv"    # Replace with your source CSV path

    # Run the merge and summary
    merged_df, summary = main()
