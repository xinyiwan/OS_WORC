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

    Format: Rows are parameters/categories, Columns are modalities (T1W, T1W_FS_CE, T2W_FS)

    Parameters:
    -----------
    merged_df : pd.DataFrame
        Merged dataframe with metadata

    Returns:
    --------
    summary_table : pd.DataFrame
        Formatted summary table with modalities as columns
    """
    print("\n" + "="*80)
    print("OVERVIEW OF INCLUDED MRI SEQUENCES AND ACQUISITION PARAMETERS")
    print("="*80)

    # Filter only included images
    included_df = merged_df[merged_df['included'] == 'yes'].copy()

    # Get unique modalities
    modalities = sorted(included_df['modality'].unique())

    # Initialize list to store rows
    summary_rows = []

    # Helper function to add a row
    def add_row(category, subcategory, values_dict):
        row = {'Category': category, 'Parameter': subcategory}
        for modality in modalities:
            row[modality] = values_dict.get(modality, '-')
        summary_rows.append(row)

    # 1. Magnetic field strength (Tesla) - use actual unique values from data
    print("\n1. Magnetic Field Strength Distribution")
    print("-" * 80)

    if 'Tesla' in included_df.columns:
        # Get unique Tesla values from the data
        tesla_values = sorted(included_df['Tesla'].dropna().unique())

        for tesla in tesla_values:
            values_dict = {}
            for modality in modalities:
                mod_data = included_df[included_df['modality'] == modality]
                count = (mod_data['Tesla'] == tesla).sum()
                total = len(mod_data)

                if count > 0 and total > 0:
                    percentage = (count / total) * 100
                    values_dict[modality] = f"{count} ({percentage:.1f}%)"
                else:
                    values_dict[modality] = '-'

            # Format Tesla label (e.g., 1.5T -> 1·5T)
            tesla_label = f"{tesla:.1f}T".replace('.', '·')
            add_row('Magnetic field strength', tesla_label, values_dict)

    # 2. Manufacturer - use actual unique values from data
    print("\n2. Manufacturer Distribution")
    print("-" * 80)

    if 'Manufacturer' in included_df.columns:
        # Get unique manufacturers from the data
        manufacturers = included_df['Manufacturer'].dropna().unique()

        # Extract manufacturer name (e.g., "PHILIPS MEDICAL SYSTEMS" -> "PHILIPS")
        manufacturer_names = set()
        for manuf in manufacturers:
            manuf_upper = str(manuf).upper()
            for key in ['SIEMENS', 'GE', 'PHILIPS', 'TOSHIBA', 'HITACHI']:
                if key in manuf_upper:
                    manufacturer_names.add(key)
                    break
            else:
                # If no known manufacturer found, add the full name
                manufacturer_names.add(str(manuf))

        # Add Unknown for missing values
        if included_df['Manufacturer'].isna().any():
            manufacturer_names.add('Unknown')

        # Sort manufacturers
        manufacturer_names = sorted(manufacturer_names)

        for manuf in manufacturer_names:
            values_dict = {}
            for modality in modalities:
                mod_data = included_df[included_df['modality'] == modality]

                if manuf == 'Unknown':
                    count = mod_data['Manufacturer'].isna().sum()
                else:
                    count = mod_data['Manufacturer'].str.upper().str.contains(manuf, na=False).sum()

                total = len(mod_data)

                if count > 0 and total > 0:
                    percentage = (count / total) * 100
                    values_dict[modality] = f"{count} ({percentage:.1f}%)"
                else:
                    values_dict[modality] = '-'

            add_row('Manufacturer', manuf.title(), values_dict)

    # 3. Settings (mean ± variance)
    print("\n3. Acquisition Settings")
    print("-" * 80)

    # Slice Thickness
    values_dict = {}
    for modality in modalities:
        mod_data = included_df[included_df['modality'] == modality]
        if 'slice_thickness' in mod_data.columns:
            valid_data = mod_data['slice_thickness'].dropna()
            if len(valid_data) > 0:
                mean_val = valid_data.mean()
                var_val = valid_data.var()
                values_dict[modality] = f"{mean_val:.1f} ± {var_val:.1f}"
            else:
                values_dict[modality] = '-'
        else:
            values_dict[modality] = '-'
    add_row('Setting (Unit)', 'Slice Thickness (mm)*', values_dict)

    # Repetition Time
    values_dict = {}
    for modality in modalities:
        mod_data = included_df[included_df['modality'] == modality]
        if 'Repetition Time (ms)' in mod_data.columns:
            valid_data = mod_data['Repetition Time (ms)'].dropna()
            if len(valid_data) > 0:
                mean_val = valid_data.mean()
                var_val = valid_data.var()
                values_dict[modality] = f"{mean_val:.1f} ± {var_val:.1f}"
            else:
                values_dict[modality] = '-'
        else:
            values_dict[modality] = '-'
    add_row('Setting (Unit)', 'Repetition time (ms)*', values_dict)

    # Echo Time
    values_dict = {}
    for modality in modalities:
        mod_data = included_df[included_df['modality'] == modality]
        if 'Echo Time (ms)' in mod_data.columns:
            valid_data = mod_data['Echo Time (ms)'].dropna()
            if len(valid_data) > 0:
                mean_val = valid_data.mean()
                var_val = valid_data.var()
                values_dict[modality] = f"{mean_val:.1f} ± {var_val:.1f}"
            else:
                values_dict[modality] = '-'
        else:
            values_dict[modality] = '-'
    add_row('Setting (Unit)', 'Echo time (ms)*', values_dict)

    # 4. Available sequences (count of images)
    print("\n4. Available Sequences and Subjects")
    print("-" * 80)

    values_dict = {}
    for modality in modalities:
        count = len(included_df[included_df['modality'] == modality])
        values_dict[modality] = count
    add_row('', 'Available sequences', values_dict)

    # 5. Available subjects (unique count)
    values_dict = {}
    for modality in modalities:
        count = included_df[included_df['modality'] == modality]['Subject'].nunique()
        values_dict[modality] = count
    add_row('', 'Available subjects', values_dict)

    # Create DataFrame
    summary_table = pd.DataFrame(summary_rows)

    # Set multi-index for better display
    summary_table = summary_table.set_index(['Category', 'Parameter'])

    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(summary_table)

    return summary_table


def save_summary_to_file(summary_table, output_path):
    """
    Save summary table to CSV and text files.

    Parameters:
    -----------
    summary_table : pd.DataFrame
        Summary table with modalities as columns
    output_path : str
        Path to save the summary file
    """
    # Save as CSV
    csv_path = output_path.replace('.txt', '.csv') if output_path.endswith('.txt') else output_path
    summary_table.to_csv(csv_path)
    print(f"\nSummary table saved to: {csv_path}")

    # Also save as formatted text file
    txt_path = csv_path.replace('.csv', '_formatted.txt')
    with open(txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("OVERVIEW OF INCLUDED MRI SEQUENCES AND ACQUISITION PARAMETERS\n")
        f.write("="*80 + "\n\n")
        f.write(summary_table.to_string())
        f.write("\n\n")
        f.write("* Values shown as: mean ± variance for continuous variables\n")
        f.write("  or count (percentage%) for categorical variables\n")

    print(f"Formatted summary saved to: {txt_path}")


def main():
    """
    Main function to run the metadata merge and summary generation.
    """
    # Define file paths (modify these as needed)
    current_csv_path = "/projects/0/prjs1425/shark/preprocessing/modality/included_images_app.csv"
    source_csv_path = "/projects/0/prjs1425/shark/image_records/Osteo_Sarcoma_xnatsort_20250319.csv"

    output_merged_path = "/projects/0/prjs1425/shark/preprocessing/meta/merged_output.csv"
    output_summary_path = "/projects/0/prjs1425/shark/preprocessing/meta/summary_report.txt"

    # Load and merge metadata
    merged_df = load_and_merge_metadata(
        current_csv_path,
        source_csv_path,
        output_path=output_merged_path
    )

    # Create summary table
    summary_table = create_summary_table(merged_df)

    # Save summary to file
    save_summary_to_file(summary_table, output_summary_path)

    return merged_df, summary_table


if __name__ == "__main__":

    # Run the merge and summary
    merged_df, summary = main()
