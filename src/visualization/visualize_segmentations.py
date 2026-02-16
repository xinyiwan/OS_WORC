"""
This script visualizes examples using semi-automatic prompt-based segmentation,
manual correction and expert approval across different versions.
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

# Configuration
modalities = ['T1W', 'T1W_FS_C', 'T2W_FS']
versions = ['v0', 'v1', 'v2']
image_file_name = 'image.nii.gz'
segmentation_file_name = 'mask.nii.gz'
base_data_dir = '/projects/0/prjs1425/Osteosarcoma_WORC/exp_data'

# Colors for different versions
colors = ['red', 'green', 'blue']
version_colors = dict(zip(versions, colors))


def find_best_slice(seg_data):
    """
    Find the slice with the most segmentation voxels.
    The shortest axis will be used as the slice axis.

    Parameters:
    -----------
    seg_data : numpy.ndarray
        3D segmentation array

    Returns:
    --------
    slice_axis : int
        The axis to slice along (the shortest dimension)
    slice_idx : int
        The index of the slice with most segmentation
    """
    # Find the shortest axis
    slice_axis = np.argmin(seg_data.shape)

    # Count segmentation voxels per slice along the shortest axis
    slice_counts = np.sum(seg_data, axis=tuple(i for i in range(3) if i != slice_axis))

    # Find the slice with maximum segmentation
    slice_idx = np.argmax(slice_counts)

    return slice_axis, slice_idx


def get_slice(data, slice_axis, slice_idx):
    """
    Extract a 2D slice from 3D data along specified axis.
    Reorient so the slice_axis is the last dimension.

    Parameters:
    -----------
    data : numpy.ndarray
        3D array
    slice_axis : int
        Axis to slice along
    slice_idx : int
        Index of the slice

    Returns:
    --------
    slice_2d : numpy.ndarray
        2D slice
    """
    if slice_axis == 0:
        return data[slice_idx, :, :]
    elif slice_axis == 1:
        return data[:, slice_idx, :]
    else:  # slice_axis == 2
        return data[:, :, slice_idx]


def load_nifti_data(file_path):
    """
    Load NIfTI file and return data array.

    Parameters:
    -----------
    file_path : str
        Path to NIfTI file

    Returns:
    --------
    data : numpy.ndarray or None
        Image data, or None if file doesn't exist
    """
    if not os.path.exists(file_path):
        return None

    try:
        nii = nib.load(file_path)
        data = nii.get_fdata()
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def get_patient_ids(modality, version='v0'):
    """
    Get all patient IDs for a given modality from a reference version.

    Parameters:
    -----------
    modality : str
        The modality (e.g., 'T1W')
    version : str
        Version to scan for patient directories (default 'v0')

    Returns:
    --------
    patient_ids : list
        List of patient directory names
    """
    version_dir = os.path.join(base_data_dir, modality, version)

    if not os.path.exists(version_dir):
        return []

    # Get all directories that look like patient IDs
    patient_ids = [d for d in os.listdir(version_dir)
                   if os.path.isdir(os.path.join(version_dir, d))]

    return sorted(patient_ids)


def plot_segs(modality, patient_id, output_dir='output_visualizations'):
    """
    Plot segmentations of 3 versions on the same image. Only plot the contour of each segmentation.
    Images are all NIfTI files.
    Make the shortest axis the last axis of the image, and choose the slice where segmentation has the most 1s.
    Segmentations for a specific version can be missing. If missing, skip that version.

    Parameters:
    -----------
    modality : str
        The modality to visualize (e.g., 'T1W')
    patient_id : str
        Patient ID directory name (e.g., 'OS_000001_01')
    output_dir : str
        Directory to save output plots
    """
    # Create output directory if it doesn't exist
    modality_output_dir = os.path.join(output_dir, modality)
    os.makedirs(modality_output_dir, exist_ok=True)

    # Load image from v0 (assuming this is the reference)
    image_path = os.path.join(base_data_dir, modality, 'v0', patient_id, image_file_name)
    image_data = load_nifti_data(image_path)

    if image_data is None:
        print(f"Warning: Could not load image for {modality}/{patient_id}")
        return

    # Load all available segmentations
    segmentations = {}
    all_segs_combined = np.zeros_like(image_data)

    for version in versions:
        seg_path = os.path.join(base_data_dir, modality, version, patient_id, segmentation_file_name)
        seg_data = load_nifti_data(seg_path)

        if seg_data is not None:
            segmentations[version] = seg_data
            all_segs_combined += seg_data
        else:
            print(f"  Warning: Segmentation for {modality}/{version}/{patient_id} not found, skipping...")

    if len(segmentations) == 0:
        print(f"  No segmentations found for {modality}/{patient_id}")
        return

    # Find the best slice to display (using combined segmentations)
    slice_axis, slice_idx = find_best_slice(all_segs_combined)

    # Extract 2D slices
    image_slice = get_slice(image_data, slice_axis, slice_idx)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Display the image
    ax.imshow(image_slice.T, cmap='gray', origin='lower')

    # Overlay contours for each version
    for version, seg_data in segmentations.items():
        seg_slice = get_slice(seg_data, slice_axis, slice_idx)

        # Plot contour
        contours = ax.contour(seg_slice.T, levels=[0.5], colors=version_colors[version],
                             linewidths=2, origin='lower')

        # Add label
        contours.collections[0].set_label(version)

    # Add legend
    ax.legend(loc='upper right', fontsize=12)

    # Add title
    axis_names = ['Sagittal', 'Coronal', 'Axial']
    ax.set_title(f'{modality} - {patient_id}\n'
                 f'{axis_names[slice_axis]} Slice {slice_idx} - '
                 f'Segmentation Comparison (v0=red, v1=green, v2=blue)',
                 fontsize=14, fontweight='bold')

    ax.axis('off')

    # Save figure
    output_path = os.path.join(modality_output_dir, f'{patient_id}_segmentation_comparison.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")

    plt.close()


def main():
    """
    Main function to process all modalities and create visualizations.
    """
    print("Starting segmentation visualization...")
    print(f"Base data directory: {base_data_dir}")
    print(f"Modalities to process: {modalities}")
    print(f"Versions to compare: {versions}")
    print("-" * 60)

    output_dir = 'output_visualizations'
    total_processed = 0

    for modality in modalities:
        print(f"\nProcessing modality: {modality}")

        # Get all patient IDs for this modality
        patient_ids = get_patient_ids(modality)

        if not patient_ids:
            print(f"  No patient directories found for {modality}")
            continue

        print(f"  Found {len(patient_ids)} patients")

        # Process each patient
        for patient_id in patient_ids:
            try:
                plot_segs(modality, patient_id, output_dir=output_dir)
                total_processed += 1
            except Exception as e:
                print(f"  Error processing {modality}/{patient_id}: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Visualization complete!")
    print(f"Total images processed: {total_processed}")
    print(f"Check the '{output_dir}' directory for results.")


if __name__ == '__main__':
    main()
