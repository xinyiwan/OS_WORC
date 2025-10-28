import os
import shutil
from pathlib import Path
import pandas as pd

def create_WIRdata(modality, version, WIR_pids):
    """
    Copy patient data from original modality/version to WIR/modality/version
    only for patients in the WIR_pids list.
    
    Parameters:
    modality (str): Modality name (e.g., 'T1W', 'T2W_FS')
    version (str): Version number (e.g., '0', '9')
    WIR_pids (list): List of patient IDs to include (e.g., ['OS_000131', 'OS_000112'])
    """
    
    # Define source and destination paths
    source_base = Path('/projects/0/prjs1425/Osteosarcoma_WORC/exp_data')
    dest_base = source_base / 'WIR' / modality / f'v{version}'
    
    source_path = source_base / modality / f'v{version}'
    
    print(f"Source: {source_path}")
    print(f"Destination: {dest_base}")
    print(f"Patients to copy: {WIR_pids}")
    
    # Create destination directory if it doesn't exist
    dest_base.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    missing_count = 0
    
    # Iterate through patient IDs in the list
    for patient_id in WIR_pids:
        # Find all folders that match this patient ID (with different suffixes like _01, _02, etc.)
        patient_folders = []
        
        if source_path.exists():
            for item in source_path.iterdir():
                if item.is_dir() and item.name.startswith(patient_id + '_'):
                    patient_folders.append(item.name)
        
        if not patient_folders:
            print(f"Warning: No folders found for patient {patient_id} in {source_path}")
            missing_count += 1
            continue
        
        # Copy each patient folder
        for folder_name in patient_folders:
            source_folder = source_path / folder_name
            dest_folder = dest_base / folder_name
            
            try:
                # Copy the entire folder recursively
                if dest_folder.exists():
                    print(f"Folder {folder_name} already exists in destination, skipping...")
                    continue
                
                shutil.copytree(source_folder, dest_folder)
                print(f"✓ Copied {folder_name}")
                copied_count += 1
                
            except Exception as e:
                print(f"✗ Error copying {folder_name}: {e}")
    
    print(f"\nCopy operation completed!")
    print(f"Successfully copied: {copied_count} patient folders")
    print(f"Missing patients: {missing_count}")
    print(f"Total patients in input list: {len(WIR_pids)}")

if __name__ == "__main__":
    # Example patient list - replace with your actual WIR_pids
    WIR = '/projects/0/prjs1425/Osteosarcoma_WORC/image_records/WIR_patient_mapping.csv'
    WIR_pids = pd.read_csv(WIR)['mapped_id'].tolist()
    
    # Define which modalities and versions you want to copy
    modalities = ['T2W_FS', 'T1W_FS_C', 'T1W']  # Add other modalities as needed
    versions = ['0', '1', '9']  # Add other versions as needed
    
    for modality in modalities:
        for version in versions:
            print(f"\n{'='*60}")
            print(f"Processing {modality}/v{version}")
            print(f"{'='*60}")
            create_WIRdata(modality, version, WIR_pids)