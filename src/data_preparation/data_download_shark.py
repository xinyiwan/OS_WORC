import os
import pandas as pd
import logging
from server_utils import test_ssh_connection, copy_with_rsync
import subprocess

IMAGE_RECORDS_DIR = '/exports/lkeb-hpc/xwan/osteosarcoma/preprocessing/dataloader/'
DATA_DIR = '/projects/prjs1779/Osteosarcoma/exp_data'
LOG_DIR = '/exports/lkeb-hpc/xwan/osteosarcoma/logs'

# Remote server configuration
REMOTE_SERVER = 'snail'
REMOTE_USER = 'xwan1'
REMOTE_BASE_PATH = '/projects/prjs1779/Osteosarcoma/exp_data'
REMOTE_PORT = 22

def create_remote_directory(remote_path, server, user, port=22):
    """Create directory on remote server if it doesn't exist."""
    try:
        # Extract directory from file path
        remote_dir = os.path.dirname(remote_path)
        
        # Use ssh to create directory
        cmd = ['ssh', '-p', str(port), f'{user}@{server}', f'mkdir -p {remote_dir}']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logging.info(f"Created remote directory: {remote_dir}")
            return True
        else:
            logging.error(f"Failed to create remote directory {remote_dir}: {result.stderr}")
            return False
    except Exception as e:
        logging.error(f"Error creating remote directory {remote_path}: {str(e)}")
        return False

def download_data(modality='T1W', version='v0', transfer_method='rsync'):
    """
    Execute the data download from SHARK to Snellius using existing CSV with paths.
    """
    # logging file
    logging.basicConfig(filename=f'{LOG_DIR}/data_download_shark2snail_{modality}.log', 
                        level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')

    # Test SSH connection first
    if not test_ssh_connection(REMOTE_SERVER, REMOTE_USER, REMOTE_PORT):
        logging.error(f"Cannot connect to {REMOTE_SERVER}. Please check credentials and connection.")
        return

    # Read the CSV file
    records_path = os.path.join(IMAGE_RECORDS_DIR, f'{modality}_df.csv')
    if not os.path.exists(records_path):
        logging.error(f"Records file not found: {records_path}")
        return
    
    records = pd.read_csv(records_path)
    
    # Filter only included records if you have that column
    seg_column = f'seg_{version}_path'
    if seg_column in records.columns:
        records = records[records[seg_column].notna()]
    else:
        logging.error(f"Segmentation column {seg_column} not found in CSV")
        return
    
    logging.info(f"Found {len(records)} records to process")

    successful_transfers = 0
    failed_transfers = 0

    for idx, row in records.iterrows():
        pid = row['pid_n']
        
        # Get image path directly from CSV
        img_file = row['image_path']
        if pd.isna(img_file) or not os.path.exists(img_file):
            logging.warning(f"Image file not found: {img_file}")
            failed_transfers += 1
            continue
        
        # Get segmentation path based on version
        seg_column = f'seg_{version}_path'
        if seg_column not in records.columns:
            logging.error(f"Segmentation column {seg_column} not found in CSV")
            failed_transfers += 1
            continue
            
        seg_file = row[seg_column]
        if pd.isna(seg_file) or not os.path.exists(seg_file):
            logging.warning(f"Segmentation file not found: {seg_file}")
            failed_transfers += 1
            continue

        # Define remote paths
        remote_dir = os.path.join(REMOTE_BASE_PATH, modality, version, pid)
        remote_img_path = os.path.join(remote_dir, 'image.nii.gz')
        remote_seg_path = os.path.join(remote_dir, 'mask.nii.gz')

        # Create remote directory first
        if not create_remote_directory(remote_img_path, REMOTE_SERVER, REMOTE_USER, REMOTE_PORT):
            logging.error(f"Failed to create remote directory for {pid}")
            failed_transfers += 1
            continue

        # Copy files to remote server
        success = False

        if transfer_method == 'rsync':
            success_img = copy_with_rsync(img_file, remote_img_path, REMOTE_SERVER, REMOTE_USER, REMOTE_PORT)
            success_seg = copy_with_rsync(seg_file, remote_seg_path, REMOTE_SERVER, REMOTE_USER, REMOTE_PORT)
            success = success_img and success_seg

            if success:
                successful_transfers += 1
                logging.info(f"Successfully transferred files for {pid}")
            else:
                failed_transfers += 1
                logging.error(f"Failed to transfer files for {pid}")

    # Summary
    logging.info(f"Transfer completed. Successful: {successful_transfers}, Failed: {failed_transfers}")
    print(f"Transfer summary for {modality} (version {version}):")
    print(f"  Successful transfers: {successful_transfers}")
    print(f"  Failed transfers: {failed_transfers}")
    print(f"  Total attempted: {successful_transfers + failed_transfers}")

if __name__ == "__main__":
    # download_data('T1W', version='v9')
    # download_data('T1W', version='v0')
    download_data('T1W', version='v1')

    # download_data('T2W_FS', version='v0')
    download_data('T2W_FS', version='v1')
    # download_data('T2W_FS', version='v9')


    # download_data('T1W_FS_C', version='v0')
    download_data('T1W_FS_C', version='v1')
    # download_data('T1W_FS_C', version='v9')