import os, glob
import pandas as pd
import logging
from server_utils import test_ssh_connection, copy_with_rsync

IMAGE_RECORDS_DIR = '/exports/lkeb-hpc/xwan/osteosarcoma/preprocessing/modality/included_images_app.csv'
DATA_DIR = '/projects/0/prjs1425/Osteosarcoma_WORC/exp_data'
RECORDS = pd.read_csv(IMAGE_RECORDS_DIR)
LOG_DIR = '/exports/lkeb-hpc/xwan/osteosarcoma/logs'

DATA_STORAGE = '/exports/lkeb-hpc-data/XnatOsteosarcoma/os_data_tmp/os_data_tmp/reorg_DCM2NII/'
SEG_DATA_STORAGE = '/exports/lkeb-hpc/xwan/osteosarcoma/OS_seg_resample'

# Remote server configuration
REMOTE_SERVER = 'snellius-lumc'  # Change this to your remote server
REMOTE_USER = 'xwan'             # Change this to your username
REMOTE_BASE_PATH = '/projects/0/prjs1425/Osteosarcoma_WORC/exp_data'  # Change this to your remote base path
REMOTE_PORT = 22  # Change if using non-standard SSH port


# logging file
logging.basicConfig(filename=f'{LOG_DIR}/data_download_shark2snellius.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


# Create new pid_n column
def create_pid_n(group):
    """
    Create a new 'pid_n' to create images for each subject with multiple scans. 
    The final format will be 'OS_0000001_01', 'OS_0000001_02', etc. 
    """
    # Sort by session and scan to ensure consistent ordering
    group = group.sort_values(['session', 'scan'])
    # Create index numbers starting from 01
    indices = [f"{i+1:02d}" for i in range(len(group))]
    group['pid_n'] = group['Subject'] + '_' + indices
    return group


def download_data(modality='T1W', transfer_method='rsync'):

    """
    Exacute the data download from SHARK.
    """

    # Test SSH connection first
    if not test_ssh_connection(REMOTE_SERVER, REMOTE_USER, REMOTE_PORT):
        logging.error(f"Cannot connect to {REMOTE_SERVER}. Please check credentials and connection.")
        return

    # Create directory if it doesn't exist @ Snellius
    # if not os.path.exists(f'{DATA_DIR}/{modality}'):
    #     os.makedirs(f'{DATA_DIR}/{modality}')
    #     logging.info(f'Created directory: {DATA_DIR}/{modality}')
    
    imgs = RECORDS[RECORDS['modality'] == modality]

    # Apply the function to create 'pid_n'
    imgs = imgs.groupby('Subject').apply(create_pid_n).reset_index(drop=True)
    imgs = imgs.sort_values('pid_n').reset_index(drop=True)

    # Reorder columns to have pid_n first
    cols = imgs.columns.tolist()
    cols = ['pid_n'] + [col for col in cols if col != 'pid_n']
    imgs = imgs[cols]

    successful_transfers = 0
    failed_transfers = 0

    for idx, row in imgs.iterrows():
        pid = row['pid_n']
        subject = row['Subject']
        session = row['session']
        scan = row['scan']

        # create subject directory if it doesn't exist @ Snellius
        # subject_dir = f'{DATA_DIR}/{modality}/{pid}'
        # if not os.path.exists(subject_dir):
        #     os.makedirs(subject_dir)
        #     logging.info(f'Created directory: {subject_dir}')
        
        # Find the image file 
        file_pattern = os.path.join(DATA_STORAGE, subject, session, f"{scan}-*.nii.gz")
        files = glob.glob(file_pattern)
        if len(files) == 0:
            logging.warning(f'No files found for pattern: {file_pattern}')
            continue
        elif len(files) > 1:
            logging.warning(f'Multiple files found for pattern: {file_pattern}. Using the first one.')
            continue
        else:
            logging.info(f'Image found following the pattern: {file_pattern}')
            img_file = files[0]
        
        # Find segmentation file
        seg_file_pattern = os.path.join(SEG_DATA_STORAGE, subject, session, 'gau_aff', f"SEG_0_{scan}-*.nii.gz")
        files = glob.glob(seg_file_pattern)
        if len(files) == 0:
            logging.warning(f'No files found for pattern: {file_pattern}')
            continue
        elif len(files) > 1:
            logging.warning(f'Multiple files found for pattern: {file_pattern}. Using the first one.')
            continue
        else:
            logging.info(f'Seg found following the pattern: {file_pattern}')
            seg_file = files[0]

        # Define remote paths
        remote_img_path = os.path.join(REMOTE_BASE_PATH, modality, pid, 'image.nii.gz')
        remote_seg_path = os.path.join(REMOTE_BASE_PATH, modality, pid, 'mask.nii.gz')

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
    print(f"Transfer summary for {modality}:")
    print(f"  Successful transfers: {successful_transfers}")
    print(f"  Failed transfers: {failed_transfers}")
    print(f"  Total attempted: {successful_transfers + failed_transfers}")
        

if __name__ == "__main__":

    download_data('T1W')