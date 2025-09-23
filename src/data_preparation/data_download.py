import os, glob
import pandas as pd
import logging

IMAGE_RECORDS_DIR = '/projects/0/prjs1425/Osteosarcoma_WORC/image_records/included_images_app.csv'
DATA_DIR = '/projects/0/prjs1425/Osteosarcoma_WORC/exp_data'
RECORDS = pd.read_csv(IMAGE_RECORDS_DIR)
LOG_DIR = '/projects/0/prjs1425/Osteosarcoma_WORC/logs'

# logging file
logging.basicConfig(filename=f'{LOG_DIR}/data_download.log', level=logging.INFO,
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


def download_data(modality='T1W'):

    # Create directory if it doesn't exist
    if not os.path.exists(f'{DATA_DIR}/{modality}'):
        os.makedirs(f'{DATA_DIR}/{modality}')
        logging.info(f'Created directory: {DATA_DIR}/{modality}')
    
    imgs = RECORDS[RECORDS['modality'] == modality]

    # Apply the function to create 'pid_n'
    imgs = imgs.groupby('Subject').apply(create_pid_n).reset_index(drop=True)
    imgs = imgs.sort_values('pid_n').reset_index(drop=True)

    # Reorder columns to have pid_n first
    cols = imgs.columns.tolist()
    cols = ['pid_n'] + [col for col in cols if col != 'pid_n']
    imgs = imgs[cols]

    for idx, row in imgs.iterrows():
        pid = row['pid_n']
        subject = row['Subject']
        session = row['session']
        scan = row['scan']

        # create subject directory if it doesn't exist
        subject_dir = f'{DATA_DIR}/{modality}/{pid}'
        if not os.path.exists(subject_dir):
            os.makedirs(subject_dir)
            logging.info(f'Created directory: {subject_dir}')
        
        # Find the image file and download it through rsync




if __name__ == "__main__":
    download_data('T1W')