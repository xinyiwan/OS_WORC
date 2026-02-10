import pandas as pd
import numpy as np
 
 
def get_first_records(subject):
    """
    Docstring for get_first_records
    
    :param subject: Subject ID 'OS_000XXX'
    """
    target_dir = '/exports/lkeb-hpc/xwan/osteosarcoma/OS_seg/{subject}/*/segmentation_history/history.json'
 
    # get all segmentation history files for the subject
    import glob
    seg_history_files = glob.glob(target_dir.format(subject=subject))
 
    if len(seg_history_files) > 1:
        # Check if there is correction record under the same folder
        for seg_file in seg_history_files:
            review_file = seg_file.replace('segmentation_history', 'review_*')
            review_files = glob.glob(review_file)
            if len(review_files) == 0:
                seg_history_files.remove(seg_file)
 
 
    return len(seg_history_files)

 
 
 
if __name__ == "__main__":
    image_data = pd.read_csv('/exports/lkeb-hpc/xwan/osteosarcoma/preprocessing/modality/included_images_app.csv')
    subjects = image_data['Subject'].unique().tolist()
 
    for subject in subjects:
        n_records = get_first_records(subject)
        if n_records > 1:
            print(f"Subject {subject} has {n_records} segmentation records.")
        if n_records == 0:
            print(f"Subject {subject} has {n_records} segmentation records.")
 
 
 