import os
import pandas as pd 
import numpy as np

modalities = ['T1W_FS_C', 'T1W', 'T2W_FS']
versions = ['v9', 'v0', 'v1' ]

for modality in modalities:
    # print(f"Check modality: {modality}.")
    included_df = pd.read_csv(f'/gpfs/work1/0/prjs1425/shark/preprocessing/dataloader/{modality}_df.csv')
    session_ids = included_df['pid_n']
    subject_ids = list(np.unique(included_df['Subject']))
    print('=' * 80)
    for version in versions:
        # print(f"Check version: {version}.")
        dir_ids = f'/projects/0/prjs1425/Osteosarcoma_WORC/exp_data/{modality}/{version}'
        exp_session_ids = os.listdir(dir_ids)
        exp_session_ids = [pid for pid in exp_session_ids if 'OS' in pid]
        exp_subject_ids = list(np.unique([pid[:-3] for pid in exp_session_ids]))
        
        non_overlap_img = set(session_ids) ^ set(exp_session_ids)
        non_overlap_sub = set(subject_ids) ^ set(exp_subject_ids)

        print(f" {len(exp_session_ids)} images and {len(exp_subject_ids)} images in exp of [{modality}][{version}]. ")

        print(non_overlap_img)
        print(non_overlap_sub)