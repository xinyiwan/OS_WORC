import pandas as pd 
import numpy as np 

def get_baseline_df(df):

    """
    Include the baseline patients and return the df with only those patients. 
    """
    
    # put the anonymized pids 
    mapping = pd.read_csv('/exports/lkeb-hpc/xwan/osteosarcoma/clinical_features/patid_xnat_lut_20250307.csv')
    mapping = mapping.rename(columns={'Original_PatientID': 'pat_nr'})
    df = pd.merge(df, mapping, on='pat_nr')

    df['Trial_PatientID'] = 'OS_' + df['Trial_PatientID'].astype(str).str.zfill(6)
    df = df.rename(columns={'Trial_PatientID': 'Subject'})

    # Move column 'Subject' to the third position
    cols = df.columns.tolist()
    cols.insert(2, cols.pop(cols.index('Subject')))
    df = df[cols]

    # Get included pids from preprocessed imaging 
    inclusion = pd.read_csv('/exports/lkeb-hpc/xwan/osteosarcoma/preprocessing/modality/included_images_app.csv')
    included_pids = sorted(np.unique(inclusion['Subject']))

    df = df[df['Subject'].isin(included_pids)].reset_index(drop=True)
    return df

