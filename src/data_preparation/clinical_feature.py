import pandas as pd
import os

CLINICAL_INFO = '/projects/0/prjs1425/Osteosarcoma_WORC/image_records/clinical_features_factorized.csv'
clinical_data = pd.read_csv(CLINICAL_INFO)

def generate_clinical_features(data_path):
    """
    Generate a clinical features CSV file from the provided clinical data CSV.
    
    Parameters:
    data_path (str): Path to the input data containing subject folders.
    """

    df = pd.DataFrame()
    
    # Get list of subject directories
    subject_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    df['pid_n'] = sorted(subject_dirs)

    # Map clinical data safely
    df['Age_Start'] = df['pid_n'].apply(lambda x: get_clinical_value(x, 'Age_Start'))
    df['geslacht'] = df['pid_n'].apply(lambda x: get_clinical_value(x, 'geslacht'))
    df['pres_sympt'] = df['pid_n'].apply(lambda x: get_clinical_value(x, 'pres_sympt'))
    df['path_fract'] = df['pid_n'].apply(lambda x: get_clinical_value(x, 'path_fract'))
    df['Tumor_location'] = df['pid_n'].apply(lambda x: get_clinical_value(x, 'Tumor_location'))
    df['Soft_Tissue_Exp'] = df['pid_n'].apply(lambda x: get_clinical_value(x, 'Soft_Tissue_Exp'))

    # save only features
    df.rename(columns={'pid_n': 'Patient'}, inplace=True)
    df.to_csv(f'{data_path}/clinical_features.csv', index=False)
    df['Huvosnew'] = df['Patient'].apply(lambda x: get_clinical_value(x, 'Huvosnew'))


    # save features with ground truth Huvos
    df.to_csv(f'{data_path}/clinical_features_with_Huvos.csv', index=False)

def get_clinical_value(pid_n, column_name):
    """Safely get clinical data value with error handling"""
    try:
        # Extract patient ID (remove the _01, _02 suffix)
        patient_id = pid_n.replace('OS_0', 'OS_00')[:-3]
        
        # Find matching patient in clinical data
        patient_data = clinical_data[clinical_data['Patient'] == patient_id]
        
        if len(patient_data) == 0:
            print(f"Warning: No clinical data found for patient {patient_id}")
            return None
        
        value = patient_data[column_name].values[0]
        
        # Handle empty/NaN values
        if pd.isna(value) or value == '':
            return None
            
        return value
        
    except Exception as e:
        print(f"Error getting {column_name} for {pid_n}: {e}")
        return None
    
if __name__ == "__main__":

    modalities = ['T2W_FS', 'T1W_FS_C']
    versions = ['0', '1', '9'] 
    # modalities = ['T1W']
    # versions = ['0', '9'] 
    for modality in modalities:
        for version in versions:
            data_path = f'/projects/0/prjs1425/Osteosarcoma_WORC/exp_data/{modality}/v{version}'
            generate_clinical_features(data_path)

    

    
