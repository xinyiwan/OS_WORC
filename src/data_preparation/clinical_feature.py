import pandas as pd
import os

CLINICAL_INFO = '/projects/0/prjs1425/Osteosarcoma_WORC/image_records/clinical_features_factorized.csv'
clinical_data = pd.read_csv(CLINICAL_INFO)

def generate_clinical_features(data_path, level='image'):
    """
    Generate a clinical features CSV file from the provided clinical data CSV.
    
    Parameters:
    data_path (str): Path to the input data containing subject folders.
    """

    df = pd.DataFrame()
    
    # Get list of subject directories
    subject_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]

    if level == 'subject':
        subject_dirs = [d[:-3] for d in subject_dirs]
        df['pid_n'] = sorted(set(subject_dirs))

    # If image level
    if level == 'image':
        df['pid_n'] = sorted(subject_dirs)

    # Map clinical data safely
    df['Age_Start'] = df['pid_n'].apply(lambda x: get_clinical_value(x, 'Age_Start', level=level))
    df['geslacht'] = df['pid_n'].apply(lambda x: get_clinical_value(x, 'geslacht', level=level))
    df['pres_sympt'] = df['pid_n'].apply(lambda x: get_clinical_value(x, 'pres_sympt', level=level))
    df['path_fract'] = df['pid_n'].apply(lambda x: get_clinical_value(x, 'path_fract', level=level))
    df['Tumor_location'] = df['pid_n'].apply(lambda x: get_clinical_value(x, 'Tumor_location', level=level))
    df['Soft_Tissue_Exp'] = df['pid_n'].apply(lambda x: get_clinical_value(x, 'Soft_Tissue_Exp', level=level))

    # save only features
    df.rename(columns={'pid_n': 'Patient'}, inplace=True)
    df.to_csv(f'{data_path}/clinical_features.csv', index=False)
    df['Huvosnew'] = df['Patient'].apply(lambda x: get_clinical_value(x, 'Huvosnew'))


    # save features with ground truth Huvos
    df.to_csv(f'{data_path}/clinical_features_with_Huvos.csv', index=False)

def get_clinical_value(pid_n, column_name, level='image'):
    """Safely get clinical data value with error handling"""
    try:
        # Extract patient ID (remove the _01, _02 suffix)
        patient_id = pid_n[:-3]
        patient_id = patient_id.replace('OS_0', 'OS_00')
        # patient_id = pid_n
        
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


def modify_split(data_path, excel_file_path, level='image'):
    """
    Modify splits based on available subject/image directories in data_path
    
    Parameters:
    - data_path: path to the data directory containing subject/image folders
    - excel_file_path: path to the existing Excel file with splits
    - level: 'subject' or 'image'
    """
    
    # Get list of directories
    all_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    
    if level == 'subject':
        # For subject level, remove the suffix to get base subject IDs
        available_ids = set([d[:-3] for d in all_dirs if '_' in d])
        id_type = "Subject"
    else:  # image level
        # For image level, use the full image IDs
        available_ids = set(all_dirs)
        id_type = "Image"
    
    print(f"Available {id_type} IDs: {len(available_ids)}")
    
    # Read the existing Excel file
    df = pd.read_csv(excel_file_path)
    
    # Create a modified dataframe
    modified_data = {}
    
    # First pass: find the maximum length needed for any column
    max_needed_length = 0
    
    for column in df.columns:
        original_ids = df[column].dropna().tolist()  # Remove empty cells
        
        if level == 'subject':
            # Direct match for subject level
            available_in_column = [pid for pid in original_ids if pid in available_ids]
        else:  # image level
            # For image level, expand each subject ID to all its images
            available_in_column = []
            for subject_id in original_ids:
                # Find all images that belong to this subject
                matching_images = [img_id for img_id in available_ids 
                                 if img_id.startswith(subject_id + '_')]
                if matching_images:
                    available_in_column.extend(sorted(matching_images))
        
        # Update the maximum length needed
        max_needed_length = max(max_needed_length, len(available_in_column))
    
    print(f"Maximum column length needed: {max_needed_length}")
    
    # Second pass: build the columns with consistent length
    for column in df.columns:
        original_ids = df[column].dropna().tolist()  # Remove empty cells
        
        if level == 'subject':
            # Direct match for subject level
            available_in_column = [pid for pid in original_ids if pid in available_ids]
            missing_in_column = [pid for pid in original_ids if pid not in available_ids]
        else:  # image level
            # For image level, expand each subject ID to all its images
            available_in_column = []
            missing_in_column = []
            
            for subject_id in original_ids:
                # Find all images that belong to this subject
                matching_images = [img_id for img_id in available_ids 
                                 if img_id.startswith(subject_id + '_')]
                if matching_images:
                    available_in_column.extend(sorted(matching_images))
                else:
                    missing_in_column.append(subject_id)
        
        print(f"\nColumn: {column}")
        print(f"  Original: {len(original_ids)} {id_type.lower()} IDs")
        print(f"  Available: {len(available_in_column)} {id_type.lower()} IDs")
        print(f"  Missing: {len(missing_in_column)} {id_type.lower()} IDs")
        if missing_in_column and len(missing_in_column) <= 10:  # Only print if not too many
            print(f"  Missing IDs: {missing_in_column}")
        
        # Pad with empty strings to match the maximum length needed
        modified_data[column] = available_in_column + [''] * (max_needed_length - len(available_in_column))
    
    # Create new DataFrame
    modified_df = pd.DataFrame(modified_data)
    
    # Save the modified Excel file
    output_file = os.path.join(data_path, 'patient_splits.csv')
    modified_df.to_csv(output_file, index=False)
    
    print(f"\nModified split file saved as: {output_file}")
    print(f"DataFrame shape: {modified_df.shape}")
    
    return modified_df, available_ids

def check_label_distribution(data_path, clinical_file='clinical_features_with_Huvos.csv', iter=20):
    clinical_file = os.path.join(data_path, clinical_file)
    data_splits = os.path.join(data_path, 'patient_splits.csv')    
    clinical_data = pd.read_csv(clinical_file)
    splits = pd.read_csv(data_splits)

    for i in range(iter):
        train_ids = splits[f'{i}_train'].dropna().tolist()
        test_ids = splits[f'{i}_test'].dropna().tolist()

        train_labels = [clinical_data[clinical_data['Patient'] == pid]['Huvosnew'].values[0] for pid in train_ids if pid in clinical_data['Patient'].values]
        test_labels = [clinical_data[clinical_data['Patient'] == pid]['Huvosnew'].values[0] for pid in test_ids if pid in clinical_data['Patient'].values]

        print(f"Iteration {i+1}:")
        print(f"  Train set - Total: {len(train_labels)}, 0: {train_labels.count(0)}, 1: {train_labels.count(1)}")
        print(f"  Test set - Total: {len(test_labels)}, 0: {test_labels.count(0)}, 1: {test_labels.count(1)}")  
        print("\n")
    
if __name__ == "__main__":

    # modalities = ['T2W_FS', 'T1W_FS_C']
    # versions = ['0', '1', '9'] 
    modalities = ['T1W']
    versions = ['0', '9'] 

    excel_file_path = '/projects/0/prjs1425/Osteosarcoma_WORC/image_records/patient_splits.csv'
    for modality in modalities:
        for version in versions:
            data_path = f'/projects/0/prjs1425/Osteosarcoma_WORC/exp_data/{modality}/v{version}'
            generate_clinical_features(data_path, level='image')
            modified_df, available_ids = modify_split(data_path, excel_file_path, level='image')
            check_label_distribution(data_path)

