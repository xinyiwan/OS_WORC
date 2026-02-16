import pandas as pd
import os
from collections import Counter

CLINICAL_INFO = '/projects/0/prjs1425/Osteosarcoma_WORC/image_records/WORC_clinical_input.csv'

clinical_data = pd.read_csv(CLINICAL_INFO)
WIRC_data = pd.read_csv('/projects/0/prjs1425/Osteosarcoma_WORC/image_records/WIR_patient_mapping.csv')

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
        subject_dirs = [d[:-3] if len(d) > 10 else d for d in subject_dirs]
        df['pid_n'] = sorted(set(subject_dirs))

    # If image level
    if level == 'image':
        df['pid_n'] = sorted(subject_dirs)

    # Map clinical data safely
    df['Age_Start'] = df['pid_n'].apply(lambda x: get_clinical_value(x, 'Age_Start', level=level))
    df['sex'] = df['pid_n'].apply(lambda x: get_clinical_value(x, 'sex', level=level))
    df['pres_sympt'] = df['pid_n'].apply(lambda x: get_clinical_value(x, 'pres_sympt', level=level))
    # df['path_fract'] = df['pid_n'].apply(lambda x: get_clinical_value(x, 'path_fract', level=level))
    df['location'] = df['pid_n'].apply(lambda x: get_clinical_value(x, 'Location_extremity_no_extremity', level=level))
    df['diagnosis'] = df['pid_n'].apply(lambda x: get_clinical_value(x, 'Diagnosis_high', level=level))
    df['metastasis'] = df['pid_n'].apply(lambda x: get_clinical_value(x, 'Distant_meta_pres', level=level))
    df['tumor_size'] = df['pid_n'].apply(lambda x: get_clinical_value(x, 'Size_primary_tumor', level=level))

    # NAC info does not belong to clinical features
    # df['NAC'] = df['pid_n'].apply(lambda x: get_clinical_value(x, 'CTX_pre_op_new', level=level))

    
    # save only features
    df.rename(columns={'pid_n': 'Patient'}, inplace=True)
    df.to_csv(f'{data_path}/clinical_features.csv', index=False)

    if 'WIR' in data_path:
        df['WIR_label'] = df['Patient'].apply(lambda x: get_WIR_label(x))
        df.to_csv(f'{data_path}/clinical_features_with_WIR.csv', index=False)
    else:
        df['Huvosnew'] = df['Patient'].apply(lambda x: get_clinical_value(x, 'Huvosnew', level=level))
        # save features with ground truth Huvos
        df.to_csv(f'{data_path}/clinical_features_with_Huvos.csv', index=False)

def generate_clinical_features_bylist(pid_list, save_path, level='subject'):
    """
    Generate a clinical features CSV file from the provided clinical data CSV.
    
    Parameters:
    pid_list (List): List of pids.
    """

    df = pd.DataFrame()
    
    # Get list of subject directories
    df['pid_n'] = sorted(set(pid_list))


    # Map clinical data safely
    df['Age_Start'] = df['pid_n'].apply(lambda x: get_clinical_value(x, 'Age_Start', level=level))
    df['sex'] = df['pid_n'].apply(lambda x: get_clinical_value(x, 'sex', level=level))
    df['pres_sympt'] = df['pid_n'].apply(lambda x: get_clinical_value(x, 'pres_sympt', level=level))
    # df['path_fract'] = df['pid_n'].apply(lambda x: get_clinical_value(x, 'path_fract', level=level))
    df['location'] = df['pid_n'].apply(lambda x: get_clinical_value(x, 'Location_extremity_no_extremity', level=level))
    df['diagnosis'] = df['pid_n'].apply(lambda x: get_clinical_value(x, 'Diagnosis_high', level=level))
    df['metastasis'] = df['pid_n'].apply(lambda x: get_clinical_value(x, 'Distant_meta_pres', level=level))
    df['tumor_size'] = df['pid_n'].apply(lambda x: get_clinical_value(x, 'Size_primary_tumor', level=level))

    # NAC info does not belong to clinical features
    # df['NAC'] = df['pid_n'].apply(lambda x: get_clinical_value(x, 'CTX_pre_op_new', level=level))

    
    # save only features
    df.rename(columns={'pid_n': 'Patient'}, inplace=True)
    df.to_csv(f'{save_path}/clinical_features.csv', index=False)

    if 'WIR' in save_path:
        df['WIR_label'] = df['Patient'].apply(lambda x: get_WIR_label(x))
        df.to_csv(f'{save_path}/clinical_features_with_WIR.csv', index=False)
    else:
        df['Huvosnew'] = df['Patient'].apply(lambda x: get_clinical_value(x, 'Huvosnew', level=level))
        # save features with ground truth Huvos
        df.to_csv(f'{save_path}/clinical_features_with_Huvos.csv', index=False)



def get_clinical_value(pid_n, column_name, level='image'):
    """Safely get clinical data value with error handling"""
    try:
        # Extract patient ID (remove the _01, _02 suffix)
        if level == 'image':
            patient_id = pid_n[:-3]

        if level == 'subject':
            patient_id = pid_n
        
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
    
def get_WIR_label(pid_n):
    """Get WIR label for a given patient ID"""
    try:
        patient_id = pid_n[:-3]
        patient_data = WIRC_data[WIRC_data['mapped_id'] == patient_id]
        
        if len(patient_data) == 0:
            print(f"Warning: No WIR data found for patient {patient_id}")
            return None
        
        value = patient_data['perfusion_label'].values[0]
        
        if pd.isna(value) or value == '':
            return None
            
        return value
        
    except Exception as e:
        print(f"Error getting WIR_label for {pid_n}: {e}")
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
        available_ids = set([d[:-3] if len(d)>10 else d for d in all_dirs if '_' in d])
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

def modify_split_bypids(pid_list, excel_file_path, save_path, level='subject'):
    """
    Modify splits based on available subject/image directories in data_path
    
    Parameters:
    - pid_list: Patient list
    - excel_file_path: path to the existing Excel file with splits
    - level: 'subject' or 'image'
    """
    
    # Get list and level
    id_type = "Subject"
    available_ids = set(pid_list)

    
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
    output_file = os.path.join(save_path, 'patient_splits.csv')
    modified_df.to_csv(output_file, index=False)
    
    print(f"\nModified split file saved as: {output_file}")
    print(f"DataFrame shape: {modified_df.shape}")
    
    return modified_df, available_ids
    
def check_label_distribution(data_path, iter=20):
    label_type = 'Huvos' if not 'WIR' in data_path else 'WIR' if 'WIR' in data_path else 'Unknown'
    if label_type == 'Unknown':
        print(f"Warning: Unrecognized clinical file '{data_path}'. Cannot determine label type.")
        return
    
    clinical_file = os.path.join(data_path, f'clinical_features_with_{label_type}.csv')
    data_splits = os.path.join(data_path, 'patient_splits.csv')    
    clinical_data = pd.read_csv(clinical_file)
    splits = pd.read_csv(data_splits)

    # Create a DataFrame to store all distribution information
    distribution_data = []
    
    for i in range(iter):
        train_ids = splits[f'{i}_train'].dropna().tolist()
        test_ids = splits[f'{i}_test'].dropna().tolist()

        if 'Huvos' in clinical_file:
            train_labels = [clinical_data[clinical_data['Patient'] == pid]['Huvosnew'].values[0] for pid in train_ids if pid in clinical_data['Patient'].values]
            test_labels = [clinical_data[clinical_data['Patient'] == pid]['Huvosnew'].values[0] for pid in test_ids if pid in clinical_data['Patient'].values]
        elif 'WIR' in clinical_file:
            train_labels = [clinical_data[clinical_data['Patient'] == pid]['WIR_label'].values[0] for pid in train_ids if pid in clinical_data['Patient'].values]
            test_labels = [clinical_data[clinical_data['Patient'] == pid]['WIR_label'].values[0] for pid in test_ids if pid in clinical_data['Patient'].values]
        else:   
            continue

        # Calculate distributions
        train_counter = Counter(train_labels)
        test_counter = Counter(test_labels)
        
        train_total = len(train_labels)
        test_total = len(test_labels)
        
        train_0_pct = round(train_labels.count(0)/train_total, 3) if train_total > 0 else 0
        train_1_pct = round(train_labels.count(1)/train_total, 3) if train_total > 0 else 0
        test_0_pct = round(test_labels.count(0)/test_total, 3) if test_total > 0 else 0
        test_1_pct = round(test_labels.count(1)/test_total, 3) if test_total > 0 else 0

        # Store distribution data
        distribution_data.append({
            'iteration': i,
            'modality': os.path.basename(os.path.dirname(data_path)),
            'version': os.path.basename(data_path),
            'train_total': train_total,
            'train_label_0': train_counter.get(0, 0),
            'train_label_1': train_counter.get(1, 0),
            'train_pct_0': train_0_pct,
            'train_pct_1': train_1_pct,
            'test_total': test_total,
            'test_label_0': test_counter.get(0, 0),
            'test_label_1': test_counter.get(1, 0),
            'test_pct_0': test_0_pct,
            'test_pct_1': test_1_pct
        })

        print(f"Iteration {i+1}:")
        print(f"  Train set - Total: {train_total}, 0: {train_0_pct}, 1: {train_1_pct}")
        print(f"  Test set - Total: {test_total}, 0: {test_0_pct}, 1: {test_1_pct}")
        print("\n")
    
    # Convert to DataFrame and save to CSV
    distribution_df = pd.DataFrame(distribution_data)
    output_file = os.path.join(data_path, 'label_distributions.csv')
    distribution_df.to_csv(output_file, index=False)
    print(f"Label distributions saved to: {output_file}")
    
    return distribution_df

def create_summary_distribution_csv(data_dir):
    """Create a summary CSV with distributions across all modalities and versions"""
    modalities = ['T1W', 'T2W_FS', 'T1W_FS_C']
    versions = ['0', '1', '9']
    
    all_distributions = []
    
    for modality in modalities:
        for version in versions:
            data_path = os.path.join(data_dir, f'{modality}/v{version}')
            distribution_file = os.path.join(data_path, 'label_distributions.csv')
            
            if os.path.exists(distribution_file):
                df = pd.read_csv(distribution_file)
                all_distributions.append(df)
    
    if all_distributions:
        summary_df = pd.concat(all_distributions, ignore_index=True)
        summary_file = os.path.join(data_dir, 'label_distributions_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary distribution CSV saved to: {summary_file}")
        return summary_df
    else:
        print("No distribution files found.")
        return None

# Add this call at the end of your main block
if __name__ == "__main__":
    modalities = ['T1W', 'T2W_FS', 'T1W_FS_C']
    versions = ['0', '1', '9'] 
    level = 'subject'
    
    
    # modalities = ['dummy']
    # versions = ['0']
    # level = 'subject'

    excel_file_path = f'/projects/0/prjs1425/Osteosarcoma_WORC/image_records/balance_datasplit/patient_splits.csv'

    for modality in modalities:
        for version in versions:
            data_path = f'/projects/0/prjs1425/Osteosarcoma_WORC/exp_data/{modality}/v{version}'
            generate_clinical_features(data_path, level=level)
            modified_df, available_ids = modify_split(data_path, excel_file_path, level=level)
            check_label_distribution(data_path)
    
    # Create summary across all runs
    create_summary_distribution_csv(data_dir='/projects/0/prjs1425/Osteosarcoma_WORC/exp_data')


    # # WIR
    # excel_file_path = f'/projects/0/prjs1425/Osteosarcoma_WORC/image_records/balance_datasplit/patient_splits_WIR.csv'

    # for modality in modalities:
    #     for version in versions:
    #         data_path = f'/projects/0/prjs1425/Osteosarcoma_WORC/exp_data/WIR/{modality}/v{version}'
    #         generate_clinical_features(data_path, level='image')
    #         modified_df, available_ids = modify_split(data_path, excel_file_path, level='image')
    #         check_label_distribution(data_path)
    
    # # Create summary across all runs
    # create_summary_distribution_csv(data_dir='/projects/0/prjs1425/Osteosarcoma_WORC/exp_data/WIR')