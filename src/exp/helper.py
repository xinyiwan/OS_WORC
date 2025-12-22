import os, glob
import pandas as pd
import shutil
from collections import Counter

CLINI_INFO = '/projects/0/prjs1425/Osteosarcoma_WORC/image_records/WORC_clinical_input.csv'
DATA_C = pd.read_csv(CLINI_INFO)

def get_imgs_by_agegroup(modality, version, age_group, exp_name):
    """
    age_group = Children: age < 16 
    age_group = AYA: age >=16, age < 40 
    age_group = Older_adults: age >= 40
    """
    DATA_C['Age_Start'] = DATA_C['Age_Start'].replace(',','.', regex=True).astype(float)
    DATA_C['Age_group'] = DATA_C['Age_Start'].apply(categorize_age)
    included_pids = DATA_C[DATA_C['Age_group'] == age_group]['Patient'].tolist()

    exp_data_dir = os.path.join('/projects/0/prjs1425/Osteosarcoma_WORC/exp_data', exp_name, modality, version)
    os.makedirs(exp_data_dir, exist_ok=True)

    image_file_name = 'image.nii.gz'
    segmentation_file_name = 'mask.nii.gz'

    imagedatadir = f'/projects/0/prjs1425/Osteosarcoma_WORC/exp_data/{modality}/{version}'
    images = glob.glob(os.path.join(imagedatadir, "*", image_file_name))

    # Filter images and segmentations to only include patients in the specified age group
    filtered_images = []
    filtered_segs = []
    
    for img_path in images:
        # Extract patient ID from the path 
        patient_id = os.path.basename(os.path.dirname(img_path))[:-3]
        if patient_id in included_pids:
            filtered_images.append(img_path)
            seg_path = img_path.replace(image_file_name, segmentation_file_name)
            if os.path.exists(seg_path):
                filtered_segs.append(seg_path)
    
    images_dict = {f"{os.path.basename(os.path.dirname(image))}_0": image for image in filtered_images}
    segs_dict = {f"{os.path.basename(os.path.dirname(seg))}_0": seg for seg in filtered_segs}
    
    # Generate CSV files with clinical features based on existing ones
    cli_csv = os.path.join(imagedatadir, 'clinical_features.csv')
    cli_csv_huvos = os.path.join(imagedatadir, 'clinical_features_with_Huvos.csv')
    patient_splits = os.path.join(imagedatadir, 'patient_splits.csv')
    
    def filter_csv_by_pids(csv_path, included_pids, is_splits_file=False):
        """Filter CSV files to only include specified patient IDs"""
        if not os.path.exists(csv_path):
            print(f"Warning: CSV file not found: {csv_path}")
            return None
        
        df = pd.read_csv(csv_path)
        
        if is_splits_file:
            # For splits files, filter each column to only include image IDs from our age group
            # First pass: find the maximum length needed for any column
            max_length = 0
            filtered_columns = {}
            
            for column in df.columns:
                original_ids = df[column].dropna().tolist()
                filtered_ids = []
                
                for image_id in original_ids:
                    # Extract base patient ID from image ID (remove _01, _02 suffix)
                    if pd.isna(image_id) or image_id == '':
                        continue
                    base_patient_id = image_id[:-3] if isinstance(image_id, str) and len(image_id) > 10 and '_' in image_id else image_id
                    if base_patient_id in included_pids:
                        filtered_ids.append(image_id)
                
                # Update maximum length
                max_length = max(max_length, len(filtered_ids))
                filtered_columns[column] = filtered_ids
            
            # Second pass: build the DataFrame with consistent length
            filtered_df = pd.DataFrame()
            for column, filtered_ids in filtered_columns.items():
                # Pad with empty strings to match the maximum length
                padded_ids = filtered_ids + [''] * (max_length - len(filtered_ids))
                filtered_df[column] = padded_ids
            
            print(f"Splits file: original shape {df.shape}, filtered shape {filtered_df.shape}")
            return filtered_df
        else:
            # For clinical feature files, filter rows based on Patient column
            if 'Patient' in df.columns:
                # Handle both subject-level and image-level patient IDs
                def get_base_patient_id(patient_id):
                    if isinstance(patient_id, str) and len(patient_id) > 10 and '_' in patient_id:
                        return patient_id[:-3]
                    return patient_id
                
                if df['Patient'].dtype == 'object':  # String type
                    df['base_patient'] = df['Patient'].apply(get_base_patient_id)
                    filtered_df = df[df['base_patient'].isin(included_pids)].copy()
                    filtered_df.drop('base_patient', axis=1, inplace=True)
                else:
                    filtered_df = df[df['Patient'].isin(included_pids)].copy()
                print(f"Clinical file: original shape {df.shape}, filtered shape {filtered_df.shape}")
                return filtered_df
            else:
                print(f"Warning: No 'Patient' column found in {csv_path}")
                return df
    
    # Filter and save all CSV files
    csv_files_to_filter = [
        (cli_csv, False),
        (cli_csv_huvos, False),
        (patient_splits, True),
    ]
    
    for csv_path, is_splits in csv_files_to_filter:
        if os.path.exists(csv_path):
            filtered_data = filter_csv_by_pids(csv_path, included_pids, is_splits)
            if filtered_data is not None:
                output_path = os.path.join(exp_data_dir, os.path.basename(csv_path))
                filtered_data.to_csv(output_path, index=False)
                print(f"Saved filtered {os.path.basename(csv_path)} to {output_path}")
    
    # Make label distribution based on image level from patient_splits
    def create_label_distribution(data_path, iter=20):
        """Create label distribution CSV based on patient splits and clinical features"""
        clinical_file = os.path.join(data_path, 'clinical_features_with_Huvos.csv')
        data_splits = os.path.join(data_path, 'patient_splits.csv')
        
        if not os.path.exists(clinical_file) or not os.path.exists(data_splits):
            print("Warning: Required files not found for label distribution")
            return None
        
        clinical_data = pd.read_csv(clinical_file)
        splits = pd.read_csv(data_splits)

        # Create a DataFrame to store all distribution information
        distribution_data = []
        
        for i in range(iter):
            train_col = f'{i}_train'
            test_col = f'{i}_test'
            
            if train_col not in splits.columns or test_col not in splits.columns:
                continue
                
            train_ids = splits[train_col].dropna().tolist()
            test_ids = splits[test_col].dropna().tolist()

            # Get labels for train and test sets
            train_labels = []
            for pid in train_ids:
                if pid and pid in clinical_data['Patient'].values:
                    label = clinical_data[clinical_data['Patient'] == pid]['Huvosnew'].values[0]
                    if pd.notna(label):
                        train_labels.append(label)
            
            test_labels = []
            for pid in test_ids:
                if pid and pid in clinical_data['Patient'].values:
                    label = clinical_data[clinical_data['Patient'] == pid]['Huvosnew'].values[0]
                    if pd.notna(label):
                        test_labels.append(label)

            # Calculate distributions
            train_total = len(train_labels)
            test_total = len(test_labels)
            
            train_0_count = train_labels.count(0)
            train_1_count = train_labels.count(1)
            test_0_count = test_labels.count(0)
            test_1_count = test_labels.count(1)
            
            train_0_pct = round(train_0_count/train_total, 3) if train_total > 0 else 0
            train_1_pct = round(train_1_count/train_total, 3) if train_total > 0 else 0
            test_0_pct = round(test_0_count/test_total, 3) if test_total > 0 else 0
            test_1_pct = round(test_1_count/test_total, 3) if test_total > 0 else 0

            # Store distribution data
            distribution_data.append({
                'iteration': i,
                'train_total': train_total,
                'train_label_0': train_0_count,
                'train_label_1': train_1_count,
                'train_pct_0': train_0_pct,
                'train_pct_1': train_1_pct,
                'test_total': test_total,
                'test_label_0': test_0_count,
                'test_label_1': test_1_count,
                'test_pct_0': test_0_pct,
                'test_pct_1': test_1_pct
            })

            print(f"Iteration {i}:")
            print(f"  Train set - Total: {train_total}, 0: {train_0_pct}, 1: {train_1_pct}")
            print(f"  Test set - Total: {test_total}, 0: {test_0_pct}, 1: {test_1_pct}")
        
        # Convert to DataFrame and save to CSV
        if distribution_data:
            distribution_df = pd.DataFrame(distribution_data)
            output_file = os.path.join(data_path, 'label_distributions.csv')
            distribution_df.to_csv(output_file, index=False)
            print(f"Label distributions saved to: {output_file}")
            return distribution_df
        else:
            print("No distribution data generated")
            return None
    
    # Create label distribution
    print("Creating label distributions...")
    distribution_df = create_label_distribution(exp_data_dir)
    
    return images_dict, segs_dict, exp_data_dir


# Categorize ages into groups
def categorize_age(age):
    if age < 16:
        return "Children"
    elif 16 <= age < 40:
        return "AYA"
    else:
        return "Older_adults"

if __name__ == '__main__':

    modalities = ['T1W', 'T1W_FS_C', 'T2W_FS']
    versions = ['v0', 'v1', 'v9']
    age_groups = ['Children', 'AYA', 'Older_adults']

    for m in modalities:
        for v in versions:
            for group in age_groups:
                get_imgs_by_agegroup(modality=m, version=v, age_group=group, exp_name=group)


def create_overfit_splits(data_path, n_splits=20):
    """Create overfit splits where all data is in training set"""
    data_splits = os.path.join(data_path, 'patient_splits.csv')
    
    if not os.path.exists(data_splits):
        print("Warning: patient_splits.csv not found for overfit splits")
        return None
    
    splits = pd.read_csv(data_splits)
    all_image_ids = []
    
    # Collect all image IDs from all iterations
    for i in range(n_splits):
        train_col = f'{i}_train'
        test_col = f'{i}_test'
        
        if train_col in splits.columns:
            all_image_ids.extend(splits[train_col].dropna().tolist())
        if test_col in splits.columns:
            all_image_ids.extend(splits[test_col].dropna().tolist())
    
    all_image_ids = list(set(all_image_ids))  # Unique IDs
    max_length = len(all_image_ids)
    
    # Create new splits DataFrame and keep the test sets to be same as given splits
    overfit_splits = pd.DataFrame()
    for i in range(n_splits):
        overfit_splits[f'{i}_train'] = all_image_ids
        overfit_splits[f'{i}_test'] = splits[f'{i}_test']
    
    output_file = os.path.join(data_path, 'patient_overfit_splits.csv')
    overfit_splits.to_csv(output_file, index=False)
    print(f"Overfit splits saved to: {output_file}")
    return overfit_splits