import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from utils import get_baseline_df
import os
from sklearn.preprocessing import LabelEncoder
from simpleML import run_simple_ML

def categorize(df, variables):

    # Initialize label encoder
    le = LabelEncoder()
    mapping_dict = {}
    # For a single column
    for column_name in variables:
        df[column_name + '_num'] = le.fit_transform(df[column_name])

        # Create mapping dictionary
        mapping_dict[column_name] = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"Mapping for {column_name}: {mapping_dict[column_name]}")
    
    return df, mapping_dict


def get_clean_data(data_dir):
    df = pd.read_csv(data_dir, sep=';')

    # Get pids
    included_df = get_baseline_df(df)
    included_df.to_csv('/gpfs/work1/0/prjs1425/shark/preprocessing/clinical_analysis/included_pid_full_cli_info_loc.csv', index=False)

    # Simple cleaning
    included_df['Age_Start'] = included_df['Age_Start'].str.replace(',', '.').astype(float).astype(int)
    included_df = included_df.rename(columns={'geslacht': 'sex'})
    return included_df

if __name__ == "__main__":

    data_dir = '/gpfs/work1/0/prjs1425/shark/clinical_features/osteosarcoma_t.csv'
    included_df = get_clean_data(data_dir)

    variables = ['Subject', 
                 'Age_Start', 'sex', 'pres_sympt', 
                 'Location_extremity_no_extremity', 
                 'loc_prim_code',
                 'Diagnosis_high',
                #  'path_fract',  # should be removed since it overlap with symptons
                 'Distant_meta_pres',
                 'Size_primary_tumor', 
                 'CTX_pre_op_new',
                 'Huvos']
    data = included_df[variables]
    
    # categorical variables
    cat = variables[2:]
    # Replace all the empty string as 'unknown'
    data = data.replace(['', ' ', '  ', '   ', '\t', '\n'], 'unknown')
    # Categorize variables
    df_test, mapping_dict = categorize(data, cat)

    # Prepare input and label for Simple ML
    label = data['Huvos_num']  # Adjust if necessary, ensure it's the label column
    input = pd.DataFrame()
    # get cat columns
    input['Age_Start'] = data['Age_Start']
    for col in variables:
        if col in cat[:-1]:
            input[col] = df_test[col + '_num']
    
    res_dir = '/gpfs/work1/0/prjs1425/shark/preprocessing/clinical_analysis/simpleML'
    input.to_csv(os.path.join(res_dir,'input_loc.csv'), index=False)
    run_simple_ML(input, label, res_dir)

    # Save a version for WORC input with 'Patient'
    input['Patient'] = data['Subject']
    input['Huvosnew'] = data['Huvos_num']
    cols = input.columns.tolist()
    cols.insert(0, cols.pop(cols.index('Patient')))
    input = input[cols]
    input.to_csv(os.path.join('/gpfs/work1/0/prjs1425/shark/preprocessing/clinical_analysis','WORC_clinical_input_loc.csv'), index=False)


    # Check simple ML for different age group
    data_C = data[data['Age_Start'] < 16]
    data_AYA = data[(data['Age_Start'] >= 16) & (data['Age_Start'] < 40)]
    data_OD = data[data['Age_Start'] >= 40]

    def simpleML(data, exp_name):
        label = data['Huvos_num']
        input = pd.DataFrame()
        input['Age_Start'] = data['Age_Start']
        for col in variables:
            if col in cat[:-1]:
                input[col] = df_test[col + '_num']
        
        run_simple_ML(input, label, res_dir, exp_name=exp_name)
    
    simpleML(data_C, 'Children')
    simpleML(data_AYA, 'AYA')
    simpleML(data_OD, 'OD')

