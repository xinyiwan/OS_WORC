import os
os.environ["NUMEXPR_MAX_THREADS"] = "192"  

import sys
sys.path.append('/projects/0/prjs1425/Osteosarcoma_WORC/WORC')
from WORC.facade.simpleworc import SimpleWORC

# These packages are only used in analysing the results
import pandas as pd
import json
import fastr
import glob


# Define the folder this script is in, so we can easily find the example data
script_path = '/gpfs/work1/0/prjs1425/Osteosarcoma_WORC/WORC_COM_OS_tmp/'

# Determine whether you would like to use WORC for binary_classification,
modus = 'binary_classification'

def main(experiment_name, modality, version, mode, type):
    """Execute WORC Tutorial experiment."""
    print(f"Running in folder: {script_path}.")
    # ---------------------------------------------------------------------------
    # Input
    # ---------------------------------------------------------------------------

    # File in which the labels (i.e. outcome you want to predict) is stated
    if type == 'Huvos':
        imagedatadir = f'/projects/0/prjs1425/Osteosarcoma_WORC/exp_data/{modality}/{version}'
        label_file = f'/projects/0/prjs1425/Osteosarcoma_WORC/exp_data/{modality}/{version}/clinical_features_with_{type}.csv'
        label_name = ['Huvosnew']
    if type == 'WIR':
        imagedatadir = f'/projects/0/prjs1425/Osteosarcoma_WORC/exp_data/WIR/{modality}/{version}'
        label_file = f'/projects/0/prjs1425/Osteosarcoma_WORC/exp_data/WIR/{modality}/{version}/clinical_features_with_{type}.csv'
        label_name = ['WIR_label']
        

    image_file_name = 'image.nii.gz'
    segmentation_file_name = 'mask.nii.gz'

    coarse = False

    # Instead of the default tempdir, let's but the temporary output in a subfolder
    # in the same folder as this script
    tmpdir = os.path.join(script_path, experiment_name)
    print(f"Temporary folder: {tmpdir}.")

    # ---------------------------------------------------------------------------
    # The actual experiment
    # ---------------------------------------------------------------------------
    experiment = SimpleWORC(experiment_name)
    # Add dummy images from the directory    
    experiment.images_from_this_directory(imagedatadir,
                                          image_file_name=image_file_name,
                                          is_training=True)
    
    experiment.segmentations_from_this_directory(imagedatadir,
                                                 segmentation_file_name=segmentation_file_name,
                                                 is_training=True)
    
    # Add the clinical features as metadata
    if type == 'Huvos':
        experiment.semantics_from_this_file(os.path.join(imagedatadir, 'clinical_features.csv'))
    experiment.labels_from_this_file(label_file)
    experiment.predict_labels(label_name)

    # Set fixed splits
    experiment.set_fixed_splits(os.path.join(imagedatadir,'patient_splits.csv'))

    experiment.set_image_types(['MRI'])
    experiment.binary_classification(coarse=coarse)
    # Set the temporary directory
    experiment.set_tmpdir(tmpdir)
    # Add evaluation
    experiment.add_evaluation()

    # Change the config for including clinical features or not
    overrides = editconfig(mode=mode)
    experiment.add_config_overrides(overrides)
        
    # Run the experiment!
    experiment.execute()

    # ---------------------------------------------------------------------------
    # Analysis of results
    # ---------------------------------------------------------------------------

    # Locate output folder
    outputfolder = fastr.config.mounts['output']
    experiment_folder = os.path.join(outputfolder, 'WORC_' + experiment_name)

    print(f"Your output is stored in {experiment_folder}.")

    # Read the features for the first patient
    # NOTE: we use the glob package for scanning a folder to find specific files
    feature_files = glob.glob(os.path.join(experiment_folder,
                                           'Features',
                                           'features_*.hdf5'))

    if len(feature_files) == 0:
        raise ValueError('No feature files found: your network has failed.')

    feature_files.sort()
    featurefile_p1 = feature_files[0]
    features_p1 = pd.read_hdf(featurefile_p1)

    # Read the overall peformance
    performance_file = os.path.join(experiment_folder, 'performance_all_0.json')
    if not os.path.exists(performance_file):
        raise ValueError(f'No performance file {performance_file} found: your network has failed.')

    with open(performance_file, 'r') as fp:
        performance = json.load(fp)

    # Print the feature values and names
    print("Feature values from first patient:")
    for v, l in zip(features_p1.feature_values, features_p1.feature_labels):
        print(f"\t {l} : {v}.")

    # Print the output performance
    print("\n Performance:")
    stats = performance['Statistics']
    for k, v in stats.items():
        print(f"\t {k} {v}.")

def editconfig(mode=0):

    """
    mode 0 - no semantic features
    mode 1 - with semantic features
    mode 2 - only use semantic features
    mode 3 - histogram + semantic features
    """

    overrides = {
                'General': {
                    'AssumeSameImageAndMaskMetadata': 'True',
                    },
                'CrossValidation': {
                   'N_iterations': '20',
                  }
        }
    
    if mode == 1:
        # Add semantic features to existing overrides
        overrides['SelectFeatGroup'] = {
            'semantic_features': 'True'
        }
    if mode == 2:
        overrides['SelectFeatGroup'] = {
            'semantic_features': 'True',
            'shape_features': 'False',
            'histogram_features': 'False',
            'orientation_features': 'False',
            'texture_Gabor_features': 'False',
            'texture_GLCM_features': 'False',
            'texture_GLDM_features': 'False',
            'texture_GLCMMS_features': 'False',
            'texture_GLRLM_features': 'False',
            'texture_GLSZM_features': 'False',
            'texture_GLDZM_features': 'False',
            'texture_NGTDM_features': 'False',
            'texture_NGLDM_features': 'False',
            'texture_LBP_features': 'False',
            'dicom_features': 'False',
            'vessel_features': 'False',
            'phase_features': 'False',
            'fractal_features': 'False',
            'location_features': 'False',
            'rgrd_features': 'False',
            'original_features': 'True',
            'wavelet_features': 'False',
            'log_features': 'False',
        }
    if mode == 3:
            overrides['SelectFeatGroup'] = {
                'semantic_features': 'True',
                'shape_features': 'False',
                'histogram_features': 'True',
                'orientation_features': 'False',
                'texture_Gabor_features': 'False',
                'texture_GLCM_features': 'False',
                'texture_GLDM_features': 'False',
                'texture_GLCMMS_features': 'False',
                'texture_GLRLM_features': 'False',
                'texture_GLSZM_features': 'False',
                'texture_GLDZM_features': 'False',
                'texture_NGTDM_features': 'False',
                'texture_NGLDM_features': 'False',
                'texture_LBP_features': 'False',
                'dicom_features': 'False',
                'vessel_features': 'False',
                'phase_features': 'False',
                'fractal_features': 'False',
                'location_features': 'False',
                'rgrd_features': 'False',
                'original_features': 'True',
                'wavelet_features': 'False',
                'log_features': 'False',
            }
    return overrides
        


if __name__ == '__main__':
    # command line argument parsing
    import argparse 
    parser = argparse.ArgumentParser(description='Run WORC for clinical features.')
    parser.add_argument('--exp_name', type=str, default='test',
                        help='Name of the experiment to run.')
    parser.add_argument('--modality', type=str, default='T1W',
                        help='Image modality to run.')
    parser.add_argument('--version', type=str, default='v0',
                        help='Version of segmentation to run.')
    parser.add_argument('--mode', type=int,
                        help='Mode to control the inclusion of clinical features')
    parser.add_argument('--label_type', type=str, default='Huvos',
                        help='Huvos or WIR')
    args = parser.parse_args()
    
    # print parsed arguments
    print(f"Experiment Name: {args.exp_name}")
    print(f"Modality: {args.modality}")
    print(f"Version: {args.version}")
    print(f"Mode: {args.mode}")
    print(f"Label Type: {args.label_type}") 

    main(experiment_name=args.exp_name, modality=args.modality, version=args.version, mode=args.mode, type=args.label_type)
