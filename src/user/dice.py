import seg_metrics.seg_metrics as sg
import os, glob
import logging
import sys
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from utils import select_v0_seg, select_v1_seg, select_v2_seg
from visualize import visualize, create_comparison_table
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Output to console
        logging.FileHandler('/gpfs/work1/0/prjs1425/shark/logs_sk_on_sn/dice_test_feb16.log')  # Output to file
    ]
)

logger = logging.getLogger(__name__)

def resample_to_common_space(gdth_img, gdth_affine, gdth_spacing,
                              pred_img, pred_affine, pred_spacing):
    """
    Resample both images to a common physical space.
    """
    # Use finest spacing
    common_spacing = np.minimum(gdth_spacing, pred_spacing)
    logger.info(f"Resampling to common spacing: {common_spacing}")
    
    # Simple approach: resample pred to gdth's physical space
    # (gdth is the reference)
    common_affine = gdth_affine
    common_shape = gdth_img.shape
    
    logger.info(f"Reference (gdth) space: shape={common_shape}, affine diagonal={np.diag(common_affine[:3,:3])}")
    
    # Resample pred to gdth space
    pred_affine_inv = np.linalg.inv(pred_affine)
    
    def resample_image(img, img_affine_inv, img_name):
        """Resample image to reference space"""
        # Create grid of voxel indices in reference (gdth) space
        idx_i, idx_j, idx_k = np.meshgrid(
            np.arange(common_shape[0]),
            np.arange(common_shape[1]),
            np.arange(common_shape[2]),
            indexing='ij'
        )
        
        coords_homo = np.array([
            idx_i.flatten(), idx_j.flatten(), idx_k.flatten(),
            np.ones(idx_i.size)
        ])
        
        # Ref voxel indices → physical space (using gdth affine)
        coords_physical = common_affine @ coords_homo
        
        # Physical space → source image voxel indices
        coords_source = img_affine_inv @ coords_physical
        source_coords = coords_source[:3, :]
        
        logger.info(f"{img_name} source coords range: {source_coords.min()} to {source_coords.max()}")
        
        # Check if coordinates are within bounds
        in_bounds = (
            (source_coords[0, :] >= 0) & (source_coords[0, :] < img.shape[0]) &
            (source_coords[1, :] >= 0) & (source_coords[1, :] < img.shape[1]) &
            (source_coords[2, :] >= 0) & (source_coords[2, :] < img.shape[2])
        )
        logger.info(f"{img_name} voxels in bounds: {in_bounds.sum()} / {in_bounds.size}")
        
        resampled = map_coordinates(img, source_coords, 
                                   order=0,  # nearest neighbor
                                   mode='constant', cval=0)
        resampled = resampled.reshape(common_shape)
        
        unique_vals = np.unique(resampled)
        logger.info(f"{img_name} resampled unique values: {unique_vals}")
        
        return resampled
    
    # Keep gdth as-is (it's the reference)
    gdth_resampled = gdth_img.copy()
    
    # Resample pred to gdth space
    pred_resampled = resample_image(pred_img, pred_affine_inv, "Prediction")
    
    return gdth_resampled, pred_resampled, common_affine


def dice_analysis_images(gdth_img_path, pred_img_path, task_name='dice',
                         metrics=['dice','jaccard','precision','recall','fpr','fnr','vs','hd','msd','mdsd','hd95']):
    """
    Dice analysis with proper registration for different spacings.
    """
    labels = [0, 1]
    
    # Setup output
    res_path = os.path.dirname(os.path.dirname(gdth_img_path))
    res_path = os.path.join(res_path, task_name)
    os.makedirs(res_path, exist_ok=True)
    csv_file = os.path.join(res_path, pred_img_path.split('/')[-1] + '.csv')
    
    # Check if already done
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            if not df.empty:
                logger.info(f"SKIP: Already analyzed with {len(df)} entries")
                return True
        except Exception as e:
            logger.warning(f"Could not read CSV: {e}")
    
    temp_files = []
    
    try:
        # Load images
        gdth_nib = nib.load(gdth_img_path)
        pred_nib = nib.load(pred_img_path)
        
        gdth_img = gdth_nib.get_fdata()
        pred_img = pred_nib.get_fdata()
        
        gdth_spacing = np.array(gdth_nib.header.get_zooms()[:3])
        pred_spacing = np.array(pred_nib.header.get_zooms()[:3])
        
        logger.info(f"Ground truth: shape={gdth_img.shape}, spacing={gdth_spacing}")
        logger.info(f"Prediction:   shape={pred_img.shape}, spacing={pred_spacing}")
        
        # Register to common space
        logger.info("Registering to common physical space...")
        gdth_img, pred_img, common_affine = resample_to_common_space(
            gdth_img, gdth_nib.affine, gdth_spacing,
            pred_img, pred_nib.affine, pred_spacing
        )
        
        # Save temporary files for seg_metrics
        
        
        # Compute metrics
        logger.info("Computing metrics on aligned images...")
        metrics_result = sg.write_metrics(
            labels=labels[1:],
            gdth_img=gdth_img,
            pred_img=pred_img,
            csv_file=csv_file,
            metrics=metrics
        )
        
        logger.info(f"Analysis completed. Results: {csv_file}")
        return True
        
    except Exception as e:
        logger.error(f"ERROR during analysis: {e}")
        logger.exception("Full traceback:")
        return False
        
    finally:
        # Cleanup
        for f in temp_files:
            try:
                os.remove(f)
                logger.info(f"Cleaned up: {f}")
            except:
                pass

def check_dice_exists(seg_path, task_name='dice'):
    """
    Check if DICE analysis already exists for a segmentation file
    """
    res_path = os.path.join(os.path.dirname(os.path.dirname(seg_path)), task_name)
    if not os.path.exists(res_path):
        return False
    
    # Look for CSV files related to this segmentation
    base_name = os.path.basename(seg_path).replace('.nii.gz', '')
    csv_files = [f for f in os.listdir(res_path) if f.endswith('.csv') and base_name in f]
    
    for csv_file in csv_files:
        csv_path = os.path.join(res_path, csv_file)
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                return True
        except:
            pass
    
    return False

SEG_DIR = '/gpfs/work1/0/prjs1425/shark/OS_seg'
if __name__ == '__main__':
    # Minimal startup logging
    print("Starting DICE analysis script...")
    print(f"Scanning directory: {SEG_DIR}")
    
    image_data = pd.read_csv('/gpfs/work1/0/prjs1425/shark/preprocessing/modality/included_images_app.csv')
    subjects = image_data['Subject'].unique().tolist()
    
    print(f"Found {len(subjects)} patients")

    # STEP 1: compute DICE for pairs of 1st segmentation and 2nd segmentation (reviewed or redo)
    processed_count = 0
    dice_0_processed = 0
    skipped_dice_exists = 0
    error_count = 0
    skipped_no_data = 0

    subject_infos = {}

    # Progress tracking
    total_patients = len(subjects)
    
    for i, patient in enumerate(subjects, 1):
        patient_path = os.path.join(SEG_DIR, patient)
        subject_infos[patient] = {}
        
        # Progress update every 10 patients
        print(f"Progress: {i}/{total_patients} patients - Processing {patient}")
    
        # Find first segmentation (v1) - FINAL segmentation
        seg_v1 = select_v1_seg(patient_path)
        subject_infos[patient]['seg_v1'] = seg_v1   
        
        if not seg_v1:
            logger.debug(f"No first segmentation found for {patient}")
            continue
        
        # Find second segmentation (v2) - review/redo FINAL
        seg_v2, cat_label = select_v2_seg(patient_path)
        subject_infos[patient]['seg_v2'] = seg_v2  
        subject_infos[patient]['seg_v2_category'] = cat_label
        
        # Check if files actually exist
        if not os.path.exists(seg_v1):
            logger.warning(f"First segmentation file not found: {seg_v1}")
            skipped_no_data += 1
            continue
        
        # Check if DICE analysis already exists (v1 vs v2)
        if check_dice_exists(seg_v1, 'dice'):
            subject_infos[patient]['dice'] = 'exists'  
            skipped_dice_exists += 1
        else:
            # Compute DICE between v1 and v2
            logger.info(f"Computing the dice for: {patient}")

            if not seg_v2 and cat_label == 'none':
                logger.debug(f"No second segmentation found for {patient}, V2 equals V1")
                seg_v2 = seg_v1
            
            success = dice_analysis_images(
                gdth_img_path=seg_v2, 
                pred_img_path=seg_v1,
                task_name='dice'
            )
            if success:
                processed_count += 1
                subject_infos[patient]['dice'] = 'exists'  
            else:
                error_count += 1
                subject_infos[patient]['dice'] = 'failed'  
        
        # Find v0 segmentation (last prompt-based segmentation before FINAL)
        # First, find the segmentation_history directory
        seg_history_dir = os.path.dirname(seg_v1)
        seg_v0 = select_v0_seg(seg_history_dir)
        subject_infos[patient]['seg_v0'] = seg_v0  
        
        # Check if seg_v0 exists and DICE_0 analysis exists
        if seg_v0 and os.path.exists(seg_v0):
            if check_dice_exists(seg_v0, 'dice_0'):
                subject_infos[patient]['dice_0'] = 'exists'  
                skipped_dice_exists += 1
            else:
                # Compute DICE between v0 and v2
                logger.info(f"Computing the dice_0 for: {patient}")
                
                if not seg_v2 and cat_label == 'none':
                    logger.debug(f"No second segmentation found for {patient}, V2 equals V1")
                    seg_v2 = seg_v1

                success = dice_analysis_images(
                    gdth_img_path=seg_v2,
                    pred_img_path=seg_v0,
                    task_name='dice_0'
                )
                if success:
                    dice_0_processed += 1
                    subject_infos[patient]['dice_0'] = 'exists'  
                else:
                    error_count += 1
                    subject_infos[patient]['dice_0'] = 'failed'  
        else:
            if not seg_v0:
                subject_infos[patient]['dice_0'] = None 
                logger.debug(f"No v0 segmentation found for {patient}")
            skipped_no_data += 1

    # Summary
    print("\n" + "=" * 50)
    print("DICE ANALYSIS SUMMARY:")
    print(f"Total patients: {len(subjects)}")
    print(f"DICE (v1 vs v2) processed: {processed_count}")
    print(f"DICE_0 (v0 vs v2) processed: {dice_0_processed}")
    print(f"Skipped (DICE already exists): {skipped_dice_exists}")
    print(f"Skipped (no data): {skipped_no_data}")
    print(f"Errors: {error_count}")
    print("Script completed!")

    # Stage 1: Analyze and categorize DICE results
    # check the overview of subject_infos
    info_df = pd.DataFrame.from_dict(subject_infos, orient='index')
    # make index a column
    info_df.reset_index(inplace=True)
    info_df.rename(columns={'index': 'Subject'}, inplace=True)
    
    # V1 is based on V2
    v1_based_on_v2 = info_df[info_df['dice'] == 'exists'].Subject.values.tolist()
    v0_based_on_v2 = info_df[info_df['dice_0'] == 'exists'].Subject.values.tolist()

    ## check if v0_based_on_v2 is same as v1_based_on_v2
    common_subjects = set(v1_based_on_v2).intersection(set(v0_based_on_v2))
    print(f"Subjects with both DICE and DICE_0 computed: {len(common_subjects)}")

    # V1 = V2 (V2 is none)
    v1_eq_v2_subjects = info_df[(info_df['seg_v2_category'] == 'none')].Subject.values.tolist()
    print(f"Subjects where V1 equals V2 (no review/redo): {len(v1_eq_v2_subjects)}")

    # V1 != V2 (V2 redo)
    v1_non_eq_v2_subjects_redo = info_df[(info_df['seg_v2_category'] != 'redo') & (info_df['dice'] == 'failed')].Subject.values.tolist()
    print(f"Subjects where V1 is no equal to V2 bcs redo): {len(v1_non_eq_v2_subjects_redo)}")

    # V1 != V2 (V2 review, but mis-use on another sequence)
    v1_non_eq_v2_subjects_review = info_df[(info_df['seg_v2_category'] != 'review') & (info_df['dice'] == 'failed')].Subject.values.tolist()
    print(f"Subjects where V1 is no equal to V2 bcs redo): {len(v1_non_eq_v2_subjects_review)}")



    # Stage 2: Visualize and compare DICE results between DICE_0 and DICE
    df_0 = visualize('dice_0')
    # df_0['id'] = df_0['id'] = df_0['filename'].apply(lambda x: x.split('/')[6])
    # unique_ids_0, counts = np.unique(df_0.id.values.tolist(), return_counts=True)

    df_1 = visualize('dice')