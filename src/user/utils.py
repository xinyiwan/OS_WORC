import pandas as pd
import numpy as np
import os
import json
import glob

def select_v2_seg(patient_path):
    """
    Find the second segmentation (review_correction or review_redo) for a patient
    """
    # Look for review_correction
    target_dir = f'{patient_path}/*/review_correction/FINAL*'
    review_files = glob.glob(target_dir)
    
    if review_files:
        # Return the first review_correction FINAL file
        return review_files[0], 'review'
    
    # If no review_correction, look for review_redo
    target_dir = f'{patient_path}/*/review_redo/FINAL*'
    redo_files = glob.glob(target_dir)
    
    if redo_files:
        # Return the first review_redo FINAL file
        return redo_files[0], 'redo'
    
    return None, 'none'

def select_v1_seg(patient_path):
    target_dir = f'{patient_path}/*/segmentation_history/FINAL*'
    seg_history_files = glob.glob(target_dir)

    if len(seg_history_files) > 1:
        # Check if there is correction record under the same folder
        for seg_file in seg_history_files:
            review_file = seg_file.replace('segmentation_history', 'review_*')
            review_files = glob.glob(os.path.dirname(review_file))
            if len(review_files) == 0:
                seg_history_files.remove(seg_file)
    return seg_history_files[0] if seg_history_files else None

def select_v0_seg(seg_history_path):
    """
    Select the last prompt-based segmentation file (v0) from the segmentation history directory.
    This is the segmentation just before the FINAL segmentation.
    
    :param seg_history_path: Path to the segmentation_history directory
    :return: Path to the last v0 segmentation file if exists, else None
    """
    if not os.path.exists(seg_history_path):
        return None
    
    # Read history.json to get the exact sequence
    history_file = os.path.join(seg_history_path, "history.json")
    
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history_data = json.load(f)
            
            # Find the index of the first FINAL action
            final_index = -1
            for i, record in enumerate(history_data):
                action = record.get('action', '').lower()
                if action == 'final' or record.get('is_final', False):
                    final_index = i
                    break
            
            if final_index == -1:
                # No FINAL found in the history
                return None
            
            # Look backward from the FINAL index to find the last prompt before it
            last_prompt_file = None
            for i in range(final_index - 1, -1, -1):
                record = history_data[i]
                action = record.get('action', '').lower()
                filename = record.get('filename')
                
                if not 'OS_000109' in history_file:
                    if (action == 'prompt') and filename :
                        file_path = os.path.join(seg_history_path, filename)
                        if os.path.exists(file_path):
                            last_prompt_file = file_path
                            break
                else:
                    if (action == 'prompt' or action == 'undo') and filename :
                        file_path = os.path.join(seg_history_path, filename)
                        if os.path.exists(file_path):
                            last_prompt_file = file_path
                            break

            
            return last_prompt_file
            
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Error reading history.json: {e}")
            # If there's an error reading JSON, fall back to file-based method
            pass
    else:
        print(f"history.json not found in {seg_history_path}")

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
    return seg_history_files

def get_second_records(subject):
    """
    Docstring for get_first_records
    
    :param subject: Subject ID 'OS_000XXX'
    """
    target_dir = '/exports/lkeb-hpc/xwan/osteosarcoma/OS_seg/{subject}/*/review_*/history.json'
 
    # get all segmentation history files for the subject
    import glob
    review_files = glob.glob(target_dir.format(subject=subject))
 
    # if len(review_files) != 1:
    #     print(f"Subject {subject} has {len(review_files)} review records.")
    
    return review_files
