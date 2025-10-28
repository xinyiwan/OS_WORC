import pandas as pd
import os 

inclusion = pd.read_csv('/projects/0/prjs1425/Osteosarcoma_WORC/image_records/included_images_app.csv')
pids = sorted(set(inclusion['Subject'].tolist()))
dummy_img = '/projects/0/prjs1425/Osteosarcoma_WORC/dummy_data/image.nii.gz'
dummy_mask = '/projects/0/prjs1425/Osteosarcoma_WORC/dummy_data/mask.nii.gz'

if __name__ == '__main__':
    
    for pid in pids:
        patient_dir = f'/projects/0/prjs1425/Osteosarcoma_WORC/exp_data/dummy_data/{pid}'
        f_image = f'{patient_dir}/image.nii.gz'
        f_mask = f'{patient_dir}/mask.nii.gz'
        
        # copy image to dummy folder
        os.makedirs(patient_dir, exist_ok=True)
        os.system(f'cp {dummy_img} {f_image}')
        os.system(f'cp {dummy_mask} {f_mask}')


    
