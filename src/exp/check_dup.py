import os
os.environ["NUMEXPR_MAX_THREADS"] = "192"  

import sys
sys.path.append('/projects/0/prjs1425/Osteosarcoma_WORC/WORC')
from WORC.facade.simpleworc import SimpleWORC
from pathlib import Path

# These packages are only used in analysing the results
import pandas as pd
import json
import fastr
import glob
from helper import get_imgs_by_agegroup, create_overfit_splits, get_imgs_by_mrigroup, get_imgs_by_subtype



def get_imgs_Simpleworc(directory):
    directory = Path(directory).expanduser()
    image_file_name = 'image.nii.gz'
    glob='*/'
    images = list(directory.glob(f'{glob}{image_file_name}'))
    return images

def get_imgid_from_path(path):
    return path.parts[-2]

versions = ['v9', 'v0', 'v1', ]
modalities = ['T1W', 'T1W_FS_C', 'T2W_FS']
types = ['Children', 'AYA', 'Older_adults']

for modality in modalities:
    for version in versions:
        imagedatadir = f'/projects/0/prjs1425/Osteosarcoma_WORC/exp_data/{modality}/{version}'
        images = get_imgs_Simpleworc(imagedatadir)
        images = [get_imgid_from_path(image) for image in images]
        subids = [image[:-3] for image in images]
        print(f'{modality} {version} has {len(set(subids))} unique subids and {len(images)} images')
       

        if version == 'v1':
            ag_images_ids = []
            # add all the images ids for each age group into a list of lists
            for type in types:
                images_dict, segs_dict, exp_data_dir = get_imgs_by_agegroup(modality=modality, version=version, age_group=type, exp_name=type)
                images_ids = [iid[:-2] for iid in images_dict.keys()]
                ag_images_ids.extend(images_ids)
            
            # check the non overlap between images and images_ids
            nids = [nid[:-3] for nid in ag_images_ids]
            overlap = set(nids) - set(subids)
            print(f'{modality} {version} has {len(overlap)} images that are not in the AGE groups data')
            

            