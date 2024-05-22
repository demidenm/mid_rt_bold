import argparse
import os
from pynv import Client

parser = argparse.ArgumentParser(description="Script to upload image maps to NeuroVault")

parser.add_argument("--token", help="File with NeuroVault Token Information")
parser.add_argument("--sample", help="Name of sample, AHRB, ABCD or MLS")
parser.add_argument("--group_files", help="list of image paths to group files")
parser.add_argument("--icc_files", help="lost of image paths to icc files")
parser.add_argument("--subsample_files", help="lost of image paths to icc files",
                    default=None)
args = parser.parse_args()

nv_token = args.token
sample = args.sample
grp_paths = args.group_files
icc_paths = args.icc_files
subsample_paths = args.subsample_files

# cog atlsa monetary incentive delay task
task = 'trm_4f23fc8c42d28'
task_name = 'monetary incentive delay task'

# Run code
with open(nv_token, 'r') as file:
    # get token info
    token_info = file.read()

api = Client(token_info.strip())
collection_name = api.create_collection(f'{sample}: MNI152 3D maps for Multiverse Reliability')

# add group images
with open(grp_paths, 'r') as file:
    est_type = 'Group'
    est = 'cohens_d'
    for img_path in file:
        clean_path = img_path.strip()
        img_basename = os.path.basename(clean_path)
        file_details = img_basename.split('_')
        subs = None
        # Loop through each part of the path
        for part in file_details:
            if part.startswith('subs-'):
                subs = part.split('-')[1]
        image_name = f'{est_type}: {img_basename}'
        image = api.add_image(collection_name['id'], clean_path, name=image_name, map_type='Other',
                              modality='fMRI-BOLD', analysis='G', sample_size={subs},
                              target_template_image='GenericMNI', type_design='event_related',
                              cognitive_paradigm_cogatlas=task, task_paradigm=task_name, estimate_type=est)

# Add ICC images
with open(icc_paths, 'r') as file:
    est_type = 'ICC'
    for img_path in file:
        clean_path = img_path.strip()
        img_basename = os.path.basename(clean_path)
        file_details = img_basename.split('_')
        subs = None
        stat = None
        # Loop through each part of the path
        for part in file_details:
            if part.startswith('subs-'):
                subs = part.split('-')[1]
            if part.startswith('stat-'):
                stat = part.split('-')[1]
            image_name = f'{est_type}: {img_basename}'
        image = api.add_image(collection_name['id'], clean_path, name=image_name, map_type='Other',
                              modality='fMRI-BOLD', analysis='G', sample_size={subs},
                              target_template_image='GenericMNI', type_design='event_related',
                              cognitive_paradigm_cogatlas=task, task_paradigm=task_name, estimate_type=stat)

if subsample_paths is not None:
    with open(subsample_paths, 'r') as file:
        est_type = 'SubsampleICC'
        for img_path in file:
            clean_path = img_path.strip()
            img_basename = os.path.basename(clean_path)
            file_details = img_basename.split('_')
            subs = None
            stat = None
            # Loop through each part of the path
            for part in file_details:
                if part.startswith('subs-'):
                    subs = part.split('-')[1]
                if part.startswith('stat-'):
                    stat = part.split('-')[1]
                image_name = f'{est_type}: {img_basename}'
            image = api.add_image(collection_name['id'], clean_path, name=image_name, map_type='Other',
                                  modality='fMRI-BOLD', analysis='G', sample_size={subs},
                                  target_template_image='GenericMNI', type_design='event_related',
                                  cognitive_paradigm_cogatlas=task, task_paradigm=task_name, estimate_type=stat)

