import argparse
import os
from pynv import Client

parser = argparse.ArgumentParser(description="Script to upload image maps to NeuroVault")

parser.add_argument("--token", help="File with NeuroVault Token Information")
parser.add_argument("--sample", help="Name of sample, AHRB or ABCD")
parser.add_argument("--group_files", help="list of image paths to group files")

args = parser.parse_args()

nv_token = args.token
sample = args.sample
grp_paths = args.group_files


# cog atlsa monetary incentive delay task
task = 'trm_4f23fc8c42d28'
task_name = 'monetary incentive delay task'

# Run code
with open(nv_token, 'r') as file:
    # get token info
    token_info = file.read()

api = Client(token_info.strip())
collection_name = api.create_collection(f'{sample}: Randomise MNI152 3D maps for MID RT Models')

# add group images
with open(grp_paths, 'r') as file:
    est_type = 'Group'
    for img_path in file:
        clean_path = img_path.strip()
        map_type = os.path.basename(clean_path).split('_')[-1].split('.')[0]
        if 'tstat' in map_type:
            est = 't_stat'
        elif 'fstat' in map_type:
            est = 'f_stat'
        elif 'cohensd' in map_type:
            est = 'cohens_d'
        elif 'pearsonr' in map_type:
            est = 'pearsons_r'

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
        
