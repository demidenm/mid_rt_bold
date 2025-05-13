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

# cog atlas monetary incentive delay task
task_id = 'trm_4f23fc8c42d28'
task_name = 'monetary incentive delay task'

# Run code
with open(nv_token, 'r') as file:
    # get token info
    token_info = file.read()

api = Client(token_info.strip())
collection_name = api.create_collection(f'{sample}: MNI152 3D maps for MID RT Models')

with open(grp_paths, 'r') as file:
    est_type = 'Group'
    for img_path in file:
        clean_path = img_path.strip()
        
        # the filename from the full path
        img_basename = os.path.basename(clean_path)
        
        # Extract elements from the filename
        file_parts = img_basename.split('_')
        
        # Empty variables to hold parsed values
        subs = None
        session = None
        task = None
        contrast = None
        model = None
        map_type = None
        diff = False
        # Parse each part of the filename
        for part in file_parts:
            if part.startswith('subs-'):
                subs = part.split('-')[1]
            elif part.startswith('ses-'):
                session = part.split('-', 1)[1]
            elif part.startswith('task-'):
                task = part.split('-', 1)[1]
            elif part.startswith('contrast-'):
                contrast = part.split('-', 1)[1]
            elif part.startswith('mod-'):
                model = part.split('-', 1)[1]
            elif part.startswith('diff-'):
                diff = True
                model = part.split('-', 1)[1]  
            elif part.startswith('perm-'):
                # map type and potential "uncorr" suffix
                perm_part = part.split('-', 1)[1]
                if perm_part.endswith('.nii.gz'):
                    map_type = perm_part[:-7]  # Remove .nii.gz
                else:
                    map_type = perm_part
                    
        # is 'uncorr' suffix in map_type
        is_uncorrected = False
        if map_type and '-uncorr' in map_type:
            is_uncorrected = True
            map_type = map_type.replace('-uncorr', '')
        
        # subs int
        sample_size = int(subs) if subs and subs.isdigit() else None
        
        # stat type
        if 'tstat' in map_type:
            map_type_param = 'T map'
            est = 't_stat'
        elif 'fstat' in map_type:
            map_type_param = 'F map'
            est = 'f_stat'
        elif 'zstat' in map_type:
            map_type_param = 'Z map'
            est = 'z_stat'
        elif 'cohensd' in map_type:
            est = 'cohens_d'
            map_type_param = 'Other'  # cohen's d is not a standard map type
        else:
            map_type_param = 'Other'


        # Descriptive image name
        # Estimated Corrected / uncorrected
        if diff:
            if is_uncorrected:
                image_name = f'{est} (uncorr): Diff {model} Contrast {contrast}'
            else:
                image_name = f'{est} (corr): Diff {model} Contrast {contrast}'
        else:
            if is_uncorrected:
                image_name = f'{est} (uncorr): Model {model} Contrast {contrast}'
            else:
                image_name = f'{est} (corr): Model {model} Contrast {contrast}'
        

        try:
            image = api.add_image(
                collection_id=collection_name['id'],
                file=clean_path,
                name=image_name,
                map_type=map_type_param,
                modality='fMRI-BOLD',
                analysis_level='G',
                sample_size=sample_size,
                target_template_image='MNI152NLin2009cAsym',
                image_type='statistic_map',
                cognitive_paradigm_cogatlas=task_id,  
                task=task_name,  
                estimate_type=est
            )
            print(f"Successfully uploaded: {image_name}")
            print(f"Image ID: {image.get('id', 'Unknown')}")
        except Exception as e:
            print(f"Error uploading {image_name}: {str(e)}")
            
