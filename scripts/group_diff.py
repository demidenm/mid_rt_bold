import os
import sys
import stat
import warnings
import subprocess
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
from nilearn.image import math_img
warnings.filterwarnings("ignore")

# Getpath to Stage2 scripts
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
from scripts.model_designmat_regressors import (
    make_randomise_files, make_4d_data_mask, generate_permutation_matrix, 
    get_cluster_2sided, find_largest_cluster_size, permutation_test_with_clustering
)

parser = argparse.ArgumentParser(description="Script to run first level task models w/ nilearn")
parser.add_argument("--sample", help="sample type, ahrb, abcd or mls?")
parser.add_argument("--task", help="task type -- e.g., mid, reward, etc")
parser.add_argument("--run", help="Run lvl -- e.g., 1 or 2 (for 01 / 02)")
parser.add_argument("--site_file", help="File to subject site data", default=None)
parser.add_argument("--ses", help="session, include the session type without prefix 'ses', e.g., 1, 01, baselinearm1")
parser.add_argument("--contrast", help="contrast label, e.g. 'LRew-Neut' or 'LPunHit-LPunMiss'")
parser.add_argument("--a_mod", help="Model to subtract from e.g. mod-Cue-None, mod-Cue-rt, mod-Cue-probexcond, mod-dairc")
parser.add_argument("--b_mod", help="Model to subtract e.g. mod-Cue-None, mod-Cue-rt, mod-Cue-probexcond")
parser.add_argument("--mask", help="path the to the binarized brain mask (e.g., MNI152 or "
                                   "constrained mask in MNI space, or None", default=None)
parser.add_argument("--input", help="input path to data")
parser.add_argument("--output", help="output folder where to write out and save information")

args = parser.parse_args()

# Now you can access the arguments as attributes of the 'args' object.
sample = args.sample
task = args.task
run = args.run
ses = args.ses
site_file = args.site_file
a_mod = args.a_mod
b_mod = args.b_mod
contrast = args.contrast
brainmask = args.mask
in_dir = args.input
scratch_out = args.output


# create z-stat map effect / sqrt(var)
# find all images for a and b, assert IDs same for each beta / var

# [a] mod
a_mod_betas = sorted(glob(f'{in_dir}/*_ses-{ses}_task-{task}_contrast-{contrast}_{a_mod}_stat-effect.nii.gz'))
a_mod_vars = sorted(glob(f'{in_dir}/*_ses-{ses}_task-{task}_contrast-{contrast}_{a_mod}_stat-var.nii.gz'))
a_beta_ids = [os.path.basename(path).split('_')[0] for path in a_mod_betas]
a_var_ids = [os.path.basename(path).split('_')[0] for path in a_mod_vars]
# [b] mod

b_mod_betas = sorted(glob(f'{in_dir}/*_ses-{ses}_task-{task}_contrast-{contrast}_{b_mod}_stat-effect.nii.gz'))
b_mod_vars = sorted(glob(f'{in_dir}/*_ses-{ses}_task-{task}_contrast-{contrast}_{b_mod}_stat-var.nii.gz'))
b_beta_ids = [os.path.basename(path).split('_')[0] for path in b_mod_betas]
b_var_ids = [os.path.basename(path).split('_')[0] for path in b_mod_vars]
assert set(a_beta_ids) == set(a_var_ids), f"Subject IDs don't match between {a_mod} beta and variance maps"
assert set(b_beta_ids) == set(b_var_ids), f"Subject IDs don't match between {b_mod} beta and variance maps"
assert set(a_beta_ids) == set(b_beta_ids), f"Subject IDs don't match between {a_mod} and {b_mod} maps"

# loop through maps and make z-stat map for each subject
a_beta_dict = {os.path.basename(path).split('_')[0]: path for path in a_mod_betas}
a_var_dict = {os.path.basename(path).split('_')[0]: path for path in a_mod_vars}
b_beta_dict = {os.path.basename(path).split('_')[0]: path for path in b_mod_betas}
b_var_dict = {os.path.basename(path).split('_')[0]: path for path in b_mod_vars}

for subject_id in a_beta_ids:
    print(f"Creating z-stat for subject {subject_id}...")

    # Calculate z-statistic maps (beta / sqrt(variance)) using math_img
    a_zstat_img = math_img(
        "beta / np.sqrt(np.maximum(var, 1e-6))",  # tiny epsilon to avoid division by zero
        beta=a_beta_dict[subject_id],
        var=a_var_dict[subject_id]
    )
    a_beta_filename = os.path.basename(a_beta_dict[subject_id])
    a_zstat_filename = os.path.join(in_dir, a_beta_filename.replace('stat-effect', 'stat-zstat'))
    a_zstat_img.to_filename(a_zstat_filename)

    b_zstat_img = math_img(
        "beta / np.sqrt(np.maximum(var, 1e-6))",
        beta=b_beta_dict[subject_id],
        var=b_var_dict[subject_id]
    )
    b_beta_filename = os.path.basename(b_beta_dict[subject_id])
    b_zstat_filename = os.path.join(in_dir, b_beta_filename.replace('stat-effect', 'stat-zstat'))
    b_zstat_img.to_filename(b_zstat_filename)


# find all contrast fixed effect maps for model permutation across subjects and get IDs to match to lists
a_mod_list = sorted(glob(f'{in_dir}/*_ses-{ses}_task-{task}_contrast-{contrast}_{a_mod}_stat-zstat.nii.gz'))
b_mod_list = sorted(glob(f'{in_dir}/*_ses-{ses}_task-{task}_contrast-{contrast}_{b_mod}_stat-zstat.nii.gz'))
a_mod_ids = [os.path.basename(path).split('_')[0] for path in a_mod_list]
b_mod_ids = [os.path.basename(path).split('_')[0] for path in b_mod_list]
num_maps = len(a_mod_list)

assert (np.array(a_mod_ids) == np.array(b_mod_ids)).all(), "Order of IDs in a_mod != b_mod."


# subset id's SITE to match BOLD paths
sub_sites_df = pd.read_csv(site_file, sep=',')
sub_ids = [os.path.basename(path).split('_')[0] for path in a_mod_list]
site_ids = sub_sites_df['subject'].values
sub_sites_df = sub_sites_df.set_index('subject').loc[sub_ids].reset_index()
assert (sub_sites_df['subject'].values == np.array(sub_ids)).all(), "Order of IDs in sub_sites_df != sub_ids."
site_vals = sub_sites_df['site_fslrand'].values

# randomise, permuted maps + corrected
tmp_rand = f'{scratch_out}/custom'
level = 'diff'
outdir = f'{tmp_rand}/outmaps/{contrast}/{level}'

if not os.path.exists(f'{outdir}'):
    os.makedirs(f'{outdir}')

# make bold mask for a & b models
make_4d_data_mask(bold_paths=a_mod_list, sess=ses, contrast_lab=contrast,
                  model_type=a_mod, tmp_dir=f'{tmp_rand}/concat_imgs_diff')
make_4d_data_mask(bold_paths=b_mod_list, sess=ses, contrast_lab=contrast,
                  model_type=b_mod, tmp_dir=f'{tmp_rand}/concat_imgs_diff')

# get concatenated images
a_mod_nii = f'{tmp_rand}/concat_imgs_diff/subs-{num_maps}_ses-{ses}_task-MID_contrast-{contrast}_{a_mod}.nii.gz'
b_mod_nii = f'{tmp_rand}/concat_imgs_diff/subs-{num_maps}_ses-{ses}_task-MID_contrast-{contrast}_{b_mod}.nii.gz'

# estimate differences for permutation test
diff_nifti = math_img('img1 - img2', img1=a_mod_nii, img2=b_mod_nii)
diff_label = f'diff-{a_mod.split("-")[-1]}-{b_mod.split("-")[-1]}'
diff_nifti_path = f'{tmp_rand}/concat_imgs_diff/subs-{num_maps}_ses-{ses}_task-MID_contrast-{contrast}_{diff_label}.nii.gz'
diff_nifti.to_filename(diff_nifti_path)


print("*** Running: *** /diff", contrast)
# run permutation test with clustering
mask_path = f'{tmp_rand}/concat_imgs_diff/subs-{num_maps}_ses-{ses}_task-MID_contrast-{contrast}_{a_mod}_mask.nii.gz'
results = permutation_test_with_clustering(data_path=diff_nifti_path, mask_path=mask_path, scanner_ids=site_vals,
cluster_forming_threshold=3.1,n_permutations=1000, n_cpus=6)

print(f"Max cluster size threshold: {results['maxclustersizethreshold']}")
print('Biggest cluster for positive')
print(find_largest_cluster_size(results['negposarrays']['pos_labeled_obs_array']))
print('Biggest cluster for negative')
print(find_largest_cluster_size(results['negposarrays']['neg_labeled_obs_array']))

# save images
for img_name, img in results.items():
    if img_name in ['maxclustersizethreshold', 'negposarrays']:
        continue
    else:
        # Create filename based on the image name
        filefull = f"subs-{num_maps}_ses-{ses}_task-MID_contrast-{contrast}_{diff_label}_{img_name}.nii.gz"
        filename = os.path.join(outdir, f"{filefull}")
        print(f"Saving {img_name} to {filename}")
        img.to_filename(filename)
        
# create cohen's d map
sigclust_tmap = math_img(
'img1*img2', img1=results['perm-tstat'], img2=results['perm-sigclustermask']
)
sqrt_nsubs = np.sqrt(num_maps)
cohend_img = math_img(f'img/{sqrt_nsubs}', img=sigclust_tmap)
cohend_img.to_filename(f'{outdir}/subs-{num_maps}_ses-{ses}_task-MID_contrast-{contrast}_{diff_label}_perm-cohensd.nii.gz')
