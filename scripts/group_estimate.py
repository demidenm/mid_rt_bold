import os
import warnings
import subprocess
import argparse
import numpy as np
import pandas as pd
import sys
from glob import glob
from nilearn.image import math_img
from nilearn.reporting import get_clusters_table
from nilearn.glm.second_level import SecondLevelModel
warnings.filterwarnings("ignore")

# Getpath to Stage2 scripts
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
from scripts.model_designmat_regressors import (
    make_randomise_grp, make_randomise_rt, make_randomise_files, make_4d_data_mask, generate_permutation_matrix, 
    get_cluster_2sided, find_largest_cluster_size, permutation_test_with_clustering
)


def group_onesample(fixedeffect_paths: list, session: str, task_type: str,
                    contrast_type: str, group_outdir: str,
                    model_lab: str, rt_array=None, mask: str = None):
    """
    This function takes in a list of fixed effect files for a select contrast and
    calculates a group (secondlevel) model by fitting an intercept to length of maps.
    For example, for 10 subject maps of contrast A, the design matrix would include an intercept length 10.

    :param fixedeffect_paths: a list of paths to the fixed effect models to be used
    :param session: string session label, BIDS label e.g., ses-1
    :param task_type: string task label, BIDS label e.g., mid
    :param contrast_type: contrast type saved from fixed effect models
    :param model_lab: complete string of model label, e.g., 'mod-Cue-rt' or 'mod-Cue-None'
    :param group_outdir: path to folder to save the group level models
    :param rt_array: array with subject mean cent RT files to use in group regressor
    :param mask: path to mask, default none
    :return: nothing return, files are saved
    """

    if not os.path.exists(group_outdir):
        os.makedirs(group_outdir)
        print("Directory created:", group_outdir)

    n_maps = len(fixedeffect_paths)
    # Create design matrix with intercept (1s) that's length of subjects/length of fixed_files
    design_matrix = pd.DataFrame([1] * n_maps, columns=['int'])
    if rt_array:
        design_matrix['rt'] = rt_array
        con_array = ['int','rt']
    else:
        con_array = ['int']

    # Fit secondlevel model
    sec_lvl_model = SecondLevelModel(mask_img=mask, smoothing_fwhm=None, minimize_memory=False)
    sec_lvl_model = sec_lvl_model.fit(second_level_input=fixedeffect_paths,
                                      design_matrix=design_matrix)
    # contrasts mean 'int' and corr with mRT 'rt
    for con in con_array:
        tstat_map = sec_lvl_model.compute_contrast(
            second_level_contrast=con,
            second_level_stat_type='t',
            output_type='stat'
        )
        tstat_out = f'{group_outdir}/subs-{n_maps}_ses-{session}_task-{task_type}_' \
                    f'contrast-{contrast_type}_{model_lab}_stat-tstat_{con}.nii.gz'
        tstat_map.to_filename(tstat_out)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="Script to run first level task models w/ nilearn")
parser.add_argument("--sample", help="sample type, ahrb, abcd or mls?")
parser.add_argument("--task", help="task type -- e.g., mid, reward, etc")
parser.add_argument("--run", help="Run lvl -- e.g., 1 or 2 (for 01 / 02)")
parser.add_argument("--rt_file", help="File to subject level rt data(for 01 / 02)", default=None)
parser.add_argument("--site_file", help="File to subject site data", default=None)
parser.add_argument("--ses", help="session, include the session type without prefix 'ses', e.g., 1, 01, baselinearm1")
parser.add_argument("--model", help="model label,"
                                    " e.g. 'mod-Cue-rt' or 'mod-Cue-None, mod-dairc'")
parser.add_argument("--contrast", help="contrast label, e.g. 'LRew-Neut' or 'LPunHit-LPunMiss'")
parser.add_argument("--mask", help="path the to the binarized brain mask (e.g., MNI152 or "
                                   "constrained mask in MNI space, or None")
parser.add_argument("--randomise", help="To run or not, randomise/custom", default="custom")
parser.add_argument("--input", help="input path to data")
parser.add_argument("--output", help="output folder where to write out and save information")

args = parser.parse_args()

# Now you can access the arguments as attributes of the 'args' object.
sample = args.sample
task = args.task
run = args.run
rt_file = args.rt_file
site_file = args.site_file
ses = args.ses
contrast = args.contrast
model = args.model
brainmask = args.mask
run_randomise = args.randomise
in_dir = args.input
scratch_out = args.output

# mRTs for subjects, averaged across runs
sub_rt_df = pd.read_csv(rt_file, sep=',')
sub_sites_df = pd.read_csv(site_file, sep=',')

# find all contrast fixed effect maps for model permutation across subjects
list_maps = sorted(glob(f'{in_dir}/*_ses-{ses}_task-{task}_*'
                        f'contrast-{contrast}_{model}_stat-effect.nii.gz'))

# subset id's RT times to match
sub_ids = [os.path.basename(path).split('_')[0] for path in list_maps]
subset_df = sub_rt_df[sub_rt_df['Subject'].isin(sub_ids)].copy()
subset_df = subset_df.set_index('Subject').loc[sub_ids].reset_index()  # ensure index sorts same as IDs
assert (subset_df['Subject'].values ==
       np.array(sub_ids)).all(), "Order of IDs in subset_df != sub_ids."

mean_rt = subset_df['Average_RT'].mean()
subset_df['Mean_Centered_RT'] = (subset_df['Average_RT'] - mean_rt).values
rt_vals = subset_df['Mean_Centered_RT'].values

# site ids match file order
site_ids = sub_sites_df['subject'].values
sub_sites_df = sub_sites_df.set_index('subject').loc[sub_ids].reset_index()
assert (sub_sites_df['subject'].values == np.array(sub_ids)).all(), "Order of IDs in sub_sites_df != sub_ids."
site_vals = sub_sites_df['site_fslrand'].values


group_onesample(fixedeffect_paths=list_maps, session=ses, task_type=task,
                contrast_type=contrast, group_outdir=scratch_out,
                model_lab=model, mask=brainmask, rt_array=None)

if run_randomise == "randomise":
    # randomise, permuted maps + corrected
    tmp_rand=f'{scratch_out}/randomise'
    make_4d_data_mask(bold_paths=list_maps, sess=ses, contrast_lab=contrast,
                      model_type=model, tmp_dir=f'{tmp_rand}/concat_imgs')

    num_maps = len(list_maps)
    comb_input_nii = f'{tmp_rand}/concat_imgs/subs-{num_maps}_ses-{ses}_task-MID_contrast-{contrast}_{model}.nii'

    for level in ['grp','int']:
        if level == 'grp':
            # Create design matrix with intercept (1s) that's length of subjects/length of fixed_files
            design_matrix = pd.DataFrame({'int': [1] * num_maps})
            make_randomise_files(desmat_final=design_matrix, regressor_names='int',site_array=site_vals,
                                 contrasts=['int'], outdir=f'{tmp_rand}/randomise/{contrast}/{level}')
            make_randomise_grp(comb_nii_path=comb_input_nii, outdir=f'{tmp_rand}/randomise/{contrast}/{level}',
            permutations=1000)

        else:
            design_matrix = pd.DataFrame({'int': [1] * num_maps, 'rt': rt_vals})
            make_randomise_files(desmat_final=design_matrix, site_array= site_vals, regressor_names=['int', 'rt'],
                                 contrasts=[['rt']], outdir=f'{tmp_rand}/randomise/{contrast}/{level}')
            make_randomise_rt(comb_nii_path=comb_input_nii, outdir=f'{tmp_rand}/randomise/{contrast}/{level}',
            permutations=1000)

        print("*** Running: *** ", level)
        script_path = f'{tmp_rand}/randomise/{contrast}/{level}/randomise_call.sh'
        subprocess.run(['bash', script_path])

elif run_randomise == "custom":
    # permuted maps + corrected using jeanette's custom algorithm
    tmp_rand=f'{scratch_out}/custom'
    make_4d_data_mask(bold_paths=list_maps, sess=ses, contrast_lab=contrast,
                      model_type=model, tmp_dir=f'{tmp_rand}/concat_imgs')

    num_maps = len(list_maps)
    comb_input_nii = f'{tmp_rand}/concat_imgs/subs-{num_maps}_ses-{ses}_task-MID_contrast-{contrast}_{model}.nii.gz'
    mask_path_nii = f'{tmp_rand}/concat_imgs/subs-{num_maps}_ses-{ses}_task-MID_contrast-{contrast}_{model}_mask.nii.gz'
    outdir=f'{tmp_rand}/outmaps/{contrast}'
    os.makedirs(outdir, exist_ok=True)
    # run permutation test with clustering
    results = permutation_test_with_clustering(data_path=comb_input_nii, mask_path=mask_path_nii, scanner_ids=site_vals,
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
            filefull = f"subs-{num_maps}_ses-{ses}_task-MID_contrast-{contrast}_{model}_{img_name}.nii"
            filename = os.path.join(outdir, f"{filefull}")
            print(f"Saving {img_name} to {filename}")
            img.to_filename(filename)
            
    # create cohen's d map
    sigclust_tmap = math_img(
    'img1*img2', img1=results['perm-tstat'], img2=results['perm-sigclustermask']
    )
    sqrt_nsubs = np.sqrt(num_maps)
    cohend_img = math_img(f'img/{sqrt_nsubs}', img=sigclust_tmap)
    cohend_img.to_filename(f'{outdir}/subs-{num_maps}_ses-{ses}_task-MID_contrast-{contrast}_{model}_perm-cohensd.nii.gz')



