import os
import sys
import warnings
import subprocess
import argparse
import numpy as np
import pandas as pd
from glob import glob
from nilearn.image import math_img
warnings.filterwarnings("ignore")

# Getpath to Stage2 scripts
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
from scripts.model_designmat_regressors import (
    make_randomise_grp, make_randomise_files, make_4d_data_mask
)

parser = argparse.ArgumentParser(description="Script to run first level task models w/ nilearn")
parser.add_argument("--sample", help="sample type, ahrb, abcd or mls?")
parser.add_argument("--task", help="task type -- e.g., mid, reward, etc")
parser.add_argument("--run", help="Run lvl -- e.g., 1 or 2 (for 01 / 02)")
parser.add_argument("--ses", help="session, include the session type without prefix 'ses', e.g., 1, 01, baselinearm1")
parser.add_argument("--contrast", help="contrast label, e.g. 'LRew-Neut' or 'LPunHit-LPunMiss'")
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
contrast = args.contrast
brainmask = args.mask
in_dir = args.input
scratch_out = args.output

# find all contrast fixed effect maps for model permutation across subjects
nonrt_list = sorted(glob(f'{in_dir}/*_ses-{ses}_task-{task}_*contrast-{contrast}_mod-Cue-rt_stat-effect.nii.gz'))
rtlist = sorted(glob(f'{in_dir}/*_ses-{ses}_task-{task}_*contrast-{contrast}_mod-Cue-None_stat-effect.nii.gz'))

# subset id's RT times to match
nonrt_ids = [os.path.basename(path).split('_')[0] for path in nonrt_list]
rt_ids = [os.path.basename(path).split('_')[0] for path in rtlist]

assert (np.array(nonrt_ids) == np.array(rt_ids)).all(), "Order of IDs in nort_ids != rt_ids."


# randomise, permuted maps + corrected
tmp_rand = f'{scratch_out}/randomise'
# make nonrt & rt 4D
make_4d_data_mask(bold_paths=nonrt_ids, sess=ses, contrast_lab=contrast,
                  model_type='mod-Cue-None', tmp_dir=f'{tmp_rand}/concat_imgs')
# make rt 4d
make_4d_data_mask(bold_paths=nonrt_ids, sess=ses, contrast_lab=contrast,
                  model_type='mod-Cue-rt', tmp_dir=f'{tmp_rand}/concat_imgs')

nonrt_nii = f'{tmp_rand}/concat_imgs/subs-500_ses-{ses}_task-MID_contrast-{contrast}_mod-Cue-None.nii'
rt_nii = f'{tmp_rand}/concat_imgs/subs-500_ses-{ses}_task-MID_contrast-{contrast}_mod-Cue-rt.nii'
# estimate differences for randomise
diff_nifti = math_img('img1 - img2', img1=nonrt_nii, img2=rt_nii)


# Create design matrix with intercept (1s) that's length of subjects/length of fixed_files
level = 'grp/diff'
design_matrix = pd.DataFrame({'int': [1] * len(nonrt_list)})
make_randomise_files(desmat_final=design_matrix, regressor_names='int',
                     contrasts=['int'], outdir=f'{tmp_rand}/randomise/{contrast}/{level}')
make_randomise_grp(comb_nii_path=diff_nifti, outdir=f'{tmp_rand}/randomise/{contrast}/{level}')

print("*** Running: *** /grp/diff", contrast)
script_path = f'{tmp_rand}/randomise/{contrast}/{level}/randomise_call.sh'
subprocess.run(['bash', script_path])

