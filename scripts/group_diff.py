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
    make_randomise_files, make_4d_data_mask
)

parser = argparse.ArgumentParser(description="Script to run first level task models w/ nilearn")
parser.add_argument("--sample", help="sample type, ahrb, abcd or mls?")
parser.add_argument("--task", help="task type -- e.g., mid, reward, etc")
parser.add_argument("--run", help="Run lvl -- e.g., 1 or 2 (for 01 / 02)")
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
a_mod = args.a_mod
b_mod = args.b_mod
contrast = args.contrast
brainmask = args.mask
in_dir = args.input
scratch_out = args.output

# find all contrast fixed effect maps for model permutation across subjects and get IDs to match to lists
a_mod_list = sorted(glob(f'{in_dir}/*_ses-{ses}_task-{task}_*contrast-{contrast}_{a_mod}_stat-effect.nii.gz'))
b_mod_list = sorted(glob(f'{in_dir}/*_ses-{ses}_task-{task}_*contrast-{contrast}_{b_mod}_stat-effect.nii.gz'))
a_mod_ids = [os.path.basename(path).split('_')[0] for path in a_mod_list]
b_mod_ids = [os.path.basename(path).split('_')[0] for path in b_mod_list]

assert (np.array(a_mod_ids) == np.array(b_mod_ids)).all(), "Order of IDs in a_mod != b_mod."


# randomise, permuted maps + corrected
tmp_rand = f'{scratch_out}/randomise'
# make nonrt & rt 4D
make_4d_data_mask(bold_paths=a_mod_list, sess=ses, contrast_lab=contrast,
                  model_type=a_mod, tmp_dir=f'{tmp_rand}/concat_imgs')
# make rt 4d
make_4d_data_mask(bold_paths=b_mod_list, sess=ses, contrast_lab=contrast,
                  model_type=b_mod, tmp_dir=f'{tmp_rand}/concat_imgs')

a_mod_nii = f'{tmp_rand}/concat_imgs/subs-500_ses-{ses}_task-MID_contrast-{contrast}_{a_mod}.nii.gz'
b_mod_nii = f'{tmp_rand}/concat_imgs/subs-500_ses-{ses}_task-MID_contrast-{contrast}_{b_mod}.nii.gz'
# estimate differences for randomise
diff_nifti = math_img('img1 - img2', img1=a_mod_nii, img2=b_mod_nii)
diff_label = f'diff-{a_mod.split("-")[-1]}-{b_mod.split("-")[-1]}'
diff_nifti_path = f'{tmp_rand}/concat_imgs/subs-500_ses-{ses}_task-MID_contrast-{contrast}_{diff_label}.nii.gz'
diff_nifti.to_filename(diff_nifti_path)


# Create design matrix with intercept (1s) that's length of subjects/length of fixed_files
level = 'grp/diff'
design_matrix = pd.DataFrame({'int': [1] * len(a_mod_list)})
make_randomise_files(desmat_final=design_matrix, regressor_names='int',
                     contrasts=['int'], outdir=f'{tmp_rand}/randomise/{contrast}/{level}')

# make group randomise run
outdir = f'{tmp_rand}/randomise/{contrast}/{level}'

if not os.path.exists(f'{outdir}'):
    os.makedirs(f'{outdir}')

randomise_call = (f'randomise_parallel -i {diff_nifti_path}'
                  f' -o {outdir}/subs-500_ses-{ses}_task-MID_contrast-{contrast}_{diff_label}_randomise'
                  f' -m {tmp_rand}/concat_imgs/subs-500_ses-{ses}_task-MID_contrast-{contrast}_{a_mod}_mask.nii.gz'
                  f' -1 -t {outdir}/desmat.con'
                  f' -f {outdir}/desmat.fts  -T -n 1000')
randomise_call_file = Path(f'{outdir}/randomise_call.sh')

with open(randomise_call_file, 'w') as f:
    f.write(randomise_call)
# This should change the file permissions to make the script executeable
randomise_call_file.chmod(randomise_call_file.stat().st_mode | stat.S_IXGRP | stat.S_IEXEC)


print("*** Running: *** /grp/diff", contrast)
script_path = f'{outdir}/randomise_call.sh'
subprocess.run(['bash', script_path])
