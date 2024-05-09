import os
import warnings
import subprocess
import argparse
import numpy as np
import pandas as pd
from glob import glob
from nilearn.glm.second_level import SecondLevelModel
warnings.filterwarnings("ignore")

# Getpath to Stage2 scripts
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
from scripts.model_designmat_regressors import (
    make_randomise_grp, make_randomise_rt, make_randomise_files, make_4d_data_mask
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
    design_matrix['rt'] = rt_array

    # Fit secondlevel model
    sec_lvl_model = SecondLevelModel(mask_img=mask, smoothing_fwhm=None, minimize_memory=False)
    sec_lvl_model = sec_lvl_model.fit(second_level_input=fixedeffect_paths,
                                      design_matrix=design_matrix)
    # contrasts mean 'int' and corr with mRT 'rt
    for con in ['int', 'rt']:
        tstat_map = sec_lvl_model.compute_contrast(
            second_level_contrast=con,
            second_level_stat_type='t',
            output_type='stat'
        )
        tstat_out = f'{group_outdir}/subs-{n_maps}_ses-{session}_task-{task_type}_' \
                    f'contrast-{contrast_type}_{model_lab}_stat-tstat_{con}.nii.gz'
        tstat_map.to_filename(tstat_out)



parser = argparse.ArgumentParser(description="Script to run first level task models w/ nilearn")
parser.add_argument("--sample", help="sample type, ahrb, abcd or mls?")
parser.add_argument("--task", help="task type -- e.g., mid, reward, etc")
parser.add_argument("--run", help="Run lvl -- e.g., 1 or 2 (for 01 / 02)")
parser.add_argument("--rt_file", help="File to subject level rt data(for 01 / 02)", default=None)
parser.add_argument("--ses", help="session, include the session type without prefix 'ses', e.g., 1, 01, baselinearm1")
parser.add_argument("--model", help="model label,"
                                    " e.g. 'mod-Cue-rt' or 'mod-Cue-None'")
parser.add_argument("--mask", help="path the to the binarized brain mask (e.g., MNI152 or "
                                   "constrained mask in MNI space, or None")
parser.add_argument("--input", help="input path to data")
parser.add_argument("--output", help="output folder where to write out and save information")

args = parser.parse_args()

# Now you can access the arguments as attributes of the 'args' object.
sample = args.sample
task = args.task
run = args.run
rt_file = args.rt_file
ses = args.ses
model = args.model
brainmask = args.mask
in_dir = args.input
scratch_out = args.output

# contrasts
contrasts = [
    # anticipatory contrasts for cue-model
    'LRew-Neut', 'ARew-Neut', 'LPun-Neut', 'APun-Neut',
    # feedback contrasts
    'ARewHit-ARewMiss', 'LRewHit-LRewMiss', 'APunHit-APunMiss',
    'LPunHit-LPunMiss', 'LRewHit-NeutHit',
    # probe maps
    'probe-base', 'rt-base'
]

sub_rt_df = pd.read_csv(rt_file, sep=',')


if model == 'mod-Cue-rt':
    contrast_list = contrasts
elif model == 'mod-Cue-None':
    # subset list to remove probe models
    contrast_list = [contrast for contrast in contrasts if contrast not in ['probe-base', 'rt-base']]
else:
    print("Model is incorrect:", model, "Should be mod-Cue-rt or mod-Cue-None")

for contrast in contrast_list:
    # find all contrast fixed effect maps for model permutation across subjects
    list_maps = sorted(glob(f'{in_dir}/*_ses-{ses}_task-{task}_*'
                            f'contrast-{contrast}_{model}_stat-effect.nii.gz'))
    # subset id's RT times to match
    sub_ids = [os.path.basename(path).split('_')[0] for path in list_maps]
    subset_df = sub_rt_df[sub_rt_df['Subject'].isin(sub_ids)].copy()
    subset_df = subset_df.set_index('Subject').loc[sub_ids].reset_index() # ensure index sorts same as IDs
    assert (subset_df['Subject'].values ==
            np.array(sub_ids)).all(), "Order of IDs in subset_df != sub_ids."

    mean_rt = subset_df['Average_RT'].mean()
    subset_df['Mean_Centered_RT'] = (subset_df['Average_RT'] - mean_rt).values
    rt_vals = subset_df['Mean_Centered_RT'].values

    # standard group maps with nilearn
    group_onesample(fixedeffect_paths=list_maps, session=ses, task_type=task,
                    contrast_type=contrast, group_outdir=scratch_out,
                    model_lab=model, mask=brainmask, rt_array=rt_vals)
    # randomise, permuted maps + corrected
    tmp_rand=f'{scratch_out}/randomise'
    make_4d_data_mask(bold_paths=list_maps, sess=ses, contrast_lab=contrast,
                      model_type=model, tmp_dir=f'{tmp_rand}/concat_imgs')

    n_maps = len(list_maps)
    comb_input_nii = f'{tmp_rand}/concat_imgs/subs-500_ses-{ses}_task-MID_contrast-{contrast}_{model}.nii'

    for level in ['grp', 'rt']:
        if level == 'grp':
            # Create design matrix with intercept (1s) that's length of subjects/length of fixed_files
            design_matrix = pd.DataFrame({'int': [1] * n_maps})
            make_randomise_files(desmat_final=design_matrix, regressor_names='int',
                                 contrasts=['int'], outdir=f'{tmp_rand}/randomise/{contrast}/{level}')
            make_randomise_grp(comb_nii_path=comb_input_nii, outdir=f'{tmp_rand}/randomise/{contrast}/{level}')

        else:
            design_matrix = pd.DataFrame({'int': [1] * n_maps, 'rt': rt_vals})
            make_randomise_files(desmat_final=design_matrix, regressor_names=['int', 'rt'],
                                 contrasts=[['rt']], outdir=f'{tmp_rand}/randomise/{contrast}/{level}')
            make_randomise_rt(comb_nii_path=comb_input_nii, outdir=f'{tmp_rand}/randomise/{contrast}/{level}')

        print("*** Running: *** ", level)
        script_path = f'{tmp_rand}/randomise/{contrast}/{level}/randomise_call.sh'
        subprocess.run(['bash', script_path])