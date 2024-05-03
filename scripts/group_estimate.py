import os
import warnings
import argparse
import pandas as pd
from glob import glob
from nilearn.glm.second_level import SecondLevelModel
warnings.filterwarnings("ignore")


def group_onesample(fixedeffect_paths: list, session: str, task_type: str,
                    contrast_type: str, group_outdir: str,
                    model_lab: str, rt_array = None, mask: str = None):
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
    :param rt_df: dataframe with subject RT files to use in group regressor
    :param mask: path to mask, default none
    :return: nothing return, files are saved
    """

    if not os.path.exists(group_outdir):
        os.makedirs(group_outdir)
        print("Directory created:", group_outdir)

    N_maps = len(fixedeffect_paths)

    if model == 'mod-Cue-rt':
        # Create design matrix with intercept (1s) that's length of subjects/length of fixed_files
        design_matrix = pd.DataFrame([1] * N_maps, columns=['int'])
        design_matrix['rt'] = rt_array

        # Fit secondlevel model
        sec_lvl_model = SecondLevelModel(mask_img=mask, smoothing_fwhm=None, minimize_memory=False)
        sec_lvl_model = sec_lvl_model.fit(second_level_input=fixedeffect_paths,
                                          design_matrix=design_matrix)
        # contrasts rt models
        for con in ['int','rt']:
            tstat_map = sec_lvl_model.compute_contrast(
                second_level_contrast=con,
                second_level_stat_type='t',
                output_type='stat'
            )
            tstat_out = f'{group_outdir}/subs-{N_maps}_ses-{session}_task-{task_type}_' \
                        f'contrast-{contrast_type}_{model_lab}_stat-tstat_{con}.nii.gz'
            tstat_map.to_filename(tstat_out)
            
    elif model == 'mod-Cue-None':
        # Create design matrix with intercept (1s) that's length of subjects/length of fixed_files
        design_matrix = pd.DataFrame([1] * N_maps, columns=['int'])
        sec_lvl_model = SecondLevelModel(mask_img=mask, smoothing_fwhm=None, minimize_memory=False)
        sec_lvl_model = sec_lvl_model.fit(second_level_input=fixedeffect_paths,
                                          design_matrix=design_matrix)
        tstat_map = sec_lvl_model.compute_contrast(
            second_level_contrast='int',
            second_level_stat_type='t',
            output_type='stat'
            )
        tstat_out = f'{group_outdir}/subs-{N_maps}_ses-{session}_task-{task_type}_' \
                    f'contrast-{contrast_type}_{model_lab}_stat-tstat_int.nii.gz'
        tstat_map.to_filename(tstat_out)
    else:
        print("Model is incorrect:", model, "Should be mod-Cue-rt or mod-Cue-non")


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
brainmask = args.mask
model = args.model
in_dir = args.input
scratch_out = args.output

# contrasts
contrasts = [
    # anticipatory contrasts for cue-model
    'LRew-Neut', 'ARew-Neut', 'LPun-Neut', 'APun-Neut',
    # feedback contrasts
    'ARewHit-ARewMiss', 'LRewHit-LRewMiss', 'APunHit-APunMiss',
    'LPunHit-LPunMiss', 'LRewHit-LNeutHit',
    # probe maps
    'probe-base', 'rt-base'
]

sub_rt_df = pd.read_csv(rt_file, sep=',')

if model == 'mod-Cue-rt':
    # find all contrast fixed effect maps for model permutation across subjects
    list_maps = sorted(glob(f'{in_dir}/*_ses-{ses}_task-{task}_*'
                            f'contrast-{contrast}_{model}_stat-effect.nii.gz'))
    # subset id's RT times to match
    sub_ids = [os.path.basename(path).split('_')[0] for path in list_maps]
    subset_df = sub_rt_df[sub_rt_df['Subject'].isin(sub_ids)].copy()
    mean_rt = subset_df['Average_RT'].mean()
    subset_df['Mean_Centered_RT'] = (subset_df['Average_RT'] - mean_rt).values
    rt_vals = subset_df['Mean_Centered_RT'].values
    # set list
    contrast_list = contrasts
    for contrast in contrast_list:
        group_onesample(fixedeffect_paths=list_maps, session=ses, task_type=task,
                        contrast_type=contrast, group_outdir=scratch_out,
                        model_lab=model, mask=brainmask, rt_array=rt_vals)

elif model == 'mod-Cue-None':
    # subset list to remove probe models
    contrast_list = [contrast for contrast in contrasts if contrast not in ['probe-base', 'rt-base']]
    rt_vals=None
    for contrast in contrast_list:
        # find all contrast fixed effect maps for model permutation across subjects
        list_maps = sorted(glob(f'{in_dir}/*_ses-{ses}_task-{task}_*'
                                f'contrast-{contrast}_{model}_stat-effect.nii.gz'))
        group_onesample(fixedeffect_paths=list_maps, session=ses, task_type=task,
                        contrast_type=contrast, group_outdir=scratch_out,
                        model_lab=model, mask=brainmask, rt_array=rt_vals)

else:
    print("Model is incorrect:", model, "Should be mod-Cue-rt or mod-Cue-None")
