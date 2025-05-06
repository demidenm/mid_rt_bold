import warnings
warnings.filterwarnings("ignore")
import sys
import os
import argparse
import pandas as pd
from glob import glob
from nilearn.glm.first_level import FirstLevelModel
import matplotlib.pyplot as plt
from nilearn.plotting import plot_design_matrix

plt.switch_backend('Agg') # turn off back end display to create plots

# Getpath to Stage2 scripts
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
from scripts.model_designmat_regressors import (
    create_design_mid, pull_regressors, fixed_effect, 
    process_data, fix_feedback_durations)

parser = argparse.ArgumentParser(description="Script to run first level task models w/ nilearn")

parser.add_argument("--sample", help="sample type, abcd, AHRB, MLS?")
parser.add_argument("--sub", help="subject name, sub-XX, include entirety with 'sub-' prefix")
parser.add_argument("--task", help="task type -- e.g., mid, reward, etc")
parser.add_argument("--ses", help="session, include the session type without prefix, e.g., 1, 01, baselinearm1")
parser.add_argument("--numvols", help="The number of volumes for BOLD file, e.g numeric")
parser.add_argument("--boldtr", help="the tr value for the datasets in seconds, e.g. .800, 2.0, 3.0")
parser.add_argument("--beh_path", help="Path to the behavioral (.tsv) directory/files for the task")
parser.add_argument("--fmriprep_path", help="Path to the output directory for the fmriprep output")
parser.add_argument("--mask", help="path the to a binarized brain mask (e.g., MNI152 or "
                                   "constrained mask in MNI space, spec-network",
                    default=None)
parser.add_argument("--mask_label", help="label for mask, e.g. mni152, yeo-network, etc",
                    default=None)
parser.add_argument("--output", help="output folder where to write out and save information")
args = parser.parse_args()

# Now you can access the arguments as attributes of the 'args' object.
sample = args.sample
subj = args.sub
task = args.task
ses = args.ses
numvols = int(args.numvols)
boldtr = float(args.boldtr)
beh_path = args.beh_path
fmriprep_path = args.fmriprep_path
brainmask = args.mask
mask_label = args.mask_label
scratch_out = args.output

rename_conds = {
    'LgReward': 'LargeWin',
    'SmallReward': 'SmallWin',
    'LgPun': 'LargeLoss',
    'SmallPun': 'SmallLoss',
    'Triangle': 'Neutral'
}

# GENERAL CONTRASTS FOR PRIMARY MODEL COMPARISONS
cue_contrasts = [
    # ANTICIPATORY
    'Cue:LW-Neut', 'Cue:W-Neut', 'Cue:LL-Neut', 'Cue:L-Neut', 'Cue:LW-Base',
    # FEEDBACK
    'FB:WHit-WMiss', 'FB:LWHit-LWMiss', 
    'FB:LWHit-NeutHit','FB:LWHit-Base',
    'FB:LHit-LMiss', 'FB:LLHit-LLMiss'
]

cue_contrasts_labs = {
    # ANTICIPATORY
    'Cue:LW-Neut': 'LargeWin - Neutral',
    'Cue:W-Neut': 'LargeWin + SmallWin - 2*Neutral',
    'Cue:LL-Neut': 'LargeLoss - Neutral',
    'Cue:L-Neut': 'LargeLoss + SmallLoss - 2*Neutral',
    'Cue:LW-Base': 'LargeWin',
    # FEEDBACK
    'FB:WHit-WMiss': 'LargeWinHit + SmallWinHit - LargeWinMiss - SmallWinMiss',
    'FB:LWHit-LWMiss': 'LargeWinHit - LargeWinMiss',
    'FB:LWHit-NeutHit': 'LargeWinHit - NeutralHit',
    'FB:LWHit-Base': 'LargeWinHit',
    'FB:LHit-LMiss': 'LargeLossHit + SmallLossHit - LargeLossMiss - SmallLossMiss',
    'FB:LLHit-LLMiss': 'LargeLossHit - LargeLossMiss',
}

# FULL MODELS WITH FIX

full_contrasts = [
    # CUE ANTICIPATORY
    'Cue:LW-Neut', 'Cue:W-Neut', 'Cue:LL-Neut', 'Cue:L-Neut', 'Cue:LW-Base',
    # FIXATION ANTICIPATORY
    'Fix:LW-Neut', 'Fix:W-Neut', 'Fix:LL-Neut', 'Fix:L-Neut', 'Fix:LW-Base',
    # FEEDBACK
    'FB:WHit-WMiss', 'FB:LWHit-LWMiss', 
    'FB:LWHit-NeutHit','FB:LWHit-Base',
    'FB:LHit-LMiss', 'FB:LLHit-LLMiss',
    # PROBE
    'Probe:All-Base', 'Probe:Win-Loss', 'Probe:Win-Neut', 'Probe:Loss-Neut',
    # PROBE RT
    'RT'
]

full_contrasts_labs = {
    # CUE ANTICIPATORY
    'Cue:LW-Neut': 'LargeWin - Neutral',
    'Cue:W-Neut': 'LargeWin + SmallWin - 2*Neutral',
    'Cue:LL-Neut': 'LargeLoss - Neutral',
    'Cue:L-Neut': 'LargeLoss + SmallLoss - 2*Neutral',
    'Cue:LW-Base': 'LargeWin',
    # FIXATION ANTICIPATORY
    'Fix:LW-Neut': 'FixLargeWin - FixNeutral',
    'Fix:W-Neut': 'FixLargeWin + FixSmallWin - 2*FixNeutral',
    'Fix:LL-Neut': 'FixLargeLoss - FixNeutral',
    'Fix:L-Neut': 'FixLargeLoss + FixSmallLoss - 2*FixNeutral',
    'Fix:LW-Base': 'FixLargeWin',
    # FEEDBACK
    'FB:WHit-WMiss': 'LargeWinHit + SmallWinHit - LargeWinMiss - SmallWinMiss',
    'FB:LWHit-LWMiss': 'LargeWinHit - LargeWinMiss',
    'FB:LWHit-NeutHit': 'LargeWinHit - NeutralHit',
    'FB:LWHit-Base': 'LargeWinHit',
    'FB:LHit-LMiss': 'LargeLossHit + SmallLossHit - LargeLossMiss - SmallLossMiss',
    'FB:LLHit-LLMiss': 'LargeLossHit - LargeLossMiss',
    # PROBE
    'Probe:All-Base': '0.33*probe_win + 0.33*probe_lose + 0.33*probe_neut',
    'Probe:Win-Loss': 'probe_win - probe_lose',
    'Probe:Win-Neut': 'probe_win - probe_neut',
    'Probe:Loss-Neut': 'probe_lose - probe_neut',
    # PROBE RT
    'RT': 'probe_rt'
}

fwhm = 5
runs = ['01', '02']
model_list = ['Saturated', 'CueYesDeriv', 'CueNoDeriv']

for run in runs:
    print(f'\tStarting {subj} {run}.')
    # import behavior events .tsv from data path, fix issue with RT column & duration onsettoonset issues
    eventsdat = pd.read_csv(f'{beh_path}/{subj}/ses-{ses}/func/{subj}_ses-{ses}_task-{task}_run-{run}_events.tsv',
                            sep='\t')
    eventsdat = fix_feedback_durations(eventsdat)
    eventsdat['Condition'] = eventsdat['Condition'].replace(rename_conds)
    events_df = process_data(eventsdat)
    events_df.to_csv(f'{scratch_out}/{subj}_ses-{ses}_task-{task}_run-{run}_events.tsv',
                     sep='\t', index=False)

    # get path to confounds from fmriprep, func data + mask, set image path
    conf_path = f'{fmriprep_path}/{subj}/ses-{ses}/func/{subj}_ses-{ses}_task-{task}_run-{run}' \
                f'_desc-confounds_timeseries.tsv'
    nii_path = glob(
        f'{fmriprep_path}/{subj}/ses-{ses}/func/{subj}_ses-{ses}_task-{task}_run-{run}'
        f'_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz')[0]
    print('\t\t 1/3 Create Regressors & Design Matrix for GLM')
    
    # get list of regressors
    # run to create design matrix
    conf_regressors = pull_regressors(confound_path=conf_path, regressor_type='opt2')

    for model in model_list:
        design_matrix = create_design_mid(events_df=events_df, bold_tr=boldtr, num_volumes=numvols,
                                          conf_regressors=conf_regressors, rt_model=model,
                                          hrf_model='spm', stc=False)
        # save design mat
        plot_design_matrix(design_matrix)
        plt.savefig(f'{scratch_out}/{subj}_ses-{ses}_task-{task}_run-{run}_mod-{model}_design-mat.png')

        print('\t\t 2/3 Mask Image, Fit GLM model ar1 autocorrelation')
        # using ar1 autocorrelation (FSL prewhitening), drift model
        fmri_glm = FirstLevelModel(subject_label=subj, mask_img=brainmask,
                                   t_r=boldtr, smoothing_fwhm=fwhm,
                                   standardize=False, noise_model='ar1', drift_model=None, high_pass=None
                                   )
        # Run GLM model using set paths and calculate design matrix
        run_fmri_glm = fmri_glm.fit(nii_path, design_matrices=design_matrix)
        print('\t\t 3/3: From GLM model, create/save contrast beta/variance maps to output path')
        if model in ['CueYesDeriv', 'CueNoDeriv']:
            contrast_list = cue_contrasts_labs
        elif model == 'Saturated':
            contrast_list = full_contrasts_labs
        else:
            print("Model provided is not of CueYesDeriv, CueNoDeriv or Saturated")

        for con_name, con in contrast_list.items():
            try:
                beta_name = f'{scratch_out}/{subj}_ses-{ses}_task-{task}_run-{run}_contrast-{con_name}_mod-{model}_stat-beta.nii.gz'
                beta_est = run_fmri_glm.compute_contrast(con, output_type='effect_size')
                beta_est.to_filename(beta_name)
                # Calc: variance
                var_name = f'{scratch_out}/{subj}_ses-{ses}_task-{task}_run-{run}_contrast-{con_name}_mod-{model}_stat-var.nii.gz'
                var_est = run_fmri_glm.compute_contrast(con, output_type='effect_variance')
                var_est.to_filename(var_name)
            except Exception as e:
                print(f'Error processing beta: {e} for {subj} and {con_name}')


print("Running Fixed effect model -- precision weight of runs for each contrast")

for model in model_list:
    if model in ['CueYesDeriv', 'CueNoDeriv']:
        contrast = cue_contrasts
    elif model == 'Saturated':
        contrast = full_contrasts
    else:
        print("Model Provide is not of CueYesDeriv, CueNoDeriv or Saturated")

    fixed_effect(subject=subj, session=ses, task_type=task,
                 contrast_list=contrast, firstlvl_indir=scratch_out, fixedeffect_outdir=scratch_out,
                 model_lab=model, save_beta=True, save_var=True, save_tstat=True)
