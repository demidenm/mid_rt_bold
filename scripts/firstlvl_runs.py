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
from scripts.model_designmat_regressors import create_design_mid, pull_regressors, fixed_effect

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

contrasts = [
    # anticipatory contrasts for cue-model
    'LRew-Neut', 'ARew-Neut', 'LPun-Neut', 'APun-Neut',
    # feedback contrasts
    'ARewHit-ARewMiss', 'LRewHit-LRewMiss', 'APunHit-APunMiss',
    'LPunHit-LPunMiss', 'LRewHit-NeutHit',
    # probe maps
    'probe-base', 'rt-base'
]

contrast_probxcond = [
    # anticipatory contrasts for cue-model
    'LRew-Neut', 'ARew-Neut', 'LPun-Neut', 'APun-Neut',
    # feedback contrasts
    'ARewHit-ARewMiss', 'LRewHit-LRewMiss', 'APunHit-APunMiss',
    'LPunHit-LPunMiss', 'LRewHit-NeutHit',
    # probe-by-condition
    'probeLRew-probeNeut', 'probeARew-probeNeut', 'probeLPun-probeNeut', 'probeAPun-probeNeut',
    'probeARewHit-probeARewMiss', 'probeLRewHit-probeLRewMiss', 'probeAPunHit-probeAPunMiss',
    'probeLPunHit-probeLPunMiss', 'probeLRewHit-probeNeutHit'
]

contrast_labs = {
    # Anticipation
    'LRew-Neut': 'LgReward - Triangle',
    'ARew-Neut': 'LgReward + SmallReward - 2*Triangle',
    'LPun-Neut': 'LgPun - Triangle',
    'APun-Neut': 'LgPun + SmallPun - 2*Triangle',
    # Feedback
    'ARewHit-ARewMiss': 'LgReward_hit + SmallReward_hit - LgReward_miss - SmallReward_miss',
    'LRewHit-LRewMiss': 'LgReward_hit - LgReward_miss',
    'APunHit-APunMiss': 'LgPun_hit + SmallPun_hit - LgPun_miss - SmallPun_miss',
    'LPunHit-LPunMiss': 'LgPun_hit - LgPun_miss',
    'LRewHit-NeutHit': 'LgReward_hit - Triangle_hit',
    # robe
    'probe-base': 'probe',
    'rt-base': 'probe_rt'
}

contrast_probxcond_labs = {
    # Anticipation
    'LRew-Neut': 'LgReward - Triangle',
    'ARew-Neut': 'LgReward + SmallReward - 2*Triangle',
    'LPun-Neut': 'LgPun - Triangle',
    'APun-Neut': 'LgPun + SmallPun - 2*Triangle',
    # Feedback
    'ARewHit-ARewMiss': 'LgReward_hit + SmallReward_hit - LgReward_miss - SmallReward_miss',
    'LRewHit-LRewMiss': 'LgReward_hit - LgReward_miss',
    'APunHit-APunMiss': 'LgPun_hit + SmallPun_hit - LgPun_miss - SmallPun_miss',
    'LPunHit-LPunMiss': 'LgPun_hit - LgPun_miss',
    'LRewHit-NeutHit': 'LgReward_hit - Triangle_hit',
    # probe-by-condition
    'probeLRew-probeNeut': 'prbhit_LgReward + prbmiss_LgReward - prbhit_Triangle - prbmiss_Triangle',
    'probeARew-probeNeut': '.5*prbhit_LgReward + .5*prbmiss_LgReward + .5*prbhit_SmallReward + .5*prbmiss_SmallReward -'
                           '1*prbhit_Triangle - 1*prbmiss_Triangle',
    'probeLPun-probeNeut': 'prbhit_LgPun + prbmiss_LgPun - prbhit_Triangle - prbmiss_Triangle',
    'probeAPun-probeNeut': '.5*prbhit_LgPun + .5*prbmiss_LgPun + .5*prbhit_SmallPun + .5*prbmiss_SmallPun - '
                           '1*prbhit_Triangle - 1*prbmiss_Triangle',
    'probeARewHit-probeARewMiss': 'prbhit_LgReward + prbhit_SmallReward - prbmiss_LgReward - prbmiss_SmallReward',
    'probeLRewHit-probeLRewMiss': 'prbhit_LgReward - prbmiss_LgReward',
    'probeAPunHit-probeAPunMiss': 'prbhit_LgPun + prbhit_SmallPun - prbmiss_LgPun - prbmiss_SmallPun',
    'probeLPunHit-probeLPunMiss': 'prbhit_LgPun - prbmiss_LgPun',
    'probeLRewHit-probeNeutHit': 'prbhit_LgReward - prbhit_Triangle'
}

fwhm = 5
runs = ['01', '02']
model_list = [None, 'rt', 'probexcond', 'dairc']
for run in runs:
    print(f'\tStarting {subj} {run}.')
    # import behavior events .tsv from data path
    events_df = pd.read_csv(f'{beh_path}/{subj}/ses-{ses}/func/{subj}_ses-{ses}_task-{task}_run-{run}_events.tsv',
                            sep='\t')

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
        plt.savefig(f'{scratch_out}/{subj}_ses-{ses}_task-{task}_run-{run}_mod-Cue-{model}_design-mat.png')

        print('\t\t 2/3 Mask Image, Fit GLM model ar1 autocorrelation')
        # using ar1 autocorrelation (FSL prewhitening), drift model
        fmri_glm = FirstLevelModel(subject_label=subj, mask_img=brainmask,
                                   t_r=boldtr, smoothing_fwhm=fwhm,
                                   standardize=False, noise_model='ar1', drift_model=None, high_pass=None
                                   )
        # Run GLM model using set paths and calculate design matrix
        run_fmri_glm = fmri_glm.fit(nii_path, design_matrices=design_matrix)
        print('\t\t 3/3: From GLM model, create/save contrast beta/variance maps to output path')
        if model in [None, 'dairc']:
            contrast_list = {key: value for key, value in contrast_labs.items() if key not in ['probe-base', 'rt-base']}
        elif model == 'rt':
            contrast_list = contrast_labs
        elif model == 'probexcond':
            contrast_list = contrast_probxcond_labs
        else:
            print("Model should be RT or None")

        for con_name, con in contrast_list.items():
            try:
                beta_name = f'{scratch_out}/{subj}_ses-{ses}_task-{task}_run-{run}_contrast-{con_name}_mod-Cue-{model}_stat-beta.nii.gz'
                beta_est = run_fmri_glm.compute_contrast(con, output_type='effect_size')
                beta_est.to_filename(beta_name)
                # Calc: variance
                var_name = f'{scratch_out}/{subj}_ses-{ses}_task-{task}_run-{run}_contrast-{con_name}_mod-Cue-{model}_stat-var.nii.gz'
                var_est = run_fmri_glm.compute_contrast(con, output_type='effect_variance')
                var_est.to_filename(var_name)
            except Exception as e:
                print(f'Error processing beta: {e} for {subj} and {con_name}')


print("Running Fixed effect model -- precision weight of runs for each contrast")

for model in model_list:
    if model in [None, 'dairc']:
        contrast = [contrast for contrast in contrasts if contrast not in ['probe-base', 'rt-base']]
    elif model == 'rt':
        contrast = contrasts
    elif model == 'probexcond':
        contrast = contrast_probxcond
    else:
        print("Model should be RT or None")

    fixed_effect(subject=subj, session=ses, task_type=task,
                 contrast_list=contrast, firstlvl_indir=scratch_out, fixedeffect_outdir=scratch_out,
                 model_lab=model, save_beta=True, save_var=True, save_tstat=False)
