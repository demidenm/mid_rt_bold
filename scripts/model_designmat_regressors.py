import os
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm import compute_fixed_effects
from glob import glob


def pull_regressors(confound_path: str, regressor_type: str = 'opt1') -> pd.DataFrame:
    """
    This function is compatible with the *confounds_timeseries.tsv file exported by fMRIprep
    When calling this function, provide the path to the confounds file for each subject (and run) and select
    the type of confounds to pull (opt_1 to opt_5).
    The functions returns a pandas dataframe with the extract confounds.

    :param confound_path: path to the *counfounds_timeseries.tsv
    :param regressor_type: Confound option from list 'conf_opt1','conf_opt2','conf_opt3','conf_opt4'
        'opt1': cosine_00 to cosine_03
        'opt2': opt1 + tran x, y, z & rot x, y, z
        'opt3': opt2 + trans x, y, z and rot x, y, z derivatives
        'opt4': opt3 + a_comp_cor 0:7 (top 8 components)
        'opt5': opt4 + motion outliers in confounds file
    :return: list of confound regressors
    """
    if not os.path.exists(confound_path):
        raise ValueError("Confounds file path not found. Check if {} exists".format(confound_path))

    confound_df = glob(confound_path)[0]
    confound_df = pd.read_csv(confound_df, sep='\t', na_values=['n/a']).fillna(0)

    # Setting up dictionary from which to pull confound list
    if regressor_type != 'opt4':
        confound_dict = {
            "opt1": ['cosine00', 'cosine01', 'cosine02', 'cosine03'],
            "opt2": ['cosine00', 'cosine01', 'cosine02', 'cosine03',
                     'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                     'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
                     'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1'],
            "opt3": ['cosine00', 'cosine01', 'cosine02', 'cosine03',
                     'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                     'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
                     'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1',
                     "a_comp_cor_00", "a_comp_cor_01", "a_comp_cor_02", "a_comp_cor_03", "a_comp_cor_04",
                     "a_comp_cor_05", "a_comp_cor_06", "a_comp_cor_07"]
        }

    else:
        confound_dict = {
            "opt1": ['cosine00', 'cosine01', 'cosine02', 'cosine03'],
            "opt2": ['cosine00', 'cosine01', 'cosine02', 'cosine03',
                     'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                     'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
                     'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1'],
            "opt3": ['cosine00', 'cosine01', 'cosine02', 'cosine03',
                     'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                     'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
                     'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1',
                     "a_comp_cor_00", "a_comp_cor_01", "a_comp_cor_02", "a_comp_cor_03", "a_comp_cor_04",
                     "a_comp_cor_05", "a_comp_cor_06", "a_comp_cor_07"]
        }

        motion_outlier_columns = confound_df.filter(regex='motion_outlier')
        # append the motion outlier columns to in dict to opt3  as opt5
        confound_dict['opt4'] = confound_dict['opt3'] + list(motion_outlier_columns.columns)

    if 'cosine03' not in confound_df.columns:
        confound_dict[regressor_type].remove('cosine03')

    return pd.DataFrame(confound_df[confound_dict[regressor_type]])


def create_design_mid(events_df: pd.DataFrame, bold_tr: float, num_volumes: int, conf_regressors: pd.DataFrame,
                      rt_model: str = None, hrf_model: str = 'spm', stc: bool = False) -> pd.DataFrame:
    """
    Creates a design matrix for each run with 5 anticipation 10 feedback conditions and
    specified regressors, such as cosine, motion and/or acompcor from fmriprep confounds.tsv

    :param events_df: this is the pandas dataframe for the events for the MID task
    :param bold_tr: TR for the BOLD volume,
    :param num_volumes: volumes in the BOLD
    :param conf_regressors: dataframe of nuisance regressors from def(regressors)
    :param rt_model: rt or None, default none
    :param hrf_model: select hrf model for design matrix, default spm
    :param stc: whether slice time correction was ran. To adjust the onsets/frame times in design matrix.
            Default False, alt True
    :return: returns a design matrix for specified model
    """

    # create a delinated hit v miss column so it is more clear
    new_feedback_label = 'Feedback.Response'
    events_df[new_feedback_label] = np.where(events_df['prbacc'] == 1.0,
                                             events_df['Condition'] + '_hit',
                                             events_df['Condition'] + '_miss')
    if rt_model is 'rt':
        try:
            # concat  cue onset/duration + feedback onset/duration + probe regressors
            conditions = pd.concat([events_df.loc[:, "Condition"],
                                    events_df.loc[:, "Feedback.Response"],
                                    pd.Series(["probe"] * len(events_df[['OverallRT', 'Probe.OnsetTime']])),
                                    pd.Series(["probe_rt"] * len(events_df[['OverallRT', 'Probe.OnsetTime']].dropna()))
                                    ], ignore_index=True)
            onsets = pd.concat([events_df.loc[:, 'Cue.OnsetTime'],
                                events_df.loc[:, "Feedback.OnsetTime"],
                                events_df.loc[:, "Probe.OnsetTime"],
                                events_df[['OverallRT', 'Probe.OnsetTime']].dropna()['Probe.OnsetTime']
                                ], ignore_index=True)
            duration = pd.concat([events_df.loc[:, 'Cue.Duration'],
                                  events_df.loc[:, "FeedbackDuration"],
                                  events_df.loc[:, "Probe.Duration"],
                                  # convert ms RT times to secs to serve as duration
                                  (events_df[['OverallRT', 'Probe.OnsetTime']].dropna()['OverallRT']) / 1000
                                  ], ignore_index=True)

            # create pandas df with events
            design_events = pd.DataFrame({
                'trial_type': conditions,
                'onset': onsets,
                'duration': duration
            })
        except Exception as e:
            print("When creating RT design matrix, an error occurred: ", e)

    elif rt_model is None:
        # concat only cue onset/duration + feedback onset + duration
        conditions = pd.concat([events_df.loc[:, "Condition"], events_df.loc[:, new_feedback_label]],
                               ignore_index=True)
        onsets = pd.concat([events_df.loc[:, 'Cue.OnsetTime'], events_df.loc[:, "Feedback.OnsetTime"]],
                           ignore_index=True)
        duration = pd.concat([events_df.loc[:, 'Cue.Duration'], events_df.loc[:, "FeedbackDuration"]],
                             ignore_index=True)
        # create pandas df with events
        design_events = pd.DataFrame({'trial_type': conditions,
                                      'onset': onsets,
                                      'duration': duration})
    else:
        print("RT model should be None or True")

    # Using the BOLD tr and volumes to generate the frame_times: acquisition time in seconds
    frame_times = np.arange(num_volumes) * bold_tr
    if stc:
        # default modulation == '1'. Offset the times due to slice time correction, see blog post:
        # https://reproducibility.stanford.edu/slice-timing-correction-in-fmriprep-and-linear-modeling /
        frame_times += bold_tr / 2

    design_matrix_mid = make_first_level_design_matrix(
        frame_times=frame_times,
        events=design_events,
        hrf_model=hrf_model,
        drift_model=None,
        add_regs=conf_regressors
        )

    return design_matrix_mid


def fixed_effect(subject: str, session: str, task_type: str,
                 contrast_list: list, firstlvl_indir: str, fixedeffect_outdir: str,
                 model_lab: str, save_beta=False, save_var=False, save_tstat=True):
    """
    This function takes in a subject, task label, set of computed contrasts using nilearn,
    the path to contrast estimates (beta maps), the output path for fixed effec tmodels and
    specification of types of files to save, the beta estimates, associated variance and t-stat (which is calculated
    based on beta/variance values)
    Several path indices are hard coded, so should update as see fit
    e.g., '{sub}_ses-{ses}_task-{task}_effect-fixed_contrast-{c}_stat-effect.nii.gz'
    :param subject: string-Input subject label, BIDS leading label, e.g., sub-01
    :param session: string-Input session label, BIDS label e.g., ses-1
    :param task_type: string-Input task label, BIDS label e.g., mid
    :param contrast_list: list of contrast types that are saved from first level
    :param model_lab: complete string of model permutation, e.g., 'mod-Cue-rt' or 'mod-Cue-None'
    :param firstlvl_indir: string-location of first level output files
    :param fixedeffect_outdir: string-location to save fixed effects
    :param save_beta: Whether to save 'effects' or beta values, default = False
    :param save_var: Whether to save 'variance' or beta values, default = False
    :param save_tstat: Whether to save 'tstat', default = True
    :return: nothing return, files are saved
    """
    for contrast in contrast_list:
        print(f"\t\t\t Creating weighted fix-eff model for contrast: {contrast}")
        betas = sorted(glob(f'{firstlvl_indir}/{subject}_ses-{session}_task-{task_type}_run-*_'
                            f'contrast-{contrast}_mod-Cue-{model_lab}_stat-beta.nii.gz'))
        var = sorted(glob(f'{firstlvl_indir}/{subject}_ses-{session}_task-{task_type}_run-*_'
                          f'contrast-{contrast}_mod-Cue-{model_lab}_stat-var.nii.gz'))

        # conpute_fixed_effects options
        # (1) contrast map of the effect across runs;
        # (2) var map of between runs effect;
        # (3) t-statistic based on effect of variance;
        fix_effect, fix_var, fix_tstat = compute_fixed_effects(contrast_imgs=betas,
                                                               variance_imgs=var,
                                                               precision_weighted=True)
        if not os.path.exists(fixedeffect_outdir):
            os.makedirs(fixedeffect_outdir)
            print("Directory created:", fixedeffect_outdir)
        if save_beta:
            fix_effect_out = f'{fixedeffect_outdir}/{subject}_ses-{session}_task-{task_type}_' \
                             f'contrast-{contrast}_mod-Cue-{model_lab}_stat-effect.nii.gz'
            fix_effect.to_filename(fix_effect_out)
        if save_var:
            fix_var_out = f'{fixedeffect_outdir}/{subject}_ses-{session}_task-{task_type}_' \
                          f'contrast-{contrast}_mod-Cue-{model_lab}_stat-var.nii.gz'
            fix_var.to_filename(fix_var_out)
        if save_tstat:
            fix_tstat_out = f'{fixedeffect_outdir}/{subject}_ses-{session}_task-{task_type}_' \
                            f'contrast-{contrast}_mod-Cue-{model_lab}_stat-tstat.nii.gz'
            fix_tstat.to_filename(fix_tstat_out)


# Jeanette Mumfords code to incorporate randomise into grp and rt models (not, separating due to randomise issues)
def make_4d_data_mask(bold_paths,sess,contrast_lab, model_type, tmp_dir):
    from nilearn.maskers import NiftiMasker

    if not os.path.exists(f'{tmp_dir}'):
        os.makedirs(f'{tmp_dir}')

    n_maps=len(bold_paths)
    data4d = nib.funcs.concat_images(bold_paths)
    masked_4d = NiftiMasker().fit(data4d).mask_img_
    filename_root = (f'{tmp_dir}/subs-{n_maps}_ses-{sess}_task-MID_contrast-{contrast_lab}_{model_type}')
    data4d.to_filename(f'{filename_root}.nii.gz')
    masked_4d.to_filename(f'{filename_root}_mask.nii.gz')

def make_randomise_files(desmat_final, regressor_names, contrasts, outdir):
    '''
    desmat_final: numpy array of the design matrix
    regressor_names: The regressor names that correspond to the columns of desmat_final
      note, the contrasts are defined using these names.  e.g. ['intercept', 'rt_centered']
    contrasts: expression_to_contrast is used, so these are entered as a list of
      string-based contrast definitions that use regressor_names: [['intercept'], ['rt_centered']]
    outdir: where you're saving all of the files
    '''
    import numpy as np
    from nilearn.glm.contrasts import expression_to_contrast_vector

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    num_input_contrasts = desmat_final.shape[0]
    num_regressors = desmat_final.shape[1]
    # make the desmat.mat file (fsl header with design matrix below)
    # First just make the file header and then append the desmat data
    desmat_path = f'{outdir}/desmat.mat'
    with open(desmat_path, 'w') as f:
        f.write(f'/NumWaves	{num_regressors} \n/NumPoints {num_input_contrasts} '
                '\n/PPheights 1.000000e+00 \n \n/Matrix \n')
        np.savetxt(f, desmat_final, delimiter='\t')
    # .grp file (required for f-tests)
    grp_path = f'{outdir}/desmat.grp'
    with open(grp_path, 'w') as f:
        f.write(f'/NumWaves  1 \n/NumPoints {num_input_contrasts}\n \n/Matrix \n')
        np.savetxt(f, np.ones(num_input_contrasts), fmt='%s', delimiter='\t')
        # save out contrasts in .con file
    contrast_matrix = []
    num_contrasts = len(contrasts)
    for contrast in contrasts:
        contrast_def = expression_to_contrast_vector(
            contrast[0], regressor_names)
        contrast_matrix.append(np.array(contrast_def))
    con_path = f'{outdir}/desmat.con'
    ppheight_and_reqeff = '\t '.join(str(val) for val in [1] * num_contrasts)
    with open(con_path, 'w') as f:
        for val, contrast in enumerate(contrasts):
            f.write(f'/ContrastName{val + 1} {contrast[0]}\n')
        f.write(f'/NumWaves  {num_regressors} \n/NumContrasts {num_contrasts}'
                f'\n/PPheights {ppheight_and_reqeff} '
                f'\n/RequiredEffect {ppheight_and_reqeff} \n \n/Matrix \n')
        np.savetxt(f, contrast_matrix, delimiter='\t')
        # fts file (used for f-test)
    fts_path = f'{outdir}/desmat.fts'
    with open(fts_path, 'w') as f:
        f.write(f'/NumWaves  {num_contrasts} \n/NumContrasts {num_contrasts}\n \n/Matrix \n')
        np.savetxt(f, np.identity(num_contrasts), fmt='%s', delimiter='\t')


def make_randomise_rt(comb_nii_path, outdir, permutations=1000):
    '''
    This hasn't been tested at all, since I launched this differently on Sherlock
    filename_input_root:  this is the filename used to make the mask and data file.  They should
    have the same names, but the mask file ends in _mask.  See the code below to clarify and edit accordingly.
    outdir: same output directoroy where the files from make_randomise_files were saved to

    '''
    from pathlib import Path
    import stat

    inp_dir, file_name = os.path.split(comb_nii_path)
    file_noext, _ = os.path.splitext(file_name)
    if not os.path.exists(f'{outdir}'):
        os.makedirs(f'{outdir}')

    randomise_call = (f'randomise -i {inp_dir}/{file_name}'
                      f' -o {outdir}/{file_noext}_randomise'
                      f' -m {inp_dir}/{file_noext}_mask.nii.gz'
                      f' -d {outdir}/desmat.mat -t {outdir}/desmat.con'
                      f' -f {outdir}/desmat.fts  -T -n {permutations}')

    randomise_call_file = Path(f'{outdir}/randomise_call.sh')
    with open(randomise_call_file, 'w') as f:
        f.write(randomise_call)
    # This should change the file permissions to make the script executeable
    randomise_call_file.chmod(randomise_call_file.stat().st_mode | stat.S_IXGRP | stat.S_IEXEC)


def make_randomise_grp(comb_nii_path, outdir, permutations=1000):
    '''
    This hasn't been tested at all, since I launched this differently on Sherlock
    filename_input_root:  this is the filename used to make the mask and data file.  They should
    have the same names, but the mask file ends in _mask.  See the code below to clarify and edit accordingly.
    outdir: same output directoroy where the files from make_randomise_files were saved to

    '''
    from pathlib import Path
    import stat

    if not os.path.exists(f'{outdir}'):
        os.makedirs(f'{outdir}')

    inp_dir, file_name = os.path.split(comb_nii_path)
    file_noext, _ = os.path.splitext(file_name)

    randomise_call = (f'randomise -i {inp_dir}/{file_name}'
                      f' -o {outdir}/{file_noext}_randomise'
                      f' -m {inp_dir}/{file_noext}_mask.nii.gz'
                      f' -1 -t {outdir}/desmat.con'
                      f' -f {outdir}/desmat.fts  -T -n {permutations}')

    randomise_call_file = Path(f'{outdir}/randomise_call.sh')
    with open(randomise_call_file, 'w') as f:
        f.write(randomise_call)
    # This should change the file permissions to make the script executeable
    randomise_call_file.chmod(randomise_call_file.stat().st_mode | stat.S_IXGRP | stat.S_IEXEC)
