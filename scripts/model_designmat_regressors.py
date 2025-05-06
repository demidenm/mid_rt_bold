import os
import stat
import itertools
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
from nilearn.glm.contrasts import expression_to_contrast_vector
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm import compute_fixed_effects
from nilearn import masking
from scipy import ndimage
from glob import glob
import multiprocessing as mp
import os
from functools import partial
from tqdm import tqdm


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
    Refactored to use OnsetToOnsetTime Durations given e-prime issues w/ Duration column as target: 
    https://support.pstnet.com/hc/en-us/articles/115000902428-TIMING-Timing-of-E-Objects-22852

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

    # create a delinated hit v miss column so it is more clear + probe specific dichotomization
    new_feedback_label = 'Feedback.Response'
    events_df[new_feedback_label] = np.where(events_df['prbacc'] == 1.0,
                                             events_df['Condition'] + 'Hit',
                                             events_df['Condition'] + 'Miss')

    events_df['Probe.Type'] = np.where(events_df['prbacc'] == 1.0,
                                       'prbhit_' + events_df['Condition'],
                                       'prbmiss_' + events_df['Condition'])

    probe_remap = {
        'SmallLoss': 'probe_lose',
        'LargeLoss': 'probe_lose',
        'SmallWin': 'probe_win',
        'LargeWin': 'probe_win',
        'Neutral': 'probe_neut'
    }
    
    events_df['Probe.Cue'] = events_df['Condition'].map(probe_remap)


    if rt_model is None:
        # CONDITIONS
        conditions = pd.concat([events_df.loc[:, "Condition"], events_df.loc[:, new_feedback_label]],
                               ignore_index=True)
        # ONSETS
        onsets = pd.concat([events_df.loc[:, 'Cue.OnsetTime'], events_df.loc[:, "Feedback.OnsetTime"]],
                           ignore_index=True)
        # DURATIONS
        duration = pd.concat([events_df.loc[:, 'Cue.OnsetToOnsetTime'], events_df.loc[:, "Feedback.OnsetToOnsetTime"]],
                             ignore_index=True)
        # create pandas df with events
        design_events = pd.DataFrame({'trial_type': conditions,
                                      'onset': onsets,
                                      'duration': duration})
    elif rt_model is 'CueNoDeriv':
        # CONDITIONS
        conditions = pd.concat([events_df.loc[:, 'Condition'],
                                events_df.loc[:, new_feedback_label]],
                               ignore_index=True)
        # ONSETS
        onsets = pd.concat([events_df.loc[:, 'Cue.OnsetTime'],
                            events_df.loc[:, 'Feedback.OnsetTime']],
                           ignore_index=True)
        # DURATIONS
        durations = pd.Series([0] * len(onsets))
        design_events = pd.DataFrame({'trial_type': conditions,
                                      'onset': onsets,
                                      'duration': durations})

    elif rt_model is 'CueYesDeriv':
        # CONDITIONS
        conditions = pd.concat([events_df.loc[:, 'Condition'],
                                events_df.loc[:, new_feedback_label]],
                               ignore_index=True)
        # ONSETS
        onsets = pd.concat([events_df.loc[:, 'Cue.OnsetTime'],
                            events_df.loc[:, 'Feedback.OnsetTime']],
                           ignore_index=True)
        # DURATIONS
        durations = pd.Series([0] * len(onsets))
        design_events = pd.DataFrame({'trial_type': conditions,
                                      'onset': onsets,
                                      'duration': durations})

    if rt_model == 'Saturated':
        try:
            # concat  cue onset/duration + feedback onset/duration + probe regressors
            # CONDITIONS
            conditions = pd.concat([events_df.loc[:, "Condition"],
                                    'Fix' + events_df.loc[:, 'Condition'],
                                    events_df.loc[:, new_feedback_label],
                                    #pd.Series(["probe"] * len(events_df[['rt_correct', 'Probe.OnsetTime']])),
                                    events_df.loc[:, "Probe.Cue"], # Probe by Cue Regressors
                                    pd.Series(["probe_rt"] * len(events_df[['rt_correct', 'Probe.OnsetTime']].dropna()))
                                    ], ignore_index=True)
            
            # ONSETS
            onsets = pd.concat([events_df.loc[:, 'Cue.OnsetTime'],
                                events_df.loc[:, 'Anticipation.OnsetTime'],
                                events_df.loc[:, "Feedback.OnsetTime"],
                                events_df.loc[:, "Probe.OnsetTime"],
                                events_df[['rt_correct', 'Probe.OnsetTime']].dropna()['Probe.OnsetTime']
                                ], ignore_index=True)
            # DURATIONS
            durations = pd.concat([events_df.loc[:, 'Cue.OnsetToOnsetTime'],
                                  events_df.loc[:, 'Anticipation.OnsetToOnsetTime'],
                                  events_df.loc[:, "Feedback.OnsetToOnsetTime"],
                                  events_df.loc[:, "Probe.OnsetToOnsetTime"],
                                  # convert ms RT times to secs to serve as duration
                                  (events_df[['rt_correct', 'Probe.OnsetTime']].dropna()['rt_correct']) / 1000
                                  ], ignore_index=True)

            # create pandas df with events
            design_events = pd.DataFrame({
                'trial_type': conditions,
                'onset': onsets,
                'duration': durations
            })
        except Exception as e:
            print("When creating RT design matrix, an error occurred: ", e)

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
        hrf_model='spm + derivative' if rt_model == 'CueYesDeriv' else hrf_model,
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
        try:
            print(f"\t\t\t Creating weighted fix-eff model for contrast: {contrast}")
            betas = sorted(glob(f'{firstlvl_indir}/{subject}_ses-{session}_task-{task_type}_run-*_'
                                f'contrast-{contrast}_mod-{model_lab}_stat-beta.nii.gz'))
            var = sorted(glob(f'{firstlvl_indir}/{subject}_ses-{session}_task-{task_type}_run-*_'
                              f'contrast-{contrast}_mod-{model_lab}_stat-var.nii.gz'))

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
                                 f'contrast-{contrast}_mod-{model_lab}_stat-effect.nii.gz'
                fix_effect.to_filename(fix_effect_out)
            if save_var:
                fix_var_out = f'{fixedeffect_outdir}/{subject}_ses-{session}_task-{task_type}_' \
                              f'contrast-{contrast}_mod-{model_lab}_stat-var.nii.gz'
                fix_var.to_filename(fix_var_out)
            if save_tstat:
                fix_tstat_out = f'{fixedeffect_outdir}/{subject}_ses-{session}_task-{task_type}_' \
                                f'contrast-{contrast}_mod-{model_lab}_stat-tstat.nii.gz'
                fix_tstat.to_filename(fix_tstat_out)
        except Exception as e:
            print(f'Error processing Fixed Effect: {e} for {subject} and {contrast}')



# Jeanette Mumfords code to incorporate randomise into grp and rt models (not, separating due to randomise issues)
def make_4d_data_mask(bold_paths, sess, contrast_lab, model_type, tmp_dir):
    from nilearn.maskers import NiftiMasker

    if not os.path.exists(f'{tmp_dir}'):
        os.makedirs(f'{tmp_dir}')

    n_maps = len(bold_paths)
    data4d = nib.funcs.concat_images(bold_paths)
    masked_4d = NiftiMasker().fit(data4d).mask_img_
    filename_root = f'{tmp_dir}/subs-{n_maps}_ses-{sess}_task-MID_contrast-{contrast_lab}_{model_type}'
    data4d.to_filename(f'{filename_root}.nii.gz')
    masked_4d.to_filename(f'{filename_root}_mask.nii.gz')


def make_randomise_files(desmat_final, site_array, regressor_names, contrasts, outdir):
    """
    desmat_final: numpy array of the design matrix
    regressor_names: The regressor names that correspond to the columns of desmat_final
      note, the contrasts are defined using these names.  e.g. ['intercept', 'rt_centered']
    contrasts: expression_to_contrast is used, so these are entered as a list of
      string-based contrast definitions that use regressor_names: [['intercept'], ['rt_centered']]
    outdir: where you're saving all of the files
    """

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
        f.write(f'/NumWaves  1 \n/NumPoints {len(site_array)}\n \n/Matrix \n')
        np.savetxt(f, site_array, fmt='%s', delimiter='\t')
    
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
    """
    This hasn't been tested at all, since I launched this differently on Sherlock
    filename_input_root:  this is the filename used to make the mask and data file.  They should
    have the same names, but the mask file ends in _mask.  See the code below to clarify and edit accordingly.
    outdir: same output directoroy where the files from make_randomise_files were saved to

    """

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
    """
    This hasn't been tested at all, since I launched this differently on Sherlock
    filename_input_root:  this is the filename used to make the mask and data file.  They should
    have the same names, but the mask file ends in _mask.  See the code below to clarify and edit accordingly.
    outdir: same output directoroy where the files from make_randomise_files were saved to

    """
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
                    f' -f {outdir}/desmat.fts  -T -n {permutations} -P'
                    )

    randomise_call_file = Path(f'{outdir}/randomise_call.sh')
    with open(randomise_call_file, 'w') as f:
        f.write(randomise_call)
    # This should change the file permissions to make the script executeable
    randomise_call_file.chmod(randomise_call_file.stat().st_mode | stat.S_IXGRP | stat.S_IEXEC)


# USE THE BELOW TO FIX RT times and Feedback onsets if not done already 

def fix_rt(row):
    """
    Fixes RT for a given row based on whether RESP is during Probe, TextDislay1 or Feedback

    Args:
        row (pd.Series): A single row of the DataFrame containing response times 
                         and onset times for probes and feedback.

    Returns:
        float or None: The fixed reaction time if certain conditions are met.
    """    
    if row['Probe.RESP'] == 1:
        return row['Probe.RT']
    elif row['TextDisplay1.RESP'] == 1:
        result = (row['Probe.OnsetToOnsetTime']*1000) + row['TextDisplay1.RT']
        return result
    elif row['Feedback.RESP'] == 1:
        result = ((row['Probe.OnsetToOnsetTime']*1000) + 
                  row['Feedback.RT'] + 
                  (row['TextDisplay1.OnsetToOnsetTime']*1000))
        return result
    else:
        return np.nan
    
def fix_feedback_durations(events_file):
    """
    For each event, this function calculates the time between the onset of feedback (t) and the next cue (t+1).
    If the event is the last one, it assigns the value from 'FeedbackDuration' instead.
    The idea is that unless programmed, e-prime holds the feedback stimulus on the screen until the next trial starts

    Args:
        events_file (pd.DataFrame): DataFrame containing event data, including 'Cue.OnsetTime', 
                                    'Feedback.OnsetTime', and 'FeedbackDuration' columns.

    Returns:
        pd.DataFrame: Updated DataFrame with 'Feedback.OnsetToOnsetTime' column adjusted.
    """
    for i in range(len(events_file)):
        if i < len(events_file) - 1:  
            events_file.loc[i, 'Feedback.OnsetToOnsetTime'] = (events_file['Cue.OnsetTime'].loc[i+1] - events_file['Feedback.OnsetTime'].loc[i]).round(3)
        else:  
            events_file.loc[i, 'Feedback.OnsetToOnsetTime'] = events_file['FeedbackDuration'].loc[i]
    
    return events_file 


def process_data(events_data):
    """
    First fix Feedback.OnsetProcesses MID ePrime data to fix reactions times for Probe, TextDisplay and Feedback

    Args:
        file (str): The path to the CSV file containing ePrime data.

    Returns:
        pd.DataFrame: A DataFrame containing the processed data with an added 
                      'rt_correct' column that contains the fixed reaction times.

    """


    resp_columns = ['Probe.RESP', 'TextDisplay1.RESP', 'Feedback.RESP']
    for column in resp_columns:
        events_data.loc[events_data[column].notnull(), column] = 1
    
    # apply fix on each ROW
    events_data['rt_correct'] = events_data.apply(fix_rt, axis=1)
    
    return events_data


def generate_permutation_matrix(scanner_ids):
    """
    [code provide by Jeanette Mumford]
    Generate a permutation matrix where each row represents a different
    permutation of sign flips for all subjects, respecting scanner grouping.

    Parameters:
    - scanner_ids: Array of length 500 containing scanner IDs (1-14) for each subject
    - n_scanners: Number of unique scanners (default=14)

    Returns:
    - permutation_matrix: Array of shape (2^n_scanners, len(scanner_ids))
                         with all possible sign permutations
    """
    # Number of subjects
    n_subjects = len(scanner_ids)
    n_scanners = len(np.unique(scanner_ids))  # Should be 14

    # Generate all possible binary combinations for scanners
    binary_patterns = list(itertools.product([0, 1], repeat=n_scanners))
    n_permutations = len(binary_patterns)  # Should be 2^n_scanners = 16,384

    # Convert to -1/+1 patterns
    scanner_permutations = np.array(binary_patterns) * 2 - 1  # Shape: (16384, 14)

    # Initialize the full permutation matrix
    permutation_matrix = np.ones((n_permutations, n_subjects))

    # For each scanner ID (1 to 14)
    for scanner in range(1, n_scanners + 1):
        # Find which subjects came from this scanner
        scanner_mask = scanner_ids == scanner

        # For all permutations, set the sign for these subjects
        # Note: scanner_permutations[:, scanner-1] is shape (16384,)
        # We use broadcasting to apply it to all matching subjects
        permutation_matrix[:, scanner_mask] = np.outer(
            scanner_permutations[:, scanner - 1], np.ones(np.sum(scanner_mask))
        )

    return permutation_matrix


def significant_cluster_mask(cluster_map_obs, max_cluster_threshold):
    """
    [code provide by Jeanette Mumford]

    Create a binary mask indicating clusters larger than the threshold.

    Parameters:
    -----------
    cluster_map_obs : ndarray
        3D array of cluster IDs (integers)
    max_cluster_threshold : int
        Minimum size threshold for clusters to include

    Returns:
    --------
    ndarray
        Binary mask where 1 indicates voxels in clusters larger than threshold
    """
    # Count the size of each cluster
    unique_labels, counts = np.unique(cluster_map_obs, return_counts=True)

    # Find which labels (excluding 0) exceed the threshold
    large_cluster_labels = unique_labels[(unique_labels != 0) & (counts > max_cluster_threshold)]

    # Create binary mask of large clusters
    large_clusters_mask = np.isin(cluster_map_obs, large_cluster_labels)

    return large_clusters_mask


def get_cluster_2sided(t_stats, mask_3d, cluster_forming_threshold=3.1):
    # Create 3D map for t-statistics
    obs_t_map_3d = np.zeros(mask_3d.shape)
    obs_t_map_3d[mask_3d] = t_stats

    # Find observed clusters (pos and neg)
    pos_thresholded_obs_map = obs_t_map_3d > cluster_forming_threshold
    # Define array for 6-connectivity, aka NN1 or "faces"
    bin_struct = ndimage.generate_binary_structure(rank=3, connectivity=1)
    pos_labeled_obs_array, num_pos_features = ndimage.label(
        pos_thresholded_obs_map, bin_struct
    )

    neg_thresholded_obs_map = obs_t_map_3d < -cluster_forming_threshold
    neg_labeled_obs_array, num_neg_features = ndimage.label(
        neg_thresholded_obs_map, bin_struct
    )
    # shift negative labels that are nonzero to avoid overlap with positive labels
    neg_labeled_obs_array[neg_labeled_obs_array > 0] += num_pos_features

    # Combine positive and negative clusters
    cluster_map = pos_labeled_obs_array + neg_labeled_obs_array
    neg_pos_arrays = {
        'pos_labeled_obs_array': pos_labeled_obs_array,
        'neg_labeled_obs_array': neg_labeled_obs_array,
    }
    return cluster_map, neg_pos_arrays


def find_largest_cluster_size(labeled_array):
    """
    [code provide by Jeanette Mumford]
    Find the size of the largest cluster in a labeled array.
    Parameters:
    - labeled_array: 3D numpy array with labeled clusters
    Returns:
    - largest_cluster_size: Size of the largest cluster (excluding background)
    """
    # Get unique labels and their counts
    unique_labels, counts = np.unique(labeled_array, return_counts=True)
    largest_cluster_size = (
        np.max(counts[1:]) if len(counts) > 1 else 0
    )  # skips the background label (0)
    return largest_cluster_size


def process_permutation(p, data, perm_matrix, mask_3d, n_subjects, cluster_forming_threshold):
    """Process a single permutation for cluster analysis.
    
    This function must be defined at module level (not inside another function)
    to be picklable for multiprocessing.
    """
    print(f'Processing permutation: {p}')
    # Apply sign flipping
    permuted_data = data * perm_matrix[p, :, np.newaxis]

    # Calculate t-statistics
    perm_mean = np.mean(permuted_data, axis=0)
    perm_std = np.std(permuted_data, axis=0, ddof=1)
    perm_t_stats = perm_mean / (perm_std / np.sqrt(n_subjects))

    # Get maximum absolute t-statistic
    max_t = np.max(np.abs(perm_t_stats))

    # Get clusters
    two_sided_clusters_perm, _ = get_cluster_2sided(
        perm_t_stats, mask_3d, cluster_forming_threshold=cluster_forming_threshold
    )
    # Get largest cluster size
    max_cluster_size = find_largest_cluster_size(two_sided_clusters_perm)
    
    return max_t, max_cluster_size


def permutation_test_with_clustering(
    data_path,
    mask_path,
    scanner_ids,
    cluster_forming_threshold=3.1,
    n_permutations=None,
    n_cpus=6,  # New parameter to control CPU usage 
):
    """
    [code provide by Jeanette Mumford; modified for parallelization]
    Sign-flipping permutation test with cluster-based thresholding for 4D NIfTI data
    With parallel processing support using a specified fraction of available CPUs

    Parameters:
    -----------
    data_path : str
        Path to 4D NIfTI file with subject data
    mask_path : str
        Path to brain mask NIfTI file
    scanner_ids : array-like
        Scanner IDs for each subject to inform permutation structure
    cluster_forming_threshold : float
        Threshold for forming clusters (default: 3.1)
    n_permutations : int, optional
        Number of permutations to run
    n_cpus : float, optional
        Fraction of available CPUs to use 

    Returns:
    --------
    dict : Dictionary containing p-values, t-statistics, and cluster information
    """
    
    # CPUs to use 
    print(f"Using {n_cpus} CPUs for parallel processing.")

    # Load the 4D NIfTI data and apply mask
    nifti_img = nib.load(data_path)
    mask_img = nib.load(mask_path)

    # Apply the mask to extract voxel data
    data = masking.apply_mask(nifti_img, mask_img)

    # Get dimensions
    n_subjects, n_voxels = data.shape

    # Calculate observed t-statistics
    mean_data = np.mean(data, axis=0)
    std_data = np.std(data, axis=0, ddof=1)
    t_stats = mean_data / (std_data / np.sqrt(n_subjects))

    # Generate permutation matrix
    perm_matrix = generate_permutation_matrix(scanner_ids)
    print(f'There are {perm_matrix.shape[0]} possible permutations.')
    if n_permutations is not None and n_permutations < perm_matrix.shape[0]:
        idx = np.random.choice(perm_matrix.shape[0], n_permutations, replace=False)
        perm_matrix = perm_matrix[idx]

    n_perms = perm_matrix.shape[0]

    # Get the original 3D shape from the mask
    mask_3d = mask_img.get_fdata().astype(bool)
    orig_shape = mask_3d.shape
    print(f"Mask shape: {orig_shape}")

    # Prepare for parallel processing
    process_func = partial(
        process_permutation,
        data=data,
        perm_matrix=perm_matrix,
        mask_3d=mask_3d,
        n_subjects=n_subjects,
        cluster_forming_threshold=cluster_forming_threshold
    )
    
    # Run permutations in parallel with progress tracking
    print(f"Starting parallel permutation processing with {n_cpus} processes...")
    
    # Using tqdm for progress bar
    with mp.Pool(processes=n_cpus) as pool:
        # imap returns results as they become available, allowing for progress tracking
        results = list(tqdm(
            pool.imap(process_func, range(n_perms)),
            total=n_perms,
            desc="Permutations"
        ))
    
    print("Completed all permutations!")
    
    # Unpack results
    perm_max_t = np.array([r[0] for r in results])
    perm_max_cluster_size = np.array([r[1] for r in results])

    # Max cluster size threshold
    max_cluster_size_threshold = np.percentile(
        perm_max_cluster_size, 95
    )  # 95th percentile of permuted cluster sizes
    print(f"95th percentile cluster size threshold: {max_cluster_size_threshold}")

    # Create 3D map of observed t-statistics
    print("Creating 3D maps of results...")
    obs_t_map_3d = np.zeros(orig_shape)
    obs_t_map_3d[mask_3d] = t_stats

    # cluster map for observed data
    cluster_map_obs, neg_pos_arrays = get_cluster_2sided(
        t_stats, mask_3d, cluster_forming_threshold=cluster_forming_threshold
    )

    sig_clusters = significant_cluster_mask(cluster_map_obs, max_cluster_size_threshold)

    # Create a t-statistic image
    t_stat_img = nib.Nifti1Image(obs_t_map_3d, mask_img.affine)

    # Create a cluster label image
    sig_cluster_mask_img = nib.Nifti1Image(
        sig_clusters.astype(np.int8), mask_img.affine
    )
    # sig_cluster_mask_img = nib.Nifti1Image(sig_clusters, mask_img.affine)

    print("Permutation test completed successfully!")
    
    return {
        'perm-tstat': t_stat_img,
        'perm-sigclustermask': sig_cluster_mask_img,
        'maxclustersizethreshold': max_cluster_size_threshold,
        'negposarrays': neg_pos_arrays,
    }