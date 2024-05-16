import os
import argparse
import pandas as pd
import numpy as np
from glob import glob
from nilearn.maskers import nifti_spheres_masker
from nilearn.signal import clean
from nilearn.masking import apply_mask, _unmask_3d, compute_brain_mask
from nilearn.image import load_img, new_img_like
import statsmodels.api as sm
from nilearn.glm.first_level import make_first_level_design_matrix


def round_cust(x):
    return np.floor(x + 0.49)


def trlocked_events(events_path: str, onsets_column: str, trial_name: str,
                    bold_tr: float, bold_vols: int, separator: str = '\t'):
    """
    Loads behavior data, creates and merges into a TR (rounded) dataframe to match length of BOLD. Trial onsets are
    matched to nearby TR using rounding when acquisition is not locked to TR.

    Parameters:
        events_path (str): Path to the events data files for given subject/run.
        onsets_column (str): Name of the column containing onset times for the event/condition.
        trial_name (str): Name of the column containing condition/trial labels.
        bold_tr (int): TR acquisition of BOLD in seconds.
        bold_vols (int): Number of time points for BOLD acquisition
        separator (str): Separator used in the events data file, default = '\t'.
    Returns:
        pandas.DataFrame: Merged dataframe with time index and events data for each event + TR delays.
    """
    if not os.path.exists(events_path):
        raise FileNotFoundError(f"File '{events_path}' not found.")

    beh_df = pd.read_csv(events_path, sep=separator)

    missing_cols = [col for col in [onsets_column, trial_name] if col not in beh_df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns: {', '.join(missing_cols)}")

    beh_df = beh_df[[onsets_column, trial_name]]

    try:
        beh_df["TimePoint"] = round_cust(
            beh_df[onsets_column] / bold_tr).astype(int)  # Per Elizabeth, avoids bakers roundings in .round()
    except Exception as e:
        print("An error occurred:", e, "Following file included NaN, dropped.", events_path)
        beh_df.dropna(inplace=True)  # cannot perform operations on missing information
        beh_df["TimePoint"] = round_cust(beh_df[onsets_column] / bold_tr).astype(int)

    time_index = pd.RangeIndex(start=0, stop=bold_vols, step=1)
    time_index_df = pd.DataFrame(index=time_index)
    # Merge behavior data with time index
    merged_df = pd.merge(time_index_df, beh_df, how='left', left_index=True, right_on='TimePoint')

    if len(merged_df) != bold_vols:
        raise ValueError(f"Merged data length ({len(merged_df)}) doesn't match volumes ({bold_vols}).")

    return merged_df


def extract_time_series_values(behave_df: pd.DataFrame, time_series_array: np.ndarray, delay: int):
    """
    Extracts time series data from the provided timeseries BOLD area for associated behavioral data
    that is acuiqred from trlocked_events w/ specified delay

    Parameters:
        behave_df (pandas.DataFrame): DataFrame containing behavioral data with a 'TimePoint' column
            indicating the starting point for each time series extraction.
        time_series_array (ndarray): Numpy Array containing time series data.
        delay (int): Number of data points to include in each extracted time series.

    Returns:
        np.ndarray: Array containing the extracted time series data for each time point in the behavioral DataFrame.
            Each row corresponds to a time point, and each column contains the extracted time series data.
    """
    extracted_series_list = []
    for row in behave_df['TimePoint']:
        start = int(row)
        end = start + delay
        extracted_series = time_series_array[start:end]
        if len(extracted_series) < delay:  # Check if extracted series is shorter than delay
            extracted_series = np.pad(extracted_series, ((0, delay - len(extracted_series)), (0, 0)), mode='constant')
        extracted_series_list.append(extracted_series)
    return np.array(extracted_series_list, dtype=object)


def extract_time_series(bold_paths: list, roi_type: str, high_pass_sec: int = None, roi_mask: str = None,
                        roi_coords: tuple = None, radius_mm: int = None, detrend=True,
                        fwhm_smooth: float = None):
    """
    For each BOLD path, extract timeseries for either a specified mask or ROI coordinate. Mask and coordinate should be
    in same space/affine as BOLD data. Function leverages NiftiLabelsMasker (mask path) NiftiSpheresMasker (coordinates)
    to achieve this.

    Parameters:
        bold_paths (list): List of paths to subjects (list should match order of subs/runs/tasks for events file list).
        roi_type (str): Type of ROI ('mask' or 'coords').
        high_pass_sec (int): High-pass filter to use, in seconds. Used to convert to filter freq using 1/secs.
        roi_mask (str or None): Path to the ROI mask image (required if roi_type is 'mask').
        roi_coords (tuple or None): Coordinates (x,y,z) for the sphere center (required if roi_type is 'coords').
        radius_mm (int or None): Radius of the sphere in mm (required if roi_type is 'coords').
        detrend: True/False, whether to use Nilearn's detrend function.
        fwhm_smooth (float or None): FWHM for spatial smoothing of data.

    Returns:
        list: List of time series for provided subjects/runs.
    """
    roi_options = ['mask', 'coords']

    if roi_type not in roi_options:
        raise ValueError("Invalid ROI type. Choose 'mask' or 'coords'.")

    if roi_type == 'mask':
        roi_series_list = []

        # Iterate over each path in bold_paths
        for bold_path in bold_paths:
            img = [load_img(i) for i in [bold_path, roi_mask]]
            assert img[0].shape[0:3] == img[1].shape, 'images of different shape, BOLD {} and ROI {}'.format(
                img[0].shape, img[1].shape)
            # Mask data by ROI and smooth and then clean data
            masked_data = apply_mask(bold_path, roi_mask, smoothing_fwhm=fwhm_smooth)
            clean_timeseries = clean(masked_data, standardize='psc', detrend=detrend,
                                     high_pass=1 / high_pass_sec if high_pass_sec is not None else None)

            # get avg at volumes across voxels, return a (volumnes,1) array
            time_series_avg = np.mean(clean_timeseries, axis=1)[:, None]
            roi_series_list.append(time_series_avg)

        return roi_series_list

    elif roi_type == 'coords':
        coord_series_list = []
        wb_mask = compute_brain_mask(bold_paths[0])

        for bold_path in bold_paths:
            _, roi = nifti_spheres_masker._apply_mask_and_get_affinity(
                seeds=[roi_coords], niimg=None, radius=radius_mm,
                allow_overlap=False, mask_img=wb_mask)
            coord_mask = _unmask_3d(X=roi.toarray().flatten(), mask=wb_mask.get_fdata().astype(bool))
            coord_mask = new_img_like(wb_mask, coord_mask, wb_mask.affine)

            img = [load_img(i) for i in [bold_path, coord_mask]]
            assert img[0].shape[0:3] == img[1].shape, 'images of different shape, BOLD {} and ROI {}'.format(
                img[0].shape[0:3], img[1].shape)
            # Mask data by ROI and smooth and then clean data
            masked_data = apply_mask(bold_path, coord_mask, smoothing_fwhm=fwhm_smooth)
            clean_timeseries = clean(masked_data, standardize='psc', detrend=detrend,
                                     high_pass=1 / high_pass_sec if high_pass_sec is not None else None)

            # get avg at volumes across voxels, return a (volumnes,1) array
            time_series_avg = np.mean(clean_timeseries, axis=1)[:, None]
            coord_series_list.append(time_series_avg)
        return coord_series_list, coord_mask

    else:
        print(f'roi_type: {roi_type}, is not in [{roi_options}]')


def extract_postcue_trs_for_conditions(events_data: list, onset: str, trial_name: str,
                                       bold_tr: float, bold_vols: int, time_series: np.ndarray,
                                       conditions: list, tr_delay: int):
    """
    Extract TR coinciding with condition onset, plus TRs for specified delay for each file. Save this to a pandas
    dataframe (long) with associated Mean Signal value, for each subject, trial of condition and cue across the range
    of TRs (1 to TR delay)

    Parameters:
        events_data (list): List of paths to behavioral data files (list should match order for
        subs/runs/tasks as bold file list).
        onset (str): Name of the column containing onset values in the behavioral data.
        trial_name (str): Name of the column containing condition values in the behavioral data.
        bold_tr (int): TR for acquisiton of BOLD data.
        bold_vols (int): Number of volumes for BOLD.
        time_series (numpy.ndarray): numpy array of time series data.
        conditions (list): List of cue conditions to iterate over, min 1.
        tr_delay (int): Number of TRs to serve as delay (post onset)

    Returns:
        pd.DataFrame: DataFrame containing mean signal intensity values, subject labels,
            trial labels, TR values, and cue labels for all specified conditions.
    """
    dfs = []
    for cue in conditions:
        cue_dfs = []  # creating separate cue dfs to accomodate different number of trials for cue types
        sub_n = 0
        for index, beh_path in enumerate(events_data):
            subset_df = trlocked_events(events_path=beh_path, onsets_column=onset,
                                        trial_name=trial_name, bold_tr=bold_tr, bold_vols=bold_vols, separator='\t')
            trial_type = subset_df[subset_df[trial_name] == cue]
            out_trs_array = extract_time_series_values(behave_df=trial_type, time_series_array=time_series[index],
                                                       delay=tr_delay)
            sub_n = sub_n + 1  # subject is equated to every event file N, subj n = 1 to len(events_data)

            # nth trial, list of TRs
            for n_trial, trs in enumerate(out_trs_array):
                num_delay = len(trs)  # Number of TRs for the current trial
                if num_delay != tr_delay:
                    raise ValueError(f"Mismatch between tr_delay ({tr_delay}) and number of delay TRs ({num_delay})")

                reshaped_array = np.array(trs).reshape(-1, 1)
                df = pd.DataFrame(reshaped_array, columns=['Mean_Signal'])
                df['Subject'] = sub_n
                df['Trial'] = n_trial + 1
                tr_values = np.arange(1, tr_delay + 1)
                df['TR'] = tr_values
                cue_values = [cue] * num_delay
                df['Cue'] = cue_values
                cue_dfs.append(df)

        dfs.append(pd.concat(cue_dfs, ignore_index=True))

    return pd.concat(dfs, ignore_index=True)


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


def create_design_mid(events_df: pd.DataFrame, bold_tr: float, num_volumes: int, conf_regressors: pd.DataFrame = None,
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
    if rt_model == 'rt':
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


def parse_args():

    parser = argparse.ArgumentParser(description="Script to dump out & plot timeseries for ROI locked to cue")
    parser.add_argument("--inp_deriv", help="BIDS fmriprep derivatives folders with subjects")
    parser.add_argument("--inp_beh", help="BIDS folder with subjects with events files within func")
    parser.add_argument("--task", help="task type -- e.g., mid, reward, etc")
    parser.add_argument("--ses", help="session, include the session type without prefix, e.g., 1, 01, baselinearm1")
    parser.add_argument("--run", help="run e.g., 01, 02")
    parser.add_argument("--output", help="output folder where to write out save timeseries df and plot")

    return parser.parse_args()


if __name__ == "__main__":

    # Parse arguments
    args = parse_args()
    inp_deriv = args.inp_deriv
    inp_beh = args.inp_beh
    task = args.task
    ses = args.ses
    run = args.run
    out_fold = args.output

    # options
    onset_col = 'Probe.OnsetTime'
    colname = onset_col.replace('.', '').replace(' ', '')
    condition_col = 'Feedback.Response'
    condition_vals = ['LgReward_hit', 'LgPun_hit', 'Triangle_hit']
    scanner = 'siemens'
    scan_tr = .8
    volumes = 403
    trdelay = 20

    roi_label = 'motor'
    roi_coordinates = (-38, -22, 56)  # left motor from neurosynth
    fwhm = 4  # from 2021 knutson paper
    roi_radius = 8  # from 2021 knutson paper
    filter_freq = 90  # 90 sec from 2021 Knutson paper


    # select paths
    bold_list = glob(f'{inp_deriv}/**/ses-{ses}/func/*_run-{run}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz')[0:100]
    beh_list = glob(f'{inp_beh}/**/ses-{ses}/func/*_run-{run}_events.tsv')[0:100]
    # check that file length and ids are same order
    bold_ids = [os.path.basename(path).split('_')[0] for path in bold_list]
    beh_ids = [os.path.basename(path).split('_')[0] for path in beh_list]
    assert len(bold_list) == len(beh_list), f"Length of bold and beh paths do not match, e.g. {len(bold_list)} & {len(beh_list)}"
    assert (np.array(bold_ids) == np.array(beh_ids)).all(), "Mismatch in order of IDs"

    print("Bold files: ", len(bold_list), "Behavioral files: ", len(beh_list))
    n_subs = len(bold_list)

    # extract timeseries
    timeseries, mask_coord = extract_time_series(bold_paths=bold_list, roi_type='coords', roi_coords=roi_coordinates,
                                                 radius_mm=roi_radius, fwhm_smooth=fwhm,
                                                 high_pass_sec=filter_freq, detrend=True)

    raw_timeseries = extract_postcue_trs_for_conditions(events_data=beh_list, onset=onset_col, trial_name=condition_col,
                                                        bold_tr=scan_tr, bold_vols=volumes, conditions=condition_vals,
                                                        tr_delay=trdelay, time_series=timeseries)
    raw_timeseries.to_csv(
        f'{out_fold}/subs-{n_subs}_run-{run}_timeseries-{roi_label}_scanner-{scanner}_{colname}_'
        f'raw_timeseries.csv', sep=',')

    # JM adding things, but you'll have to make this work
    #  Estimate the GLM for each time series
    # Goal: For each timeseries run the first level model on it and generate the predicted values
    # you can skip confound regressors here, I think.  I believe the data have already been highpass filtered.

    glm_predicted_timeseries = []  # not sure if a list of  lists will work for you, but hopefully!

    for sub in range(n_subs):
        events_loop = pd.read_csv(beh_list[sub], sep='\t')
        dat_loop = timeseries[sub]
        des_loop = create_design_mid(events_df=events_loop, bold_tr=scan_tr, num_volumes=volumes,
                                     conf_regressors=None, rt_model='rt', hrf_model='spm', stc=False)
        regression = sm.OLS(dat_loop, des_loop).fit()
        glm_predicted_timeseries.append(regression.fittedvalues)

    glm_timeserise = np.array(glm_predicted_timeseries)[:, :, None]


    predicted_timeseries = extract_postcue_trs_for_conditions(events_data=beh_list, onset=onset_col, trial_name=condition_col,
                                                              bold_tr=scan_tr, bold_vols=volumes, conditions=condition_vals,
                                                              tr_delay=trdelay, time_series=glm_timeserise)

    predicted_timeseries.to_csv(f'{out_fold}/subs-{n_subs}_run-{run}_timeseries-{roi_label}_scanner-{scanner}_{colname}_'
                                f'predicted_timeseries.csv', sep=',')
