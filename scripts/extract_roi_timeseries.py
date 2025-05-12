import os
import shutil
import argparse
import pandas as pd
import numpy as np
from glob import glob
from nilearn.maskers import NiftiMasker


def parse_args():

    parser = argparse.ArgumentParser(description="Script to dump out & plot timeseries for ROI locked to cue")
    parser.add_argument("--sub", help="Subject ID, e.g., sub-01")
    parser.add_argument("--ses", help="session, include the session type without prefix, e.g., 1, 01, baselinearm1")
    parser.add_argument("--inp_dir", help="BIDS fmriprep derivatives folders with subjects")
    parser.add_argument("--task", help="task type -- e.g., mid, reward, etc")
    parser.add_argument("--mask_dir", help="directory to location of ROI masks in MNI space")
    parser.add_argument("--output", help="output folder where to write out save timeseries df and plot")

    return parser.parse_args()


if __name__ == "__main__":

    # Parse arguments
    args = parse_args()
    sub = args.sub
    inp_deriv = args.inp_dir
    task = args.task
    ses = args.ses
    mask_dir = args.mask_dir
    out_fold = args.output

    scanner = "siemens"
    outpath = f'{out_fold}/{sub}/ses-{ses}/func'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    for mask_lab in ["Left_NAcc", "Right_NAcc", "Left_motor"]:
        maskname = mask_lab.replace('_','-')
        mask_path = f'{mask_dir}/{mask_lab}.nii.gz'

        for run in ['01', '02']:
            # select paths
            bold_file = f'{inp_deriv}/{sub}/ses-{ses}/func/{sub}_ses-{ses}_task-{task}_run-{run}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'

            # mask data by ROI and smooth and extract timeseries
            maskermodel = NiftiMasker(mask_img=mask_path, smoothing_fwhm=5, standardize="psc", detrend=False,low_pass=None, high_pass=None)
            time_series = maskermodel.fit_transform(bold_file)
            # in above result, each column is a voxel (across N voxels in mask) for ROI. So axis=1 means we are taking the mean across all voxels in the ROI for each timepoint (row)
            time_series_avg = np.mean(time_series, axis=1)[:, None] # get mean time series for each run
            time_series_df = pd.DataFrame(time_series_avg) # converting to df so can save as csv, otherwise get a ndarray 
            time_series_df.to_csv(
                f'{outpath}/{sub}_ses-{ses}_task-{task}_run-{run}_mask-{maskname}_scanner-{scanner}_est-psctimeseries.csv', sep=',', index=False, header=False)

