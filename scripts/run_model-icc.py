import warnings
warnings.filterwarnings("ignore")
import sys
import os
import nibabel as nib
import argparse
from pyrelimri import brain_icc
from glob import glob


parser = argparse.ArgumentParser(description="Script for between run ICC on First Lvl estimates")

parser.add_argument("--mod", help="model type suffix for mod-, e.g. Saturated, CueNoDeriv, CueYesDeriv")
parser.add_argument("--input", help="location is lvl output folders")
parser.add_argument("--contrast", help="suffix con label")
parser.add_argument("--output", help="icc output folder")
args = parser.parse_args()

# Now you can access the arguments as attributes of the 'args' object.
model = args.mod
inp_dir = args.input
con_name = args.contrast
out_dir = args.output

# get subject's paths
run1 = sorted(glob(f'{inp_dir}/**/*_task-MID_run-01_contrast-{con_name}_mod-{model}_stat-beta.nii.gz'))
run2 = sorted(glob(f'{inp_dir}/**/*_task-MID_run-02_contrast-{con_name}_mod-{model}_stat-beta.nii.gz'))

match_string_position = all(a.split('_')[0] == b.split('_')[0] for a, b in zip(run1, run2))

assert len(run1) == len(run2), f'Lengths of set1 [{len(run1)}] and set2 [{len(run2)}] are not equal.'
assert match_string_position, "Values at path-positions 0:3 and 5: do not match. Subjects may be misaligned."

print(f"Running ICC(3,1) on {len(run1)} subjects")
brain_models = brain_icc.voxelwise_icc(multisession_list = [run1, run2],
                                       mask=None, icc_type='icc_3')

for img_type in ['est', 'btwnsub', 'wthnsub']:
    out_icc_path = f'{out_dir}/subs-{len(run1)}_contrast-{con_name}_mod-{model}_stat-{img_type}.nii.gz'
    nib.save(brain_models[img_type], out_icc_path)