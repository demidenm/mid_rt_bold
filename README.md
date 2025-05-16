# Code for Computing BOLD Estimates on Real-Data for Manuscript: "Unintended bias in the pursuit of collinearity solutions in fMRI analysis" 

This code contains the analyses used for the Monetary Incentive Delay (MID) task for fMRI data.

As described in the [simulations code provided by Jeanette Mumford & Russ Poldrack](https://github.com/poldrack/MID_simulations): "Use [uv](https://docs.astral.sh/uv/getting-started/installation/) to generate and/or sync the dependencies to the local virtual environment.  The following will create the virtual environment, `.venv`, in the root directory. "

```
git clone https://github.com/demidenm/mid_rt_bold.git
cd mid_rt_bold
uv sync
```

The code utilizes the python scripts in the `./scripts/` folder with the templates for first and group-level models in `./scripts/cluster_jobs/templates` to create *batch jobs* to run on high-performance computers via the `.batch` submission scripts in `./scripts/cluster_jobs`. The subjects that are ran with the prefix 'sub-' are stored in the `./scripts/cluster_jobs/subject_ids`.

## Steps to regenerate analyses

There are separate make scripts in `./scripts/cluster_jobs`. In each, modify the output path (currently uses *scatch.global* and `./templates/*.txt` use */tmp/*)
1. make_first.sh : generate batch jobs for first and fixed effects models. 
2. make_group.sh : generate batch jobs for group (simple one-sample and site-permuted) averaged statistical maps. Note, before running, for site adjustment need to export the site information: See section 6.1 in `./scripts/notebook_mid-rt-results.ipynb`
2. make group_diff.sh : Same as above, but generates model paired difference batch jobs


### Run- & Subject-level models

to generate the models for each of the subjects, run the bash command:
```bash
./make_first.sh ./subject_ids/ids_sub_2yr.tsv
```

The script will generate files named `first*` [0 to N subjects] in the `./scripts/cluster_jobs/batch_jobs` folder. If running on a virtual / non-virtual desktop, you can pilot by running from the directory `./scripts/cluster_jobs/`

```bash
bash ./batch_jobs/first0

```

To submit jobs to HPC, update system/user specification information in the `first_msi.batch` file, such as:
```bash
#SBATCH --mail-user=email@umn.edu
#SBATCH -p msismall,agsmall
#SBATCH -A faird #feczk001 
```
and then submit the job using:
```bash
sbatch first_msi.batch
```
Note: your `#SBATCH --array=0-499` should reflect the N number of *first* level models that are in batch and will be run. So 0-499 for 500 and, in event of failures, *329,405,499* subject subsets.

### Group level outputs

The group average and group difference of models are ran using the same steps just differ in `make_group.sh` versus `make_group_diff.sh`. Here, only the former will be reviewed.

to generate the models for each of the group models, run the bash command (subjects is provided here becase the fixed effect values for each subject are pulled into a *tmp* folder for analyses):
```bash
./make_group.sh ./subject_ids/ids_sub_2yr.tsv
```

The script will generate files named `group*` [0 to N models] in the `./scripts/cluster_jobs/batch_jobs` folder. If running on a virtual / non-virtual desktop, you can pilot by running from the directory `./scripts/cluster_jobs/`

```bash
bash ./batch_jobs/group0

```

To submit jobs to HPC, update system/user specification information in the `group_msi.batch` file, such as:
```bash
#SBATCH --mail-user=email@umn.edu
#SBATCH -p msismall,agsmall
#SBATCH -A faird #feczk001 
```

and then submit the job using:
```bash
sbatch group_msi.batch
```

### Figures in Jupyter Notebook

The figures are created in a Jupyter notebook. If you are working on a OnDemand notebook system, update the `Custom Python Environment` with the `uv` source. Fir example,

```bash
export UV_PATH="/home/feczk001/mdemiden/analyses/mid_rt_bold"
cd $UV_PATH
source .venv/bin/activate
```

If not using OnDemand, you can run jupyter by selection `.venv` from VSCode or opening notebook via using [uv described here](https://github.com/poldrack/MID_simulations):

```bash
uv run --with jupyter jupyter lab
```