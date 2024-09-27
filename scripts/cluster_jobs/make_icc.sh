#!/bin/bash

curr_dir=`pwd`
script_dir=${curr_dir}/..
ses=ses-2YearFollowUpYArm1
inpfold=/scratch.global/${USER}/mid_rt_mod/firstlvl/${ses}
outfold=/scratch.global/${USER}/mid_rt_mod/icc
mods=('Saturated' 'CueYesDeriv' 'CueNoDeriv')
cue_cons=('Cue:LW-Neut Cue:W-Neut Cue:LL-Neut Cue:L-Neut Cue:LW-Base FB:WHit-WMiss FB:LWHit-LWMiss FB:LWHit-NeutHit FB:LWHit-Base FB:LHit-LMiss FB:LLHit-LLMiss')
counter_start=0


mkdir -p ${outfold}

n=${counter_start}
for model in ${mods[@]} ; do
	for con in ${cue_cons[@]} ; do
		output_file="./batch_jobs/icc${n}"

		echo "#!/bin/bash
echo \"Group input: ${inpfold}\"
echo \"Group output: ${outfold}\"
python ${script_dir}/run_model-icc.py \\
	--mod ${model} \\
        --input ${inpfold} \\
        --contrast ${con} \\
        --output ${outfold}

grp_error=\$?

if [ \${grp_error} -eq 0 ]; then
	echo \"Python group level completed successfully!\"
else
	echo \"Python group level failed.\"
	exit 1
fi" > ${output_file}

		n=$((n + 1))	
	done
done

chmod +x ./batch_jobs/icc*
