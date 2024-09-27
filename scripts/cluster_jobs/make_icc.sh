#!/bin/bash

curr_dir=`pwd`
script_dir=${curr_dir}/..
ses=2YearFollowUpYArm1
inpfold=/scratch.global/${USER}/mid_rt_mod/firstlvl/{ses}
outfold=/scratch.global/${USER}/mid_rt_mod/icc
mods=('Saturated' 'CueYesDeriv' 'CueNoDeriv')
cue_cons=('Cue:LW-Neut Cue:W-Neut Cue:LL-Neut Cue:L-Neut Cue:LW-Base FB:WHit-WMiss FB:LWHit-LWMiss FB:LWHit-NeutHit FB:LWHit-Base FB:LHit-LMiss FB:LLHit-LLMiss')
counter_start=0

if [ -z "$1" ]; then
        echo
	echo "Error: Missing list. Provide subject list w 'sub-' prefix in positon 1."
        echo
	exit 1
fi

mkdir -p ${outfold}

n=${counter_start}
for model in ${mods[@]} ; do
    for con in ${cue_cons[@]} ; do
        output_file="./batch_jobs/icc${n}"

        {
            echo "Group scratch out: ${group_scratch}"
            echo "Group oak out: ${group_out}"

            python ${script_dir}/group_estimate.py \
                --mod ${model} \
                --input ${inpfold} \
                --contrast ${con} \
                --output ${outfold}

            grp_error=$?

            if [ ${grp_error} -eq 0 ]; then
                echo "Python group level completed successfully!"
            else
                echo "Python group level failed."
                exit 1
            fi
        } >> ${output_file} 2>&1  

        n=$((n + 1))
    done
done

chmod +x ./