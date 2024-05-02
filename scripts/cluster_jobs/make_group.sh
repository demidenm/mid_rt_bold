#!/bin/bash

curr_dir=`pwd`
sample=abcd # abcd, ahrb or mls
task=MID
ses=baselineYear1Arm1
type=session # run or session
run=1 # 1 or 2, not used here
subj_list=${1}
inpfold=/scratch.global/${USER}/mid_rt_model/firstlvl
outfold=/scratch.global/${USER}/mid_rt_model/group
counter_start=0
rt = ['mod-Cue-rt', 'mod-Cue-rt']

if [ -z "$1" ]; then
        echo
	echo "Error: Missing list. Provide subject list w 'sub-' prefix in positon 1."
        echo
	exit 1


n=${counter_start}
for model in ${rt[@]} ; do
	sed -e "s|MODEL|${model}|g; \
		s|RUN|${run}|g; \
		s|SESSION|${ses}|g; \
		s|TASK|${task}|g; \
		s|TYPE|${type}|g;  \
		s|INPUT|${inpfold}|g; \
		s|OUTPUT|${outfold}|g; \
		s|SUBJ_IDS|${subj_list}|g; \
		s|SAMPLE|${sample}|g;" ./templates/group.txt > ./batch_jobs/group${n}
        n=$((n+1))
done

chmod +x ./batch_jobs/group*
