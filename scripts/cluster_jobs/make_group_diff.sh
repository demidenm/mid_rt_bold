#!/bin/bash

curr_dir=`pwd`
sample=abcd # abcd, ahrb or mls
task=MID
ses=2YearFollowUpYArm1
type=session # run or session
run=1 # 1 or 2, not used here
subj_list=${1}
a_mod="mod-Cue-None" # mod-Cue-rt mod-Cue-None mod-Cue-probexcond
b_mod="mod-Cue-probexcond" # mod-Cue-rt mod-Cue-None mod-Cue-probexcond
inpfold=/scratch.global/${USER}/mid_rt_mod/firstlvl
outfold=/scratch.global/${USER}/mid_rt_mod/group
counter_start=9
contrasts=('LRew-Neut' 'ARew-Neut' 'LPun-Neut' 'APun-Neut' 'ARewHit-ARewMiss' 'LRewHit-LRewMiss' 'APunHit-APunMiss' 'LPunHit-LPunMiss' 'LRewHit-NeutHit')

if [ -z "$1" ]; then
        echo
	echo "Error: Missing list. Provide subject list w 'sub-' prefix in positon 1."
        echo
	exit 1
fi

n=${counter_start}
for con in ${contrasts[@]} ; do
      sed -e "s|RUN|${run}|g; \
        s|SESSION|${ses}|g; \
        s|TASK|${task}|g; \
        s|AMOD|${a_mod}|g; \
        s|BMOD|${b_mod}|g; \
        s|CONTRAST|${con}|g; \
        s|TYPE|${type}|g;  \
        s|INPUT|${inpfold}|g; \
        s|OUTPUT|${outfold}|g; \
        s|SUBJ_IDS|${subj_list}|g; \
        s|SAMPLE|${sample}|g;" ./templates/abcd_group_diff.txt > ./batch_jobs/group${n}
        n=$((n+1))
done

chmod +x ./batch_jobs/group*
