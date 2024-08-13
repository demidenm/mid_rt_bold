#!/bin/bash

curr_dir=`pwd`
sample=abcd 
task=MID
ses=2YearFollowUpYArm1
type=session # run or session
run=1
subj_list=${1}
inpfold=/scratch.global/${USER}/mid_rt_mod/firstlvl
outfold=/scratch.global/${USER}/mid_rt_mod/group
counter_start=0
run_randomise=True  # True or False
rt_mods=('mod-Cue-rt' 'mod-Cue-None' 'mod-Cue-probexcond' 'mod-Cue-rtfull')
rtfull_contrast=('LRew-Neut' 'ARew-Neut' 'LRew-Base' 'LRew-Neut-fix' 'ARew-Neut-fix' 'LRew-Base-fix' 'LRewHit-LRewMiss' 'LRewHit-Base' 'probe-base' 'rt-base')
cuemod_contrasts=('LRew-Neut' 'ARew-Neut' 'LPun-Neut' 'APun-Neut' 'ARewHit-ARewMiss' 'LRewHit-LRewMiss' 'APunHit-APunMiss' 'LPunHit-LPunMiss' 'LRewHit-NeutHit' 'LRew-Base' 'LRewHit-Base')
rtmod_contrasts=('LRew-Neut' 'ARew-Neut' 'LPun-Neut' 'APun-Neut' 'ARewHit-ARewMiss' 'LRewHit-LRewMiss' 'APunHit-APunMiss' 'LPunHit-LPunMiss' 'LRewHit-NeutHit' 'probe-base' 'rt-base' 'LRew-Base' 'LRewHit-Base')
#probexcond=('LRew-Neut' 'ARew-Neut' 'LPun-Neut' 'APun-Neut' 'ARewHit-ARewMiss' 'LRewHit-LRewMiss' 'APunHit-APunMiss' 'LPunHit-LPunMiss' 'LRewHit-NeutHit')
probexcond=('probeLRew-probeNeut' 'probeARew-probeNeut' 'probeLPun-probeNeut' 'probeAPun-probeNeut' 'probeARewHit-probeARewMiss' 'probeLRewHit-probeLRewMiss' 'probeAPunHit-probeAPunMiss' 'probeLPunHit-probeLPunMiss' 'probeLRewHit-probeNeutHit' 'LRew-Base' 'LRewHit-Base')

if [ -z "$1" ]; then
        echo
	echo "Error: Missing list. Provide subject list w 'sub-' prefix in positon 1."
        echo
	exit 1
fi

n=${counter_start}
for model in ${rt_mods[@]} ; do
  if [ "$model" == 'mod-Cue-rt' ] ; then
    for con in ${rtmod_contrasts[@]} ; do
      sed -e "s|MODEL|${model}|g; \
        s|RUN|${run}|g; \
        s|RRAND|${run_randomise}|g; \
        s|SESSION|${ses}|g; \
        s|TASK|${task}|g; \
        s|CONTRAST|${con}|g; \
        s|TYPE|${type}|g;  \
        s|INPUT|${inpfold}|g; \
        s|OUTPUT|${outfold}|g; \
        s|SUBJ_IDS|${subj_list}|g; \
        s|SAMPLE|${sample}|g;" ./templates/abcd_group.txt > ./batch_jobs/group${n}
        n=$((n+1))
    done
  elif [ "$model" == 'mod-Cue-None' ]; then
    for con in ${cuemod_contrasts[@]} ; do
      sed -e "s|MODEL|${model}|g; \
        s|RUN|${run}|g; \
        s|RRAND|${run_randomise}|g; \
        s|SESSION|${ses}|g; \
        s|TASK|${task}|g; \
        s|CONTRAST|${con}|g; \
        s|TYPE|${type}|g;  \
        s|INPUT|${inpfold}|g; \
        s|OUTPUT|${outfold}|g; \
        s|SUBJ_IDS|${subj_list}|g; \
        s|SAMPLE|${sample}|g;" ./templates/abcd_group.txt > ./batch_jobs/group${n}
        n=$((n+1))
    done
  elif [ "$model" == 'mod-Cue-rtfull' ]; then
    for con in ${rtfull_contrast[@]} ; do
      sed -e "s|MODEL|${model}|g; \
        s|RUN|${run}|g; \
        s|RRAND|${run_randomise}|g; \
        s|SESSION|${ses}|g; \
        s|TASK|${task}|g; \
        s|CONTRAST|${con}|g; \
        s|TYPE|${type}|g;  \
        s|INPUT|${inpfold}|g; \
        s|OUTPUT|${outfold}|g; \
        s|SUBJ_IDS|${subj_list}|g; \
        s|SAMPLE|${sample}|g;" ./templates/abcd_group.txt > ./batch_jobs/group${n}
        n=$((n+1))
    done
  elif [ "$model" == 'mod-Cue-probexcond' ]; then
    for con in ${probexcond[@]} ; do
      sed -e "s|MODEL|${model}|g; \
        s|RUN|${run}|g; \
        s|RRAND|${run_randomise}|g; \
        s|SESSION|${ses}|g; \
        s|TASK|${task}|g; \
        s|CONTRAST|${con}|g; \
        s|TYPE|${type}|g;  \
        s|INPUT|${inpfold}|g; \
        s|OUTPUT|${outfold}|g; \
        s|SUBJ_IDS|${subj_list}|g; \
        s|SAMPLE|${sample}|g;" ./templates/abcd_group.txt > ./batch_jobs/group${n}
        n=$((n+1))
    done
  fi

done

chmod +x ./batch_jobs/group*
