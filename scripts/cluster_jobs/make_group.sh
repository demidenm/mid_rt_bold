#!/bin/bash

curr_dir=`pwd`
sample=abcd 
task=MID
# 2YearFollowUpYArm1
ses=2YearFollowUpYArm1
type=session # run or session
run=1
subj_list=${1}
inpfold=/scratch.global/${USER}/mid_rt_mod/firstlvl
outfold=/scratch.global/${USER}/mid_rt_mod/group
counter_start=20
run_randomise=custom  # randomise or custom --> custom site cluster correction (randomise doesnt handled imbalanced site Ns)
mods=('mod-Saturated' 'mod-CueYesDeriv')
sat_cons=('Cue:LW-Neut Cue:W-Neut Cue:LL-Neut Cue:L-Neut Cue:LW-Base Fix:LW-Neut Fix:W-Neut Fix:LL-Neut Fix:L-Neut Fix:LW-Base FB:WHit-WMiss FB:LWHit-LWMiss FB:LWHit-NeutHit FB:LWHit-Base FB:LHit-LMiss FB:LLHit-LLMiss Probe RT')
cue_cons=('Cue:LW-Neut Cue:W-Neut Cue:LL-Neut Cue:L-Neut Cue:LW-Base FB:WHit-WMiss FB:LWHit-LWMiss FB:LWHit-NeutHit FB:LWHit-Base FB:LHit-LMiss FB:LLHit-LLMiss')
#rtmod_contrasts=('LRew-Neut' 'ARew-Neut' 'LPun-Neut' 'APun-Neut' 'ARewHit-ARewMiss' 'LRewHit-LRewMiss' 'APunHit-APunMiss' 'LPunHit-LPunMiss' 'LRewHit-NeutHit' 'probe-base' 'rt-base' 'LRew-Base' 'LRewHit-Base')
#probexcond=('LRew-Neut' 'ARew-Neut' 'LPun-Neut' 'APun-Neut' 'ARewHit-ARewMiss' 'LRewHit-LRewMiss' 'APunHit-APunMiss' 'LPunHit-LPunMiss' 'LRewHit-NeutHit')
#probexcond=('probeLRew-probeNeut' 'probeARew-probeNeut' 'probeLPun-probeNeut' 'probeAPun-probeNeut' 'probeARewHit-probeARewMiss' 'probeLRewHit-probeLRewMiss' 'probeAPunHit-probeAPunMiss' 'probeLPunHit-probeLPunMiss' 'probeLRewHit-probeNeutHit' 'LRew-Base' 'LRewHit-Base')

if [ -z "$1" ]; then
        echo
	echo "Error: Missing list. Provide subject list w 'sub-' prefix in positon 1."
        echo
	exit 1
fi

n=${counter_start}
for model in ${mods[@]} ; do
  if [ "$model" == 'mod-Saturated' ] ; then
    for con in ${sat_cons[@]} ; do
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
  elif [ "$model" == 'mod-CueYesDeriv' ]; then
    for con in ${cue_cons[@]} ; do
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
  elif [ "$model" == 'mod-CueNoDeriv' ]; then
    for con in ${cue_cons[@]} ; do
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
