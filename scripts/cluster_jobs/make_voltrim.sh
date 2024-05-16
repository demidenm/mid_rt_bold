#!/bin/bash

subj_ids=$1
ses=2YearFollowUpYArm1 #2YearFollowUpYArm1 #baselineYear1Arm1
count_start=0

if [ -z "$1" ]; then
	echo
	echo "Error: Missing list. Provide subject list w 'sub-' prefix in positon 1."
	echo
	exit 1
fi

n=${count_start}
cat $subj_ids | while read line ; do
	subj=$(echo $line | awk -F" " '{ print $1 }' | awk -F"-" '{ print $2 }')
	sed -e "s|SUBJECT|${subj}|g; " ./templates/abcd_trimmedpreproc.sh > ./batch_jobs/trimvols${n}
		n=$((n+1))

done

chmod +x ./batch_jobs/trimvols*
