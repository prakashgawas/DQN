#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=500M
#SBATCH --time=01:20:00
#SBATCH --partition=optimum
#SBATCH --array=1-500
#SBATCH --output=output/arrayjob_%A_%a.out
#SBATCH --begin=now+0hour

i=1
loc="server"
N=$1
STE=$2
lm=$3
tw=$4

# for n in 2 3 4 20
# do   
# 	for d in 120 180
# 	do
# 		for q in 40 60 80
# 	do
# 	#for k in $(seq -1 1 -1)
# 	#do
# 	        if [ $SLURM_ARRAY_TASK_ID -eq $i ]
# 		then
# 	    	python OFL_Sim.py --id  $n --D $d  --q $q --dist Weibull
# 	    	#python OFL_Sim.py --id $n --D $d  --dist data-interacted
# 		fi
# 	    	(( i = $i +1 ))
# 	#done
# done
# done
# done

#101 102  regression
#103 104 105 112 # max mean prob binary
#107 0.5 
#106 108 109 110 111  # 0.4 0.6 0.7 0.75 0.3
#201 202 203 208 # 0.5 0.4 0.6 mean
# 204 205 206 207 209

for n in $N
do   
	for d in  180
	do
		#for q in 40 60 80
	#do
	for mc in 5 2 1
	do
		for eoc in 0 1
		 do
		 	for step in $STE
		 		do
		 	for k in  {-1..80}
		 	do
	        if [ $SLURM_ARRAY_TASK_ID -eq $i ]
		then
	    	#python OFL_Sim.py --id $n --D $d  --q $q --dist Weibull
	    	#python OFL_Sim.py --id $n --D $d  --dist data-interacted 
	    	python OFL_Sim.py --id $n --D $d  --eoc $eoc --max_calls $mc --dist 'data-interacted' --model $k --tf 'server' --lm $lm --tw $tw --kcap $step
		fi
	    	(( i = $i +1 ))
	done
done
done
done
#done
done
done

sleep 60
