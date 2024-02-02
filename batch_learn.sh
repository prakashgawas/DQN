#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=150:00:00
#SBATCH --partition=optimumlong
#SBATCH --array=1-7
#SBATCH --output=output/arrayjob_%A_%a.out

module load gurobi


i=1

# for n in 2 3 4 20
# do
# 	for d in 120 180
# 	do

# 		for q in 40 60 80
# 		do

#         	if [ $SLURM_ARRAY_TASK_ID -eq $i ]
# 		then
# 	    		python OFL_learn.py --id $n --D $d --q $q --dist Weibull > output/outputW$n$d$q.txt
# 	    		#python OFL_learn.py --id $n --D $d   > output/outputd$n$d.txt	
# 		fi
# 	    	(( i = $i +1 ))

# 	    done
# 	done
# done

#101 102 103 104 105 106 107 112
#201 202 203 204 205 206 207 208 209
loc="server"
lm=0
tw=0

for n in  114
do
	for d in  180
	do
		#for q in 40 60 80
		#do
		 for eoc in 0 1
		 do

		 	for mc in 2 1 5
		 	do
		 		for step in 3
		 		do

        	if [ $SLURM_ARRAY_TASK_ID -eq $i ]
		then
	    		#python OFL_learn.py --id $n --D $d --q $q --dist Weibull > output/outputW$n$d$q.txt
	    		python OFL_learn.py --id $n --D $d  --eoc $eoc --max_calls $mc --tf $loc --kcap $step --lm $lm --tw $tw > output/output$tw$lm$n$d$i.txt	
		fi
	    	(( i = $i +1 ))
	    done
	done
	    done
	#done
	done
done

sleep 60