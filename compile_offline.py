#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 17:54:45 2023

@author: prakashgawas
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 13:55:49 2022

@author: Prakash
"""

import pandas as pd
import numpy as np
from glob import glob
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=150)
args = parser.parse_args()

#folder = "/home/gawpra/Merinio2020/Learn/Instances/"
folder = 'InstancesACR_Max_calls_5/'
#folder = 'DQN/Instances_Max_calls_5_Max_wait_20/'
#folder = "Instances_Max_calls_2/"
#folder = "/home/gawpra/Merinio2020/DECTP/DECTP_Results/"
#folder2 = "../DECTP/DECTP_Results/"

all_folds = glob(folder +"*/", recursive = True)
all_folds.sort(key=len, reverse=    False) 
all_folds.sort()
print(len(all_folds))
all_folds = all_folds[::-1]


group_cols = ['time', "N", "H", "S", "D", "Q","dist"]
group_cols_user = ['seniority', "N", "H", "S", "D", "Q","dist"]
metric_cols = ["calls_made", "interacted", "in_delay", "in_delay_at_end", "in_cutoff_at_end",
               "in_cutoff", "bumps" ,"cum_calls_made",  "cum_interacted_at_end",
               "sum_cutoff", "cum_bumps", 'sum_cutoff_at_end', 'time_since_last_call']
param_cols = ["cum_calls_made","in_cutoff_at_end",  'sum_cutoff_at_end', 'sum_cutoff', 'in_cutoff']
#N = 100
#H = 300
N = args.N
#ext = '_'  + str(N) + '_' + str(H)
buckets = {1:[10, 0], 2:[30, 10], 3:[60, 30], 4:[150, 60], 5: [170, 150], 6:[180, 170]}

for fold in all_folds:
    ext = fold.split('/')[1]
    print("At - ", fold)

    #if os.path.isfile( folder + "schedule_stats_avg" + ext + ".csv"):
    #    print(ext, " present")
    #    continue

    #if fold[0:12] != folder + str(N)[0:2]:
    #    continue
    if 'data' not in fold and 'Weibull' not in fold:
        continue
    # if 'Weibull'  in fold:
    #     N = 100 
    # if 'data'  in fold :
    #     N = 150
    if str(N) not in fold:
        continue
    #if '120_120' not in fold:
    #    continue
    #data = pd.DataFrame(columns = ["id", "N", "H", "S",  "D", "dist", "t", "in_cutoff", "sum_cutoff", "calls_made"])

    files = glob(fold + "*.csv")
    run_file = [i for i in files if "Runs" in i]
    solution_files = sorted([i for i in files if "Solution" in i])
    counter = 0
    

    if len(solution_files) == 0:
        continue

        
    print("Total Files - ", len(solution_files))
    summary = pd.DataFrame(columns = ["N", "H", "S", "D", "Q", "dist", "id", "Avg_bumps", "Avg_Vacancy"])
    for sol in solution_files:
        counter += 1
        path = sol[len(folder):]
        if 'data' in path:
            path = path.replace("data_response", "data-response")
            path = path.replace("data_interacted", "data-interacted")
        inst = path.split('/')[1]
        inst = inst.replace(".", "_" )
        vals = inst.split("_") 
        stats = pd.DataFrame(index = np.arange(0,int(vals[2])+1), 
                             columns = ["id", "N", "H", "S", "D", "Q","dist"] + metric_cols).fillna(0)
        schedule = pd.read_csv(sol)
        summary.loc[len(summary)] = vals[1:-1] + [np.sum(schedule.bumps_caused)] + [np.sum(schedule.Assigned)]
        
        
        
    summary.to_csv(folder + "Average/summary" + ext + '.csv')
       
    
