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
folder = 'InstancesACR_Max_calls_2/'
#folder = 'Instances_Max_calls_5_Max_wait_20/'
#folder = 'Instances_Max_calls_1/'


all_folds = glob(folder +"*/", recursive = True)
all_folds.sort(key=len, reverse=    False) 
all_folds.sort()
print(len(all_folds))
all_folds = all_folds[::-1]
summary = pd.DataFrame(columns = ["N", "H", "S", "D", "Q", "dist", "Avg_bumps", "Avg_time"])
#schedule_stats = pd.DataFrame(columns = ["id", "N", "H", "S",  "D", "dist", "calls_made", "interacted", "in_delay", "in_cutoff", "bumps", 'cum_cutoff' , 'sum_cutoff'])
user_stats = pd.DataFrame(columns = ["id", "N", "H", "S",  "D", "Q","dist", "bumps_caused", "bumps_suffered", "pushes_out", "assigned"])
schedule_stats = pd.DataFrame()

group_cols = ['time', "N", "H", "S", "D", "Q","dist"]
group_cols_user = ['seniority', "N", "H", "S", "D", "Q","dist"]
metric_cols = ["calls_made", "interacted", "in_delay", "in_delay_at_end", "in_cutoff_at_end",
               "in_cutoff", "bumps" ,"cum_calls_made",  "cum_interacted_at_end",
               "sum_cutoff", "cum_bumps", 'sum_cutoff_at_end', 'time_since_last_call', 'last_response']
param_cols = ["cum_calls_made","in_cutoff_at_end",  'sum_cutoff_at_end', 'sum_cutoff', 'in_cutoff']
#N = 100
#H = 300
N = args.N
#ext = '_'  + str(N) + '_' + str(H)


for fold in all_folds:
    ext = fold.split('/')[1]
    print("At - ", fold)
    schedule_stats = pd.DataFrame()
    user_stats = pd.DataFrame()
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
    if len(run_file) > 0:
        path = run_file[0][len(folder):]
        if 'data' in path:
            path = path.replace("data_response", "data-response")
            path = path.replace("data_interacted", "data-interacted")
        inst = path.split('/')[0]
        vals = inst.split("_")
        result = pd.read_csv(run_file[0], header=None, names = ["srno", "N", "H", "S", "D", "Q", "bumps", "status", "time"])
        temp = [np.mean(result.bumps), np.mean(result.time)]
        summary.loc[len(summary.index)] = vals + temp
    print("Total Files - ", len(solution_files))
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
        
        stats.id = vals[7]
        stats.D = vals[4]
        stats.N = vals[1]
        stats.H = vals[2]
        stats.S = vals[3]
        stats.dist = vals[6]
        stats.Q = vals[5]
        
        
        if int(vals[4]) == 180:
            buckets = {1:[10, 0], 2:[30, 10], 3:[60, 30], 4:[150, 60], 5: [170, 150], 6:[180, 170]} 
        else:
            buckets = {1:[10, 0], 2:[30, 10], 3:[60, 30], 4:[90, 60], 5: [110, 90], 6:[120, 110]}

        
        schedule =  schedule[schedule['Call_At'] <= int(vals[2])]
        
        if counter % 20 == 0:
            print(counter, vals[7])
        stats['sum_cutoff_at_start'] = 0
        stats['in_delay_at_start'] = 0
        stats['in_cutoff_at_start'] = 0
        
        schedule['Actual_cutoff'] = schedule['Call_At'] + int(vals[4])
        schedule['Actual_cutoff'] = schedule['Actual_cutoff'].clip(0, int(vals[2]))
        temp = schedule['Call_At'].value_counts()
        stats.loc[ list(temp.index), 'calls_made' ] = temp.values
        temp = schedule['Response_At'].value_counts()
        stats.loc[ temp[temp.index <= int(vals[2])].index, 'interacted' ] = temp[temp.index <= int(vals[2])].values
        
        stats["cum_calls_made_at_start"] = stats.calls_made.cumsum() - stats.calls_made
        stats['cum_calls_made_at_start'] =  stats['cum_calls_made_at_start'].clip(0, int(vals[1])) 
        stats["cum_interacted_at_end"] = stats.interacted.cumsum()
        stats["cum_interacted_at_start"] =  stats["cum_interacted_at_end"] -  stats["interacted"]    
        
        for i in buckets:
            stats[ 'delay_bucket_' +str(i)] = 0

        stats[ 'continuous_calls'] = 0
        stats['time_to_wait'] = 0
        stats[ 'break'] = 0
        stats[ 'last_calls'] = 0
        stats[ 'no_calls_for'] = 0
        for i in schedule.index:
            ran = np.arange(schedule.Call_At.loc[i] , min(int(vals[2]),schedule.Response_At.loc[i]) )
            stats.loc[ran, 'in_delay_at_end'] = stats.in_delay_at_end.loc[ran] + 1
            ran = np.arange(schedule.Call_At.loc[i] , min(int(vals[2]),schedule.Cutoff_Time.loc[i]))
            stats.loc[ran, 'in_cutoff_at_end'] = stats.in_cutoff_at_end.loc[ran] + 1
            if schedule.Response_At.loc[i] <= int(vals[2]):
                stats.bumps.at[schedule.Response_At.loc[i]] = stats.bumps.at[schedule.Response_At.loc[i]] + schedule.bumps_caused.loc[i]
                            
            
        for t in stats.index:
            schedule.loc[:,'delay'] = schedule.Actual_cutoff - t
            
            if stats.loc[t, 'calls_made'] > 0 : 
                stats.loc[t, 'time_to_wait'] = 0
            elif stats.loc[t - 1, 'time_to_wait'] > 0:
                stats.loc[t, 'time_to_wait'] = stats.loc[t - 1, 'time_to_wait'] - 1
            else:
                i = 1
                while t + i <= int(vals[2]):
                    if stats.loc[t + i, 'calls_made'] > 0:
                        break
                    else:
                        i += 1
                stats.loc[t, 'time_to_wait'] = i
            
            
            if t != 0:
                stats.loc[t, 'continuous_calls'] = stats.loc[t - 1, 'continuous_calls'] + 1 if stats.loc[ t - 1 , 'calls_made' ] > 0 else 0
                stats.loc[t, 'break'] = stats.loc[t, 'calls_made'] == stats.loc[t - 1, 'calls_made']
                stats.loc[t, 'last_calls'] = stats.loc[t - 1, 'calls_made']
                stats.loc[t, 'no_calls_for'] = stats.loc[t - 1, 'no_calls_for'] + 1 if stats.loc[ t - 1 , 'calls_made' ] == 0 else 0
                
            temp = schedule[(schedule.Cutoff_Time >= t) & (schedule.Call_At < t)] 
            
            stats.loc[t, "sum_cutoff"] = sum(temp.delay) + stats.loc[ t, 'calls_made' ] * min(int(vals[4]), int(vals[2]) - t)#
            stats.loc[t, "sum_cutoff_at_start"] = sum(temp.delay) #
            #now = temp[temp.Call_At == t]
            #for tau in now.index:
            temp = schedule[(schedule.Cutoff_Time > t) & (schedule.Call_At <= t)] 
            stats.loc[t, "sum_cutoff_at_end"] = sum(temp.delay) - len(temp) 
            if t != int(vals[2]):
                for i in buckets:
                    stats.loc[t, 'delay_bucket_' + str(i)] = sum(((schedule.Cutoff_Time >= t) & (schedule.Call_At >= t - (buckets[i][0])) & (schedule.Call_At < t - buckets[i][1] )))
                    #stats.loc[t, 'delay_bucket_60'] = sum(((schedule.Cutoff_Time >= t) & (schedule.Call_At >= t - 59) & (schedule.Call_At < t - 9 )))
                    #stats.loc[t, 'delay_bucket_180'] = sum(((schedule.Cutoff_Time >= t) & (schedule.Call_At >= t - 179) & (schedule.Call_At < t - 59 )))
            if t != 0:
                stats.loc[ t , 'last_response' ] = stats.loc[ t , 'cum_interacted_at_start' ]  - stats.loc[ t - 1 , 'cum_interacted_at_start' ] 
                if stats.loc[ t - 1 , 'calls_made' ] > 0:
                    stats.loc[ t , 'time_since_last_call' ] = 1
                else:
                    stats.loc[ t , 'time_since_last_call' ] = stats.loc[ t - 1 , 'time_since_last_call' ] + 1
                
                stats.loc[t, "in_delay_at_start"] = stats.loc[t - 1 ,"in_delay_at_end"]
                stats.loc[t, "in_cutoff_at_start"] = stats.loc[t - 1 ,"in_cutoff_at_end"]
                

                    
        stats["sum_cutoff_at_end"] = stats[ "sum_cutoff_at_end"].clip(0, int(vals[1])*int(vals[4]))
       
        stats['shifts_available'] =  int(vals[3]) - stats['cum_interacted_at_start']
        stats['shifts_available'] =  stats['shifts_available'].clip(0, int(vals[3])) 
        
        #stats['delay_bucket_1_by_shifts_filled'] = stats['delay_bucket_10'] / (int(vals[3]) - stats['shifts_available'] + 1)
        #stats['delay_bucket_2_by_shifts_filled'] = stats['delay_bucket_60'] / (int(vals[3]) - stats['shifts_available'] + 1)
        #stats['delay_bucket_3_by_shifts_filled'] = stats['delay_bucket_180'] / (int(vals[3]) - stats['shifts_available'] + 1)
        #stats['delay_bucket_1_by_shifts_filled'] = stats['delay_bucket_1_by_shifts_filled'].fillna(0)
        #stats['delay_bucket_2_by_shifts_filled'] = stats['delay_bucket_2_by_shifts_filled'].fillna(0)
        #stats['delay_bucket_3_by_shifts_filled'] = stats['delay_bucket_3_by_shifts_filled'].fillna(0)
        # user = pd.DataFrame(index = np.arange(0,int(vals[1])+1), columns = [ "id","N", "H", "S", "D", "dist", "Q", "bumps_caused", "bumps_suffered", "push_out", "assigned"]).fillna(0)
        # user.id = vals[7]
        # user.D = vals[4]
        # user.N = vals[1]
        # user.H = vals[2]
        # user.S = vals[3]
        # user.Q = vals[5]
        # user.dist = vals[6]
        # user.bumps_caused = schedule.bumps_caused
        # user.bumps_suffered = schedule.bumps_suffered
        # user.push_out = 1 - schedule.Assigned
        # user.assigned = schedule.Assigned
        
        stats["cum_calls_made"] = stats.calls_made.cumsum()
        stats["cum_interacted_at_end"] = stats.interacted.cumsum()
        
        stats["in_cutoff"] =  stats["in_cutoff_at_start"] + stats["calls_made"] #
        stats["in_delay"] =  stats["in_cutoff_at_start"] +  stats["calls_made"] #

        #stats["in_cutoff"] = stats["in_cutoff"].clip(0, None)
        #stats["in_delay"] = stats["in_delay"].clip(0, None)
        stats["cum_bumps_at_start"] = stats.bumps.cumsum() - stats.bumps
        stats["cum_bumps"] = stats.bumps.cumsum() 
        stats["t_rem"] = stats.H.astype(int) - stats.index

        stats.index.name = "time"
        stats[["calls_made", "interacted", "in_delay", "bumps" ,  "in_cutoff"]] = stats[["calls_made", "interacted", "in_delay", "bumps" ,  "in_cutoff"]].apply(pd.to_numeric)
    

        schedule_stats = pd.concat([schedule_stats, stats])
        
        # user_stats = pd.concat([user_stats, user])
        
        # user_stats.index.name = "seniority"
        
        
    percentiles =  [np.round(0.3 + 0.05 * i,2) for i in range(13)] + [0.95, 0.98] 
    #quantile_funcs = [(str(p), lambda x: np.quantile(x,p)) for p in percentiles]
    quant = pd.DataFrame()
    for p in percentiles:
        temp = schedule_stats.groupby(group_cols)[param_cols].quantile(p) 
        for col in param_cols:
            
            quant[col + "_" + str(p)] = temp[col]
       
    
    #quant.columns = quant.columns.droplevel(1)
    base = schedule_stats.groupby(group_cols)[metric_cols].agg( ["mean", "var"])
    base.columns = base.columns.map('_'.join)
    quant = quant.reset_index()
    base = base.reset_index()
    schedule_stats_avg = pd.merge(base, quant, on=group_cols)
    #summary.to_csv(folder + "Average/summary_" + ext + ".csv")
    # user_stats_avg = user_stats.groupby(group_cols_user)[["bumps_caused", "bumps_suffered", "push_out", "assigned"]].mean()
    schedule_stats = schedule_stats.reset_index()
    schedule_stats.to_csv(folder + "Average/schedule_stats_" + ext + ".csv")
    # user_stats_avg.to_csv(folder + "Average/user_stats_avg_" + ext + ".csv")
    #schedule_stats_avg[["calls_made", "interacted", "in_delay", "bumps" ,  "in_cutoff"]] = schedule_stats_avg[["calls_made", "interacted", "in_delay", "bumps" ,  "in_cutoff"]].apply(pd.to_numeric)

    schedule_stats_avg['exclude'] = 0
    
    schedule_stats_avg = schedule_stats_avg.reset_index()
    schedule_stats_avg.H = schedule_stats_avg.H.astype(int)
    schedule_stats_avg.loc[schedule_stats_avg.H == schedule_stats_avg.time,'exclude'] = 1
    print("Save at - ", folder + "Average/schedule_stats_avg_" + ext + ".csv")
    schedule_stats_avg.to_csv(folder + "Average/schedule_stats_avg_" + ext + ".csv")
    #break
    #break