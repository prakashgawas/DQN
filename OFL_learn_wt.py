

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 02:52:10 2023

@author: Prakash
"""

import numpy as np
import pandas as pd
import os
from SECTP import SECTP
import json
from config import config
import itertools
import glob
import copy
from Learning_modules import IO_learning
from collections import deque
import time
#'../../../../../scratch/gawpra/Data'
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--id', type=int, default=115 )
parser.add_argument('--learn_iter', type=int, default=25)
parser.add_argument('--kcap', type=int, default=3)
parser.add_argument('--tf', type=str, default='local')
parser.add_argument('--dist', type=str, default='data-interacted')
parser.add_argument('--D', type=int, default=180)
parser.add_argument('--q', type=int, default=40)
parser.add_argument('--eoc', type=int, default=0)
parser.add_argument('--max_calls', type=int, default=2)
parser.add_argument('--lm', type = float,  default=0)
parser.add_argument('--tw', type=int, default=0)
args = parser.parse_args()


#resample = False
class_weight = False

lambda_ = args.lm
temporal_weight = args.tw
dist = args.dist##data-interacted Weibull
N = 150 #if dist == 'Weibull' else 150
H = 360
M = 50
D = args.D
q = args.q
eoc = True if args.eoc else False
max_calls = args.max_calls
train = True
_ = '_'

folder_name = str(args.kcap) + _ + str(lambda_) 
folder_name +=  str('_TW') if temporal_weight else '' 

train_folder = 'Data_Dagger_' + folder_name + '/' if args.tf == 'local' else '../../../../../scratch/gawpra/Data_Dagger_' + folder_name + '/'
if not os.path.exists( train_folder ):
   os.makedirs( train_folder )

max_wait = H

delay_bucket_intervals = {1:[10, 0], 2:[30, 10], 3:[60, 30], 4:[150, 60], 5: [170, 150], 6:[180, 170]} if D == 180 else {1:[10, 0], 2:[30, 10], 3:[60, 30], 4:[90, 60], 5: [110, 90], 6:[120, 110]}

config = config[args.id]
config['learn_for'] = 100 ##offline instances
config['V_C'] = 50
config['delay_bucket_intervals'] = delay_bucket_intervals

if 'wt' in config['ext']:
    ex = '_wt'
if eoc == False:    
    loc =  "Instances_Max_calls_" + str(max_calls) +"/Average/schedule_stats_150_360_50_" + str(D) + _ + str(q) + _ + dist + ex + ".csv"
    ocp_loc = "Instances_Max_calls_" + str(max_calls) +"/Average/schedule_avg.csv"
else:
    loc =  "InstancesACR_Max_calls_" + str(max_calls) +"/Average/schedule_stats_150_360_50_" + str(D) + _ + str(q) + _ + dist + ex +".csv"
    ocp_loc = "InstancesACR_Max_calls_" + str(max_calls) +"/Average/schedule_avg.csv"
    

config['ocp_at'] = ocp_loc
config['eoc'] = eoc
config['D'] = D
config['q'] = q
config['dist'] = dist
config['max_calls'] = max_calls
config['k_cap'] = args.kcap
config['learn_iter'] = args.learn_iter
config['class_weight'] = class_weight
config['temporal_weight'] = temporal_weight
config['ocp_at'] = ocp_loc
config["max_wait" ] = H
config['data_loc'] = loc
#config['resample'] = resample
if max_calls == 1 and config['type'] != 'reg':
    config['objective'] = 'binary:logistic'
config['train' ] = True
config['lambda'] = lambda_
config['store_sim_stats'] =  True
if 'time_to_wait' not in config:
    config['time_to_wait']  = H

config['seed'] = 10#1001

print(config)

def run_func(param = None):
    #for N in [100]:
    #for q in q_set: #40, 60, 80
        #for dist_name in [dist]:#[ 'data-response', 'data-interacted']:#['Uniform', 'Weibull','Normal']
        #    for D in [ 180]:# 120, 180
                #if D > Q:
                #    continue

    quantile = q/100
    #D = int(f * H)
    _ = "_"
    #name = str(N) + _ + str(H) + _ + str(M) + _ + str(D) + _ + str(q) + _ + dist_name 
    if config['policy'] == 'NN' or config['policy'] == 'xgb' or config['policy'] == 'corn':
        name = config['policy'] + _ + str(config['learn_iter']) + _ + str(config['runs']) + _ + str(max_calls) + _ + str(eoc) + _ + str(D) + _ + str(q) + _ + dist + _ +  config['ext']
    else:
        name = str(config['learn_iter']) + _ + str(config['runs']) + _ + str(max_calls)
    #if train == False:
    #    name = name +  'validation' 
    folder = policy_name + "_" + name + "/"
    
    statistics = pd.DataFrame()
    
    
    if not os.path.exists( train_folder + folder):
       os.makedirs( train_folder + folder)
    
    json.dump( config, open(train_folder + folder + "config.json", 'w' ) )
    sim = SECTP(N, H, M = M, quantile = quantile, D = D, dist = dist, name = name, det = train, seed = config['seed'])
    #policy = [1 for i in range(H)]
    sim.set_max_calls(max_calls)
    sim.set_wait(config['max_wait'])
    sim.set_vacancy_weight(config['V_C'])
    
    if config['search_files']:
        read_files = glob.glob(train_folder + folder + "Saved_data_*.csv")
        try:
            statistics = pd.read_csv( train_folder + folder + "/Stats" + policy_name + "_" + name + ".csv")
        except:
            print("No stats")
        pass_run = True
    else:
        read_files = []
        pass_run = False

    d = IO_learning(sim, config, name, read_files = read_files, folder = train_folder + folder)
    

    print("Running - ", name , ", param = ", param)
    start = time.time()
    
    for n in range(config['learn_iter']):
        print("Iteration - ", n)
        reward = 0
        
        if n >= len(read_files):
            pass_run = False
        
        for k in range(config['runs']):
            print("Episode - ", k)
            state = sim.initialize(k, policy_name, rd_values = 'generate')   
            sim.save_instance(train_folder + folder + '/Instance_' + str(n) + ".csv")
            if config['type'] == 'class':
                if config['target'] == 'called':
                    d.pred_history = pd.DataFrame(columns=[ i for i in range(2 * max_calls)] + ['time', 'action', 'run'])
                else:
                    d.pred_history = pd.DataFrame(columns=[ i for i in range(max_calls + 1)] + ['time', 'action', 'run'])
            elif config['type'] == 'reg':
                d.pred_history = pd.DataFrame(columns=[ 'pred'] + ['time', 'action', 'run'])

            if pass_run:
                continue                      
            
            sim.define_model(time_limit = 60, eoc=eoc)
            #print(sim.user_response_duration)
            end = False
            d.iteration = k
            d.solve(state)
            d.old_max_user = d.max_user
            print("Need - ", d.old_max_user)
            #d.enough = False

            time_to_wait = 0

            call = 1      
            d.set_to_call(call)
            while not end:
                
                if len(state['assignment']) < M and state['last_called'] != sim.N - 1:
                    
                    while call < max_calls and time_to_wait == 0 :         
                        time_to_wait = d.get_action(state)
                        expert_action, avg_wait = d.get_avg_expert_action_wt(state )             
                        print("wait - ", time_to_wait)
                        if time_to_wait == 0:
                            call += 1
                            d.set_to_call(call)        

                    #call = d.add_new_data(state, policy_action, expert_action)
                        _ = d.add_new_data_avg(state, time_to_wait, expert_action, avg_wait, wt = True)
                    #d.add_features(k, n)
                    
                #print(call)
                old_assign = len(sim.assignment)
                print("Call ", call , " at ", state['time'], ' with assignemnt ', len(state['assignment']))
                state, reward, end = sim.call_users(call)
                call = 0
                d.set_to_call(0)
                if  old_assign < len(sim.assignment):
                    time_to_wait = 0
                elif time_to_wait == 1:
                    call = 1
                    d.set_to_call(1)
                time_to_wait = max(time_to_wait - 1 , 0)

                # if state['time'] == 5:
                #     break
            
            #action_history = pd.concat([action_history, p.eps_history])  
            #print("Shifts scheduled = ", len(state['assignment']))
            #sim.save_events_log(k)
            print("Stats = ", sim.stats)
            sim.collect_stats()
            sim.stats ['iter'] = n
            df_new = pd.DataFrame([sim.stats])
            #self.statistics = self.statistics.append(M.stats, ignore_index=True)
            statistics = pd.concat([statistics, df_new])
            #action_history = pd.concat([action_history, p.eps_history])  
            if config['store_sim_stats'] == True:
                #if not os.path.exists( train_folder + folder + 'Sim_stats_' + str(config['model'])):
                #    os.makedirs( train_folder + folder + 'Sim_stats_' + str(config['model']))
                d.get_sim_data(n)


            #print("total_bumps = ", sim.stats['Num_users_bumped'], "shifts_vacant = ", sim.stats['shifts_vacant'])
            #print("Sim = ", k, "Avg total bumps = ", np.mean(bump_hist), "Avg shifts vacant = ", np.mean(vac_hist), "Avg Eps Reward = ", np.mean(r_hist))
            d.make_data(n, k)
        d.beta_update()
        
        #d.feature_data.to_csv(train_folder + folder + '/Feature_data.csv')
        
        if config['store_sim_stats'] == True:
            #if not os.path.exists( train_folder + folder + 'Sim_stats_' + str(config['model'])):
            #    os.makedirs( train_folder + folder + 'Sim_stats_' + str(config['model']))

            d.sim_stats.to_csv(train_folder + folder + '/Sim_Learn_Data_' + str(n) + '.csv')
        
        if os.path.isfile(train_folder + folder + "/Saved_data_" + str(n) + ".csv") and config['search_files']:
            continue
        #{k: v for k, v in sorted(sim.responses.items(), key=lambda item: item[0])}
        d.all_data['iteration'] = n
        d.all_data.to_csv(train_folder + folder + "/Saved_data_" + str(n) + ".csv")
        d.update(folder = train_folder + folder, n = n)
        d.all_data = pd.DataFrame()
        statistics.to_csv( train_folder + folder + "/Stats" + policy_name + "_" + name + ".csv")
    
    if config['policy'] == 'Linear':
        d.save_history(address = train_folder + folder)
        print("Policy = " , d.policy)
        print("Number of Employees called =", sum(d.policy))

    #d.policy.save_model(train_folder + folder, -1)    
        
    #print("Mean_bumps = ", np.mean(statistics['Num_users_bumped']))
    print("Total time = ", time.time() - start)

    
    #d.policy.df.to_csv(train_folder + folder + "All_data.csv")
    return statistics
        
if __name__ == "__main__":

    learn = 1

    policy_name = 'IOL'
    statistics = run_func()
    #statistics.to_csv("Data/Saved_stats" + str(learn_iter) + '_' + str(runs) + ".csv")
