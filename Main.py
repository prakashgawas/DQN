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
import itertools
from Policies import Policy

    
is_training = 1
runs = 300

sim = {}
dist = 'data-interacted'##data-interacted Weibull
N = 150# if dist == 'Weibull' else 150
H = 360
M = 50
D = 180
q_set = [ 40,60 ,80] if dist == 'Weibull' else [40] #, , 60, 80
#dist_name = 'Weibull'#Uniform #Weibull#Normal
max_calls = 2
train = False
save_actions = True
store_sim_stats = 0
seed = 12
V_C = 50
base_folder = "../../DECTP/" 

def run_func(param = None):
    #for N in [100]:
    for q in q_set: #40, 60, 80
        for dist_name in [dist]:#[ 'data-response', 'data-interacted']:#['Uniform', 'Weibull','Normal']
            for D in [ 180]:# 120, 180
                #if D > Q:
                #    continue
                
                seed = 11#10 while train#11 for validation#12 test
                quantile = q/100
                #D = int(f * H)
                _ = "_"
                name = str(N) + _ + str(H) + _ + str(M) + _ + str(D) + _ + str(q) + _ + str(max_calls)  + _ + dist_name 
                
                if train == False:
                    name = name +  'validation'#'validation' 'test'
                folder = policy_name + "_" + name + "/"
                
                statistics = pd.DataFrame()
                
                if not os.path.exists(base_folder + train_folder + folder):
                   os.makedirs(base_folder + train_folder + folder)
                            
                
                sim = SECTP(N, H, M = M, quantile = quantile, D = D, dist = dist_name,  name = name, det = train,  seed = seed)
                p = Policy(sim, policy_name)
                sim.set_costs(vacancy_cost=1, bump_cost=1)
                action_history = pd.DataFrame()
                print("Running - ", name , ", param = ", param)
                for k in range(runs):
                    if (k % 25) == 0:
                        print("Sim ",k)
                    p.set_policy_funct( policy_at, param = param )
                    
                    state = sim.initialize(k, policy_name, rd_values = 'generate')
                   
                    end = False
                    while not end:
                        
                        if len(state['assignment']) < M:
                            call = p.policy_funct(state)
                        else:
                            call = 0
                        #print(call)
                        
                        state, reward, end = sim.call_users(min(call, max_calls))
                        
                    #action_history = pd.concat([action_history, p.eps_history])  
                    #print("Shifts scheduled = ", len(state['assignment']))
                    #sim.save_events_log(k)
                    #print("Stats = ", sim.stats)
                    sim.collect_stats()
                    df_new = pd.DataFrame([sim.stats])
                    #self.statistics = self.statistics.append(M.stats, ignore_index=True)
                    statistics = pd.concat([statistics, df_new])
                    #if runs < 200:
                    if store_sim_stats:
                        p.get_sim_data(k, param)
                        
                    
                    #if not os.path.exists("../DECTP/DECTP_Results/" + folder):
                    #   os.makedirs("../DECTP/DECTP_Results/" + folder)
                       
                    #sim.schedule.to_csv("../DECTP/DECTP_Results/" + folder + "Schedule_" + name + "_" + str(k) + ".csv")
                if store_sim_stats :
                    p.sim_stats.to_csv(base_folder + train_folder + folder + 'Sim_Data' + str(policy_name + "_" + name) + '.csv')
                
                if is_training == 1:
                    if learn == 1:
                        name = name + param[0].replace(".","_").split("/")[-2]
                    elif policy_name == 'Linear':
                        name = name
                    else:
                        name = name + str(param)
                        
                #if save_actions == 1:
                #    action_history.to_csv('../../DECTP/' + train_folder + folder + policy_name + "_actions_" + name + ".csv")
                statistics.to_csv(base_folder + train_folder + folder + policy_name + "_" + name + ".csv")
                print("Mean_bumps = ", np.mean(statistics['Num_users_bumped']))
                print("Mean_vacancy = ", np.mean(statistics['shifts_vacant']))
                print("Cost = ", np.mean(statistics['Num_users_bumped']) + np.mean(statistics['shifts_vacant']) * V_C)
    
if is_training == 1:
    eoc = False
    if eoc == True:
        train_folder = 'DECTP_Train_max_calls' + str(max_calls) + 'ACR/'
        policy_at = 'InstancesACR_Max_calls_' + str(max_calls) + '/Average/schedule_avg.csv'
    else:
        train_folder = 'DECTP_Train_max_calls' + str(max_calls) + '/'
        policy_at = 'Instances_Max_calls_' + str(max_calls) + '/Average/schedule_avg.csv'
    learn = 0
    
    # policy_name = 'Call_and_Wait'
    # #a = [[i+1 for i in range(5)], [i+1 for i in range(15)]]
    # a = [[5], [1]]
    # params = list(itertools.product(*a))
    # print("Running - ", policy_name)
    # for param in params:
    #     run_func(list(param))
        
    # policy_name = 'Cutoff_buffer'
    # params = [[i] for i in range(9,50)]
    # print("Running - ", policy_name)
    # for param in params:
    #     run_func(list(param))
    
    # policy_name = 'call_all'
    # params = [[N]]
    # print("Running - ", policy_name)
    # for param in params:
    #     run_func(list(param))
        
    policy_name = 'Deterministic_call_policy'
    params = [[0.90]]#[[0.98], [ 0.95], ['mean']] + [[np.round(0.3 + 0.1 * i,2)] for i in range(7)]#[[0.9]]#
    print("Running - ", policy_name)
    for param in params:
        run_func(list(param))

    
    # policy_name = 'Deterministic_cutoff_policy'
    # params = [[0.98], [ 0.95],  ['mean']] + [[np.round(0.3 + 0.1 * i,2)] for i in range(7)]#[[0.9]]#
    # print("Running - ", policy_name)
    # for param in params:
    #     run_func(list(param))
        
    # policy_name = 'Deterministic_sum_cutoff_data_policy'
    # params = [[0.98], ['mean'],[ 0.95]] + [[np.round(0.3 + 0.1 * i,2)] for i in range(7)]#[[0.9]]#
    # print("Running - ", policy_name)
    
    # for param in params:
    #    run_func(list(param))
       
    # policy_name = 'Call_late'
    # params = [[3]]#[[0.9]]#
    # print("Running - ", policy_name)
    
    # for param in params:
    #    run_func(list(param))
        
    # policy_name = 'Linear'
    # params = [np.array([5, 1, 2, 2, 2, 3, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 2, 1, 1, 2, 1, 2,
    #    1, 2, 2, 1, 2, 2, 1, 2, 1, 2, 0, 2, 1, 2, 1, 1, 2, 1, 1, 1, 2, 2,
    #    1, 1, 1, 2, 1, 0, 1, 2, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0,
    #    0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0,
    #    0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    #    0, 0, 0, 0, 0, 1, 1, 2, 1, 1, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0,
    #    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #    0, 0, 0, 0, 0, 0, 0, 0, 0])]#[[0.9]]#
    # print("Running - ", policy_name)
        
    # for param in params:
    #     run_func(list(param))
    
    #learn = 1
    #policy_name = 'Linear_Regression'
    #params = [['Instances/Average/finalized_model.sav']]
    #print("Running - ", policy_name)
    
    # policy_name = 'XGB'
    # params = [['xgb.pkl']]
    
    # print("Running - ", policy_name)
    
    # for param in params:
    #     run_func(list(param))
        
    # policy_name = 'XGB_C'
    # params = [['xgb_c1.pkl', 'xgb_c2.pkl']]
    
    # print("Running - ", policy_name)
    
    # for param in params:
    #     run_func(list(param))
    
    # policy_name = 'NN'
    # params = [['NN']]
    
    # print("Running - ", policy_name)
    
    # for param in params:
    #     run_func(list(param))
#else:
    
    
