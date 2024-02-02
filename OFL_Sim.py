
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 14:58:56 2023


"""

import numpy as np
import pandas as pd
import os
from SECTP import SECTP
import json

import itertools
from glob import glob
import copy
from Learning_modules import IO_learning
from collections import deque
from config import config
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=int, default= 114)
parser.add_argument('--model', type=int, default=64)
parser.add_argument('--learn_iter', type=int, default=200)
parser.add_argument('--D', type=int, default=180)
parser.add_argument('--q', type=int, default=40)
parser.add_argument('--kcap', type=int, default=3)
parser.add_argument('--tf', type=str, default='local')
parser.add_argument('--dist', type=str, default='data-interacted')
parser.add_argument('--eoc', type=int, default=0)
parser.add_argument('--max_calls', type=int, default=2)
parser.add_argument('--lm', type = float,  default=0.0)
parser.add_argument('--tw', type=int, default=0)
#parser.add_argument('--q', type=int, default=40)
args = parser.parse_args()

_ = '_'


sims = 300
class_weight = False
temporal_weight =args.tw
lambda_ = args.lm
folder_name = str(args.kcap) + _ + str(lambda_) 
folder_name +=  str('_TW') if temporal_weight else '' 

train_folder = 'Data_Dagger_' + folder_name + '/' if args.tf == 'local' else '../../../../../scratch/gawpra/Data_Dagger_' + folder_name + '/'
if not os.path.exists( train_folder ):
   os.makedirs( train_folder )


V_C = 50
dist = args.dist##data-interacted Weibull
N = 150 
H = 360
M = 50
D = args.D
q = args.q
learn_iter = args.learn_iter

eoc = True if args.eoc else False
max_calls = args.max_calls
#dist_name = 'Weibull'#Uniform #Weibull#Normal
_ = '_'

if eoc == False:    
    loc = "Instances_Max_calls_" + str(max_calls) +"/Average/schedule_stats_150_360_50_" + str(D) + _ + str(q) + _ + dist + ".csv"
    ocp_loc = "Instances_Max_calls_" + str(max_calls) +"/Average/schedule_avg.csv"

else:
    loc =  "InstancesACR_Max_calls_" + str(max_calls) +"/Average/schedule_stats_150_360_50_" + str(D) + _ + str(q) + _ + dist + ".csv"
    ocp_loc = "InstancesACR_Max_calls_" + str(max_calls) +"/Average/schedule_avg.csv"

delay_bucket_intervals = {1:[10, 0], 2:[30, 10], 3:[60, 30], 4:[150, 60], 5: [170, 150], 6:[180, 170]} #if D == 180 else {1:[10, 0], 2:[30, 10], 3:[60, 30], 4:[90, 60], 5: [110, 90], 6:[120, 110]}


config = config[args.id]
config['delay_bucket_intervals'] = delay_bucket_intervals
config['ocp_at'] = ocp_loc
config['eoc'] = eoc
config['D'] = D
config['q'] = q
config['dist'] = dist
config['max_calls'] = max_calls
config['k_cap'] = 3
config['lambda'] = 0
config['learn_iter'] = learn_iter
config['sims'] = sims
config['class_weight'] = class_weight
config['temporal_weight'] = temporal_weight
config['ocp_at'] = ocp_loc
config["max_wait" ] = H
config['data_loc'] = loc
config['model'] = [args.model]#[i for i in range(1, 99)]#
config['V_C'] = 50

config['train' ] = False
if max_calls == 1 and config['type'] != 'reg':
    config['objective'] = 'binary:logistic'
    #config['action_type'] = 'max'
config['lambda'] = lambda_
config['store_sim_stats'] =  True
config['solve_first'] = False       

#remove
#config['prob_threshold'] = 0.7
#config['action_type'] =  'quant'

models = config['model']

print(config)

def run_func(param = None):
    #for N in [100]:
    #for q in q_set: #40, 60, 80
    #    for dist_name in [dist]:#[ 'data-response', 'data-interacted']:#['Uniform', 'Weibull','Normal']
    #        for D in [ 180]:# 120, 180
                #if D > Q:
                #    continue
    for model_id in models:
        config['model'] = model_id
        avg_bumps = []
        avg_vac = []
        seed = 11#10 while train#11 for test
        quantile = q/100
        #feature_sim_data = pd.DataFrame()
        #D = int(f * H)
        _ = "_"
        #name = str(N) + _ + str(H) + _ + str(M) + _ + str(D) + _ + str(q) + _ + dist_name 
        if config['policy'] == 'NN' or config['policy'] == 'xgb' or config['policy'] == 'corn':
            name = config['policy'] + _ + str(config['learn_iter']) + _ + str(config['runs']) + _ + str(max_calls) + _ + str(eoc) + _ + str(D) + _ + str(q) + _ + dist + _ + config['ext']
        else:
            name = str(config['learn_iter']) + _ + str(config['runs']) + _ + str(max_calls)
        #if train == False:
        #    name = name +  'validation' 
        folder = policy_name + "_" + name + "/"
        
        statistics = pd.DataFrame()
        
        if not os.path.exists( train_folder + folder):
           os.makedirs( train_folder + folder)
        if not os.path.exists( train_folder + folder + 'Schedules/'):
           os.makedirs( train_folder + folder + 'Schedules/')
        
        
        sim = SECTP(N, H, M = M, quantile = quantile, D = D, dist = dist, name = name, det = False, seed = seed)
        #policy = [1 for i in range(H)]
        sim.set_max_calls(max_calls)
        sim.set_vacancy_weight(V_C)
        sim.set_wait(config['max_wait'])

        d = IO_learning(sim, config, name, folder = train_folder + folder,  sim_type = 'sim')
        

        print("Running - ", name , ", param = ", param)
        start = time.time()
        
        for n in range(1):
            print("Iteration - ", n)
            reward = 0

            
            for k in range(config['sims']):
                print("Episode - ", k)
                state = sim.initialize(k, policy_name) #rd_values='Instances_Max_calls_2/150_360_50_180_40_data-interacted/Instance_' + str(k + 1) + '.csv'   
                sim.store_urd(_id = k)
                
                #if k != 71:
                #    continue

                if config['type'] == 'class':
                    if config['target'] == 'called':
                        d.pred_history = pd.DataFrame(columns=[ i for i in range(2 *  max_calls)] + ['time', 'action', 'run'])
                    else:
                        d.pred_history = pd.DataFrame(columns=[ i for i in range(max_calls + 1)] + ['time', 'action', 'run'])
                elif config['type'] == 'reg':
                    d.pred_history = pd.DataFrame(columns=[ 'pred' ,'time', 'action', 'run'])

                #sim.define_model(time_limit = 30)
                end = False
                d.iteration = k
                
                if config['solve_first']:
                    sim.define_model(eoc = config['eoc'])
                    obj, status, t = sim.solve(k)    
                    sim.post_process(name + "_" + str(k), itr = k, print_output = False, save_solution = True, plot = False, folder = train_folder + folder)
                    continue
   
                #d.enough = False

                while not end:
                    
                    if len(state['assignment']) < M and state['last_called'] != sim.N - 1:
                        call = d.get_action(state, k)
                        if config['store_sim_stats'] == True:
                            d.add_features(k, n)
                    else:
                        call = 0
                        
                    #print(call)
                    
                    state, reward, end = sim.call_users(call)
                    #if d.enough == False:
                    #    d.check_optimal(state)
                    #if d.enough == True:
                    #    print("here")
                    
                #action_history = pd.concat([action_history, p.eps_history])  
                #print("Shifts scheduled = ", len(state['assignment']))
                #sim.save_events_log(k)
                print("Stats = ", sim.stats)
                avg_bumps.append(sim.stats['Num_users_bumped'])
                avg_vac.append(sim.stats['shifts_vacant'])
                print("Running Average Bumps = ", np.mean(avg_bumps))
                print("Running Average Vacancy = ", np.mean(avg_vac))
                sim.collect_stats()
                sim.stats ['iter'] = n
                df_new = pd.DataFrame([sim.stats])
                

                statistics = pd.concat([statistics, df_new])
                if config['store_sim_stats'] == True:
                    #if not os.path.exists( train_folder + folder + 'Sim_stats_' + str(config['model'])):
                    #    os.makedirs( train_folder + folder + 'Sim_stats_' + str(config['model']))
                    d.get_sim_data(k)
                    
                #if k <= 10:
                #d.save_schedule(name = train_folder + folder + 'Schedules/schedule_' + str(config['model']) + _ + str(k) + ".csv")               
            if config['store_sim_stats'] == True:
                d.sim_stats.to_csv(train_folder + folder + '/Sim_Data' + str(model_id) + '.csv')
                d.feature_data.to_csv(train_folder + folder + '/Feature_sim_data_' + str(model_id) + ".csv")
                d.feature_data = pd.DataFrame(columns = config['features'] +  ['pred'] + [ 'action', 'expert_action', 'decision_rule', 'vacancy', 'bumps' , 'beat_action', 'run', 'iter'])

            statistics.to_csv( train_folder + folder + "/Stats" + policy_name + "_sim_" + name + _ + str(config['model']) + ".csv")
            sim.urd_store.reset_index(drop=True).to_csv(train_folder + folder + "/URD.csv")
            
        if config['policy'] == 'Linear':
            d.save_history(address = train_folder + folder)
            print("Policy = " , d.policy)
            print("Number of Employees called =", sum(d.policy))

            
            
        #print("Mean_bumps = ", np.mean(statistics['Num_users_bumped']))
        print("Total time = ", time.time() - start)
        
        json.dump( config, open(train_folder + folder + "config_sim.json", 'w' ) )
                    
                    # all_folds = glob( train_folder + folder + "*.csv", recursive = True)

                    # data = pd.DataFrame()
                    # for file in all_folds:
                    #     if 'Saved_data' in file:
                    #         temp = file.replace(".", "_")
                    #         _id = temp.split("_")[-2]
                    #         df = pd.read_csv(file)
                    #         df['id'] = _id
                    #         data = pd.concat([data, df])
                        
                        
                    # data.to_csv(train_folder + folder + folder[:-1] + '_Data.csv')
                    

                    # data = pd.DataFrame()
                    # for file in all_folds:
                    #     if 'Sim_Data' in file:
                    #         temp = file.replace(".", "_")
                    #         _id = temp.split("_")[-2]
                    #         df = pd.read_csv(file)
                    #         df['id'] = _id
                    #         data = pd.concat([data, df])
                        
                        
                    # data.to_csv( train_folder + folder + folder[:-1] + '_Sim_Data.csv')
                    
                    
                    # data = pd.DataFrame()
                    # for file in all_folds:
                    #     if 'sim_xgb' in file:
                    #         temp = file.replace(".", "_")
                    #         _id = temp.split("_")[-2]
                    #         df = pd.read_csv(file)
                    #         df['id'] = _id
                    #         data = pd.concat([data, df])
                        
                        
                    # data.to_csv( train_folder + folder + folder[:-1] + '_sim_stats.csv')

        
if __name__ == "__main__":
    #train_folder = 'Data/'
    #learn = 1

    policy_name = 'IOL'
    run_func()
    #statistics.to_csv("Data/Saved_stats" + str(learn_iter) + '_' + str(runs) + ".csv")
#else:
    
    
#history = np.load('Data/history.npy')
