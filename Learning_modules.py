#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 10:49:29 2023

@author: Prakash
"""

import numpy as np

import time
import math
from collections import deque
from torch import autograd, tensor , float32, argmin, manual_seed, from_numpy
import matplotlib.pylab as plt
from matplotlib import pyplot
from xgboost import plot_importance
import torch
from imblearn.over_sampling import SMOTE
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import torch.nn.functional as F
import pickle
#from coral_pytorch.losses import corn_loss
import pandas as pd
from scipy.stats import bernoulli
from collections import Counter
import xgboost as xgb
import copy
from torch.distributions import Categorical
import sys
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import Dataset
import shap
from sklearn.linear_model import LinearRegression
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
#from utils import set_seed
from buffer import ReplayBuffer, PrioritizedReplayBuffer
from ONCSS_Actual_pyomo import ONCSS
seed = 2
#tf.random.set_seed(seed)
manual_seed(seed)
#torch.set_num_threads(4)

    
class IO_learning():
    def __init__(self,  sim, config, name, alpha = 0.7, read_files = [], folder = None, sim_type = 'train'):
        
        self.sim = sim
        self.policy_type = config['policy']
        self.feature_names = config['features']
        self.k_cap = config['k_cap'] if config['k_cap'] != None else self.H
        self.pred_history = pd.DataFrame()
        self.feature_data = pd.DataFrame(columns = config['features'] + ['pred'] + [ 'action', 'expert_action', 'decision_rule', 'best_action', 'time', 'vacancy', 'bumps' ,  'run', 'iter'])
        self.sim_type = sim_type
        self.curr_kpi = [0,0]
        self.lambda_ = config['lambda']
        self.beta =  self.lambda_ 
        self.criteria = self.check_obj_sol if config['criteria'] == 'check_obj' else self.check_direct_calls
        self.eoc = config['eoc']
            
        global ber_randomGen
        ber_randomGen = bernoulli
        ber_randomGen.random_state= np.random.Generator(np.random.PCG64(10))

        
        self.ocp = pd.read_csv(config['ocp_at'])
        self.ocp = self.ocp[(self.ocp.dist == self.sim.dist_name) & (self.ocp.D == self.sim.D) &
                                            (self.ocp.N == self.sim.N) & (self.ocp.H == self.sim.H) & (self.ocp.Q == self.sim.q) ]
        self.ocp.index = self.ocp.time
        self.ocp = self.ocp['cum_calls_made_0.9']
        
        self.feature_fn = {'cum_calls_made_at_start' : self.get_ccmas,
                           't_rem' : self.get_t_rem,
                           'delay_bucket_1' : self.get_delay_bucket_1,
                           'delay_bucket_2' : self.get_delay_bucket_2,
                           'delay_bucket_3' : self.get_delay_bucket_3,
                           'delay_bucket_4' : self.get_delay_bucket_4,
                           'delay_bucket_5' : self.get_delay_bucket_5,
                           'delay_bucket_6' : self.get_delay_bucket_6,
                           'time_since_last_call' : self.get_time_since_last_call,
                           'shifts_available' : self.get_shifts_available, 
                           'sum_cutoff_at_start': self.get_sum_cutoff_at_start ,
                           'last_calls' : self.get_last_calls,
                           'time_by_shifts_available' : self.get_time_by_shifts_available ,
                           'empl_by_shifts_available' : self.get_empl_by_shifts_available,
                           'last_response': self.get_num_last_response,
                           'cum_calls_made_0.9': self.get_ocp,
                           'to_call': self.get_to_call
                        
                           #'delay_bucket_1_by_shifts_filled' : self.get_delay_bucket_1_by_shifts_filled,
                           #'delay_bucket_2_by_shifts_filled' : self.get_delay_bucket_2_by_shifts_filled,
                           #'delay_bucket_3_by_shifts_filled' : self.get_delay_bucket_3_by_shifts_filled
                           }
        
        self.normalize = config['normalize']
        self.scenario = []
        for i in range(config['scenarios']):
            self.scenario.append(        ONCSS(self.sim.N, self.sim.H, M = self.sim.M, quantile = self.sim.quantile, 
                                   D = self.sim.D,  dist = self.sim.dist_name, 
                                   name = self.sim.name, det = self.sim.det, seed = i))
            self.scenario[i].set_max_calls(config['max_calls'])
            self.scenario[i].set_wait(config['max_wait'])
            self.scenario[i].set_vacancy_weight(config['V_C'])

        data = pd.DataFrame()
        
        
        if config['train'] and not config['fit']:
            data = pd.read_csv(config['data_loc'])
            #data = data.sample(frac=0.05, random_state=1)
            data = data[data.id <= config['learn_for']]
            data['best_action'] = data['calls_made']
            data['run'] = data['id']
            #data = data.rename(columns={"delay_bucket_10": "delay_bucket_1", "delay_bucket_60": "delay_bucket_2", "delay_bucket_180": "delay_bucket_3"})
            data = data[data.shifts_available > 0 ]
            data = data[data.time_to_wait <= config['time_to_wait']] 
            data['time_by_shifts_available'] = data.t_rem  /(data.shifts_available )
            data = data.merge(self.ocp, how = 'left', left_on = 'time', right_index = True )
            #data['empl_by_shifts_available'] = (self.sim.N - data.cum_calls_made_at_start) /(data.shifts_available)
            data = data[self.feature_names + ['cum_calls_made', 'best_action' , 'run', 'time_to_wait']]
            data['iteration'] = -1
            data['time'] = self.sim.H - data['t_rem']
            
        
            if len(read_files) > 0:
                for file in read_files:
                    df = pd.read_csv(file)
                    df = df[self.feature_names + [config['target']] + [ 'run' , 'iteration']]
                    data = pd.concat([data, df])


            data[['no_calls_for', 'continuous_calls', 'curr_bumps', 'policy_action', 'acted', 'decision_rule', 'branch']] = 0
        elif config['fit']:
            data = pd.read_csv( folder + "/All_data.csv")
                
        
        print("Data Available = ", len(data))
        
        #if config['resample']:
        #    data = self.get_resampled(data)
            
        #print("Data Available After resampling = ", len(data))
        
        if self.policy_type == 'NN':
            if self.normalize and config['train']:
                data = self.data_normalize(data)
            self.policy = Supervised_NN(self.feature_names, 32, self.sim.max_calls + 1, data = data)
            self.get_action = self.get_action_NN
            self.update_policy = self.policy.update_model
            
            
        elif self.policy_type == 'xgb':
            if self.normalize and config['train']:
                data = self.data_normalize(data)
            self.policy = Xgboost(self.sim,  self.feature_names, config, data = data, folder = folder)
            self.get_action = self.get_action_xgboost
            
            if config['action_type'] == 'prob':
                self.policy.action_fn = self.policy.random_action
            elif config['action_type'] == 'quant':
                self.policy.action_fn = self.policy.quantile_action
                self.policy.prob_threshold = config['prob_threshold']
            elif config['action_type'] == 'mean':
                self.policy.action_fn = self.policy.mean_action
            else:
                self.policy.action_fn = self.policy.max_prob_action
            self.update_policy =  self.policy.update_model
        
        elif self.policy_type == 'corn':
            if self.normalize and config['train'] and not config['fit']:
                data = self.data_normalize(data)
            self.policy = CORN(self.sim,  self.feature_names, config, data = data, folder = folder)
            if self.sim_type == 'train':
                self.get_action = self.get_action_corn
            else:
                self.get_action = self.get_action_corn#_sim
            self.update_policy = self.policy.update_model
            
        else:
            self.policy = self.default_offline_policy()
            self.get_action = self.get_action_linear
            self.history = self.policy.reshape(1, -1)
            self.update_policy = self.update_model


        self.delay_bucket_intervals = config['delay_bucket_intervals']

        
        self.iteration = 0
        self.all_data = pd.DataFrame()
        self.alpha = alpha
        if config['scenarios'] == 1:
            self.data = pd.DataFrame(index = pd.MultiIndex(levels=[[],[]],codes=[[],[]], names=['time', '_id']),
                                 columns = ['interacted', 'sum_cutoff_at_start', 'in_delay_at_start', 'last_calls',
                                            'in_cutoff_at_end', 'in_delay_at_end', 'in_cutoff_at_start',
                                            'no_calls_for', 'bumps', 'continuous_calls', 'curr_bumps',
                                            'time_since_last_call', 'calls_made', 'cum_calls_made_at_start',
                                            'last_response', 'cum_interacted_at_end', 'cum_interacted_at_start',
                                            'delay_bucket_1', 'delay_bucket_2', 'delay_bucket_3', 'delay_bucket_4',
                                            'delay_bucket_5', 'delay_bucket_6', 'sum_cutoff_at_end',
                                            'shifts_available', 'cum_calls_made', 't_rem', 'cum_calls_made_0.9'])
            self.data = self.data.astype('float')
        else:
            self.data = {}
            if config['aggregator'] == 'min':
                self.aggregator = np.min
            else:
                self.aggregator = np.mean
        self.data_id = 0
        self.name = name
        self.folder = folder
        self.sim_stats = pd.DataFrame()
        self.solve_pb = True
        self.obj = 0
   
        self.action_history = pd.DataFrame()
        
    # def get_resampled(self, data):
    #     col = list(data.columns)
    #     col.remove('best_action')
    #     X = data[col]
    #     y = data['best_action']
    #     counts = dict(Counter(y))
    #     max_count = int(max(counts.values()) * 0.1)
    #     counts = {i : max(max_count, counts[i]) for i in counts}
    #     model = SMOTE(sampling_strategy=counts)
    #     X_resampled, y_resampled = model.fit_resample(X, y)
    #     data = X_resampled
    #     data['best_action'] = y_resampled
    #     print("Counter :", Counter(y_resampled))
    #     return data
       
            
    def get_action_linear(self, state):
            self.get_feature_vector(state)
            p = math.modf(self.policy[state['time']])[0]
            
            return min(int(self.policy[state['time']]) + bernoulli.rvs(p), self.sim.N - 1 - state['last_called'])
        
    def get_action_xgboost(self, state, k = 0):
        self.get_feature_vector(state)
        x = self.x
        if self.normalize:
            x = self.data_normalize(pd.DataFrame.from_dict([x]))
        action, pred = self.policy.get_action(x, state )
        self.x['predict'] = list(np.round(pred,3))
        self.x['action'] = action
        self.x['expert_action'] = 0
        self.x['decision_rule'] = 0
        self.x['best_action'] = 0
        self.x['time'] = state['time']
 
        pred = np.round(pred, 3)
        self.pred_history.loc[len(self.pred_history)] =  list(pred.flatten()) + [state['time'] , action , k]
        return min(action, self.sim.N - 1 - state['last_called'])
    
    def get_action_corn(self, state, k = 0):
        
        self.get_feature_vector(state)
        x = self.x
        if self.normalize:
            x = self.data_normalize(pd.DataFrame.from_dict([x]))
        action, pred = self.policy.get_action(x)
        self.x['predict'] = list(pred)
        self.x['action'] = action
        self.x['expert_action'] = 0
        self.x['decision_rule'] = 0
        self.x['best_action'] = 0
        self.x['time'] = state['time']
        
        pred = np.round(pred, 3)
        self.pred_history.loc[len(self.pred_history)] =  list(pred) + [1]  + [state['time'] , action , k]
        return min(action, self.sim.N - 1 - state['last_called'])
    
    # def get_action_corn_sim(self, state, k = 0):
        
    #     self.get_feature_vector(state)
    #     x = self.x
    #     if self.normalize:
    #         x = self.data_normalize(pd.DataFrame.from_dict([x]))
        
    #     if self.sim_type == 'sim' and self.get_t_rem(state) <= (self.sim.N - len(state['trace_calls']))/self.sim.max_calls and self.get_t_rem(state) > 350:
    #         action = self.sim.max_calls
    #         pred = np.array([0,0,0,0,0])
    #     else:
    #         action, pred = self.policy.get_action( x )
    #     self.x['predict'] = pred
    #     pred = np.round(pred, 3)
    #     self.pred_history.loc[len(self.pred_history)] =  list(pred) + [1]  + [state['time'] , action , k]
    #     return min(action, self.sim.N - 1 - state['last_called'])
         
    
    def get_action_NN(self, state, k):
        self.get_feature_vector(state)
        x = self.x
        if self.normalize:
            x = self.data_normalize(pd.DataFrame.from_dict([x]))
        return min( self.policy.get_action(x), self.sim.N - 1 - state['last_called'])
    
    
    def get_feature_vector(self, state):
        self.x = {}
        for f in self.feature_names:
            self.x[f] = self.feature_fn[f]( state)
            if self.x[f] < 0:
                print("here")
                sys.exit()
                
    def beta_update(self):
        self.beta = self.lambda_ * self.beta 
                
    def get_decision_rule(self):
        return ber_randomGen.rvs(self.beta)
        
                
    def add_features(self, k, n):
        #print(len(list(self.x.values()) + self.curr_kpi + [k, n]), len(self.feature_data.columns))
        self.feature_data.loc[len(self.feature_data)] = list(self.x.values()) + self.curr_kpi + [k, n]
    
    def get_expert_action(self, state):
        
        if state['last_called'] == self.sim.N - 1:# or self.enough:
            return 0
        elif state['time'] == self.call_at[state['last_called'] + 1]:
            return self.get_best_action(state, self.call_at)
        else:
            return 0
        
    def get_avg_expert_action(self, state):
        actions = []
        waits = []
        print(self.sim.user_response_duration)
        print("\nTime = ", state['time'])
        if state['last_called'] == self.sim.N - 1 or len(state['assignment'] ) == self.sim.M:# or self.enough:
            return 0, 0
        else:
             for i in range(len(self.scenario)):
                self.scenario[i].regenerate(state, copy.deepcopy(self.sim.user_response_duration))
                print(self.scenario[i].user_response_duration)
                self.scenario[i].define_model(time_limit = 60, eoc=self.eoc)
                users = list(state['trace_calls'].keys())
                times = list(state['trace_calls'].values())
                self.scenario[i].fix_call(users,  times, state['trace_calls'])
                if state['last_called'] < self.sim.N:
                    self.scenario[i].call_after(state['last_called'] + 1 , state['time'])
                obj, status, _ = self.scenario[i].solve( ret = False, tee = True)
                call_times = self.scenario[i].get_call_times()
                wait = call_times[state['last_called'] + 1] - state['time']
                action = self.get_best_action(state, call_times)
                actions.append(action)
                waits.append(wait)
             print(actions, waits)
             return np.round(self.aggregator(actions), 3), self.aggregator(waits)
         
    def get_avg_expert_action_wt(self, state):
        actions = []
        waits = []
        print(self.sim.user_response_duration)
        print("\nTime = ", state['time'])
        if state['last_called'] == self.sim.N - 1 or len(state['assignment'] ) == self.sim.M:# or self.enough:
            return 0 , 0
        else:
             for i in range(len(self.scenario)):
                self.scenario[i].regenerate(state, copy.deepcopy(self.sim.user_response_duration))
                #print(self.scenario[i].user_response_duration)
                self.scenario[i].define_model(time_limit = 60, eoc=self.eoc)
                users = list(state['trace_calls'].keys()) + [i + state['last_called'] + 1 for i in range(self.to_call)]
                times = list(state['trace_calls'].values()) + [state['time'] for i in range(self.to_call)]
                self.scenario[i].fix_call(users,  times, state['trace_calls'])
                if state['last_called'] < self.sim.N:
                    self.scenario[i].call_after(state['last_called'] + 1 , state['time'])
                obj, status, _ = self.scenario[i].solve( ret = False, tee = True)
                call_times = self.scenario[i].get_call_times()
                wait = call_times[state['last_called'] + 1 + self.to_call] - state['time']
                action = self.get_best_action(state, call_times)
                actions.append(action)
                waits.append(wait)
             print(actions, waits)
             return np.round(self.aggregator(actions), 3), self.aggregator(waits)
        
        
    def add_new_data_avg(self, state, policy_action, expert_action, avg_wait = 0, wt = False): 
        best_action = expert_action
        self.x['expert_action'] = expert_action
        decision_rule = self.get_decision_rule()
        if decision_rule:
            action = int(np.round(expert_action))
        else:
            action = policy_action
        self.x['decision_rule'] = decision_rule
        self.x['time_to_wait'] = avg_wait
        #print(policy_action, expert_action)     
          
        self.x['best_action'] = best_action
        #x['action'] = action
        self.add_feature_to_dataset_avg(state, wt)
        #self.data[state['time']] = x
        return action

        
    def add_new_data(self, state, policy_action, expert_action): 
        
        self.x['expert_action'] = expert_action
        decision_rule = self.get_decision_rule()
        if decision_rule:
            action = expert_action
        else:
            action = policy_action
        self.x['decision_rule'] = decision_rule

        #print(action, x['best_action'])
        if action >= 1:
            #if action + state['last_called'] >= self.sim.N:
            #    print(action, state['last_called'])
            users = [state['last_called'] + 1 + i for i in range(action)]
            times = [state['time'] for i in range(action)]
            self.sim.fix_call(users,  times, state['trace_calls'])
        if state['last_called'] + action + 1 < self.sim.N:
            self.sim.call_after(state['last_called'] + action + 1 , state['time'] +1)
        

        print(policy_action, expert_action)   
        best_action, branch = self.criteria(state, action, expert_action)
            
        self.x['best_action'] = best_action
        #self.x['action'] = action
        self.add_feature_to_dataset(state, policy_action, decision_rule, branch)
        #self.data[state['time']] = x
        return action
    
    def check_direct_calls(self, state, action, expert_action):
        branch = 0
        if action == expert_action:#action + state['last_called'] + 1 == self.stats.loc[(state['time'], self.data_id), 'cum_calls_made']:
            best_action = action
            pass
        else:
            if state['time'] + 1 < self.sim.H:
                #if action >=0 :
                #    self.det_feasibility_check(state, action)
                
                self.data_id += 1
                self.add_trajectory(state, self.stats)
                self.solve(state, state['time'] + 1)
                best_action = expert_action
                branch = 1
            else:
                best_action = self.sim.max_calls
            
        return best_action, branch
    
        
    def check_obj_sol(self, state, action, expert_action):
        branch = 0
        if action == expert_action:
            best_action = action
            pass
        else:
            if state['time'] + 1 < self.sim.H:
                if action >=0 :
                    self.det_feasibility_check(state, action)
                
                old_stats = self.stats
                old_obj = self.obj
                old_alloted_users = self.sim.alloted_users
                self.old_max_user = max(self.old_max_user, self.max_user)
                self.data_id += 1
                self.solve(state, state['time'] + 1)
                
                #print(self.obj, old_obj)
                #print(old_alloted_users, "\n", self.sim.alloted_users)
                #print(self.old_max_user, self.max_user)
                if old_obj == self.obj and self.old_max_user >= self.max_user:# np.all(old_alloted_users == self.sim.alloted_users)
                    best_action = action
                else:
                    
                    self.add_trajectory(state, old_stats)
                    best_action = expert_action
                    branch = 1
            else:
                best_action = self.sim.max_calls
        
        return best_action, branch
    
    def det_feasibility_check(self, state, action):
        call_users = [state['last_called'] + i for i in range(1, action+1)]
        response_at = self.sim.response_at
        for i in call_users:
            response_at[i] = state['time'] + self.sim.user_response_duration[i]
        
        total = sum(np.array(list(response_at.values())) <= self.sim.H)
        print(total)
        if total > self.sim.M:
            self.sim.alter_callback(total - self.sim.M)
    
    def add_feature_to_dataset_avg(self, state, wt):
        if wt:
            self.data[len(self.data)] = self.x
            self.data[len(self.data) - 1]['curr_bumps'] = self.sim.curr_bumps
        else:
            self.data[state['time']] = self.x
            self.data[state['time']]['curr_bumps'] = self.sim.curr_bumps
    
    
    def add_feature_to_dataset(self, state, policy_action, decision_rule, branch = 0):
        if (state['time'], self.data_id - branch) not in self.data.index:
            #self.data.loc[(state['time'], self.data_id ),:] = self.stats.loc[(state['time'], self.data_id)]
            self.data.loc[(state['time'], self.data_id),:] = self.stats.loc[(state['time'], self.data_id)]
        self.data.loc[(state['time'], self.data_id - branch), 'policy_action'] = policy_action
        self.data.loc[(state['time'], self.data_id - branch), 'acted'] = 1
        self.data.loc[(state['time'], self.data_id - branch), 'decision_rule'] = decision_rule
        self.data.loc[(state['time'], self.data_id - branch), 'branch'] = branch
        self.data.loc[(state['time'], self.data_id - branch), 'time'] = state['time']
        #if 96 in self.data.columns:
        #    print("here")        
        
    # def check_optimal(self, state):
    #     cb = 0
    #     for i,j in self.sim.event_order:
    #         if j == 1 and self.sim.event_order[(i,j)] <= self.sim.H:
    #             cb += 1
    #     self.enough = cb + len(state['assignment']) >= self.sim.M
        
    def save_schedule(self, name = ""):
        pd.DataFrame.from_dict(self.sim.schedule, orient='index').to_csv(name)
    
        
    def get_best_action(self, state, call_at):
        action = 0
        #users = []
        user = state['last_called'] + 1
        while user < self.sim.N:
            if state['time'] == call_at[user]:
                action += 1
                user += 1
            else:
                break
        return action
        
    def solve(self, state, t = 0):
        call_times = 0
        print("\nTime = ", t)
        #try:
        obj, status, _ = self.sim.solve( ret = False, tee = True)
        #except:
        #    obj, status, _ = self.sim.solve( ret = False, tee = True, log_file = True)
        if status == 'infeasible':
            print("Here")
            print(state)
            #self.sim.save_model()
            print(self.call_at)
            print(self.iteration)
            print(self.sim.user_response_duration)
            #self.save_schedule("infeasible.csv")
            sys.exit()
        else:
            call_times = self.sim.get_call_times()
            #response_times = self.sim.get_response_times(call_times)
        print("Obj = ", obj)
        self.call_at = call_times
        self.obj = obj
        self.curr_kpi = [self.sim.get_vacant_shifts() , self.sim.get_total_bumps()]
        
        self.stats = self.get_curr_schedule(state, self.sim.curr_bumps)
        self.max_user = np.max(self.sim.alloted_users)
        #return obj, status
    
        
    def get_ccmas(self, state):
        return len(state['trace_calls'])
    
    def get_delay_bucket_1(self, state):
        return self.delay_bucket(state, self.delay_bucket_intervals[1])
    
    def get_delay_bucket_2(self, state):
        return self.delay_bucket(state, self.delay_bucket_intervals[2])
    
    def get_delay_bucket_3(self, state):
        return self.delay_bucket(state, self.delay_bucket_intervals[3])
    
    def get_delay_bucket_4(self, state):
        return self.delay_bucket(state, self.delay_bucket_intervals[4])
    
    def get_delay_bucket_5(self, state):
        return self.delay_bucket(state, self.delay_bucket_intervals[5])
    
    def get_delay_bucket_6(self, state):
        return self.delay_bucket(state, self.delay_bucket_intervals[6])
    
    def get_num_last_response(self, state):
        return self.sim.current_cb
    
    def get_ocp(self, state):
        return self.ocp.loc[state['time']]
    
    def set_to_call(self, to_call):
        self.to_call = to_call
    
    def get_to_call(self, to_call):
        return self.to_call

    
    # def get_delay_bucket_1_by_shifts_filled(self, state):
    #     return self.x['delay_bucket_1'] / (len(state['assignment']) + 1)
    
    # def get_delay_bucket_2_by_shifts_filled(self, state):
    #     return self.x['delay_bucket_2'] / (len(state['assignment']) + 1)
    
    # def get_delay_bucket_3_by_shifts_filled(self, state):
    #     return self.x['delay_bucket_3'] / (len(state['assignment']) + 1)
    
    def get_t_rem(self, state):
        return self.sim.H - state['time']
    
    def get_time_since_last_call(self, state):
        return (state['time'] - np.max(list(state['trace_calls'].values()), initial=0 ))
    
    def get_shifts_available(self, state):
        return ( self.sim.M - len(state['assignment']))
    
    def get_empl_by_shifts_available(self, state):
        return (self.sim.N - len(state['trace_calls']))  / ( (self.sim.M - len(state['assignment'])) )
    
    def get_time_by_shifts_available(self, state):
        #return (self.sim.H - state['time'] )  /( (self.sim.M - len(state['assignment'])) )

        return (self.sim.H - state['time'] )  / (self.sim.M - len(state['assignment'])) 
    
    def get_sum_cutoff_at_start(self, state):
        return sum([ state['cutoff_times'][i] - state['time'] for i in state['cutoff_times']])
        
    # def get_feature(self, f, state):
    #   if f == 'cum_calls_made':
    #       return len(state['trace_calls']) #/self.sim.N
    #   elif f == 'cum_calls_made_at_start':
    #       return len(state['trace_calls']) #/self.sim.N
    #   elif 'delay_bucket' in f:
    #       u = int(f.split("_")[-1])
    #       return self.delay_bucket(state, self.delay_bucket_intervals[u])#/self.sim.N
    #   elif f =='t_rem':
    #       return self.sim.H - state['time']#/self.sim.H
    #   elif f == 'time_since_last_call':
    #       return (state['time'] - np.max(list(state['trace_calls'].values()), initial=0 )) #/ self.sim.H
    #   elif f == 'shifts_available':
    #       return ( self.sim.M - len(state['assignment']))#/ self.sim.M )
    #   elif f == 'sum_cutoff_at_start':
    #       return sum([ state['cutoff_times'][i] - state['time'] for i in state['cutoff_times']])#/(self.sim.D * self.sim.N)
    #   elif f =='continuous_calls':
    #       return self.get_consecutive_calls(list(state['trace_calls'].values()), state['time'])
    #   elif f == 'last_calls':
    #       return self.get_last_calls(state)
    #   elif f == 'time_by_shifts_available':
    #       return (self.sim.H - state['time'] ) /( self.sim.M - len(state['assignment']))
    #   #elif '_by_shifts_filled' in f:
              
    def delay_bucket(self, state, interval ):
        size = 0
        l = interval[1]
        u = interval [0]
        for i in state['trace_calls']:
            if state['trace_calls'][i] <= (state['time'] - l ) and state['trace_calls'][i] >= (state['time'] - u + 1):
                if i not in state['assignment']:
                    size += 1 
        return size
    
    def get_last_calls(self, state):
        trace = list(state['trace_calls'].values())
        count = Counter(trace)
        if state['time'] - 1 in count:
            return count[state['time'] - 1]
        return 0
    
    def get_consecutive_calls(self, arr, t):
        if len(arr) == 0:
            return 0
        arr = arr[::-1]
        if arr[0] != t-1:
            return 0
        consecutive_integers = 1
        current = arr[0]
    
        for num in arr[1:]:
            if num == current - 1:
                consecutive_integers += 1
                current = num
            else:
                break
        return consecutive_integers
    
    def default_policy(self ):
    # Generate a random binary array with N '1's and (N-length) '0's
        policy = []
        while True:
            policy.append(min(self.sim.N - sum(policy), self.sim.max_calls))
            if sum(policy) >= self.sim.N:
                break
        policy = np.concatenate( (policy, [0] * (self.sim.H + 1 - len(policy))))
        #df = df[df.dist == 'data-interacted']
        
        return policy
    
    def default_offline_policy(self):
        df = pd.read_csv("InstancesMax_calls_5/Average/schedule_avg.csv")
        policy = df['calls_made_mean']
        return np.array(policy)
        
    # def policy_maker(self, calls):
    #     policy = [int(np.ceil(calls[0]))]
    #     base = calls[0] - policy[-1]
    #     for i in range(1, len(calls)):
    #         #print(calls[i])

    #         #print(calls[i])
    #         call = min(np.clip(np.floor(calls[i] + base) ,0, 5 ), self.sim.N - sum(policy))
    #         policy.append(call)
    #         base = np.round(base +  calls[i] - policy[-1], 2)

    #         if sum(policy) >= self.sim.N:
    #             break
        
    #     posn =  np.max(np.nonzero(policy))
    #     while np.round(sum(policy), 2) < self.sim.N and  posn <self.sim.H + 1:
    #         posn += 1
    #         policy[posn ] = min(self.sim.max_calls, self.sim.N - sum(policy))

    #     policy = np.concatenate( (policy, [0] * (self.sim.H + 1 - len(policy))))

    #     if max(policy) > self.sim.max_calls or len(policy) > self.sim.H + 1 :#or sum():
    #         print("here")
    #         print(calls, policy)
    #         sys.exit()

    #     return policy.astype(int)
    
    
    
    
    
    def make_data(self, n, k):
        if len(self.scenario) > 1:
            self.data = pd.DataFrame.from_dict(self.data, orient = 'index')
            #self.data['time'] = self.data.index
        self.data['run'] = k
        #self.data['cum_calls'] = self.data['action'].cumsum()
        self.all_data = pd.concat([self.all_data, self.data])
        self.solve_pb = True
        if len(self.scenario) > 1:
            self.data = {}
        else:
            self.data = pd.DataFrame(index = pd.MultiIndex(levels=[[],[]],codes=[[],[]], names=['time', '_id']),
                                 columns = ['interacted', 'sum_cutoff_at_start', 'in_delay_at_start', 'last_calls',
                                            'in_cutoff_at_end', 'in_delay_at_end', 'in_cutoff_at_start',
                                            'no_calls_for', 'bumps', 'continuous_calls', 'curr_bumps',
                                            'time_since_last_call', 'calls_made', 'cum_calls_made_at_start',
                                            'last_response', 'cum_interacted_at_end', 'cum_interacted_at_start',
                                            'delay_bucket_1', 'delay_bucket_2', 'delay_bucket_3', 'delay_bucket_4',
                                            'delay_bucket_5', 'delay_bucket_6', 'sum_cutoff_at_end',
                                            'shifts_available', 'cum_calls_made', 't_rem', 'cum_calls_made_0.9'])
            self.data = self.data.astype('float')
        self.data_id = 0
        self.save_schedule(self.folder + "Schedule_Learn_" + str(n) +".csv" )
        self.pred_history = pd.DataFrame()
        #self.curr_bumps = 0

    def update(self, folder , n = ''):
        #self.all_data['called'] = self.all_data['calls_made'] > 0
        self.all_data['time_by_shifts_available'] = self.all_data.t_rem /self.all_data.shifts_available
        self.all_data['empl_by_shifts_available'] = (self.sim.N - self.all_data.cum_calls_made_at_start) /self.all_data.shifts_available
        #print(self.all_data.columns)
        if self.normalize:
            df = self.data_normalize(self.all_data)
        df = df.rename(columns={"calls_made": "best_action"})
        
        self.update_policy(df, folder, n)
        #df.drop(columns = ['sum_cutoff_at_end', "in_delay_at_start", 'in_cutoff_at_start','in_delay_at_end',
                                              #'in_cutoff_at_end', 'bumps', 'interacted', 'cum_interacted_at_start',
                                              #'cum_interacted_at_end'])

    def update_model(self, data, folder):
        new_policy = np.array(data.groupby('time')['best_action'].agg( ["sum"])).flatten()
        new_policy = new_policy/(max(data.run) + 1)
        new_policy = np.concatenate( (new_policy, [0] * (self.sim.H + 1 - len(new_policy))))
        #print("Old Policy - ", self.policy)
        self.policy = (1 - self.alpha) * self.policy + (self.alpha) * new_policy
        
        #self.policy = self.policy_maker(policy)
        self.history = np.vstack( (self.history, new_policy, self.policy))
        self.alpha_update()
        #print("New Policy - ", self.policy)
        
        
    def data_normalize(self, df):
        for i in df.columns:
            if 'delay_bucket' in i:
                df[i] = df[i] / self.sim.N
            elif i == 'cum_calls_made_at_start':
                df[i] = df[i] / self.sim.N
        #self.df['cum_bumps'] = df['cum_bumps'] / N
            elif i == 'last_calls':     
                df['last_calls'] = df['last_calls']/ self.sim.max_calls
            elif i == 'sum_cutoff_at_start':
                df['sum_cutoff_at_start'] = df['sum_cutoff_at_start'] / (self.sim.D * self.sim.N)
            elif i == 'time_since_last_call':
                df['time_since_last_call'] = df['time_since_last_call'] / (self.sim.max_wait)
            elif i == 't_rem':
                df['t_rem'] = df['t_rem'] / (self.sim.H)
            elif i == 'shifts_available':
                df['shifts_available'] = df['shifts_available'] / self.sim.M
            elif i == 'time_by_shifts_available':
                df['time_by_shifts_available'] = df['time_by_shifts_available'] * self.sim.M /self.sim.H
            elif i == 'empl_by_shifts_available':
                df['empl_by_shifts_available'] = df['empl_by_shifts_available'] *self.sim.M/ self.sim.N
            elif i == 'cum_calls_made_0.9':
                df['cum_calls_made_0.9'] = df['cum_calls_made_0.9']/self.sim.N
            elif i == 'last_response':
                df['last_response'] = df['last_response'] / self.sim.max_calls
            elif i == 'to_call':
                df['to_call'] = df['to_call'] / self.sim.max_calls
                
        return df
    
    def save_history(self, address):
        
        np.save(address + "history.npy", self.history)
    
        names = []
        for i in range(self.history.shape[0]):
            if i % 2 == 0: 
                names.append('policy' + str(i))
            #elif i % 3 == 1: 
            #    names.append('opolicy' + str(i))
            else:
                names.append('upolicy' +str(i))
        df = pd.DataFrame(self.history.transpose(), columns = names)
        df2 = df.cumsum()
        df2.columns = ['c' + col for col in df2.columns]
        df = pd.concat([df, df2], axis=1)
        df.index.name = "Time"
        df.to_csv(address + "/policy_history.csv")
            
    def get_curr_schedule(self, state, curr_bumps):
        self.sim.process_output()
        self.sim.save_instance_solution()
        #arrays = [np.arange(0, self.sim.H+1), np.arange(self.data_id, self.sim.H+1)]
        stats = pd.DataFrame(index = np.arange(0, self.sim.H+1),  columns = ["interacted", "sum_cutoff_at_start",
                                                                             "in_delay_at_start", "last_calls",
                                                                             "in_cutoff_at_end", "in_delay_at_end",
                                                                             "in_cutoff_at_start", "no_calls_for",
                                                                             "bumps", "continuous_calls", "curr_bumps",
                                                                             "time_since_last_call", "calls_made",
                                                                             "cum_calls_made_at_start", "last_response",
                                                                             'time_to_wait'] ).fillna(0)

        self.sim.Solution =  self.sim.Solution[self.sim.Solution['Call_At'] <= self.sim.H]
        temp = self.sim.Solution['Call_At'].value_counts()
        
        
        stats.loc[ list(temp.index), 'calls_made' ] = temp.values
        temp = self.sim.Solution['Response_At'].value_counts()
        stats.loc[ temp[temp.index <= self.sim.H].index, 'interacted' ] = temp[temp.index <= self.sim.H].values
        stats["cum_interacted_at_end"] = stats.interacted.cumsum()
        stats["cum_interacted_at_start"] =  stats["cum_interacted_at_end"] -  stats["interacted"] 
        for i in self.delay_bucket_intervals:
            stats['delay_bucket_' +str(i)] = 0

        #stats[ 'continuous_calls'] = 0


        for i in self.sim.Solution.index:
            ran = np.arange(self.sim.Solution.Call_At.loc[i] , min(self.sim.H, self.sim.Solution.Response_At.loc[i]) )
            stats.loc[ran, 'in_delay_at_end'] = stats.in_delay_at_end.loc[ran] + 1
            ran = np.arange(self.sim.Solution.Call_At.loc[i] , min(self.sim.H, self.sim.Solution.Cutoff_Time.loc[i]))
            stats.loc[ran, 'in_cutoff_at_end'] = stats.in_cutoff_at_end.loc[ran] + 1
            if self.sim.Solution.Response_At.loc[i] <= self.sim.H:
                stats.bumps.at[self.sim.Solution.Response_At.loc[i]] = stats.bumps.at[self.sim.Solution.Response_At.loc[i]] + self.sim.Solution.bumps_caused.loc[i]
                            
            
        for t in stats.index:
            self.sim.Solution.loc[:,'delay'] = self.sim.Solution.Actual_cutoff - t
            if t != 0:
                stats.loc[t, 'continuous_calls'] = stats.loc[t - 1, 'continuous_calls'] + 1 if stats.loc[ t - 1 , 'calls_made' ] > 0 else 0
                #stats.loc[t, 'break'] = stats.loc[t, 'calls_made'] == stats.loc[t - 1, 'calls_made']
                stats.loc[t, 'last_calls'] = stats.loc[t - 1, 'calls_made']
                stats.loc[t, 'no_calls_for'] = stats.loc[t - 1, 'no_calls_for'] + 1 if stats.loc[ t - 1 , 'calls_made' ] == 0 else 0
                
            temp = self.sim.Solution[(self.sim.Solution.Cutoff_Time >= t) & (self.sim.Solution.Call_At < t)] 
            
            #stats.loc[t, "sum_cutoff"] = sum(temp.delay) + stats.loc[ t, 'calls_made' ] * min(int(vals[4]), int(vals[2]) - t)#
            stats.loc[t, "sum_cutoff_at_start"] = sum(temp.delay) #
            #now = temp[temp.Call_At == t]
            #for tau in now.index:
            temp = self.sim.Solution[(self.sim.Solution.Cutoff_Time > t) & (self.sim.Solution.Call_At <= t)] 
            stats.loc[t, "sum_cutoff_at_end"] = sum(temp.delay) - len(temp) 
            
            if t != self.sim.H:
                for i in self.delay_bucket_intervals:
                    stats.loc[t, 'delay_bucket_' + str(i)] = sum(((self.sim.Solution.Cutoff_Time >= t) & (self.sim.Solution.Call_At >= t - (self.delay_bucket_intervals[i][0])) & (self.sim.Solution.Call_At < t - self.delay_bucket_intervals[i][1] )))
                    #stats.loc[t, 'delay_bucket_60'] = sum(((schedule.Cutoff_Time >= t) & (schedule.Call_At >= t - 59) & (schedule.Call_At < t - 9 )))
                    #stats.loc[t, 'delay_bucket_180'] = sum(((schedule.Cutoff_Time >= t) & (schedule.Call_At >= t - 179) & (schedule.Call_At < t - 59 )))
                    
                if stats.loc[t, 'calls_made'] > 0:
                    stats.loc[t, 'time_to_wait'] = 0
                elif stats.loc[t - 1, 'time_to_wait'] > 0:
                    stats.loc[t, 'time_to_wait'] = stats.loc[t - 1, 'time_to_wait'] - 1
                else:
                    i = 1
                    while  t + i <= self.sim.H and stats.loc[t, 'cum_interacted_at_start'] <= self.sim.M:
                        if stats.loc[t + i, 'calls_made'] > 0 :
                            break
                        else:
                            i += 1
                    stats.loc[t, 'time_to_wait'] = i 
                    
            if t != 0:
                stats.loc[ t , 'last_response' ] = stats.loc[ t , 'cum_interacted_at_start' ]  - stats.loc[ t - 1 , 'cum_interacted_at_start' ] 
                if stats.loc[ t - 1 , 'calls_made' ] > 0:
                    stats.loc[ t , 'time_since_last_call' ] = 1
                    
                else:
                    stats.loc[ t , 'time_since_last_call' ] = stats.loc[ t - 1 , 'time_since_last_call' ] + 1
                
                stats.loc[t, "in_delay_at_start"] = stats.loc[t - 1 ,"in_delay_at_end"]
                stats.loc[t, "in_cutoff_at_start"] = stats.loc[t - 1 ,"in_cutoff_at_end"]
                    
        # stats["sum_cutoff_at_end"] = stats[ "sum_cutoff_at_end"].clip(0, int(vals[1])*int(vals[4]))
        stats['sum_cutoff_at_end'] =  stats['sum_cutoff_at_end'].clip(0, self.sim.N * self.sim.H) 
        stats["cum_calls_made_at_start"] = stats.calls_made.cumsum() - stats.calls_made
        stats['cum_calls_made_at_start'] =  stats['cum_calls_made_at_start'].clip(0, self.sim.N) 
        
          
        stats['shifts_available'] =  self.sim.M - stats['cum_interacted_at_start']
        stats['shifts_available'] =  stats['shifts_available'].clip(0, self.sim.M) 
                
        stats["cum_calls_made"] = stats.calls_made.cumsum()        
        # stats["in_cutoff"] =  stats["in_cutoff_at_start"] + stats["calls_made"] #
        # stats["in_delay"] =  stats["in_cutoff_at_start"] +  stats["calls_made"] #
        # stats["cum_bumps_at_start"] = stats.bumps.cumsum() - stats.bumps
        # stats["cum_bumps"] = stats.bumps.cumsum() 

        
        stats["t_rem"] = self.sim.H - stats.index
        
        
        #stats = stats[stats.index >= state['time']]
        stats.loc[state['time'] ,"curr_bumps"] = curr_bumps
        stats['data_id'] = self.data_id
        stats['time'] = stats.index
        stats.index.name = "t"
        stats = stats.merge(self.ocp, how = 'left', left_index = True, right_index = True)
        #if 96 in stats.columns:
        #    print("here")
        
        stats = stats.set_index('data_id', append = True)
        stats = stats[stats.shifts_available > 0]
        
        return stats
        
    def add_trajectory(self, state, stats):
        stats = stats.loc[state['time']: state['time'] + self.k_cap - 1]
        #stats.index = stats.index.set_levels(stats.index.levels[1] + 1, level=1)
        self.data = pd.concat([self.data, stats])   

        
    def get_sim_data(self, _id):
        stats = pd.DataFrame(index = np.arange(0, self.sim.H+1),  columns = ["in_delay_at_start",  "in_cutoff_at_end",
                                                                             "in_delay_at_end", "interacted",
                                                                             "in_cutoff_at_start",
                                                                             "bumps", "cum_bumps",
                                                                             "calls_made", "cum_interacted_at_start",
                                                                             "cum_calls_made_at_start", "time_since_last_call",
                                                                             "time_to_wait"] ).fillna(0)

        schedule = pd.DataFrame.from_dict(self.sim.schedule, orient='index')
        schedule['Actual_cutoff'] = schedule['Call_At'] + int(self.sim.D)
        schedule['Actual_cutoff'] = schedule['Actual_cutoff'].clip(0, self.sim.H)
        self.schedule = schedule
        temp = schedule['Call_At'].value_counts()
        stats.loc[ list(temp.index), 'calls_made' ] = temp.values
        temp = schedule['Response_At'].value_counts()
        stats.loc[ temp[temp.index <= self.sim.H].index, 'interacted' ] = temp[temp.index <= self.sim.H].values
        
        for i in self.delay_bucket_intervals:
            stats['delay_bucket_' +str(i)] = 0
        stats['last_calls'] = 0 
        
        
        for i in schedule.index:
            ran = np.arange(schedule.Call_At.loc[i] , min(self.sim.H, schedule.Response_At.loc[i]) )
            stats.loc[ran, 'in_delay_at_end'] = stats.in_delay_at_end.loc[ran] + 1
            ran = np.arange(schedule.Call_At.loc[i] , min(self.sim.H, schedule.Cutoff_Time.loc[i]))
            stats.loc[ran, 'in_cutoff_at_end'] = stats.in_cutoff_at_end.loc[ran] + 1
            if schedule.Response_At.loc[i] <= self.sim.H:
                stats.bumps.at[schedule.Response_At.loc[i]] = stats.bumps.at[schedule.Response_At.loc[i]] + schedule.bumps_caused.loc[i]
                            
            
        for t in stats.index:
            schedule.loc[:,'delay'] = schedule.Actual_cutoff - t
            
            if stats.loc[t, 'calls_made'] > 0:
                stats.loc[t, 'time_to_wait'] = 0
            elif stats.loc[t - 1, 'time_to_wait'] > 0:
                stats.loc[t, 'time_to_wait'] = stats.loc[t - 1, 'time_to_wait'] - 1
            else:
                i = 1
                while t + i <= self.sim.H and stats.loc[t, 'cum_interacted_at_start'] <= self.sim.M:
                    if stats.loc[t + i, 'calls_made'] > 0:
                        break
                    else:
                        i += 1
                stats.loc[t, 'time_to_wait'] = i 
            
            if t != 0:
                stats.loc[t, "in_delay_at_start"] = stats.loc[t - 1 ,"in_delay_at_end"]
                stats.loc[t, "in_cutoff_at_start"] = stats.loc[t - 1 ,"in_cutoff_at_end"]
                stats.loc[t, 'last_calls'] = stats.loc[t - 1, 'calls_made']
                
                if stats.loc[ t - 1 , 'calls_made' ] > 0:
                    stats.loc[ t , 'time_since_last_call' ] = 1
                else:
                    stats.loc[ t , 'time_since_last_call' ] = stats.loc[ t - 1 , 'time_since_last_call' ] + 1
                    
            temp = schedule[(schedule.Cutoff_Time >= t) & (schedule.Call_At < t)] 
            
            #stats.loc[t, "sum_cutoff"] = sum(temp.delay) + stats.loc[ t, 'calls_made' ] * min(int(vals[4]), int(vals[2]) - t)#
            stats.loc[t, "sum_cutoff_at_start"] = sum(temp.delay)
                
            if t != self.sim.H:
                for i in self.delay_bucket_intervals:
                    stats.loc[t, 'delay_bucket_' + str(i)] = sum(((schedule.Cutoff_Time >= t) & (schedule.Call_At >= t - (self.delay_bucket_intervals[i][0])) & (schedule.Call_At < t - self.delay_bucket_intervals[i][1] )))

            
        stats["cum_calls_made_at_start"] = stats.calls_made.cumsum() - stats.calls_made
        stats['cum_calls_made_at_start'] =  stats['cum_calls_made_at_start'].clip(0, self.sim.N) 
        stats["cum_interacted_at_end"] = stats.interacted.cumsum()
        stats["cum_interacted_at_start"] =  stats["cum_interacted_at_end"] -  stats["interacted"]    
        stats['shifts_available'] =  self.sim.M - stats['cum_interacted_at_start']
        stats['shifts_available'] =  stats['shifts_available'].clip(0, self.sim.M) 
        stats["cum_bumps"] = stats.bumps.cumsum()
        stats["cum_calls_made"] = stats.calls_made.cumsum()        
        
        stats["t_rem"] = self.sim.H - stats.index
        stats['time_by_shifts_available'] = stats.t_rem /(stats.shifts_available )
        stats['empl_by_shifts_available'] = (self.sim.N - stats.cum_calls_made_at_start) /(stats.shifts_available)
        #stats.index.name = "time"
        stats['run'] = _id
        stats['time'] = stats.index
        
        self.pred_history['time'] = self.pred_history['time'].astype(int)
        self.pred_history['run'] = self.pred_history['run'].astype(int)
        
        stats =  stats.merge(self.pred_history, on = ['time', 'run'], how = 'left')
        self.sim_stats = pd.concat([self.sim_stats, stats])
        

    def save_model(self, folder):
        self.policy.save_model(folder)
        
    def alpha_update(self):
        self.alpha = min(self.alpha * 0.95, 0.02)
        


class Xgboost():
    def __init__(self, sim, feature_names, config, data = pd.DataFrame(), folder = None):
       self.feature_names = feature_names
       self.model_stats = pd.DataFrame()
       #self.actions = [i for i in range(1, sim.max_calls + 1)]
       self.max_action = sim.max_calls
       self.class_weight = config['class_weight']
       self.temporal_weight = config['temporal_weight']
       self.state_counter = []
       self.model = []
       self.featur_imp = pd.DataFrame()
       
       self.target = config['target']
       if config['target'] == 'best_action' :  
           if config['max_calls'] == 1:
               self.num_class = 1
           else:
               self.num_class = sim.max_calls + 1  
               
           self.max_pred = sim.max_calls
           self.model_count = 1
           
           if config['type'] == 'reg':
               self.get_action = self.reg_call_cat
           elif config['type'] == 'class':
               self.get_action = self.class_call_cat
            
           
       elif config['target'] == 'called':
           self.num_class = 1 
           self.max_pred = sim.max_calls
           self.get_action = self.called_action
           self.model_count = sim.max_calls
           
       elif config['target'] == 'cum_calls_made':
           self.max_pred = sim.N
           self.get_action = self.cum_calls_made_action
           self.model_count = 1
           
       elif config['target'] == 'time_to_wait':
               self.max_pred = config['max_wait']
               self.get_action = self.time_to_wait_action
               self.model_count = 1
       if len(data):
           self.df = data
           #self.df = self.df.iloc[:1000]

           if config['type'] == 'reg':
               for i in range(self.model_count):
                   self.model.append(xgb.XGBRegressor(n_estimators=1000,
                                     max_depth=7,
                                     eta=0.3,
                                     subsample=0.7,
                                     eval_metric=config['eval_metric'],
                                     objective = config['objective'],
                                     colsample_bytree=0.8))
                   #print("classes ",self.num_class)
                   self.fit_model(self.df, i)
                   
               
           elif config['type'] == 'class':
               for i in range(self.model_count):
                   self.model.append(xgb.XGBClassifier( learning_rate = 0.3,
                                    n_estimators  = 800,
                                    max_depth     = 8,
                                    eval_metric=config['eval_metric'],
                                    #eta = 0.1,
                                    colsample_bytree= 1,
                                    objective= config['objective'],
                                    subsample= 1.0,
                                    min_child_weight= 4))
                   data = pd.DataFrame()
                   if self.target == 'called':  
                       data = self.df.loc[self.df['best_action'] >= i].copy()
                       data.loc[:,'called'] = (data['best_action'] > i).astype('uint8')
                   else:
                       data = self.df
                   
           
                   self.fit_model(data, i)
           print('Number of models - ', self.model_count)
           self.save_model(folder, '-1')
           self.save_data(folder, empty=True)
           
       if not config['fit'] and not config['train']:
           self.load_model(folder, config)         
       pass
       
    def splitDataTarget(self, data):       
        target = data[self.target]
        return data[self.feature_names], target
    
    def update_model(self, df, folder, n = ''):
        
        #df = self.normalize(df, sim)
        self.df = pd.read_csv(folder + "All_data.csv")
        #print(self.df.dtypes)
        self.df = self.df[ self.feature_names + [ 'cum_calls_made', 'best_action', 'iteration',
       'no_calls_for', 'continuous_calls', 'curr_bumps',
       'run', 'time', 'policy_action', 'acted', 'decision_rule', 'branch']]#, 'time_to_wait'
        df['iteration'] = n
        #print(self.df.dtypes, df.dtypes)
        self.df = pd.concat([self.df, df])    
        #print(self.df.dtypes)
        print("New Data Size = ", len(self.df))
        #self.save_data(folder)
        
        
        
        
        for i in range(self.model_count):
            data = pd.DataFrame()
            if self.target == 'called':  
                data = self.df.loc[self.df['best_action'] >= i].copy()
                data.loc[:,'called'] = (data['best_action'] > i).astype('uint8')
            else:
                data = self.df
            
            self.fit_model(data, i)
            
        self.save_data(folder)
        self.save_model(folder, n)
        self.df = pd.DataFrame()
        pd.DataFrame(self.state_counter).to_csv( folder + "state_counter.csv")
        
            
    def fit_model(self, data, model_id):
        

        t_counts = 1000/data["time"].value_counts()
    
        wt = t_counts.to_dict()
        wt = data["time"].map(wt)
        #print(wt, data['time'])
        
        if self.class_weight:
            called = data[self.target] > 0
            class_weight = compute_class_weight('balanced', classes=np.unique(called), y = called)
            
            wt1 = {i:class_weight[i] for i in range(self.max_action + 1)}
            wt1 = data[self.target].map(wt1)
        
            wt = wt * wt1
            
        if self.temporal_weight:
            wt1 = { i : 1 / (0.05 * np.exp( (model_id - (i+1)) * 0.08) + 0.95) for i in range(min(data.iteration),  max(data.iteration + 1))}
            wt1 = data['iteration'].map(wt1)
            wt = wt * wt1

        X, y = self.splitDataTarget(data)
        #print(X.dtypes)
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)
        #print(y.unique())

        self.state_counter.append(len(X.drop_duplicates()))
        print("Fitting XGB model :- ", model_id, len(data))
        self.model[model_id].fit(X, y, verbose=True, sample_weight=wt)
        predict_train = self.model[model_id].predict(X)
        
        self.get_importance(model_id)
        

        #explainer = shap.TreeExplainer(self.model)
        #shap_values = explainer.shap_values(X)
        #plot = shap.summary_plot(shap_values, X, plot_type="bar", feature_names = ['tr', 'db1', 'db2', 'db3', 'db4', 'db5',
        #                                   'db6', 'tslc', 'sv',  'cl ', 'la', 'rtbysv','cc'], show=False)
        
        #['Time remaining', 'Delay bucket 1', 'Delay bucket 2',
        #                 'Delay bucket 3', 'Delay bucket 4', 'Delay bucket 5',
        #                 'Delay bucket 6', 'Time since last call', 'Shifts vacant',
        #                 'Cumulative leftover cutoff of employees in delay ',
        #                 'Action at last epoch', 'Ratio of time remaining and shifts vacant',
        #                 'Cumulative employees called ']
        #plt.savefig("Sense_ana.png",dpi=150, bbox_inches='tight')
        
        #plot = shap.summary_plot(shap_values, X,
        #                  feature_names = ['tr', 'db1', 'db2', 'db3', 'db4', 'db5',
        #                                   'db6', 'tslc', 'sv',   'cl ', 'la', 'rtbysv',    'cc'],  show=False)

        #plt.savefig("Sense_ana.png",dpi=150, bbox_inches='tight')
        
        #self.df['predict'] = predict_train
        #self.df['predict'] = self.df['predict'].clip(0, self.max_pred)
        

        # dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True, feature_names=X_test.columns)
        #predict_test = model.predict(X_test)

        print('\t\ttrain dataset score {:.4f} '.format(self.model[model_id].score(X, y)))
        #print('\t\ttest dataset score {:.4f} '.format(model.score(X_test, y_test)))
        
        if isinstance(self.model[model_id], xgb.XGBClassifier):
            print("Classification Report")
            print(classification_report(y, predict_train))
            
            stats = {'precision_micro':precision_score(y, predict_train, average = 'micro'),
                     'precision_macro':precision_score(y, predict_train, average = 'macro'),
                     'recall_micro':recall_score(y, predict_train, average = 'micro'),
                     'recall_macro':recall_score(y, predict_train, average = 'macro'),
                     'accuracy': accuracy_score(y, predict_train)}
            
            if self.target == 'called':
                stats['precision_binary']  = precision_score(y, predict_train, average = 'binary')
                stats['recall_binary'] = recall_score(y, predict_train, average = 'binary')
                
            
            
        if isinstance(self.model[model_id], xgb.XGBRegressor):
            print("Regression Report")
            print("MAE - ", mean_absolute_error(y, predict_train))
            print("MSE - ", mean_squared_error(y, predict_train))
            
            stats = {'mae':mean_absolute_error(y, predict_train),
                     'mse':mean_squared_error(y, predict_train)}
        stats['data'] = len(data)
        df_new = pd.DataFrame([stats])  
        self.model_stats = pd.concat([self.model_stats, df_new])
        
    def get_importance(self, model_id):
        imp = self.model[model_id].get_booster().get_score(importance_type = "gain")
        print(self.model[model_id].get_booster().get_score(importance_type = "weight"))
        print(imp)
        #plot_importance(self.model, max_num_features=10) # top 10 most important features
        
        imp = pd.DataFrame([imp])
        self.featur_imp = pd.concat([self.featur_imp, imp], ignore_index=True)
        
        #plt.show()
        
    def class_call_cat(self, x, state):
        #self.model.predict_proba(pd.DataFrame.from_dict([x]))
        prob = self.model[0].predict_proba(x)[0]
        action = self.action_fn(prob)
        #print(x.values, int(action))
        return int( action ) , prob #+ bernoulli.rvs(p)
    
    def reg_call_cat(self, x, state):
        #self.model.predict_proba(pd.DataFrame.from_dict([x]))
        pred = self.model[0].predict(x)
        p = np.clip(math.modf(pred)[0], 0, 1)
        action = np.round(pred[0])#np.round(pred[0])
       
        action = int( action ) #+ p > 0.5#bernoulli.rvs(p)
        action = np.clip(action, 0, self.max_action)
        return action, pred #+ bernoulli.rvs(p)
        
    def called_action(self, x, state):
        prob = []
        action = 0
        for i in range(self.model_count):
            pred = self.model[i].predict_proba(x)[0]
            prob.extend(list(pred))
        
            choice = self.random_action(pred)
            if choice == 0:
                break
            action += choice
        
        prob = prob + [0] * (2 * self.max_action - len(prob))
        
        return action , prob
    
    
    def cum_calls_made_action(self, x, state):
        pred = np.round(np.clip(self.model[0].predict(x)[0], 0, self.max_pred), 3)
        action = np.clip(int(pred - state['last_called'] - 1), 0, self.max_action)
        return action, np.array([pred])
    
    def time_to_wait_action(self, x, state):
        pred = np.round(np.clip(self.model[0].predict(x)[0], 0, self.max_pred), 3)
        wait = int(pred)
        return wait, np.array([pred])
        
    def random_action(self, prob):
        return np.random.choice(len(prob) , p=prob)
    
    def quantile_action(self, prob):
        cum = np.cumsum(prob)
        return list(np.array(cum) > self.prob_threshold).index(True)
    
    def mean_action(self, prob):
        mean = np.round(sum([i * prob[i] for i in range(len(prob))]), 3)
        
        p = math.modf(mean)[0]
        return int(mean) + bernoulli.rvs(p)
    
    def max_prob_action(self, prob):
        return np.argmax(prob)
            
    
    def save_data(self, folder, empty = False):
        if empty:
            self.df[:0].reset_index(drop = True).to_csv(folder + "All_data.csv")
        else:
            self.df.reset_index(drop = True).to_csv(folder + "All_data.csv")
        self.model_stats.to_csv(folder + "Model_Stats.csv")
        self.featur_imp.to_csv(folder + "Feature_Imp.csv")
        #sys.exit()
    
    
    def save_model(self, folder, n):
        for i in range(self.model_count):
            #pickle.dump(self.model[i], open(folder + "xgb_" + str(i) + '_' + str(n) +".pkl", "wb"))
            self.model[i].save_model(folder + "xgb_" + str(i) + '_' + str(n) +".json")


    def load_model(self, folder, config):
        n = config['model']
        for i in range(self.model_count):
            if config['type'] == 'reg':
                self.model.append(xgb.XGBRegressor())
            else:
                self.model.append(xgb.XGBClassifier())
            self.model[i].load_model(folder + "xgb_" + str(i) + '_' +  str(n) +".json")
            #self.model.append(pickle.load(open(folder + "xgb_" + str(i) + '_' +  str(n) +".pkl", "rb")))
        #self.model = obj.model
        
        
class CORN():
    def __init__(self, sim, feature_names, config, data = pd.DataFrame(), folder = None):
       self.feature_names = feature_names
       self.model_stats = pd.DataFrame()
       #self.actions = [i for i in range(1, sim.max_calls + 1)]
       self.max_action = sim.max_calls
       
       self.target = config['target']
       self.learning_rate = config['learning_rate']
       self.num_class = sim.max_calls + 1  
       self.max_pred = sim.max_calls
       self.num_epochs = 150
       self.batch_size = 128
       self.wt = config['wt']
       self.prob = config['prob_threshold']
       self.class_weight = config['class_weight']
       self.mean_action = config['ext'] == 'class_corn_mean'
       self.is_train = config['train']
       
       self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       print('Training on', self.DEVICE)

       if len(data) and config['train']:
           self.df = data
           #self.df = self.df.iloc[:1000]
           if not config['fit']:
               self.model = MLP(in_features=len(self.feature_names), num_classes = self.num_class)
               self.model.to(self.DEVICE)
               self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
               self.fit_model()
               self.save_model(folder, '-1')
               self.save_data(folder)
           elif config['fit']:
               self.load_model(folder, config['load_model'])
       if not config['fit'] and not config['train']:
           self.load_model(folder, config['model'])
               
       pass
       
    def splitDataTarget(self):       
        target = self.df[self.target]
        return self.df[self.feature_names], target
    
    def update_model(self, df, folder, n = ''):
        
        #df = self.normalize(df, sim)
        #self.df = pd.read_csv(folder + "All_data.csv")
        #self.df = self.df[ self.feature_names + ['cum_calls_made', 'best_action', 'iteration',
        #                                         'called', 'predict', 'no_calls_for', 'continuous_calls', 'curr_bumps',
        #                                         'action', 'run']]
        df['iteration'] = n
        self.df = pd.concat([self.df, df])    
        print("New Data Size = ", len(self.df))

        
        self.fit_model()
        self.save_data(folder)
        self.save_model(folder, n)
        #self.df = pd.DataFrame()
        
    def get_weights(self):
        self.df['t_rem_actual'] = (self.df.t_rem * 360).round().astype(int)
        self.df['t_rem_actual'] = self.df['t_rem_actual'].astype(int) 
        t_counts = self.wt/self.df["t_rem_actual"].value_counts()
        
        wt = t_counts.to_dict()
        wt = self.df["t_rem_actual"].map(wt)
        return wt
    
    
            
    def fit_model(self):
        print("Fitting NN model.")
        
        wt = self.get_weights()

        X, y = self.splitDataTarget()
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)
        #print(y.unique())
        
        wt = wt.values
        X = X.values
        y = y.values
        
        train_dataset = MyDataset(X, y, wt)
        cw = None
        if self.class_weight:
            cw = self.get_class_weight(y)
        
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True, # want to shuffle the dataset
                                  num_workers=0)

        self.train(train_loader, cw)


        logits = self.model(torch.tensor(X).float())
        predict_train = self.corn_label_from_logits(logits)

        self.df['predict'] = predict_train
           
        self.df['predict'] = self.df['predict'].clip(0, self.max_pred)
        #print("Classification Report")
        #print(classification_report(y, predict_train))
            
        # stats = {'precision_micro':precision_score(y, predict_train, average = 'micro'),
        #              'precision_macro':precision_score(y, predict_train, average = 'macro'),
        #              'recall_micro':recall_score(y, predict_train, average = 'micro'),
        #              'recall_macro':recall_score(y, predict_train, average = 'macro'),
        #              'accuracy': accuracy_score(y, predict_train)}

        # df_new = pd.DataFrame([stats])  
        
        # #df_new['called_sum'] = sum(self.df['called'])
        # self.model_stats = pd.concat([self.model_stats, df_new])
        
        
    def corn_label_from_logits(self, logits, return_prob = False):
        """
        Returns the predicted rank label from logits for a
        network trained via the CORN loss.

        Parameters
        ----------
        logits : torch.tensor, shape=(n_examples, n_classes)
            Torch tensor consisting of logits returned by the
            neural net.

        Returns
        ----------
        labels : torch.tensor, shape=(n_examples)
            Integer tensor containing the predicted rank (class) labels


        Examples
        ----------
        >>> # 2 training examples, 5 classes
        >>> logits = torch.tensor([[14.152, -6.1942, 0.47710, 0.96850],
        ...                        [65.667, 0.303, 11.500, -4.524]])
        >>> corn_label_from_logits(logits)
        tensor([1, 3])
        """
        probas = torch.sigmoid(logits)
        probas = torch.cumprod(probas, dim=1)
        
        
        predict_levels = probas > self.prob
        predicted_labels = torch.sum(predict_levels, dim=1)
    
        if self.mean_action:
            predicted_labels = self.get_mean_action(probas)
        
        #print(predicted_labels)
        if return_prob == False:
            return predicted_labels
        else:
            return predicted_labels, probas
        
        
    def get_mean_action(self, prob):
        
        def f(prob):
            action = self.max_action
            for k in range(len(prob)-1,-1,-1):
                action = prob[k]* action + (1-prob[k])*(k)
            return action
        
        predicted_labels = np.array(list(map(f, prob.detach().numpy())))
        
        if not self.is_train:
            p = np.modf(predicted_labels)[0]
            predicted_labels = np.floor(predicted_labels).astype(int) + bernoulli.rvs(p)
            
        return predicted_labels
        
        
    def train(self, train_loader, cw):
        
        self.model = self.model.train()
        for epoch in range(self.num_epochs):
            
            
            for batch_idx, (features, class_labels, weight) in enumerate(train_loader):

                class_labels = class_labels.to(self.DEVICE)
                features = features.to(self.DEVICE)
                weight = weight.to(self.DEVICE)
                if features.size()[0] <= 1:
                    continue
                logits = self.model(features)
                
                #### CORN loss 
                loss = self.corn_loss(logits, class_labels, self.num_class, weight, cw)
                ###--------------------------------------------------------------------###   
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()
                
                ### LOGGING
                if not batch_idx % 1000:
                    print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
                           %(epoch+1, self.num_epochs, batch_idx, 
                             len(train_loader), loss))
                    
    def get_class_weight(self, y):
        # wt = []
        # y_subset = np.array(y)
        # for i in range(self.num_class - 1):
        #     temp = y_subset > i
        #     if i == 0:
        #         class_weight = compute_class_weight('balanced', classes=np.unique(temp), y = temp)
        #     else:
        #         class_weight = np.ones(2)
            
        #     wt.append(class_weight)
        #     y_subset = y_subset[temp]
        temp = y > 0
        class_weight = compute_class_weight('balanced', classes=np.unique(temp), y = temp)
        wt = np.ones((self.num_class,2))
        wt[0, 0] = class_weight[0]
        wt[0, 1] = class_weight[1]
        # for i in range(1,self.num_class):
        #     wt[i, 0] = class_weight[1]
        #     wt[i, 1] = class_weight[1]
        return wt
    
    
    def corn_loss(self, logits, y_train, num_classes, importance_weights=None, class_weight = None):
        """Computes the CORN loss described in our forthcoming
        'Deep Neural Networks for Rank Consistent Ordinal
        Regression based on Conditional Probabilities'
        manuscript.

        Parameters
        ----------
        logits : torch.tensor, shape=(num_examples, num_classes-1)
            Outputs of the CORN layer.

        y_train : torch.tensor, shape=(num_examples)
            Torch tensor containing the class labels.

        num_classes : int
            Number of unique class labels (class labels should start at 0).

        Returns
        ----------
            loss : torch.tensor
            A torch.tensor containing a single loss value.

        Examples
        ----------
        >>> import torch
        >>> from coral_pytorch.losses import corn_loss
        >>> # Consider 8 training examples
        >>> _  = torch.manual_seed(123)
        >>> X_train = torch.rand(8, 99)
        >>> y_train = torch.tensor([0, 1, 2, 2, 2, 3, 4, 4])
        >>> NUM_CLASSES = 5
        >>> #
        >>> #
        >>> # def __init__(self):
        >>> corn_net = torch.nn.Linear(99, NUM_CLASSES-1)
        >>> #
        >>> #
        >>> # def forward(self, X_train):
        >>> logits = corn_net(X_train)
        >>> logits.shape
        torch.Size([8, 4])
        >>> corn_loss(logits, y_train, NUM_CLASSES)
        tensor(0.7127, grad_fn=<DivBackward0>)
        """
        
        if importance_weights is None:
            importance_weights = torch.ones(len(y_train))
        
        if class_weight is None:
            class_weight = torch.ones(num_classes, 2)

        
        sets = []
        for i in range(num_classes-1):
            label_mask = y_train > i-1
            label_tensor = (y_train[label_mask] > i).to(torch.int64)
            label_weight = (importance_weights[label_mask])
            sets.append((label_mask, label_tensor, label_weight))

        num_examples = 0
        losses = 0.
        for task_index, s in enumerate(sets):
            train_examples = s[0]
            train_labels = s[1]
            train_weight = s[2]

            if len(train_labels) < 1:
                continue

            num_examples += len(train_labels)
            pred = logits[train_examples, task_index]

            loss = -torch.sum((F.logsigmoid(pred) * train_labels * class_weight[task_index][1] + (F.logsigmoid(pred) - pred)*(1-train_labels)) * train_weight * class_weight[task_index][0])
            losses += loss

        return losses/num_examples

    
    def get_action(self, x):
        #self.model.predict_proba(pd.DataFrame.from_dict([x]))
        self.model.eval()
        logits = self.model.forward(torch.tensor(x.values).float())

        action, probas = self.corn_label_from_logits(logits, return_prob=True)
        p = math.modf(action)[0]
        return int( action ) + bernoulli.rvs(p) , probas.tolist()[0] #+ bernoulli.rvs(p)
            
    
    def save_data(self, folder):
        self.df.to_csv(folder + "All_data.csv")
        self.model_stats.to_csv(folder + "Model_Stats.csv")
        #sys.exit()
    
    
    def save_model(self, folder, n):
        #pickle.dump(self.model, open(folder + "corn" + str(n) +".pkl", "wb"))
        torch.save(self.model.state_dict(), folder + "corn" + str(n) )


    def load_model(self, folder, n = '0'):
        #self.model = pickle.load(open(folder + "corn" + str(n) +".pkl", "rb"))
        
        self.model = MLP(in_features=len(self.feature_names), num_classes = self.num_class)
        self.model.load_state_dict(torch.load(folder + "corn" + str(n) ))


class MLP(torch.nn.Module):

    def __init__(self, in_features, num_classes, num_hidden_1=300, num_hidden_2=300):
        super().__init__()
        
        self.my_network = torch.nn.Sequential(
            
            # 1st hidden layer
            torch.nn.Linear(in_features, num_hidden_1, bias=False),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.BatchNorm1d(num_hidden_1),
            
            # 2nd hidden layer
            torch.nn.Linear(num_hidden_1, num_hidden_2, bias=False),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.BatchNorm1d(num_hidden_2),
            
            # 3rd hidden layer
            # torch.nn.Linear(num_hidden_2, num_hidden_3, bias=False),
            # torch.nn.LeakyReLU(),
            # torch.nn.Dropout(0.2),
            # torch.nn.BatchNorm1d(num_hidden_3),
            
            ### Specify CORN layer
            torch.nn.Linear(num_hidden_2, (num_classes-1))
            ###--------------------------------------------------------------------###
        )
                
    def forward(self, x):
        logits = self.my_network(x)
        return logits
    
class MyDataset(Dataset):

    def __init__(self, feature_array, label_array, weights_array, dtype=np.float32):
    
        self.features = feature_array.astype(np.float32)
        self.labels = label_array
        self.weights = weights_array

    def __getitem__(self, index):
        inputs = self.features[index]
        label = self.labels[index]
        weight = self.weights[index]
        return inputs, label, weight

    def __len__(self):
        return self.labels.shape[0]
    
class Supervised_NN(nn.Module):
    def __init__(self, feature_names, hidden_size, output_size, target_name = 'best_action', data = pd.DataFrame()):
       super(Supervised_NN, self).__init__()
       self.feature_names = feature_names
       self.target_name = target_name
       self.num_class = output_size 
       self.fc1 = nn.Linear(len(feature_names), hidden_size)
       self.relu = nn.ReLU()
       self.fc2 = nn.Linear(hidden_size, output_size)
       self.sigmoid = nn.Sigmoid()
       self.softmax = nn.Softmax(dim = 1)
       
       self.loss_fn = nn.CrossEntropyLoss()
       self.optimizer = optim.Adam(self.parameters(), lr=0.001)
       self.num_epochs = 100
       self.batch_size = 64
       self.threshold = 0.5
      
       
       if len(data):
           self.df = data
           #self.create_data(data)
           self.update_model(data)
             
       
    def forward(self, x):
       x = self.fc1(x)
       x = self.sigmoid(x)
       x = self.fc2(x)
       x = self.softmax(x)
       return x
   

   
    # def create_data(self, data):

    #     self.dataset = ConcatDataset([self.dataset, TensorDataset(torch.tensor(data[self.feature_names].values), y)])
    #     self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
   
    def update_model(self, data):
        #data = self.df.sample(n = self.batch_size)
        data = torch.tensor(self.df[self.feature_names].values,  dtype = float32)
        y = torch.zeros((len(data), self.num_class), dtype = float32)
        for i in range(self.num_class):
            y[:, i] = torch.tensor(self.df['best_action'].values == i, dtype = float32)
            
            
        for epoch in range(self.num_epochs):
            #for batch_inputs, batch_labels in self.dataloader:
            # Zero the gradients
            idx = np.random.randint(len(self.df), size=self.batch_size)
            batch_inputs = data[idx]
            batch_labels = y[idx]
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.forward(batch_inputs)
            
            # Compute the loss
            loss = self.loss_fn(outputs, batch_labels)
            
            # Backpropagation
            loss.backward()
        
            # Update the weights
            self.optimizer.step()
            
            if (epoch + 1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}')
                
        return
        
    def get_action(self, x):
        x = x.values
        x = torch.tensor(x, dtype = float32)
        output = self.forward(x)
        status = np.where(output < self.threshold)[0]
        
        if len(status) == 0:
            return 0
        return status[0]