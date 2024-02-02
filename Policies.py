#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 06:06:32 2023

@author: Prakash
"""

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from RL_modules import NN_Learning_modules

class Policy():
    
    def __init__(self, sim,  policy_name = 'Cutoff_buffer' ):
        self.policy_name = policy_name
        self.sim = sim
        self.sim_stats = pd.DataFrame()
        self.delay_bucket_intervals = {1:[10, 0], 2:[30, 10], 3:[60, 30], 4:[150, 60], 5: [170, 150], 6:[180, 170]} #if D == 180 else {1:[10, 0], 2:[30, 10], 3:[60, 30], 4:[90, 60], 5: [110, 90], 6:[120, 110]}

    
    
    def set_policy_funct(self, policy_at,  param = None):
        self.call_func = pd.read_csv(policy_at)

        self.call_func = self.call_func[(self.call_func.dist == self.sim.dist_name) & (self.call_func.D == self.sim.D) &
                                            (self.call_func.N == self.sim.N) & (self.call_func.H == self.sim.H) & (self.call_func.Q == self.sim.q) ]
        self.call_func.index = self.call_func.time
        self.call_all_at = 1
        
        
        
        #self.model_loc = 0 
        if self.policy_name == "Cutoff_buffer":
            self.cutoff_buffer = param[0]
            self.policy_funct = self.cutoff_buffer_policy
            

        elif self.policy_name == 'Call_and_Wait':
            self.call_X = param[0]
            self.wait_t = param[1]
            self.policy_funct = self.call_and_wait
        
    
            
        elif self.policy_name == 'Deterministic_call_policy':
            self.policy_funct = self.deterministic_call_policy
            #self.acc = param[0]
            #self.call_func = self.call_func[self.call_func.y == 'cum_calls_made']
            #self.t_func = self.call_func.t_min.values
            #self.set_model()
            self.threshold = self.call_func['cum_calls_made' + '_' + str(param[0])].values

          

        elif self.policy_name == 'Deterministic_sum_cutoff_data_policy':
            self.policy_funct = self.deterministic_sum_cutoff_data_policy
            #self.acc = param[0]
            
            self.threshold = self.call_func['sum_cutoff' + '_' + str(param[0])].values
            #self.set_model() 
            #sum_cutoff_at_end
            
        elif self.policy_name == 'Deterministic_cutoff_policy':
            self.policy_funct = self.deterministic_cutoff_policy

            #self.acc = param[0]
            self.threshold = self.call_func['in_cutoff' + '_' + str(param[0])].values
            #in_cutoff_at_end
            #self.set_model()
                        
        elif self.policy_name == 'Cutoff_time_buffer':
            self.policy_funct = self.cutoff_time_buffer
            self.max_num_shift = param[0]
            self.allowed_cutoff = param[1]
            
        elif self.policy_name == 'Linear':
            self.policy_funct = self.linear_policy
            self.policy = param
            
                
        elif self.policy_name == 'random':
            self.policy_funct = self.random
            self.rand_dist = param[0]
            self.max_calls = param[1]
            self.interval = param[2]
            
        elif self.policy_name == 'call_all':
            self.policy_funct = self.call_all
            self.call = param[0]
         
        elif self.policy_name == 'Call_late':
            self.policy_funct = self.call_late
            self.call = param[0]
        
            
        elif self.policy_name == 'NN':
            self.d = NN_Learning_modules(self.sim, self.policy_name)
            self.d.load_model(param[0])
            self.policy_funct = self.d.get_best_decision

            
        elif self.policy_name == 'Linear_Regression':
            self.policy_funct = self.learned_model
            self.model = pickle.load(open(param[0], 'rb'))
            
            temp = [0] + [ int(f.split("_")[-1]) for f in self.model.feature_names_in_ if "delay_bucket" in f]
            self.feature_names = self.model.feature_names_in_ 
            self.delay_bucket_intervals = {temp[i+1] : temp[i] for i in range(len(temp) - 1)}
            self.process = self.lr_reg
            
        elif self.policy_name == 'XGB' or self.policy_name == 'XGB_C':
            self.policy_funct = self.learned_model
            self.model = []
            for file in param:
                self.model.append(pickle.load(open(param[0], 'rb')))
            self.feature_names = self.model[0].get_booster().feature_names#self.model.feature_names
            temp = [0] + [ int(f.split("_")[-1]) for f in self.feature_names if "delay_bucket" in f]
            
            self.delay_bucket_intervals = {temp[i+1] : temp[i] for i in range(len(temp) - 1)}
            self.process = self.lr_reg#self.xgb_process
            self.eps_history = pd.DataFrame(columns = ['time', 'calls_made'] + self.feature_names , index = np.arange(self.sim.H + 1 ))
            
    def learned_model(self, state):
        if state['time'] == 0:
            return 1
        else:
            x = {}
            for f in self.feature_names:
                x[f] = [self.get_features(f, state)]
            
            #print(x)
            #if state['time'] == 200 :
            #    print("here")
            
            x = self.process(x)
            if state['time'] < 180:
                
                target = self.model[0].predict(x)
            else:
                target = self.model[1].predict(x)
            
            #print(np.multiply(self.model.coef_ , np.array(list(x.values())).flatten()))
            if self.policy_name == 'Linear_Regression' or self.policy_name == 'XGB' :
                #print(target[0] , self.current_buffer(state), np.round(target[0]- self.current_buffer(state)).astype(int))
                return max(0 , np.round(target[0] - self.current_buffer(state)).astype(int))#*self.sim.N
            
            elif self.policy_name == 'XGB_C':
                #print(target[0])
                #prob = self.model.predict_proba(x)
                self.eps_history.loc[state['time']] = [state['time'],  target[0]] + list(x.values[0])
                return target[0]
                #return (prob[0][1]  > 0.8) *1
            
    def xgb_process(self, x):
        return xgb.DMatrix(np.array(list(x.values())).reshape(1, -1),feature_names=self.feature_names )
    
    def lr_reg(self, x):
        return pd.DataFrame(x)
    
    
    
    def get_features(self, f, state):
        if f == 'cum_calls_made':
            return len(state['trace_calls']) #/self.sim.N
        elif f == 'cum_calls_made_at_start':
            return len(state['trace_calls']) #/self.sim.N
        elif 'delay_bucket' in f:
            u = int(f.split("_")[-1])
            return self.delay_bucket(state, u, self.delay_bucket_intervals[u])#/self.sim.N
        elif f =='t_rem':
            return self.sim.H - state['time']#/self.sim.H
        elif f == 'time_since_last_call':
            return (state['time'] - max(state['trace_calls'].values())) #/ self.sim.H
        elif f == 'shifts_available':
            return ( self.sim.M - len(state['assignment']))#/ self.sim.M )
        elif f == 'sum_cutoff_at_start':
            return sum([ state['cutoff_times'][i] - state['time'] for i in state['cutoff_times']])#/(self.sim.D * self.sim.N)
        elif f =='continuous_calls':
            return self.get_consecutive_calls(list(state['trace_calls'].values()), state['time'])
            
    def get_consecutive_calls(self, arr, t):
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
        
    def delay_bucket(self, state, u, l ):
        size = 0
        for i in state['trace_calls']:
            if state['trace_calls'][i] >= (state['time'] - u + 1) and state['trace_calls'][i] <= (state['time'] - l):
                if i not in state['assignment']:
                    size += 1 
    
        return size
    
    def current_buffer(self, state):
        buffer = 0
        for i in state['trace_calls']:
            if i not in state['assignment']:
                buffer += 1
            
        return buffer
    
    def cutoff_buffer_policy(self, state):
        if state['last_called'] + 1 < self.sim.N:
            return min(self.cutoff_buffer - len(state['cutoff_times']), self.sim.N - state['last_called'] - 1, self.sim.max_calls)
        else:
            return 0
        
        
    def call_all(self, state):
        return self.call
    
    def call_late(self, state):
        call = 5 if self.sim.H - state['time'] <= self.sim.N /self.call else 0
        return call
        
    
    def call_and_wait(self, state):
        if state['time'] % self.wait_t == 0:
            return min (self.call_X, self.sim.N - state['last_called'] - 1)
        else:
            return 0       
        
        
    def deterministic_call_policy(self, state):
        if ( state['time'] >= self.sim.H - self.call_all_at ):
           return int((self.sim.N - state['last_called'] - 1))
        
        #diff = (t)*(t)*(t) * self.coeff[0] +   (t)*(t) * self.coeff[1] + t * self.coeff[2] + self.coeff[3] - (self.last_called + 1)
        diff = self.threshold[state['time']]
        #if (t > self.H/2):
        #    diff = diff * self.acc 
            #diff += int(t /30)* self.acc
        diff = diff - (state['last_called'] + 1)
        #diff = max(1 if t==0 else 0, diff)
        return int(min(self.sim.N - state['last_called'] - 1, int(np.ceil(diff ))) )
    
        
    def deterministic_cutoff_policy(self, state):
        if ( state['time'] >= self.sim.H - self.call_all_at ):
           return int((self.sim.N - state['last_called'] - 1))

        diff = self.threshold[state['time']]
        #if (t > self.H/2):
        #    diff = diff * self.acc 
        diff = diff - len(state['cutoff_times'].keys())
        diff = max( 0, diff)
        return min(self.sim.N - state['last_called'] - 1, int(np.ceil(diff ))) 
    
    
    def deterministic_sum_cutoff_data_policy(self, state):
        if ( state['time'] >= self.sim.H - self.call_all_at):
           return int((self.sim.N - state['last_called'] - 1))

        total = sum([state['cutoff_times'][i] - state['time'] for i in state['cutoff_times']])
        buffer = self.threshold[state['time']]
        #if (t > self.H/2):
        #    buffer = buffer * self.acc 
        return min(self.sim.N - state['last_called'] - 1, int(np.round((buffer - total)/self.sim.D ))) 
        #pass
        
        
    def cutoff_time_buffer(self, state):
        total = sum([state['cutoff_times'][i] - state['time'] for i in state['cutoff_times']])
        buffer = min(self.max_num_shift, self.sim.M - len(state['assignment'])) * self.allowed_cutoff
        
        return min(self.sim.N - state['last_called'] - 1, int((buffer - total)/self.sim.D))
    
    def random(self, state):
        if state['time'] % self.interval == 0:
            if self.rand_dist == "uniform":
                return int(min(np.random.randint(0, self.max_calls), self.sim.N - self.last_called - 1))
            elif self.rand_dist == "geometric":
                return int(min(np.random.geometric(0.2), self.sim.N - state['last_called'] - 1, self.max_calls))
        else:
            return 0

    def linear_policy(self, state):
        return self.policy[state['time']]
    
    def get_sim_data(self, _id, model):
        stats = pd.DataFrame(index = np.arange(0, self.sim.H+1),  columns = ["in_delay_at_start",  "in_cutoff_at_end",
                                                                             "in_delay_at_end", "interacted",
                                                                             "in_cutoff_at_start",
                                                                             "bumps", "cum_bumps",
                                                                             "calls_made", "cum_interacted_at_start",
                                                                             "cum_calls_made_at_start", "time_since_last_call"] ).fillna(0)

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
        stats[ 'time_to_wait'] = 0
        
        for i in schedule.index:
            ran = np.arange(schedule.Call_At.loc[i] , min(self.sim.H, schedule.Response_At.loc[i]) )
            stats.loc[ran, 'in_delay_at_end'] = stats.in_delay_at_end.loc[ran] + 1
            ran = np.arange(schedule.Call_At.loc[i] , min(self.sim.H, schedule.Cutoff_Time.loc[i]))
            stats.loc[ran, 'in_cutoff_at_end'] = stats.in_cutoff_at_end.loc[ran] + 1
            if schedule.Response_At.loc[i] <= self.sim.H:
                stats.bumps.at[schedule.Response_At.loc[i]] = stats.bumps.at[schedule.Response_At.loc[i]] + schedule.bumps_caused.loc[i]
                            
            
        for t in stats.index:
            schedule.loc[:,'delay'] = schedule.Actual_cutoff - t
            
            if stats.loc[t, 'calls_made'] > 0 : 
                stats.loc[t, 'time_to_wait'] = 0
            elif stats.loc[t - 1, 'time_to_wait'] > 0:
                stats.loc[t, 'time_to_wait'] = stats.loc[t - 1, 'time_to_wait'] - 1
            else:
                i = 1
                while t + i <= self.sim.H:
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
        
        #stats =  stats.merge(self.pred_history, on = ['time', 'run'], how = 'left')
        self.sim_stats = pd.concat([self.sim_stats, stats])
    