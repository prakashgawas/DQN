
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 18:29:35 2022

@author: Prakash
"""

import os
#from pyomo.environ import *
import math
import numpy as np
#from pyomo.opt import SolverFactory
#from pyomo.util.infeasible import log_infeasible_constraints
#import logging
import itertools
import pandas as pd
#import matplotlib.pyplot as plt
#import time
#from scipy.stats import weibull_min
from ONCSS_Actual_pyomo import ONCSS

class SECTP(ONCSS):
    
    def __init__(self, N, H, M = 50, quantile = 0.5, D = 90, dist = 'Uniform',  name = None, det = False, seed = 10):
                
        #print("Integer Program Model")
        ONCSS.__init__(self, N, H, M = M, quantile = quantile, 
                       D = D,  dist = dist, 
                       name = name, det = det, seed = seed)
        
        self.constant_delay = D
        self.set_costs()
        
        
        
    def initialize(self, itr, policy_name, adjust_reward = False, rd_values = 'generate'):

        self.rd_values = rd_values
        self.iteration = itr
        self.set_data(self.rd_values)
        #self.save_CB_data()
        
        self.events  = dict()
        self.counter = 0
        
        #self.personnel_count = 0
        self.last_called = -1
        #current time
        self.t = 0
        self.event_order = {}
        self.delay_users = []
        self.cutoff_users = []
        self.callback_users = []
        self.eligible_users = [i for i in range(self.N)]
        self.assignment = np.array([]).astype(int)
        self.call_time = dict()
        self.cb_time = dict()
        self.cutoff_times = dict()
        self.curr_bumps = 0
        self.current_cb = 0 
        self.net_bumps = 0
        self.bumps_by = dict()
        self.trace_calls = dict()
        
        self.stats = {}
        #self.responses = {i: self.H + 1 for i in range(self.N)}
        self.schedule = {}
        self.last_call_at = []
        self.initialize_stats(policy_name)
        self.adjust_reward = adjust_reward
        
        
        
        return {'time' :self.t,
                'last_called':self.last_called, 
                #'last_call_at':self.last_call_at,
                'assignment': self.assignment,
                'trace_calls':self.trace_calls, 
                'cutoff_times':self.cutoff_times}
            
    
    def save_CB_data(self):
            
        df = pd.DataFrame.from_dict(self.user_response_duration, orient='index', columns=['Response_delay'])
        #df['Eligibility'] = sum(self.state[3], [])
        name = str(self.N) + "_" + str(self.H) + "_" +  str(self.M) + "_" +  str(self.constant_delay  ) + "_" + self.dist_name + "_" + str(self.iteration)
        df.to_csv("Stoch_instances/RD_" + name  +".csv")
    
        
        
    def initialize_stats(self, policy_name):
        self.stats['iteration'] = self.iteration
        #self.stats['run_counter'] = b
        #self.stats['list_id'] = list_id
        #self.stats['Last_Shifts_filled_at'] = self.end_simulation
        #self.stats['Shifts_filled_at'] = {}
        self.stats['policy_name'] = policy_name
        self.stats['Eligible users'] = self.N
        self.stats['horizon'] = self.H
        self.stats['Max_cutoff_buffer'] = 0
        

    
    def collect_stats(self):
                
        self.stats['Num_users_not_called'] = len(self.eligible_users)
        self.stats['Num_users_late'] = len(self.delay_users)
        self.stats['Num_users_called'] = self.N - len(self.eligible_users)
        self.stats['Num_users_cb'] = len(self.callback_users)
        self.stats['Num_users_bumped'] = sum(self.bumps_by.values())
        self.stats['shifts_vacant'] = self.M - len(self.assignment)
        user_response_duration = np.array(list(self.user_response_duration.values()))
        self.stats['avail_users'] = sum(user_response_duration < self.H)
        self.stats['effective_avail_users'] = sum(user_response_duration < self.H / 1.5)
       

        # for i in self.assignment:
        #     self.schedule[i]['Assigned'] = 0           
        # self.schedule = pd.DataFrame.from_dict(self.schedule, orient='index')
        
    def set_costs(self, vacancy_cost = 0, bump_cost = 1):
        self.c_v = vacancy_cost
        self.c_b = bump_cost
    
    def call_users(self, call):
        
        self.curr_bumps  = 0
        self.current_cb = 0 
        if self.N  < self.last_called + 1 + call:
            call = self.N - self.last_called - 1
        
        for i in range(call):
            self.last_called += 1
            self.trace_calls[self.last_called] = self.t
            
            if self.t + self.user_response_duration[self.last_called] <= self.H:
                self.event_order[(self.last_called,1)] = self.t + self.user_response_duration[self.last_called]
                self.event_order = {k: v for k, v in sorted(self.event_order.items(), key=lambda item: item[1])}
            self.delay_users.append( self.last_called)
            self.add_bump_cutoff()
            
            self.eligible_users.remove(self.last_called)
            self.events[self.counter] = {'time':self.t ,'log' : "user called", 'user' : self.last_called, 'delay':self.user_response_duration[self.last_called] , 'users_in_delay': list(self.delay_users),'cutoff_users':list(self.cutoff_users) }
            #print("log = user 1 delay ",self.events,self.counter)
            self.counter += 1   
            
            self.stats['Max_cutoff_buffer'] = max(self.stats['Max_cutoff_buffer'], len(self.cutoff_users))

            
            self.schedule[self.last_called] = {'Call_At': self.t, 'Response_Delay' :self.user_response_duration[self.last_called],
                                               'Response_At':self.t + self.user_response_duration[self.last_called],
                                               'Assigned': 1, 'bumps_suffered': 0, 'bumps_caused': 0,
                                               'Cutoff_Time': self.t + np.minimum(self.D, self.user_response_duration[self.last_called])}
            self.last_call_at.append(self.t)
            self.stats['last_call_at'] = self.last_call_at[-1] 
        
        if len(self.event_order) > 0:
            self.next_events()
            
        self.t += 1
        
        if self.t == self.H:
            self.collect_stats()


        #if len(self.assignment) == self.M:
        #    self.sim_callbacks()
        reward = self.curr_bumps * self.c_b  + self.c_v * (self.t == self.H) * (self.M - len(self.assignment))
        
        return {'time' :self.t,
                'last_called':self.last_called, 
                #'last_call_at': self.last_call_at,
                'assignment': self.assignment,
                'trace_calls': self.trace_calls, 
                'cutoff_times':self.cutoff_times}, reward * (1 - self.adjust_reward), self.t == self.H
            
            
    def add_bump_cutoff(self):
        self.event_order[(self.last_called,2)] = self.t  + self.constant_delay
        self.cutoff_users.append( self.last_called)
        self.cutoff_times[self.last_called] = min(self.t  + self.constant_delay, self.H) 
        self.event_order = {k: v for k, v in sorted(self.event_order.items(), key=lambda item: item[1])}
        
    def next_events(self):
        event_list = list(self.event_order)
        
        for i in event_list:
            # if i[0] == 37:
            #     print("here")
            if i not in self.event_order:
                continue
            if self.event_order[i] == self.t:
                if i[1] == 1:
                    self.curr_bumps += self.callback(i[0])
                    self.delay_users.remove( i[0])
                    if i[0] in self.cutoff_users:
                        self.cutoff_users.remove( i[0])
                        

                elif i[1] == 2:
                    self.cutoff_users.remove( i[0])
                    self.events[self.counter] = {'time': self.t, 'log' : "user cutoff passed", 'user' : i[0], 'users_in_delay': list(self.delay_users),'cutoff_users':list(self.cutoff_users)}
                    #print("log = user 1 delay ",self.events,self.counter)
                    self.counter += 1 
                    self.cutoff_times.pop(i[0])
                self.event_order.pop(i)
            else:
                break
            
    def callback(self, user):
        self.callback_users.append(user)
        #self.responses[user] = self.t
        bumps = self.adjust_schedule(user)
        self.net_bumps += bumps
        
        self.bumps_by[user] = bumps

        self.schedule[user]['bumps_caused'] = bumps
        

        if user in self.cutoff_times:
            self.cutoff_times.pop(user)
            self.cutoff_users = list(self.cutoff_times.keys())
        
        if (user, 2) in self.event_order:
            self.event_order.pop((user, 2))
            
        self.events[self.counter] = {'time': self.t , 'log' : "user callback", 'user' : user, 'user_cutoff': self.trace_calls[user] + self.constant_delay , 'users_in_delay': list(self.delay_users),'cutoff_users':list(self.cutoff_users), 'assignment': list(self.assignment.astype(int)), 'bumps': bumps }
        #print("log = user 1 delay ",self.events,self.counter)
        self.counter += 1 
        self.current_cb += 1
        return bumps
    
    def adjust_schedule(self, user):
        #if user == 14:
        #    print("here")
        bumps = 0
        temp = user
        #if user == 2:
        #    print("here")
        if user in self.cutoff_users:
            for i in range(len(self.assignment)):
                if temp < self.assignment[i]:
                    #print(self.assignment[i])
                    self.schedule[self.assignment[i]]['bumps_suffered'] += 1
                    self.assignment[i], temp = temp, self.assignment[i]
                    bumps += 1
        self.assignment = np.append(self.assignment, temp)
            
        if len(self.assignment) > self.M:
            self.assignment = self.assignment[0: self.M]

        return bumps
        
        
    def save_events_log(self, _id ):
        df = pd.DataFrame.from_dict(self.events, orient='index')
        
        #df.user = df.user.astype(int)
        df.to_csv(   '../DECTP/Simulation_events_' + str(_id) + '.csv')

