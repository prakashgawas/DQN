
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 14:38:38 2022

@author: gawpra
"""
import os
from pyomo.environ import ConcreteModel, Objective, Constraint, ConstraintList, Set, Var, Param
from pyomo.environ import NonNegativeReals, Binary, NonNegativeIntegers 
from pyomo.environ import minimize, value 
from pyomo.environ import SolverStatus, TransformationFactory
import numpy as np
from pyomo.opt import SolverFactory
from pyomo.util.infeasible import log_infeasible_constraints
import logging
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.stats import weibull_min, norminvgauss
from scipy.special import erfinv
import math

class ONCSS():
    def __init__(self, N, H,  M = None, D = None, dist = 'data',  name = None, det = True, quantile = 0.5, folder = "", seed = 10):
        #print("Integer Program Model")
        self.N = N
        self.H = H
        self.M = N if M == None else M 
        self.D = H if D == None else D
        self.Q = D
        self.dist_name = dist
        self.quantile = quantile
        self.q = int(quantile * 100)
        self.U = [i for i in range(self.N)]
        self.S = [i+1 for i in range(self.M)]
        self.T = [t for t in range(self.H+1)]
        
        
        self.next_of = {}
        u = -1
        for i in self.U:
            if u == -1:
                u = i
            else:
                self.next_of[u] = i
                u = i
                
        self.name = name
        self.det = det
        self.data = pd.read_csv("response_delay2.csv", index_col=0)
        self.data = self.data.sort_values(by=['interacted']).reset_index(drop = True)
        self.data = self.data[self.data['mask'] == self.det]
        self.data = self.data['interacted']    
        self.data = self.data[~np.isnan(self.data)]
        self.folder = folder
        self.max_calls = N
        self.cb_generator = np.random.default_rng(seed)
        self.cb_regenerator = np.random.default_rng(seed + 1)
        self.urd_store = pd.DataFrame()
        
    def set_max_calls(self, max_calls = np.inf):
        self.max_calls = max_calls
        
    def set_wait(self, max_wait):
        self.max_wait = max_wait
        
    def set_vacancy_weight(self,  wt = 100):
        self.C_V = wt
        
    def set_data(self, rd_values = 'generate'):
        
        if rd_values  == 'generate':
            cbs = self.response_dist()
        else:
            cbs = pd.read_csv(rd_values)
            cbs = np.array(cbs.Response_delay.values)

        self.user_response_duration = { i : cbs[i] for i in self.U}
        #self.user_response_duration = {0: 0, 1: 1000, 2: 121, 3: 0, 4: 3, 5: 1000, 6: 171, 7: 1062, 8: 0, 9: 0, 10: 1000, 11: 1000, 12: 56, 13: 0, 14: 1035, 15: 1185, 16: 2, 17: 1000, 18: 0, 19: 0, 20: 0, 21: 1146, 22: 3, 23: 339, 24: 1000, 25: 0, 26: 1000, 27: 1000, 28: 0, 29: 2, 30: 761, 31: 0, 32: 897, 33: 817, 34: 2, 35: 0, 36: 1145, 37: 1000, 38: 11, 39: 4, 40: 0, 41: 0, 42: 1000, 43: 1000, 44: 1000, 45: 791, 46: 578, 47: 63, 48: 0, 49: 1000, 50: 1000, 51: 2, 52: 1000, 53: 43, 54: 851, 55: 969, 56: 3, 57: 899, 58: 0, 59: 1, 60: 1166, 61: 1000, 62: 937, 63: 1000, 64: 41, 65: 1000, 66: 0, 67: 1000, 68: 1000, 69: 49, 70: 337, 71: 16, 72: 376, 73: 16, 74: 15, 75: 705, 76: 0, 77: 997, 78: 0, 79: 1000, 80: 1163, 81: 59, 82: 1199, 83: 963, 84: 1000, 85: 0, 86: 0, 87: 0, 88: 179, 89: 208, 90: 1102, 91: 1000, 92: 851, 93: 1000, 94: 4, 95: 122, 96: 953, 97: 0, 98: 1000, 99: 1000, 100: 1000, 101: 1, 102: 777, 103: 1000, 104: 1000, 105: 912, 106: 18, 107: 1000, 108: 27, 109: 1000, 110: 0, 111: 1000, 112: 389, 113: 1000, 114: 46, 115: 863, 116: 0, 117: 0, 118: 1118, 119: 976, 120: 0, 121: 1000, 122: 0, 123: 0, 124: 1149, 125: 1000, 126: 0, 127: 162, 128: 940, 129: 1000, 130: 0, 131: 330, 132: 0, 133: 0, 134: 0, 135: 953, 136: 1023, 137: 1000, 138: 0, 139: 1067, 140: 0, 141: 4, 142: 830, 143: 91, 144: 1000, 145: 45, 146: 0, 147: 0, 148: 0, 149: 199}
        #urd = pd.read_csv("Instance_217.csv").to_dict()['Response_delay']
        #for i in range(40):
        #    self.user_response_duration[i] = urd[i]
        #self.user_response_duration = .to_dict()['Response_delay']
        #self.user_response_duration[1] = 0
        #print(self.user_response_duration)
        
        self.set_delta()
        
    def set_delta(self):
                
        self.delta = {}
        for u in self.U:
            for v in self.U:
                if u < v:
                    self.delta[u,v] = self.user_response_duration[u] - self.user_response_duration[v]
                    
    def regenerate(self, state, urd):
        curr_urd = urd
        temp = self.cb_regenerator.choice(self.data, size = self.N).astype(int)
        for i in self.U:
            if i not in state['assignment'] and i <= state['last_called']:
               curr_urd[i]  = self.cb_regenerator.choice(self.data[self.data > (state['time'] - state['trace_calls'][i]) ], size = 1).astype(int)[0]
            elif i > state['last_called']:
               curr_urd[i] = temp[i]
        self.user_response_duration = curr_urd
        self.set_delta()
               
                    
    def response_dist(self):
        if self.dist_name == "Uniform":
            return np.random.randint(0, self.Q/self.quantile, size = self.N)
        
        elif self.dist_name == "Weibull":
            c = 1.02
            scale = self.Q/(math.pow(-math.log(1 - self.quantile), 1/c))
            temp = weibull_min.rvs(c = c, scale = scale, size = self.N)
            temp[ temp > self.H] = self.H
            return temp.astype(int)
        
        elif self.dist_name == "Normal":
            loc = 0.8 * self.Q
            scale = (self.Q - loc)/(math.sqrt(2) * erfinv(self.quantile * 2 - 1)) 
            temp = np.random.normal(loc = loc, scale = scale, size = self.N)
            temp[ temp > self.H] = self.H
            temp[ temp < 0] = 0
            return temp.astype(int)
        
        elif self.dist_name == "Triangular":
            left = 0
            p = 0.8
            b = - 2 * self.Q
            a = self.quantile + (1 - self.quantile) * p 
            right = (- b + (math.sqrt(b**2 - 4 * a * self.Q * self.Q) ))/(2* a) 
            temp = np.random.triangular(left, mode = p * right, right = right, size = self.N)
            return temp.astype(int)

        elif "data" in self.dist_name:

            temp = self.cb_generator.choice(self.data, size = self.N)
            temp[temp > 1000] = 1000
            return temp.astype(int)
        
        elif self.dist_name == 'norminvgauss':
            a = 2.203
            b = 2.202
            loc = 0
            scale = 2.05
            temp = norminvgauss.rvs(a, b, loc = loc, scale = scale, size=self.N)# * scale + loc
            return temp.astype(int)
            #remember to do int delay times
    
    def define_model(self, time_limit = 300, eoc = False):
        self.model = ConcreteModel()
        self.eoc = eoc
        self.define_data()
        self.define_variables()
        self.define_constraints()
        if self.max_calls != np.inf:
            self.define_call_constraints()
        if self.max_wait != self.H:    
            self.define_fast_calls_constraints()
            
            
        #users = [i for i in range(self.N)]
        #time = [0,0,0,0,7,8,8,8,9,10,11,12,37,37,37]
        #self.fix_call(users, time)    
        if eoc == True:  
            
            self.define_eoc_constraints()
            
        #self.add_constraints()
        self.define_objective()
        self.setup_solver(time_limit=time_limit)
        
    def define_data(self):
        self.slack = int(np.ceil(self.N/self.max_calls))
        
        self.model.Users = Set(initialize = self.U)
        self.model.Shifts = Set(initialize = self.S)
        self.model.Time = Set(initialize= self.T)
        
        self.model.valid_pairs = Set(initialize = self.delta.keys())
        
        self.model.Response = Param(self.model.Users, initialize=self.user_response_duration)
        self.model.Delta = Param(self.model.valid_pairs, initialize = self.delta, default = 0)
        #self.model.wait = Param(self.model.Users, initialize=1, mutable=True)
        self.model.C_V = Param(initialize= self.C_V)
    
    def define_variables(self):
        #start times
        self.model.s = Var(self.model.Users, within=NonNegativeIntegers, name = 'call_u_at')#NonNegativeReals
        self.model.bump = Var(self.model.valid_pairs, within=Binary, name = 'u_bumps_v')
        #self.model.p = Var(self.model.valid_pairs, within=Binary, name = 'p')
        #variable for call to be made at time t
        #self.model.y = Var(self.model.Users, self.model.Time, within=Binary, name = 'call_u_at_t')
        
        if self.N > self.M:
            #variable for push out
            self.model.z = Var(self.model.Users, within=Binary, name = "Push_out_u")
            
            self.model.V = Var( within=NonNegativeIntegers, name = "VacantShifts")
            #self.model.r = Var(self.model.Users,  self.model.Time, within = Binary, name='responded')
            #self.model.a = Var(self.model.Users,  self.model.Time, within = Binary, name='assignment')
        
    def define_objective(self):

        self.model.objective = Objective(expr = self.model.C_V *self.model.V + sum(self.model.bump[u,v] for u in self.model.Users for v in self.model.Users if u < v), sense=minimize)
    
    def define_constraints(self):
        
        
        if self.N == self.M:
            
            def priority(model, u, v):
                return (-self.H - self.slack, model.s[u] - model.s[v], 0 ) 
                
            self.model.con1 = Constraint(self.model.valid_pairs, rule=priority, name="priority_constraint"  )
            
             
            def end_times(model, u):
                return (0 , model.s[u] + self.user_response_duration[u], self.H)
                
            self.model.con2 = Constraint(self.model.Users, rule=end_times, name="end_before_horizon"  )
            
            def set_bumps(model, u, v):
                return (-self.H, model.s[u] - model.s[v] + model.Delta[u,v] - self.H * model.bump[u,v] , 0)
                 
            self.model.con3 = Constraint(self.model.valid_pairs, rule=set_bumps, name="set_bumps"  )
           
        
        elif self.N > self.M:
            
            def priority(model, u, v):
                return (-self.H - self.slack, model.s[u] - model.s[v], 0 ) 
                
            self.model.con1 = Constraint(self.model.valid_pairs, rule=priority, name="priority_constraint"  )
        
            
 
            def end_times(model, u):
                return model.s[u] + self.user_response_duration[u]  - (self.H + 1) * model.z[u] >= 0
               
            self.model.con2 = Constraint(self.model.Users, rule=end_times, name="end_before_horizon"  )
            
            def end_times2(model, u):
                return   model.s[u] + self.user_response_duration[u] - (self.user_response_duration[u] + self.slack) * model.z[u] <= self.H 
                
            self.model.con5 = Constraint(self.model.Users, rule=end_times2, name="set_callbacks"  )
            
            def call_before_end(model, u):
                return   model.s[u]  <= (self.H - 1) +  (self.slack + 1) * (  model.z[u])
                
            self.model.con9 = Constraint(self.model.Users, rule=call_before_end, name="call_before_end"  )
       
            
            def set_bumps(model, u, v):
                if self.user_response_duration[v] >= self.user_response_duration[u] or self.user_response_duration[u] >= self.D:
                    return model.bump[u,v] == 0
                else:
                    return model.s[u] - model.s[v] + model.Delta[u,v] <= model.Delta[u,v] * model.bump[u,v] + (self.H + self.user_response_duration[u]) * model.z[u]
                    
            self.model.con3 = Constraint(self.model.valid_pairs, rule=set_bumps, name="set_bumps"  )
            
            #scheduling constarint
            #self.model.con4 = Constraint(expr = (sum(self.model.z[u] for u in self.model.Users ) <= self.N - self.M), name = 'need_M_cbs')
            self.model.con4 = Constraint(expr = (self.M - self.N + sum(self.model.z[u] for u in self.model.Users ) <= self.model.V), name = 'need_M_cbs')
            self.model.con15 = Constraint(expr = (self.model.s[0] == 0), name = 'call_early')

            #self.model.con16 = Constraint(expr = (self.N - self.M <= sum(self.model.z[u] for u in self.model.Users ) ), name = 'M-callbacks')
                               
            def bump_and_push_out_u(model, u, v):
                return  model.bump[u,v] <= 1 - model.z[u] 
            self.model.con7 = Constraint(self.model.valid_pairs, rule=bump_and_push_out_u, name="bump_and_push_out_u"  )
            
            def bump_and_push_out_v(model, u, v):
                return  model.bump[u,v] <= 1 - model.z[v] 
            self.model.con8 = Constraint(self.model.valid_pairs, rule=bump_and_push_out_v, name="bump_and_push_out_v"  )          
        
            #self.model.calling_constraint = {}        
            
            
    # def add_constraints(self):
    #         def set_response(model, u, t):
    #             return  t - (model.s[u] + self.user_response_duration[u]) <= self.H * model.r[u, t] 
            
    #         self.model.con10 = Constraint(self.model.Users, self.model.Time, rule=set_response, name="set_response"  )
            
    #         def set_assign1(model, u, t):
    #             return   model.a[u, t]  <=  model.r[u, t] 
            
    #         self.model.con11 = Constraint(self.model.Users, self.model.Time, rule=set_assign1, name="set_assign1"  )
            
    #         def set_assign2(model, u, t):
    #             return   sum(self.model.r[i, t] for i in range(u+1) ) - self.M  <=  self.M * (1 - self.model.a[u,t]) 
            
    #         self.model.con12 = Constraint(self.model.Users, self.model.Time, rule=set_assign2, name="set_assign2"  )
            
    #         def set_assign3(model, u, t):
    #             return self.M -  sum(self.model.r[i, t] for i in range(u) )   <=  self.M * (1 - self.model.r[u,t]) + self.M * (self.model.a[u,t]) 
            
    #         self.model.con13 = Constraint(self.model.Users, self.model.Time, rule=set_assign3, name="set_assign3"  )
        
    
    def define_call_constraints(self):
        def max_calls(model, u):
            if u + self.max_calls < self.N:
                return  model.s[u] + 1 <= model.s[u + self.max_calls] 
            else:
                return Constraint.Skip
        self.model.call_time = Constraint( self.model.Users, rule=max_calls, name="max_calls"  )
        
    def define_eoc_constraints(self):
        self.eoc1_keys = {}
        self.eoc2_keys = {}
        def end_of_call1(model, u, v):
            if self.user_response_duration[u] < self.D and self.user_response_duration[v] <=self.H and self.user_response_duration[v] <= self.user_response_duration[u]:       
                
                return  model.s[u] - model.s[v]  + model.Delta[u,v] <= -1 + (1001) * (model.bump[u,v] + model.z[v] + model.z[u]) 
            else:
                return Constraint.Skip
            
        self.model.eoc1 = Constraint(self.model.valid_pairs, rule=end_of_call1, name="end_of_call1"  )  
        
        def end_of_call2(model, u, v):
            if self.user_response_duration[u] < self.D and self.user_response_duration[v] <=self.H and self.user_response_duration[v] <= self.user_response_duration[u]:
                return  model.s[u] - model.s[v]  + model.Delta[u,v] >= 1  - 1001 * (model.z[v] + model.z[u] +  (1 - model.bump[u,v]))
            else:
                return Constraint.Skip
            
        self.model.eoc2 = Constraint(self.model.valid_pairs, rule=end_of_call2, name="end_of_call2"  )  
        
    def alter_callback(self, x):    
        self.model.con16.set_value(self.model.con16.lower - x <=  self.model.con16.body)
        
    def define_fast_calls_constraints(self):
        def fast_calls(model, u):
            if u < self.N - 1:
                return  model.s[u + 1] - model.s[u] <= self.max_wait 
            else:
                return Constraint.Skip
        self.model.fast_calls = Constraint( self.model.Users, rule=fast_calls, name="fast_calls"  )
        
        
    def fix_call(self, users, times, called_users):
        for k in range(len(users)):
            self.model.s[users[k]].fix(times[k])
                  
        called_users = set(list(called_users.keys()) + users)
            
        if self.eoc: 
            for  u,v in self.model.eoc1.keys():
                if u in called_users and v in called_users :
                    self.model.eoc1[u,v].deactivate()
            for  u,v in self.model.eoc2.keys():
                if u in called_users and v in called_users  :
                    self.model.eoc2[u,v].deactivate()
                
        #if user in self.model.calling_constraint:
        #    self.model.calling_constraint[user].deactivate()
        #def fix_call_at(model ):
        #    return model.s[user]  == time
        #self.model.calling_constraint[user] = Constraint( rule = fix_call_at, name = 'fix_' + str(user) + '_at_' + str(time))
        
    def call_after(self, user, time):
        #def call_after_time(model ):
        #    return model.s[user]  >= time
        #self.model.calling_constraint[user] = Constraint( rule = call_after_time, name = 'fix_' + str(user) + '_at_' + str(time))
        self.model.s[user].setlb(time)
        # if self.max_calls == 1:
        #     self.model.wait[user] = self.model.wait[user] + 1
        
        
    def save_model(self):
        with open('Model.txt', 'w') as output_file:
            self.model.pprint(output_file)
            
    def setup_solver(self,  time_limit = 300 ):
        self.opt = SolverFactory("gurobi")#executable = '/home/gawpra/gurobi/9.5.0/linux64/bin'
        self.opt.options["Threads"] = 2
        #opt.options["OutputFlag"]  = 1
        self.opt.options['TimeLimit'] = time_limit
        self.opt.options['LogToConsole'] = 0
        self.opt.options['OutputFlag'] = 0
        #self.opt.options['slog'] = 1

    
    def solve(self, k = '',  ret = True, tee = True, log_file = None):
        #opt = SolverFactory("cplex", executable = "/Applications/CPLEX_Studio201/cplex/bin/x86-64_osx/cplex") ##SolverFactory("cplex")
        
        print("Solving...")
        if log_file == True:
            #self.opt.options['LogFile'] = '' 
            log_file = self.folder + self.name + "_" + str(k) + ".log"
        #   start = time.time()


        #    result = self.opt.solve(self.model, report_timing = True, tee=tee, logfile=log_file) ##,tee=True to check solver info
        #    end = time.time()
        #else:
        start = time.time()

        result = self.opt.solve(self.model, report_timing = True, tee=tee) ##,tee=True to check solver info
        end = time.time()
            
        if result.solver.status == SolverStatus.aborted: #max time limit reached 
            result.solver.status = SolverStatus.warning  #change status so that results can be loaded
        
        print("Time Elapsed - ",end - start)
        self.model.solutions.load_from(result)   
        
        #result.solver.Time
        #print(start , end)
        if ret == True:
            
            log_infeasible_constraints(self.model, log_expression=True, log_variables=True)
            logging.basicConfig(filename='infeasible.log',  level=logging.INFO)
            
            print("Solve Status - ",result.solver.termination_condition)
            print("Store result...")
        if result.solver.termination_condition == 'infeasible':
            return 0, result.solver.termination_condition, end - start
        else:
            return value(self.model.objective), result.solver.termination_condition, end - start

        
    def post_process(self, name, itr= None, print_output = False, save_solution = False, plot = False, folder = None):
        self.process_output()
        if print_output == True:
            self.print_output(name)
            
        if plot == True or save_solution == True:
            self.save_instance_solution()
        if plot == True:
            
            self.plot(name)
            
        if save_solution == True:
            if folder == None:
                folder = self.folder
            

            #name = str(self.N) + "_" + str(self.M) +  "_" + str(self.H)
            self.Solution.to_csv(folder + "/Solution_" + name +".csv")
            
        #self.save_instance()
        
        
    def print_output(self, name):
        string = "*" * 50
        print(string)
        print("Problem <<<<", name)
        print(string)
        print("Output :-")
        print("Call Status = ", self.call_at)
        print("Response Status = ", self.response_at)
        print("Response_Delay = ", self.user_response_duration)
        print("Total Bumps = ", self.total_bumps)
        print("Total Pushed Out = ", self.pushout_total)
        if self.N > self.M:
            print("Assignment = ", self.allotment)
        print("Vacant Shifts = ", self.model.V.value)
        print(string)
        
        
    def get_call_times(self):
        return {k: round(v, 2) for k, v in self.model.s.extract_values().items()}
    
    def get_response_times(self, call_at):
        return {u: call_at[u] + self.user_response_duration[u] for u in call_at}
    
    def process_output(self):
        self.bumps = self.model.bump.extract_values()
        
        self.call_at =  self.get_call_times()
       
        self.response_at = self.get_response_times(self.call_at)
        self.total_bumps = value(self.model.objective)
        
        self.bump_set = {i for i in self.bumps if self.bumps[i] > 0}
        #bumped = [i[1] for i in self.bump_set]
        self.bumps_suffered = [0] * self.N
        self.bumps_caused = [0] * self.N
        
        
        for i in self.bump_set:
            self.bumps_suffered[i[1]] += 1
        
        for i in self.bump_set:
            self.bumps_caused[i[0]] += 1
            
        if self.N > self.M:
            self.allotment = self.model.z.extract_values()
            self.alloted_users = np.array([i for i in self.allotment if self.allotment[i] < 1])
            
        self.pushout_total = sum(self.allotment.values())
        
        # for u, v in self.model.valid_pairs:
        #     if u <= 5 and v <= 5:
        #         print( u, v, self.call_at[u] - self.call_at[v]  + self.delta[u,v] , " <= ",  -1 + (1001)* (self.allotment[u] + self.allotment[v] + (self.bumps[u,v]) ))
        #         print( u, v, self.call_at[u] - self.call_at[v]  + self.delta[u,v] ," >= ", 1 - (1001)* (self.allotment[u] + self.allotment[v] + (1 - self.bumps[u,v]) ))


    def save_instance_solution(self):
        self.Solution = pd.DataFrame(columns= ['Call_At', 'Response_Delay', 'Response_At'])
        self.Solution['Call_At'] = self.call_at.values()
        self.Solution['Response_Delay'] = self.user_response_duration.values()
        if self.N > self.M:
            self.Solution['Assigned'] = self.allotment.values()
        self.Solution['Response_At'] = self.response_at.values()
        self.Solution['bumps_suffered'] = self.bumps_suffered#.values()
        self.Solution['bumps_caused'] = self.bumps_caused
        self.Solution['Cutoff_Time'] = self.Solution['Call_At'] + np.minimum(self.D, self.Solution['Response_Delay'])
        #Solution['Response_At'] = pd.to_numeric(Solution['Response_At'])
        self.Solution = self.Solution.reset_index()
        self.Solution = self.Solution.astype(int)
        self.Solution['index'] = 'E' + self.Solution['index'].astype(str)
        self.Solution['Actual_cutoff'] = self.Solution['Call_At'] + self.D
        self.Solution['Actual_cutoff'] = self.Solution['Actual_cutoff'].clip(0, self.H)
        self.Solution = self.Solution.reset_index()
        #self.Solution = self.Solution.iloc[::-1]
        
    def store_urd(self, _id = 0):
        urd =  pd.DataFrame.from_dict(self.user_response_duration, orient='index',
                                columns=['Response_delay'])
        urd['_id'] = _id
        urd['user'] = urd.index
        self.urd_store = pd.concat([self.urd_store, urd])
        
    def get_vacant_shifts(self):
        return self.model.V.value
    
    def get_total_bumps(self):
        return sum(self.model.bump.extract_values().values())
        
    def plot(self, name):
        
        
        #fig = px.timeline(Solution, x_start="Call_At", x_end="Response_At", y="index")
        #fig.update_yaxes(autorange="reversed") # otherwise tasks are listed from the bottom up
        #fig.show()
        #fig.write_image("solution.jpeg")
        
        fig, ax = plt.subplots(1, figsize=(16,16))
        ax.barh(self.Solution['index'], self.Solution.Response_Delay, left=self.Solution.Call_At)
        #self.Solution = self.Solution.iloc[::-1]
        for idx, row in self.Solution.iterrows():
            ax.text(row.Response_At, len(self.Solution) - idx, f"{int(row.Response_Delay)}", 
                    va='top', ha = 'center', alpha=0.8)
            
        ax.axvline(x=self.H,color = 'r')
        plt.gca().set_xlim(left=0)
        plt.grid()
        ax.xaxis.get_ticklocs(minor=True)     # []

        # Initialize minor ticks
        ax.minorticks_on()
        ax.yaxis.set_tick_params(which='minor', bottom=False)
        #plt.show()
        plt.savefig(self.folder + '/Plot_' + name + '.png', dpi=300)
        
    def solve_linear_relaxation(self, k):
        
        opt = SolverFactory("gurobi")#executable = '/home/gawpra/gurobi/9.5.0/linux64/bin'
        opt.options["threads"] = 1
        opt.options["OutputFlag"]  = 1
        opt.options['TimeLimit'] = 200
        xfrm = TransformationFactory('core.relax_integer_vars')
        self.lr_model = xfrm.apply_to(self.model)
        start = time.time()

        result = opt.solve(self.model, report_timing = True, tee=True, logfile=self.folder + self.name + "_" + str(k) + ".log") ##,tee=True to check solver info
        end = time.time()
        
        return value(self.model.objective), result.solver.termination_condition, end - start
        
    def save_instance(self, name):
        pd.DataFrame.from_dict(self.user_response_duration, orient='index',
                               columns=['Response_delay']).to_csv(name)    