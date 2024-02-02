#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 18:37:23 2022

@author: Prakash
"""

import numpy as np
import pandas as pd
import os
from ONCSS_Actual_pyomo import ONCSS

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=150)
parser.add_argument('--k', type=int, default=217)
parser.add_argument('--H', type=int, default=360)
parser.add_argument('--D', type=int, default=180)
parser.add_argument('--q', type=int, default=40)
parser.add_argument('--M', type=int, default=50)
parser.add_argument('--dist', type=str, default='data-interacted')#data-interacted norminvgauss
#parser.add_argument('--q', type=int, default=40)
args = parser.parse_args()

#runs = 1000
sim = {}
N = args.N
H = args.H
k = args.k
q = args.q
M = args.M
D = [args.D]
check_current = 0
max_wait = args.H

eoc = False
folder_name = 'InstancesX'
if eoc == True:    
    folder_name = folder_name + 'ACR'

summary = pd.DataFrame(columns = ['id', 'obj', 'vacancy', 'bumps', 'status'])
for max_calls in [5]:
    for q in [q]:
        for dist_name in  [args.dist]: #'Uniform','Normal', 'Weibull','Triangular'
            for d in D:
                
                #D = int(f * H)
                _ = "_"
                max_calls_str = "_Max_calls_" + str(max_calls) if max_calls != np.inf else "" 
                max_wait_str = "_Max_wait_" + str(max_wait) if max_wait != H else ""
                name = str(N) + _ + str(H) + _ + str(M) + _ + str(d) + _ + str(q) + _ + dist_name
                folder = folder_name + max_calls_str + max_wait_str +  "/" + name 
                print(folder)
                if not os.path.exists(folder):
                   os.makedirs(folder)
                O = ONCSS(N, H, M = M, D = d, dist = dist_name, name = name, det = True, quantile = q/100, folder = folder, seed = k)
                O.set_max_calls(max_calls)
                O.set_wait(max_wait)
                O.set_vacancy_weight(50)
                #for k in range(runs):     
                print("RUN :-", k)
                #rd_file = "Stoch_Instances/RD_" + name + _ + str(k) + ".csv"
                np.random.seed(k)
                O.set_data()
                O.define_model(eoc = eoc)
                if check_current == 1 :
                    if os.path.isfile( folder + "/Solution_" + name + "_" + str(k) + ".csv"):
                        continue
                    if k >= 500:
                        continue
                O.save_instance(folder + '/Instance_' + str(k) + ".csv")
    
                obj, status, t = O.solve(k)           
                if status != 'infeasible':
                    sim[k] = (N, H, M, D, q, obj, status, t)
                    O.post_process(name + "_" + str(k), itr = k, print_output = True, save_solution = True, plot = True)
             
                summary.loc[0] = [k, obj, O.get_vacant_shifts(), O.get_total_bumps(),  status]
                summary.to_csv(folder + '/summary_' + str(k) + '.csv' )