#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:13:12 2024

@author: prakashgawas
"""

import numpy as np
import copy
from glob import glob
import os
import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=150)
parser.add_argument('--H', type=int, default=360)
args = parser.parse_args()

#folder = "/home/gawpra/Merinio2020/Learn/Instances/"
folder = 'Instances_Max_calls_2/Average/'

all_files = glob(folder +"*", recursive = True)
max_calls = 2

for file in all_files:
    if 'schedule_stats_' + str(args.N) in file and '_tw' not in file:
        df = pd.read_csv(file, index_col=0)
        df['to_call'] = 0
        df1 = df[df.calls_made >= 1].reset_index(drop = True)
        df2 =  df[df.calls_made == 0].reset_index(drop = True)
        data = df1.to_dict(orient='index')
        counter = 0
        new_data = {}
        df.index = df.time
        df = df.set_index('id', append = True)
        for i in data:
            for j in range(min(data[i]['calls_made'], max_calls - 1)):
                
                new_data[counter] = copy.deepcopy(data[i])
                new_data[counter]['time_to_wait'] = 0
                new_data[counter]['to_call'] = j
                
                counter += 1

            data[i]['to_call'] = min(data[i]['calls_made'], max_calls - 1)
            if data[i]['calls_made'] < max_calls and data[i]['time'] < args.H:
                data[i]['time_to_wait'] = df.loc[(data[i]['time'] + 1, data[i]['id']), 'time_to_wait'] + 1
            
        data = pd.DataFrame.from_dict(data, orient = 'index')
        new_data = pd.DataFrame.from_dict(new_data, orient = 'index')
        df = pd.concat([df2, new_data, data])
        df = df.sort_values(by=['id', 'time', 'to_call'])
        df.to_csv(file[:-4] + '_wt.csv')
            
