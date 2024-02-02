#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 09:36:21 2024

@author: prakashgawas
"""


from glob import glob
import pandas as pd
import json
import os
loc = '../../../../../scratch/gawpra/'


settings = glob(loc + 'Data_Dagger_3_0.0*' )
all_sim = pd.DataFrame(columns = ['lm','tw', 'step', 'm_count', 's_count', 'name'])

for setting in settings:
    print(setting)
    folders = glob(setting + '/*' )
    
    param = setting.split('/')[-1]
    tw = 1 if 'TW' in param else 0
    lm = param.split('_')[3]
    step = param.split('_')[2]

    for folder in folders:
        print(folder)
        if ".csv" in folder:
            continue
        m_count = 0
        s_count = 0
        name = folder.split('/')[-1]
        name = name.replace('IOL', 'StatsIOL_sim')
        for i in range(-1, 200):
            if os.path.exists( folder + '/' + name + '_' + str(i) + '.csv'):
                s_count += 1
            if os.path.exists( folder + '/xgb_0' + '_' + str(i) + '.json'):
                m_count += 1
                
        all_sim.loc[len(all_sim)] = [lm, tw, step, m_count, s_count, name]

           

all_sim.to_csv("All_sim.csv")
