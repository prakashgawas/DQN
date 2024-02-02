 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 15:21:37 2023

@author: prakashgawas
"""

from glob import glob
import pandas as pd
import json
import os
loc = '../../../../../scratch/gawpra/'
#folders = glob('../../../../../scratch1/gawpra/Data_Dagger/*' ) 


settings = glob(loc + 'Data_Dagger_3_0.0*' )
all_sim = pd.DataFrame()

for setting in settings:
    print(setting)
    folders = glob(setting + '/*' )
    all_data = pd.DataFrame()
    param = setting.split('/')[-1]
    tw = 1 if 'TW' in param else 0
    lm = param.split('_')[3]
    step = param.split('_')[2]
    
    for fold in folders:
        print(fold)
        if ".csv" in fold:
            continue
        if '200' not in fold:
            continue
        all_folds = glob(fold + "/*.csv", recursive = True)
        if not os.path.exists( fold + '/config.json'):
           continue
        
        with open(fold + '/config.json', 'r') as f:
            config = json.load(f)
        
        data = pd.DataFrame()
        for file in all_folds:
    
            if 'StatsIOL_sim' in file:
                temp = file.replace(".", "_")
                _id = temp.split("_")[-2]
                df = pd.read_csv(file)
                df['model_id'] = _id
                df['model'] = config['policy']
                df['eoc'] = config['eoc']
                df['type'] = config['type'] + '_' + config['action_type']
                df["D"] = config['D']
                df["q"] = config['q']
                df["dist"] = config['dist']
                df['target'] = config['target']
                df['action_type'] = config['action_type']
                if config['action_type'] == 'threshold' or config['action_type'] == 'quant':
                    df['threshold'] = config['prob_threshold']
                else:
                    df['threshold'] = None
                df['max_calls'] = config['max_calls']
                df['cost'] = df['shifts_vacant'] * 50 + df['Num_users_bumped']
                data = pd.concat([data, df])
                all_data = pd.concat([all_data, df])
            
        name = fold.split('/')[-1]
        data.to_csv( fold + '/' + name + '_sim_stats.csv')
        all_data['lm'] = lm
        all_data['tw'] = tw
        all_data['step'] = step
        all_sim = pd.concat([all_sim, all_data])
        
        
        data = pd.DataFrame()
        for file in all_folds:
            if 'Saved_data' in file:
                temp = file.replace(".", "_")
                _id = temp.split("_")[-2]
                df = pd.read_csv(file)
                df['id'] = _id
                data = pd.concat([data, df])
            
            
        data.to_csv(fold + '/' + name + '_Data.csv')
        
    all_data.to_csv( setting + '/All_sim_stats.csv')

all_sim.to_csv( loc +  '/All_sim_stats.csv')
    
    
    # data = pd.DataFrame()
    # for file in all_folds:
    #     if 'Sim_Data' in file:
    #         temp = file.replace(".", "_")
    #         _id = temp.split("_")[-2]
    #         df = pd.read_csv(file)
    #         df['id'] = _id
    #         data = pd.concat([data, df])
        
        
    # data.to_csv('Data/' + fold + fold[:-1] + '_Sim_Data.csv')
    
    
    
