#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 16:00:51 2023

@author: prakashgawas
"""


from glob import glob
import pickle
import torch
import xgboost as xgb
import os

location = "Data_Dagger/"
# directory 
folders = glob('Data_Dagger/*' )
folders = sorted(folders)


for fold in folders:
    if 'csv' not in fold:
        all_folds = glob(fold+ "/*", recursive = True)
        print(fold)
        for model in all_folds:
            if ".pkl" in model:
                bst = xgb.XGBClassifier()
                bst = pickle.load(open(model, "rb"))
                #nn = pickle.load(open(model, "rb"))
                name = model.split("/")[-1]
                name = name.split(".")[0]
                if not os.path.isfile(fold + "/" + name +".json"):  
                    bst.save_model(fold + "/" + name +".json")
                os.remove(model)
                #torch.save(nn.state_dict(), 'Data/IOL_corn_80_1_5_360_class_corn_0.3_/' + name)
    
    