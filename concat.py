#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 17:35:44 2023

@author: prakashgawas009
"""

import pandas as pd
import numpy as np
from glob import glob

#folder = 'Instances/Average/'
folder = 'InstancesACR_Max_calls_2/Average/'
#folder = 'InstancesACR_Max_calls_2/Average/'


files = glob(folder +"*.csv")
summary = pd.DataFrame()
schedule_avg = pd.DataFrame()
    
    
for file in files:
    if 'summary' in file:
        summary = pd.concat([summary, pd.read_csv(file)])
    elif 'schedule_stats_avg' in file:
        if "Weibull" in file or 'data' in file:
            schedule_avg = pd.concat([schedule_avg, pd.read_csv(file)])
        
summary.to_csv(folder + "summary.csv")
schedule_avg.to_csv(folder + "schedule_avg.csv")


# 180	180	data-interacted	2.696813977389517
# 180	180	data-interacted	0.224049331963001
# 180	180	data-interacted	0.2076053442959917
# 180	180	data-interacted	0.1901336073997944
# 180	180	data-interacted	0.1017471736896197
# 180	180	data-interacted	0.1109969167523124
# 180	180	data-interacted	0.1438848920863309
# 180	180	data-interacted	0.1140801644398766
# 180	180	data-interacted	0.1428571428571428
# 180	180	data-interacted	0.1027749229188078
# 180	180	data-interacted	0.0719424460431654
# 180	180	data-interacted	0.0904419321685508
# 180	180	data-interacted	0.0760534429599177
# 180	180	data-interacted	0.0688591983556012
# 180	180	data-interacted	0.0606372045220966
# 180	180	data-interacted	0.1171634121274409
# 180	180	data-interacted	0.0647482014388489
# 180	180	data-interacted	0.092497430626927
# 180	180	data-interacted	0.119218910585817
# 180	180	data-interacted	0.1109969167523124
# 180	180	data-interacted	0.0914696813977389
# 180	180	data-interacted	0.0719424460431654
# 180	180	data-interacted	0.0524152106885919
# 180	180	data-interacted	0.08016443987667
# 180	180	data-interacted	0.0760534429599177
# 180	180	data-interacted	0.0966084275436793
