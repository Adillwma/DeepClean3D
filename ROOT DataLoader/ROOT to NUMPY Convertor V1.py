# -*- coding: utf-8 -*-
"""
ROOT to NUMPY Convertor
@author: Max Carter
Created on Tue Nov 23 2022
"""

#%% - Dependencies
import matplotlib.pyplot as plt
import numpy as np
import uproot

###Optimisation information: 
# The uproot.open function has many options, including alternate handlers for each input type, num_workers to control 
# parallel reading, and caches (object_cache and array_cache). The defaults attempt to optimize parallel processing, 
# caching, and batching of remote requests, but better performance can often be obtained by tuning these parameters.



path = #Path for root file
with uproot.open(path) as file:
    for data_item in file:
        #np_data_item = ???? #Turn data_item into numpy array if it is not already    
        
        savepath = path + '/processed'
        np.save(np_data_item, savepath)

