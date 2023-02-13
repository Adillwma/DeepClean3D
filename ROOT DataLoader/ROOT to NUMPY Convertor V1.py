# -*- coding: utf-8 -*-
"""
ROOT to NUMPY Convertor
@author: Adill Al-Ashgar
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



path = "C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/data/tuple_1.root" #Path for root file
with uproot.open(path) as file:
    for data_item in file:
        np.save(data_item, path)
print("Conversion Successfull")