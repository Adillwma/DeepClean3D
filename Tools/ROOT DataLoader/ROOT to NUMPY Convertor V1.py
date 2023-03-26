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

# path of root data:
# Adill path
path = "C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/data/tuple_1.root" #Path for root file
# Max path
# path = r"C:\Users\maxsc\OneDrive - University of Bristol\3rd Year Physics\Project\Autoencoder\2D 3D simple version\Root Data\Root Files\tuple_1.root"

# path to save to:
# Adill path
output_path = "C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/data/root2numpy/" #Path for processed file output
# Max path
# output_path = r"C:\Users\maxsc\OneDrive - University of Bristol\3rd Year Physics\Project\Autoencoder\2D 3D simple version\Root Data\Numpy Files/"

# set number of files counter to both print at end and label datasets produced:
num_files = 0

with uproot.open(path) as file:
    for data_item in file:
        num_files += 1
        np.save(output_path + str(num_files), data_item)
 
print(f"Conversion Successfull, saved {num_files} files")