# -*- coding: utf-8 -*-
"""
Created on Thursday March 2 2023
Standalone Dataset Test V1.0.0
Author: Adill Al-Ashgar
University of Bristol

Created to avoid messing around with broken datasets any longer, now dataset can immidetly 
be verified for correct dtype, correct image size and correct min-max value range distribution.

### Improvements to be made
# Want to add this (or just the raw intergrity and dist test files) into the actual dataset generator so that: 1) it automatically can report if the dataset ccontains an error during creation. 2) it automatically saves the dataset report (the integrity check vals etc and the distribution image) to a folder related to the datsaset.
# Improve formatting and readbility of termian prints during tests
# No need to double load the data if running both tests! (even though second time is very fast as data already in ram or cached from first load)
# Improve dataset dist tester, add red or green test passed/failed to the stats on graph edge, remove uneeded stats, add max and mix
"""

#%% - User Inputs
time_dimension = 100
dataset_full_filepath = r"C:\Users\Student\Documents\UNI\Onedrive - University of Bristol\Yr 3 Project\Circular and Spherical Dummy Datasets\Dataset 15_MultiX_80X_20X2"

#%% - Load Tests
from DC3D_V3.Helper_files.Dataset_distribution_tester_V1 import dataset_distribution_tester
from DC3D_V3.Helper_files.Dataset_Integrity_Check_V1 import dataset_integrity_check

#%% - Run Tests
dataset_integrity_check(dataset_full_filepath, full_test=True, print_output=True)
dataset_distribution_tester(dataset_full_filepath, time_dimension=time_dimension, ignore_zero_vals_on_plot=True)
