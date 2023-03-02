# -*- coding: utf-8 -*-
"""
Created on Thursday March 2 2023
Standalone Dataset Test V1.0.0
Author: Adill Al-Ashgar
University of Bristol

Created to avoid messing around with broken datasets any longer, now dataset can immidetly 
be verified for correct dtype, correct image size and correct min-max value range distribution.
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
