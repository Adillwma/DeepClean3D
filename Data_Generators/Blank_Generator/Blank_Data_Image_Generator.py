# -*- coding: utf-8 -*-
"""
Created on Thurs Feb 23 2022
Blank Data Gen V1.0.0
Author: Adill Al-Ashgar
University of Bristol
"""


"""   # User inputs for standalone use uncomment this block
#%% - User Inputs
ydim = 128
xdim = 88
number_of_files = 8000
output_dir = "C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/Yr 3 Project/Circular and Spherical Dummy Datasets/Dataset 15_X_10K_Blanks/Data/"
# Ada: "C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/Yr 3 Project/Circular and Spherical Dummy Datasets/Dataset 15_X_10K_Blanks/Data/"
# Max:
"""

#%% - Dependencies
import os
import numpy as np
from tqdm import tqdm 
import torch 

#%% - Function
def generate_blanks(xdim, ydim, number_of_files):

    print("\nCreating Blank Images...")

    # Generate tensor of zeros
    arr = torch.zeros(number_of_files, 1, ydim, xdim, dtype=torch.float64)

    return arr

