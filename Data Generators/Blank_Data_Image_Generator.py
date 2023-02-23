# -*- coding: utf-8 -*-
"""
Created on Thurs Feb 23 2022
Blank Data Gen V1.0.0
Author: Adill Al-Ashgar
University of Bristol
"""
#%% - User Inputs
n = 128
m = 88
number = 3000
output_dir = "C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/Yr 3 Project/Circular and Spherical Dummy Datasets/Dataset 15_X_10K_Blanks/Data/"
# Ada: "C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/Yr 3 Project/Circular and Spherical Dummy Datasets/Dataset 15_X_10K_Blanks/Data/"
# Max:


#%% - Dependencies
import os
import numpy as np
from tqdm import tqdm 

#%% - Compute
# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Generate L arrays of zeros and save them to disk
for i in tqdm(range(number)):                       #Display pregress bar during generation for user feedback

    # Generate array of zeros
    arr = np.zeros((n, m))
    
    # Generate filename
    filename = f"blankdata_{i+1}.npy"
    
    # Save array to disk
    np.save(os.path.join(output_dir, filename), arr)

print(f"\n Generation of {number} images completed successfully\n")
