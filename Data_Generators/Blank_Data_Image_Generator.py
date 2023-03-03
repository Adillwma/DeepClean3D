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

#%% - Function
def generate_blanks(xdim, ydim, number_of_files, output_dir):

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print("\nCreating Blank Images...")
    # Generate 'number_of_files' arrays of zeros and save them to disk
    for i in tqdm(range(number_of_files), desc="Blank Image"):                       #Display pregress bar during generation for user feedback

        # Generate array of zeros
        arr = np.zeros((ydim, xdim))
        
        # Generate filename
        filename = f"blankdata_{i+1}.npy"
        
        # Save array to disk
        np.save(os.path.join(output_dir, filename), arr)

    # Feedback to user on task completed
    print(f"Generation of {number_of_files} blank images completed successfully\n")
