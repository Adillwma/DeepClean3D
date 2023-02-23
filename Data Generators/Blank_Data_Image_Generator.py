# -*- coding: utf-8 -*-
"""
Created on Thurs Feb 23 2022
Blank Data Gen V0.0.1
Author: Adill Al-Ashgar
University of Bristol
"""


import os
import numpy as np

# Define user arguments
n = 10
m = 5
L = 3
output_dir = "output/"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Generate L arrays of zeros and save them to disk
for i in range(L):
    # Generate array of zeros
    arr = np.zeros((n, m))
    
    # Generate filename
    filename = f"zeros_{i}.npy"
    
    # Save array to disk
    np.save(os.path.join(output_dir, filename), arr)
