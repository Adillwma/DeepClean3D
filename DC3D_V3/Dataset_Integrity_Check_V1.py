# -*- coding: utf-8 -*-
"""
Dataset Integrity Checker V1.0.0
Created: 20:56 18 Feb 2023
Author: Adill Al-Ashgar
University of Bristol

# To import:
from Dataset_Integrity_Check_V1 import dataset_integrity_check

# To then run, call function with: 
dataset_integrity_check(folder_path, full_test=True/False, print_output=True/False)

#### Improvements to be made
# Fix integrity check conditions especially for single test case (maybe make the one random plot have to match at least one other random one?)
# Simplify code for the single vs full test cases
# Improve plots and add descriptions
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np

##V2 moved from glob to os scan dir - is now faster
def dataset_integrity_check(folder_path, full_test=False, print_output=True):
    """
    Perform an integrity check on a directory containing numpy arrays (.npy files).
    This function loads the numpy arrays from the directory and performs checks on their shape and type.
    If `full_test` is True, all numpy arrays in the directory will be tested. If False, only one random numpy array is tested.
    If `print_output` is True, the function will print information on the check results.
    If the integrity check passes, the function will print the shape and type of all the numpy arrays in the directory,
    as well as the overall minimum and maximum values found in all the arrays.

    Args:
        folder_path (str): The path to the folder containing the numpy arrays.
        full_test (bool, optional): If True, test all numpy arrays in the folder. If False, test only one random array. Defaults to False.
        print_output (bool, optional): If True, print the results of the check. Defaults to True.
    Returns:
        None
    """    
    integrity_check = "Integrity Check Failed"    # Message to be printed at end of test, this is updated if the test passes
    if print_output:
        print(folder_path)

    # Get a list of all .npy files in the folder using os.scandir
    npy_files = [entry.path for entry in os.scandir(folder_path) if entry.name.endswith('.npy')]
    number_of_files = len(npy_files)

    # If no .npy files were found, return None
    if number_of_files == 0:
        print("Error: No files found in dataset dir! \n(files must be inside a folder that is inside the dataset dir specified)\n")
        return None

    def npy_file_loader(chosen_file):
        # Load the numpy array from the chosen file
        file_path = os.path.join(folder_path, chosen_file)
        arr = np.load(file_path)
        return arr

    mins = []
    maxs = []

    # Load the first file outside the loop to avoid loading it multiple times
    first_file = npy_file_loader(npy_files[0])
    first_file_shape = np.shape(first_file)
    first_file_type = type(first_file)


    if full_test:   # Test all files in directory sequentially by index
        mins.append(np.amin(first_file))
        maxs.append(np.amax(first_file))
        for file_number in range(1, number_of_files):
            test_file = npy_file_loader(npy_files[file_number])
            if np.shape(test_file) != first_file_shape or type(test_file) != first_file_type:
                print(f"File {file_number} has a different shape or type than the first file.")
            mins.append(np.amin(test_file))
            maxs.append(np.amax(test_file))
        integrity_check = "Integrity Check Passed"
    
    else:  # Test 1 file only - loads random npy file from dir
        test_file = npy_file_loader(np.random.choice(npy_files))
        mins.append(np.amin(test_file))
        maxs.append(np.amax(test_file))
        if np.shape(test_file) == first_file_shape and type(test_file) == first_file_type:
            integrity_check = "Integrity Check Passed"
    
    if print_output:
        print(integrity_check)
    
    if integrity_check == "Integrity Check Passed" and print_output == True:
        print("All files have shape:", first_file_shape)
        print("All files are of type:", first_file_type)
        print("Overall Min TOF value in all files scanned:", np.amin(mins))
        print("Overall Max TOF value in all files scanned:", np.amax(maxs))

        plt.imshow(test_file)
        plt.title("Single image from dataset for visual reference")
        plt.show()

#%% - Demo Driver

dataset_title = "Dataset 12_X10K"
data_path = "C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/Yr 3 Project/Circular and Spherical Dummy Datasets/"
dir = (data_path + dataset_title + "/Data/")

dataset_integrity_check(dir)
