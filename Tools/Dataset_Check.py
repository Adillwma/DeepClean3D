
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
from SNR_Test.DragRace import drag_race

#####Imporvemnts to be made
#Fix integrity check conditions
#Simplify 
#tweak plots


def dataset_integrity_check(folder_path, full_test=False, print_output = True):
    integrity_check = "Integrity Check Failed"    # Message to be printed at end of test, this is updated if the test passes
    if print_output:
        print(folder_path)

    # Get a list of all .npy files in the folder
    npy_files = glob.glob(f"{folder_path}/*.npy")
    number_of_files = len(npy_files)

    # If no .npy files were found, return None
    if number_of_files == 0:
        print("Error: No files found in dataset dir! \n(files must be inside a folder that is inside the dataset dir specified)\n")
        return None

    def npy_file_loader(npy_files, i=None):
        if i is None:
            # If index is not passed, choose a random file
            chosen_file = np.random.choice(npy_files)
        else:
            # Load the file at the specified index
            chosen_file = npy_files[i]
        
        # Load the numpy array from the chosen file
        file_path = os.path.join(folder_path, chosen_file)
        arr = np.load(file_path)
        return arr

    mins = []
    maxs = []

    ### SIMPLIFY SECTION !!!!
    if full_test:   # Test all files in directory sequenctially by index
        for file_number in range(number_of_files):
            test_file = npy_file_loader(npy_files, file_number)
            if file_number == 0:
                first_file_shape = np.shape(test_file)
                first_file_type = type(test_file)
            else:
                if np.shape(test_file) != first_file_shape or type(test_file) != first_file_type:
                    print(f"File {file_number} has a different shape or type than the first file.")
            mins.append(np.amin(test_file))
            maxs.append(np.amax(test_file))
        integrity_check = "Integrity Check Passed"
    
    else:  # Test 1 file only - loads random npy file from dir
        test_file = npy_file_loader(npy_files)
        mins.append(np.amin(test_file))
        maxs.append(np.amax(test_file))
        integrity_check = "Integrity Check Passed"  ## Should be some condition? although i guess integrity is over the datset so for 1 makes no sense>?
    if print_output:
        print(integrity_check)

    if integrity_check == "Integrity Check Passed" and print_output == True:
        print("All files have shape:", np.shape(test_file))
        print("All files are of type:", type(test_file))
        print("Overall Min TOF value in all files scanned:", np.amin(mins))
        print("Overall Max TOF value in all files scanned:", np.amax(maxs))

        plt.imshow(test_file)
        plt.title("Single image from dataset for visual reference")
        plt.show()

##V1 stopped loading first test file multiple times to check shape etc, now done once and stored - speed increase negligable aswas only one data file (tested speed) also changed np.shape(array) to array.shape
def dataset_integrity_check1(folder_path, full_test=False, print_output = True):
    integrity_check = "Integrity Check Failed"    # Message to be printed at end of test, this is updated if the test passes
    if print_output:
        print(folder_path)

    # Get a list of all .npy files in the folder
    npy_files = glob.glob(f"{folder_path}/*.npy")
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

        plt.imshow(first_file)
        plt.title("Single image from dataset for visual reference")
        plt.show()

##V2 moved from glob to os scan dir - is now faster
def dataset_integrity_check2(folder_path, full_test=False, print_output=True):
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

#%% - Driver
dataset_title = "Dataset 12_X10K"
data_path = "C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/Yr 3 Project/Circular and Spherical Dummy Datasets/"
dir = (data_path + dataset_title + "/Data/")

#dataset_integrity_check(dir)

#dataset_integrity_check1(dir)

#dataset_integrity_check2(dir)

#dataset_integrity_check3(dir)


functions_to_race = [dataset_integrity_check, dataset_integrity_check1, dataset_integrity_check2] #, bilateral_filter_denoise, trilateral_filter_denoise, wiener_filter_denoise, wavelet_denoise, nlm_denoise, tv_regularization_denoise, PCA_denoise, LRA_SVD_denoise, BM3D_denoise, WNNM_denoise]
args_list = [[dir, True, False]]
drag_race(1, 1, functions_to_race, args_list, 3, share_args=True)