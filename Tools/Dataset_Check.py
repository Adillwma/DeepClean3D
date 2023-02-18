
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

def load_random_npy_file(folder_path):
    """
    Loads a random .npy file from the specified folder path.

    Parameters:
    folder_path (str): The path to the folder containing the .npy files.

    Returns:
    A numpy array loaded from a randomly chosen .npy file in the folder.
    """
    # Get a list of all .npy files in the folder
    npy_files = glob.glob(f"{folder_path}/*.npy")

    # If no .npy files were found, return None
    if len(npy_files) == 0:
        print("No files found in dir!")
        return None
    
    # Choose a random .npy file from the list
    chosen_file = np.random.choice(npy_files)
    
    # Load the numpy array from the chosen file
    file_path = os.path.join(folder_path, chosen_file)
    arr = np.load(file_path)
    return arr




dataset_title = "Dataset 10_X"
data_path = "C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/Yr 3 Project/Circular and Spherical Dummy Datasets/"
dir = (data_path + dataset_title + "/Data/")
print(dir)

mins = []
maxs = []

for _ in range(200):
    test_file = load_random_npy_file(dir)
    mins.append(np.amin(test_file))
    maxs.append(np.amax(test_file))

print("Shape:", np.shape(test_file))
print("Type:", type(test_file))
print("Min:", np.amin(mins))
print("Max:", np.amax(maxs))


plt.imshow(test_file)
plt.show()