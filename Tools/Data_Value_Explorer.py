# -*- coding: utf-8 -*-
"""
Data Value Explorer V1.0.0
Created on Mon Feb 26 2023
Author: Adill Al-Ashgar
University of Bristol   

This script checks a given dataset (filepath given by user) and plots histogram of the values of 
one random sample from the set and also of the entore set combined. Then calculates distribution
statistics for each case. USed to verify dataset TOF values are in correct places whilst designing
normalisation and reconstructions.

### Improvemnts to be made
# for statistics fucntion add a check for if input is 1D and if not then flatten it 
# must find faster way of loading in the numpy files and then flattening them all into one long np array
#
"""

#%% - Dependencies
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
import statistics
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

#%% - Function
def file_loader(folder_path, load_full_set=False, print_output=True):
    if print_output:
        print(folder_path)

    # Updated above to now get a list of all .npy files in the folder and its subfolders using os.walk
    npy_files = []
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith('.npy'):
                npy_files.append(os.path.join(dirpath, filename))
    
    number_of_files = len(npy_files)

    # If no .npy files were found, return None
    if number_of_files == 0:
        print("Error: Either specified dir does not exist or no files found! \n(files must be inside a folder that is inside the dataset dir specified)\n")
        return None

    def npy_file_loader(chosen_file):
        # Load the numpy array from the chosen file
        file_path = os.path.join(folder_path, chosen_file)
        arr = np.load(file_path)
        return arr

    if load_full_set:   # Test all files in directory sequentially by index
        test_files = [] # initialize an empty list
        for file_number in tqdm(range(1, number_of_files), desc='Loading files'):
            test_file = npy_file_loader(npy_files[file_number])
            test_file = test_file.flatten().tolist()
            test_files.append(test_file)  # append each numpy file to the array
        output = np.array(test_files).flatten()

    else:  # Test 1 file only - loads random npy file from dir
        test_file = npy_file_loader(np.random.choice(npy_files))
        test_file = test_file.flatten()
        output = test_file

    return output

def calculate_statistics(data):   #Need to improve this by adding a check at begining to see if input is realla a 1D list/array and if not to then flatten it to 1D
    """
    Calculate various statistical measures for a given 1D list or array with values from a distribution.

    Parameters:
    -----------
    data : list
        A list of values that make up the distribution of pixel activation times.

    Returns:
    --------
    dict
        A dictionary containing the computed statistical measures as keys and their corresponding values. 
        The keys include "Mean", "Median", "Mode", "Range", "Stdev", "Variance", "Coefficient of variation", 
        "Quartiles1 (25%)", "Quartile 2 (50%)", "Quartile 3 (75%)", "Skewness", "Kurtosis", "Fourier", 
        "Power Spectrum", "Wavelet Transform", and "PCA".
    """

    # Calculate mean using numpy function
    if len(data) == 0:
        mean = 0 
    else:    
        mean = np.mean(data)

    # Calculate median using numpy function
    median = np.median(data)

    # Calculate mode using numpy function
    mode =  statistics.mode(data) 

    # Calculate range using numpy function
    data_range = np.ptp(data)

    # Calculate standard deviation using numpy function
    stdev = np.std(data)

    # Calculate Variance
    variance = np.var(data)
    
    # Calculate Coefficient of Variation
    if mean == 0:
        cv = np.nan
    else:
        cv = stdev / mean
    
    # Quartiles
    q1 = np.percentile(data, 25)
    q2 = np.percentile(data, 50)
    q3 = np.percentile(data, 75)

    # Calculate skewness and kurtosis using numpy function
    skewness = skew(data)
    kurt = kurtosis(data)

    # Return the results as a dictionary
    results = {
        "Range": data_range,
        "Mean": mean,
        "Median": median,
        "Mode": mode,
        "Stdev": stdev,
        'Variance': variance,
        'Coefficient of variation': cv,
        'Quartiles1 (25%)': q1,
        'Quartile 2 (50%)': q2,
        'Quartile 3 (75%)': q3,
        "Skewness": skewness,
        "Kurtosis": kurt,
    }
    return results

#%% - User Inputs
time_dimension = 100
dataset_title = "Dataset 15_X_10K_Blanks"
data_path = "C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/Yr 3 Project/Circular and Spherical Dummy Datasets/"
dir = (data_path + dataset_title)

ignore_zero_vals_on_plot = True

#%% - Load Files
single_file = file_loader(dir)
full_set = file_loader(dir, load_full_set=True)

single_file_stats = calculate_statistics(single_file)
full_set_stats = calculate_statistics(full_set)

#%% - Output Plots
fig = plt.figure(figsize=(16,9), constrained_layout=True)
gs = GridSpec(2, 5, figure=fig)
ax1 = fig.add_subplot(gs[0, :4])
ax2 = fig.add_subplot(gs[1, :4])
ax3 = fig.add_subplot(gs[0, 4])
ax4 = fig.add_subplot(gs[1, 4])

if ignore_zero_vals_on_plot:
    single_file = single_file[np.where(single_file != 0)]
    full_set = full_set[np.where(full_set != 0)]

# Plot the histogram and extract the bin counts and bin edges
nsingle, binssingle, patches = ax1.hist(single_file, bins=time_dimension+1, density=False, color='blue', alpha=0.5, edgecolor='black', linewidth=1.2)
nfull, binfull, patches = ax2.hist(full_set, bins=time_dimension+1, density=False, color='blue', alpha=0.5, edgecolor='black', linewidth=1.2)
###Could collect this bin number and width data for checking against after the test is complete???

# Add a title and axis labels
fig.suptitle("Histograms showing the value of every pixel in the input data")
ax1.set_title('Single file')
ax2.set_title('Entire Dataset')
ax2.set_xlabel('Data Values')
ax1.set_ylabel('Density')
ax2.set_ylabel('Density')

# Add grid lines, limits and legend
ax1.grid(axis='y', alpha=0.75)
ax2.grid(axis='y', alpha=0.75)
ax1.set_ylim(bottom=0)
ax2.set_ylim(bottom=0)
ax1.legend(['Data'])
ax2.legend(['Data'])

# Set the text properties
text_props = dict(horizontalalignment='right', verticalalignment='baseline', fontsize=9)
text_x_position = 0.98

# Iterate over the dictionary and print each statistic as a line of text
for i, (key, value) in enumerate(single_file_stats.items()):
    text = f"{key}: {value}"
    ax3.text(text_x_position, 1 - (i * 0.05)-0.2, text, transform=ax3.transAxes, **text_props)
    ax3.set_axis_off()

# Iterate over the dictionary and print each statistic as a line of text
for i, (key, value) in enumerate(full_set_stats.items()):
    text = f"{key}: {value}"
    ax4.text(text_x_position, 1 - (i * 0.05)-0.2, text, transform=ax4.transAxes, **text_props)
    ax4.set_axis_off()

# Save or show the plot
plt.show()  #swap for plot_or_Save_function

print("\nProgram Completed\n")
