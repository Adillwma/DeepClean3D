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
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pywt
import os
from tqdm import tqdm
###Ploting confidence of each pixel as histogram per epoch with line showing the detection threshold
def data_histogram(data, time_dimension, ax=None):
    """
    Plot a histogram showing the TOF of each pixel.

    Parameters
    ----------
    data : numpy.ndarray
        The input data to plot.
    time_dimension : int
        The number of bins in the histogram.
    ax : matplotlib.axes.Axes, optional
        The subplot on which to plot the histogram. If not provided, a new
        figure and subplot will be created.

    Returns
    -------
    matplotlib.axes.Axes
        The subplot on which the histogram was plotted.
    """
    #data2 = data.flatten()  # Flatten the input data   only perform this fater check if inp is 1D
    if ax is None:
        fig, ax = plt.subplots()  # Create a new subplot if none is provided
    _, _, bars = ax.hist(data, time_dimension, histtype='bar')  # Create the histogram
    ax.bar_label(bars, fontsize=10, color='navy')  # Add labels to the bars
    return ax  # Return the subplot object

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
    mean = np.mean(data)

    # Calculate median using numpy function
    median = np.median(data)

    # Calculate mode using numpy function
    mode, count = np.unique(data, return_counts=True)
    mode = mode[count.argmax()] if count.max() > 1 else "No mode found"

    # Calculate range using numpy function
    data_range = np.ptp(data)

    # Calculate standard deviation using numpy function
    stdev = np.std(data)

    # Calculate Variance
    variance = np.var(data)
    
    # Calculate Coefficient of Variation
    cv = stdev / mean
    
    # Quartiles
    q1 = np.percentile(data, 25)
    q2 = np.percentile(data, 50)
    q3 = np.percentile(data, 75)

    # Calculate skewness and kurtosis using numpy function
    skewness = skew(data)
    kurt = kurtosis(data)
    
    # Compute the Fourier transform
    fourier_transform = np.fft.fft(data)
    
    # Compute the power spectrum using the Fourier transform
    power_spectrum = np.abs(fourier_transform) ** 2

    # Compute the wavelet transform using the Daubechies 4 wavelet
    wavelet_transform, _ = pywt.dwt(data, 'db4')
    
    # Compute the principal components using PCA
    pca = PCA()
    principal_components = pca.fit_transform(data.reshape(-1, 1))

    # Return the results as a dictionary
    results = {
        "Mean": mean,
        "Median": median,
        "Mode": mode,
        "Range": data_range,
        "Stdev": stdev,
        'Variance': variance,
        'Coefficient of variation': cv,
        'Quartiles1 (25%)': q1,
        'Quartile 2 (50%)': q2,
        'Quartile 3 (75%)': q3,
        "Skewness": skewness,
        "Kurtosis": kurt,
        "Fourier": fourier_transform,
        "Power Spectrum": power_spectrum,
        "Wavelet Transform": wavelet_transform,
        "PCA": principal_components
    }
    return results

#%% - User Inputs
time_dimension = 100
dataset_title = "Dataset 15_X_10K_Blanks"
data_path = "C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/Yr 3 Project/Circular and Spherical Dummy Datasets/"
dir = (data_path + dataset_title)

#%% - Load Files
single_file = file_loader(dir)
full_set = file_loader(dir, load_full_set=True)

print(np.shape(single_file))
print(type(single_file))

print(np.shape(full_set))
print(type(full_set))

single_file_stats = calculate_statistics(single_file)
full_set_stats = calculate_statistics(full_set)

#%% - Output Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

data_histogram(single_file, time_dimension, ax=ax1)
ax1.set_title("Single file")

data_histogram(full_set, time_dimension, ax=ax2)
ax2.set_title("Entire Dataset")

fig.suptitle("Histograms showing the value of every pixel in the input data")

# Set the text properties
text_props = dict(horizontalalignment='left', verticalalignment='baseline', fontsize=12)

# Iterate over the dictionary and print each statistic as a line of text
for i, (key, value) in enumerate(single_file_stats.items()):
    text = f"{key}: {value}"
    ax1.text(0.02, 1 - (i * 0.05), text, transform=ax1.transAxes, **text_props)

# Iterate over the dictionary and print each statistic as a line of text
for i, (key, value) in enumerate(full_set_stats.items()):
    text = f"{key}: {value}"
    ax2.text(0.02, 1 - (i * 0.05), text, transform=ax2.transAxes, **text_props)

# Show the plot
plt.show()

