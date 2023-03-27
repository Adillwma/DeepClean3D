#%% - Dependencies
import os
import numpy as np
import matplotlib.pyplot as plt

#%% - User inputs
### NOTE Set Below Dirs and then run!
input_dir = "N:/Yr 3 Project Datasets/Test"
output_dir = "N:/Yr 3 Project Datasets/3D_Test"


#%% - Function
def dataset_2D_to_3D_Expansion(input_dir, output_dir, time_dimension=100):
    input_dir = input_dir + "/Data//"
    output_dir = output_dir + "/Data//"

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop over each file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".npy"):
            # Load the data from the file
            filepath = os.path.join(input_dir, filename)
            data = np.load(filepath)

            # Apply the processing functions to the data
            shape = data.shape
            processed_data = np.zeros((shape[0], shape[1], time_dimension))

            i, j = np.nonzero(data)           # Compute the indices for the non-zero elements of data in the third dimension of array_3D
            k = data[i, j].astype(int)        # Convert the values to integers
            processed_data[i, j, k-1] = 1     # array_3D is now a 3D numpy array of size n by m by time_dimension_max, with the non-zero values from the original 2D array set to 1 in the appropriate location 

            # Save the processed data to a new file in the output directory
            output_filepath = os.path.join(output_dir, "3D -" + filename)
            np.save(output_filepath, processed_data)

#%% - Driver
dataset_2D_to_3D_Expansion(input_dir, output_dir)