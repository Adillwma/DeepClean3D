#%% - Dependencies
import numpy as np

#%% - Function
def to_3d_transform(data, time_dimension=100):

    # Apply the processing functions to the data
    shape = data.shape
    processed_data = np.zeros((shape[0], shape[1], time_dimension))

    i, j = np.nonzero(data)           # Compute the indices for the non-zero elements of data in the third dimension of array_3D
    k = data[i, j].astype(int)        # Convert the values to integers
    processed_data[i, j, k-1] = 1     # array_3D is now a 3D numpy array of size n by m by time_dimension_max, with the non-zero values from the original 2D array set to 1 in the appropriate location 
    #output = processed_data[np.newaxis, ...]   # add dim of 1 to start for channle dims
    #print(np.shape(output))
    return processed_data
