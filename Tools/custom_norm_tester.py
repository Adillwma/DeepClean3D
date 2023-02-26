# -*- coding: utf-8 -*-
"""
Custom Normalistaion/Renormalisation Tester V1.0.0
Created on Mon Feb 26 2023
Author: Adill Al-Ashgar
University of Bristol   

This script tests normalisation and renormalisation returns the same data as oroginal.
also test edge cass of 0 val TOF becuas ei supsect we have created a misteak in how we encode 0 vals


### Improvemnts to be made
#  

"""


#%% - Dependencies
import numpy as np
import matplotlib.pyplot as plt

#%% - Tester 
reconstruction_threshold = 0.5
time_dimension = 100
print_data_to_console = True   # Default [False] 

# Define the size of the array
size = int(np.sqrt(time_dimension))

# Create a NumPy array with the specified size and fill it with values from 0 to 100
data = np.arange(size ** 2).reshape(size, size)
data[size-1][size-2] = time_dimension-1
data[size-1][size-1] = time_dimension

#%% - Functions to test
def custom_normalisation(data, reconstruction_threshold, time_dimension=100):
    """
    # Notes   - The 0 slice problem, the recon_thresh vs 1-reconthresh problem

    the data starts off with values between 0 and time dim (is the 0 an error that needs fixing? confused with no hit)
    First step is to normalise that to between 0 and 1 so dividing the values by the max possible value (time_dimension)

    Next we want to set a reconstruction threshold i.e 0.5 and compress all the actual hits above this point so moving 0-1 to 0.5-1 
    This means the subspace of the original 0-1 norm that is left for us is (1-reconstruction threshold)
    
    So we squash the data into this subspace i.e 0-1 to 0-subpace (i.e 0.5) [!!! i think currently we are sqahin it to recon threshold as in the case of 0.5 that we used it is actaully the same, but for all other numbers it actually should be to the val of subspace ie 1-recon thresh not recon thresh itself]
    Then we raise the data up to max vals so from 0 to subsapce (0.5) to (1-subspace) to 1 i.e 0.5 to 1 (carfull for other numbers 1-subspace is not recon thresh)
    
    Then we move any vals that have the min val i.e 1-subspace to 0 as thay are assumed to have been the 0 vals (although i realise now they also include the real hits that happen in the first timeslice (the 0 slice) this needs fixing!!!)
   
   ####ANOTHER PROBLEM fixed
       data = (data / ( 2 * time_dimension)) + reconstruction_threshold

    changed to 

        data = (data / ( (1/reconstruction_threshold) * time_dimension)) + reconstruction_threshold
   
    """
    #data = (data / ( (1/reconstruction_threshold) * time_dimension)) + reconstruction_threshold
    data = ((data / time_dimension) / (1/(1-reconstruction_threshold))) + reconstruction_threshold

    
    
    for row in data:
        for i, ipt in enumerate(row):
            if ipt == reconstruction_threshold:
                row[i] = 0
    return data

def custom_renormalisation(data, reconstruction_threshold, time_dimension=100):
    """
    the data starts off with values between (1-subspace) to 1 i.e 0.5 to 1 (carfull for other numbers 1-subspace is not recon thresh)
    
    so first want to move value of 1-subspace down to rest on 0    #??????????? this could present a big error problem????? (perhaps its okay becuas eno vals exist with the exact number 1-subspace that woud casue the error becuse they have already all been set to zeros)
    so all values must have 1-subspcae taken away from them

    now the numbers are in range 0 - subspace 

    so to normalise it back to 0-1 we can multiply it by (1/reconstruction_threshold) which gives the number of subspaces that fit in the full 0-1 space

    then we mutiply all values by time_dimesnison to gte back to a range of 0-time dim

 
    """
    data = np.where(data > reconstruction_threshold, ((data - reconstruction_threshold)*(1/(1-reconstruction_threshold)))*(time_dimension), data)
    return data
"""
3D recon

data_output = []
for cdx, row in enumerate(data):
    for idx, num in enumerate(row):
        if num > reconstruction_threshold:
            num -= reconstruction_threshold
            num = num * (time_dimension-1*2)
            data_output.append([cdx,idx,num])
return np.array(data_output)


"""


#%% - Results
data_norm = custom_normalisation(data, reconstruction_threshold, time_dimension)
data_renorm = custom_renormalisation(data_norm, reconstruction_threshold, time_dimension)

if print_data_to_console:
    print("\nInput Data\n", data)
    print("\nNorm Data\n", data_norm)
    print("\nRenorm Data\n", data_renorm)

# compare the arrays using np.isclose()
isclose = np.isclose(data, data_renorm, atol=1e12)

# check if all elements in isclose are True using np.all()
if np.all(isclose):
    print("\nTest Passed. All elements are close within the specified tolerances.")
else:
    print("Test Failed. Not all elements are close within the specified tolerances.")




