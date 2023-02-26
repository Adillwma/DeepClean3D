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
time_dimension=100
print_data_to_console = False   # Default [False] 
data = np.arrange(0,100)

print("simplest test") #trying to find out if the 2 here is because we usre using recon thresh of 0.5? mayby 2 needs to be determied programatically? maybe 1/recon
test = (data / (2 * time_dimension))
print(test)


#%% - Functions to test
def custom_normalisation(data, reconstruction_threshold, time_dimension=100):
    """
    # Notes   - The 0 slice problem, the recon_thresh vs 1-reconthresh problem

    the data starts off with values between 0 and time dim (is the 0 an error that needs fixing? confused with no hit)
    First step is to normalise that to between 0 and 1 so dividing the values by the max possible value (time_dimension)

    Next we want to set a reconstruction threshold i.e 0.5 and compress all the actual hits above this point so moving 0-1 to 0.5-1 
    This means the subspace of the original 0-1 norm that is left for us is (1-reconstruction threshold)
    
    So we squash the data into this subspace i.e 0-1 to 0-subpace (i.e 0.5) [!!! i think currently we are sqahin it to recon threshold as in the case of 0.5 that we used it is actaully the same, but for all other numbers it actually should be to the val of subspace ie 1-recon thresh not recon thresh itself]
    Then we raise the data up to max vals so from 0 to subsapce (0.5) to (1-subspace) to subspace i.e 0.5 to 1 (carfull for other numbers 1-subspace is not recon thresh)
    
    Then we move any vals that have the min val i.e 1-subspace to 0 as thay are assumed to have been the 0 vals (although i realise now they also include the real hits that happen in the first timeslice (the 0 slice) this needs fixing!!!)
    """
    data = (data / (2 * time_dimension)) + reconstruction_threshold
    for row in data:
        for i, ipt in enumerate(row):
            if ipt == reconstruction_threshold:
                row[i] = 0
    return data

def custom_renormalisation(data, reconstruction_threshold, time_dimension=100):
    data_output = []
    for cdx, row in enumerate(data):
        for idx, num in enumerate(row):
            if num > reconstruction_threshold:
                num -= reconstruction_threshold
                num = num * (time_dimension-1*2)
                data_output.append([cdx,idx,num])
    return np.array(data_output)

#%% - Results
data_norm = custom_normalisation
data_renorm = custom_renormalisation

if print_data_to_console:
    print(data)
    print(data_norm)
    print(data_renorm)

if np.isclose(data, data_renorm, atol=1e12):
    print("Test Pass")
else:
    print("Test Failed")


