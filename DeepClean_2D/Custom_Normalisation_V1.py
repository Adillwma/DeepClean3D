# -*- coding: utf-8 -*-
"""
Custom Normalisation Transform V1
@author: Adill Al-Ashgar
Created on Tue Nov 15 19:02:32 2022

USER NOTICE!
Input data must be tensor, output is tensor with values scaled betwen 0.5 and 1. Additional arguments required specifies the max and min possible values of the input tensor
"""
import torch
import numpy as np

def numpy_normalisatation(data):     #old, takes 0 - 100 data and turns it into 127.5-255 with 0's remaining 0
    output = ((data / 100) * 127.5) + 127.5
    output = np.where(data==0,data, output)
    return output

def custom_np_to_tensor_no_norm(data):
    data = np.expand_dims(data, axis=0)
    data = torch.tensor(data)
    return(data)
    
def custom_normalisation(data, min=0, max=100):
    output = data / (max*2)    #Divides all values in the input tensor (data), by the maximum allowed value in time dimension multiplied by 2 (max*2) this normalises the data between 0 and 0.5
    output = output + 0.5
    output = np.where(data == 0, data, output)
    return(output) 


def custom_tensor_normalisation(data, min=0, max=100):
    #data = np.expand_dims(data, axis=0)
    #data = torch.tensor(data)

    output = torch.div(data, max*2)    #Divides all values in the input tensor (data), by the maximum allowed value in time dimension multiplied by 2 (max*2) this normalises the data between 0 and 0.5
    output = torch.add(output, 0.5)
    output = torch.where(data == 0, data, output)
    return(output) 


#ti = (np.array([0,1,10,100]))
def custom_normalisation_oldv2(data, min=0, max=100):
    data = np.expand_dims(data, axis=0)
    #print(np.shape(data))
    data = torch.tensor(data)
    #1.0 - ((100-val)/100) * 0.5
    #print("IN",data)
    output = torch.negative(data)   #TURNS VAL TO -VAL
    #print("1",output)
    output = torch.add(output, max) # -VAL + 100
    #print("2",output)
    output = torch.div(output, max) # /100
    #print("3",output)
    output = torch.mul(output, 0.5) # * 0.5
    #print("4",output)
    output = torch.negative(output) #TURNS RESULT TO -RESULT
    #print("5",output)
    output = torch.add(output, 1.0) # -RESULT + 1.0
    #print("6",output)
    output = torch.where(data == 0, data, output)
    #print("OUT",output)

    #extra bodge for now
    #output = torch.mul(output, 127.5)  + 127.5 
    #output = torch.add(output, 127.5 )
    #output = torch.where(data == 0, data, output)
    return(output)
#custom_normalisation(ti)

#Function
def custom_normalisation_oldv1(data, min=0, max=100):
    output = torch.div(data, max*2)    #Divides all values in the input tensor (data), by the maximum allowed value in time dimension multiplied by 2 (max*2) this normalises the data between 0 and 0.5
    nonzero_id = torch.nonzero(output)    #Finda all index's of nonzero values in the tensor

    for id in nonzero_id:   #iterates through all the nonzero tensor index's 
        output[id[0]][id[1]][id[2]] = output[id[0]][id[1]][id[2]] + 0.5      #adds 0.5 to all non zero tensor indexs (could have doen this other way round but this is faster as the input is mostly zeros) 
    
    return(output)                        #Outputs tensor of same dimensions and size as input but scaled between 0.5 and 1 unless value is zero in which case it remains 0


#output = custom_normalisation_V1(test_tensor)
#print(output)

def normalisation_reconstruction(data, reconstruction_threshold=0.5, time_dimension=100):
    output_data = data.numpy()
    print(data)
    out = np.zeros([3,3])
    for row, row_data in enumerate(output_data):
        for column, TOF in enumerate(row_data):
            if TOF >= 0.5:
                TOF_denorm = (TOF - reconstruction_threshold) * 2 * time_dimension
                out[row][column] = (TOF_denorm)
    return(out)


#reconstructed_data = normalisation_reconstruction(output)
#print(reconstructed_data)