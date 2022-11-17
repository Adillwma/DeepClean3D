# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 02:45:06 2022

@author: Student
"""
#import numpy as np 
import matplotlib.pyplot as plt

def conv_outputs_h(H_in, P_h, D_h, K_h, S_h):
    H_out = ((H_in + 2*P_h - D_h * (K_h - 1) - 1 )/S_h) + 1
    return(H_out)

def conv_outputs_w(W_in, P_w, D_w, K_w, S_w):
    W_out = ((W_in + 2*P_w - D_w * (K_w - 1) - 1)/S_w) + 1
    return(W_out)

def square_conv_outputs(I, P, D, K, S):
    out = ((I + 2*P - D * (K - 1) - 1 )/S) + 1
    return(out)

def square_conv_T_outputs(I, P, D, K, S, O):
    out = (I - 1) *S - 2 *P + D * (K - 1) + O + 1 
    return(out)

C_in = 16           # number of input channels
C_out = 32          # number of output channels


H_in = 64         # height of the inputs
W_in = 44          # width of the inputs
K = 3              # kernel size (can be an integer of a two-value-integer tuple)
P = 1              # padding  (can be an integer of a two-value-integer tuple)
S = 2              # stride   (can be an integer of a two-value-integer tuple)
D = 1              # dilation (can be an integer of a two-value-integer tuple)
O = 1              # Output padding (used only in the conv Transpose )

print("Height:", conv_outputs_w(H_in, P, D, K, S))
print("Width:",conv_outputs_w(W_in, P, D, K, S))

print(".T Height:", square_conv_T_outputs(H_in, P, D, K, S, O))
print(".T Width:",square_conv_T_outputs(W_in, P, D, K, S, O))
run_tests = 0 #Set to 1 to run the tests, 0 to turn off

if run_tests == 1:
    #####TESTS#####
    input_size_range = (1,100) #Range to test accross
    kernal_size_range = (1,100) #Range to test accross
    stride_range = (1,100) #Range to test accross
    padding_range = (0,100) #Range to test accross
    
    input_size = 5 #Default value to use when held as constant
    kernal_size = 3 #Default value to use when held as constant
    stride = 1 #Default value to use when held as constant
    padding = 1 #Default value to use when held as constant
    dilation = 1 #Default value to use when held as constant
    square_conv_outputs(input_size, padding, dilation, kernal_size, stride)
    
    insize_data = []
    kersize_data = []
    stride_data = []
    pad_data = []
    
    idx_data = []
    kdx_data = []
    sdx_data = []
    pdx_data = []
    
    irange = range(input_size_range[0], input_size_range[1]+1)
    krange = range(kernal_size_range[0], kernal_size_range[1]+1)
    srange = range(stride_range[0], stride_range[1]+1)
    prange = range(padding_range[0], padding_range[1]+1)
    
    for idx, input_size in enumerate(irange):
        insize_data.append(square_conv_outputs(input_size, padding, dilation, kernal_size, stride))
        idx_data.append(idx)
        
    plt.plot(idx_data, insize_data)
    plt.title("input_size")
    plt.show()
    
    for kdx, kernal_size in enumerate(krange):
        kersize_data.append(square_conv_outputs(input_size, padding, dilation, kernal_size, stride))
        kdx_data.append(kdx)
        
    plt.plot(kdx_data, kersize_data)
    plt.title("kernal_size")
    plt.show()
            
    for sdx, stride in enumerate(srange):
        stride_data.append(square_conv_outputs(input_size, padding, dilation, kernal_size, stride))
        sdx_data.append(sdx)
        
    plt.plot(sdx_data, stride_data)
    plt.title("stride")
    plt.show()
        
    for pdx, padding in enumerate(prange):
        pad_data.append(square_conv_outputs(input_size, padding, dilation, kernal_size, stride))
        pdx_data.append(pdx) 
        
    plt.plot(pdx_data, pad_data)
    plt.title("padding")
    plt.show()