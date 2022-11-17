# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 02:45:06 2022
Convoloutional Layer Output Size Calculator V2.0.1
@author: Adill Al-Ashgar
"""
#%% - Dependencies
import matplotlib.pyplot as plt

#%% - User Inputs
conv_type = 2 #Select conv type: 0=conv2D, 1=conv2D.Transpose, 2=conv3D, 3=conv3D.Transpose (WARNING: Values other than 0-3 will select conv3D.Transpose)

H_in = 128          # height of the inputs
W_in = 88          # width of the inputs
D_in = 100          # depth of the input (Only used if one of the 3D conv types is selected above)
K = 3              # kernel size (can be an integer of a two-value-integer tuple)
P = 1              # padding  (can be an integer of a two-value-integer tuple)
S = 2              # stride   (can be an integer of a two-value-integer tuple)
D = 1              # dilation (can be an integer of a two-value-integer tuple)
O = 1              # Output padding (used only in the conv Transpose )

#Note: Currently channels are unused in this script, possible improvment? Can add layer channel/parameter calculation from it?
C_in = 16           # number of input channels
C_out = 32          # number of output channels



#%% - Functions
#Conv2D
def conv_outputs_2d(I, P, D, K, S):
    out = ((I + 2*P - D * (K - 1) - 1 )/S) + 1
    return(out)

#Conv2D.Transpose
def conv_T_outputs_2d(I, P, D, K, S, O):
    out = (I - 1) *S - 2 *P + D * (K - 1) + O + 1  
    return(out)

#Conv3D
def conv_outputs_3d(I, P, D, K, S):
    out = ((I + 2*P - D * (K - 1) - 1 )/S) + 1
    return(out)

#Conv3D.Transpose
def conv_T_outputs_3d(I, P, D, K, S, O):
    out = (I - 1) *S - 2 *P + D * (K - 1) + O + 1  
    return(out)

#%% - Outputs
print("\nCompleted... Remember to round down non integer values.")
if conv_type == 0:      #conv2D
    print("Height:", conv_outputs_2d(H_in, P, D, K, S))
    print("Width:", conv_outputs_2d(W_in, P, D, K, S))

elif conv_type == 1:    #conv2D.Transpose
    print("Height:", conv_T_outputs_2d(H_in, P, D, K, S, O))
    print("Width:", conv_T_outputs_2d(W_in, P, D, K, S, O))

elif conv_type == 2:    #conv3D
    print("Height:", conv_outputs_3d(H_in, P, D, K, S))
    print("Width:", conv_outputs_3d(W_in, P, D, K, S))
    print("Depth:", conv_outputs_3d(D_in, P, D, K, S))

else:                   #conv3D.Transpose
    print("Height:", conv_T_outputs_3d(H_in, P, D, K, S, O))
    print("Width:", conv_T_outputs_3d(W_in, P, D, K, S, O))
    print("Depth:", conv_T_outputs_3d(D_in, P, D, K, S, O))
print("\n")


#%% - Automated Testing    
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