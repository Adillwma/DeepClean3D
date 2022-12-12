# -*- coding: utf-8 -*-
"""
Convolutional Layer Output Size Calculator V3.0
@author: Adill Al-Ashgar
Created on Fri Nov 11 02:45:06 2022

USER NOTICE!
x To use this code standalone as a calculator, just set the parameters below in 'User Inputs' section, 
  then run the entire code, it should then print your results to terminal.

x You can also import this function into another page by putting it into the same 
  root dir and importing conv_calculator
"""

#%% - Dependencies
import matplotlib.pyplot as plt
from math import floor

#%% - User Inputs
conv_type = 1 #Select conv type: 0=conv2D, 1=conv2D.Transpose, 2=conv3D, 3=conv3D.Transpose (WARNING: Values other than 0-3 will select conv3D.Transpose)

H_in = 7         # height of the inputs
W_in = 4         # width of the inputs
D_in = 81         # depth of the input (Only used if one of the 3D conv types is selected above)
K = 3              # kernel size (can be an integer of a two-value-integer tuple)
P = 0            # padding  (can be an integer of a two-value-integer tuple)
S = 2              # stride   (can be an integer of a two-value-integer tuple)
D = 1              # dilation (can be an integer of a two-value-integer tuple)
O = (0,1)           # Output padding (used only in the conv Transpose )

#Note: Currently channels are unused in this script, possible improvment? Can add layer channel/parameter calculation from it?
C_in = 16           # number of input channels
C_out = 32          # number of output channels



#%% - Helper Functions
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

#%% - Wrapper function
def conv_calculator(conv_type, K, P, S, D, H_in, W_in, D_in=0, O=1):
    """
    conv_type = Select convolution type: \n0=conv2D, 1=conv2D.Transpose, 2=conv3D, 3=conv3D.Transpose \n(WARNING: Values other than 0-3 will select conv3D.Transpose)\n
    H_in = height of the inputs\n
    W_in = width of the inputs\n
    D_in = depth of the input (Only used if one of the 3D conv types is selected above)\n
        
    Following values can be input as either an integer or an integer tuple of same dimension as convoloution 2 or 3)\n
    K = kernel size \n
    P = padding \n
    S = stride  \n
    D = dilation \n
    O = output padding (used only in the Transposed convolutions)\n
    """
    if type(K) == int:
        K=[K,K,K]
    print("K =",K)

    if type(P) == int:
        P=[P,P,P]
    print("P =",P)

    if type(S) == int:
        S=[S,S,S]
    print("S =",S)

    if type(D) == int:
        D=[D,D,D]
    print("D =",D)

    if type(O) == int:
        O=[O,O,O]
    print("O =",O)


    print("\nCompleted... Remember to round down non integer values.")
    if conv_type == 0:      #conv2D
        H = conv_outputs_2d(H_in, P[0], D[0], K[0], S[0])
        W = conv_outputs_2d(W_in, P[1], D[1], K[1], S[1])
        print("Height:", H)
        print("Width:", W)

    elif conv_type == 1:    #conv2D.Transpose
        H = conv_T_outputs_2d(H_in, P[0], D[0], K[0], S[0], O[0])
        W = conv_T_outputs_2d(W_in, P[1], D[1], K[1], S[1], O[1])
        print("Height:", H)
        print("Width:", W)

    elif conv_type == 2:    #conv3D
        H = conv_outputs_3d(H_in, P[0], D[0], K[0], S[0])
        W = conv_outputs_3d(W_in, P[1], D[1], K[1], S[1])
        Dep = conv_outputs_3d(D_in, P[2], D[2], K[2], S[2])
        print("Height:", H)
        print("Width:", W)
        print("Depth:", Dep)

    else:                   #conv3D.Transpose
        H = conv_T_outputs_3d(H_in, P[0], D[0], K[0], S[0], O[0])
        W = conv_T_outputs_3d(W_in, P[1], D[1], K[1], S[1], O[1])
        Dep = conv_T_outputs_3d(D_in, P[2], D[2], K[2], S[2], O[2])
        print("Height:", H)
        print("Width:", W)
        print("Depth:", Dep)
    print("\n")

    if conv_type == 0 or conv_type == 1:
        outputs = floor(H), floor(W)
    else:
        outputs = floor(H), floor(W), floor(Dep)
    return(outputs)

#%% - Outputs
#conv_calculator(conv_type, K, P, S, D, H_in, W_in, D_in, O)
start_size = 128, 88
Kernal = (3,3)
CL1 = conv_calculator(conv_type=0, K=Kernal, P=0, S=2, D=1, H_in=start_size[0], W_in=start_size[1], D_in=0, O=0)
CL2 = conv_calculator(conv_type=0, K=Kernal, P=0, S=2, D=1, H_in=CL1[0], W_in=CL1[1], D_in=0, O=0)
CL3 = conv_calculator(conv_type=0, K=Kernal, P=(0,1), S=2, D=1, H_in=CL2[0], W_in=CL2[1], D_in=0, O=0)
CL4 = conv_calculator(conv_type=0, K=Kernal, P=0, S=2, D=1, H_in=CL3[0], W_in=CL3[1], D_in=0, O=0)

CLT1 = conv_calculator(conv_type=1, K=Kernal, P=0, S=2, D=1, H_in=CL4[0], W_in=CL4[1], D_in=0, O=0)
CLT2 = conv_calculator(conv_type=1, K=Kernal, P=(0,1), S=2, D=1, H_in=CLT1[0], W_in=CLT1[1], D_in=0, O=0)
CLT3 = conv_calculator(conv_type=1, K=Kernal, P=0, S=2, D=1, H_in=CLT2[0], W_in=CLT2[1], D_in=0, O=0)
CLT4 = conv_calculator(conv_type=1, K=Kernal, P=0, S=2, D=1, H_in=CLT3[0], W_in=CLT3[1], D_in=0, O=1)

print("Initial Size:", start_size)
print()
print ("Conv 1 Size:", CL1)
print ("Conv 2 Size:", CL2)
print ("Conv 3 Size:", CL3)
print ("Conv 4 Size:", CL4)
print()
print ("Conv.T 1 Size:", CLT1)
print ("Conv.T 2 Size:", CLT2)
print ("Conv.T 3 Size:", CLT3)
print ("Conv.T 4 Size:", CLT4)
#%% - Automated Testing    
run_debugging_tests = 0 #Set to 1 to run the tests, 0 to turn off

if run_debugging_tests == 1:
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