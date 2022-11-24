# -*- coding: utf-8 -*-
"""
Convolutional Layers Finder V1
@author: Adill Al-Ashgar
Created on Wed Nov 23 2022
"""

#%% - Dependencies
import matplotlib.pyplot as plt
from Conv_layers_output_size_calculator_V2 import conv_calculator

#%% - User Inputs
conv_type = 3 #Select conv type: 0=conv2D, 1=conv2D.Transpose, 2=conv3D, 3=conv3D.Transpose (WARNING: Values other than 0-3 will select conv3D.Transpose)

H_in = 63        # height of the inputs
W_in = 43        # width of the inputs
D_in = 49          # depth of the input (Only used if one of the 3D conv types is selected above)
K = 3              # kernel size (can be an integer of a two-value-integer tuple)
P = 0              # padding  (can be an integer of a two-value-integer tuple)
S = 2              # stride   (can be an integer of a two-value-integer tuple)
D = 1              # dilation (can be an integer of a two-value-integer tuple)
O = 0             # Output padding (used only in the conv Transpose )

#Note: Currently channels are unused in this script, possible improvment? Can add layer channel/parameter calculation from it?
C_in = 16           # number of input channels
C_out = 32          # number of output channels



####

def least_padding_solver(number_of_layers, input_dimensions):
    output_size = 4 
    for i in range (0,number_of_layers):
        if i == 0:
            input_size = 1
        else:
            input_size = output_size
        output_size = output_size*2
        print(input_size, output_size)
        #result = conv_calculator


least_padding_solver(8,[128,88,100])

# %%
