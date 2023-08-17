# -*- coding: utf-8 -*-
"""
DC3D Autoencoder V8 Protected with dynamic input helper 
Created on Mon Feb 20 2022
author: Adill Al-Ashgar
University of Bristol   

# Convoloution + Linear Autoencoder

### Possible Improvements
"""

#%% - Dependancies
import numpy as np
import torch
from torch import nn

#%% - Encoder
class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim, encoder_debug):
        super().__init__()

        # Enables encoder layer shape debug printouts
        self.encoder_debug = encoder_debug

        self.dynamic_input_multiple = 8

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=0),       #Input_channels, Output_channels, Kernal_size, Stride, Padding
            nn.ReLU(True),                                 #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
            nn.Conv2d(8, 16, 3, stride=2, padding=0),      #Input_channels, Output_channels, Kernal_size, Stride, Padding
            nn.BatchNorm2d(16),                            #BatchNorm normalises the outputs as a batch? #!!!. argument is 'num_features' (expected input of size)
            nn.ReLU(True),                                 #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
            nn.Conv2d(16, 32, 3, stride=2, padding=0),     #Input_channels, Output_channels, Kernal_size, Stride, Padding
            nn.ReLU(True)                                  #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
        )
 
        ###Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
    
        ###Linear Encoder Layers
        self.encoder_lin = nn.Sequential(
            nn.Linear(32 * 16 * 11, 128),                   #!!! linear network layer. arguuments are input dimensions/size, output dimensions/size. Takes in data of dimensions 3* 3 *32 and outputs it in 1 dimension of size 128
            nn.ReLU(True),                                #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
            nn.Linear(128, encoded_space_dim)             #Takes in data of 1 dimension with size 128 and outputs it in one dimension of size defined by encoded_space_dim (this is the latent space? the smalle rit is the more comression but thte worse the final fidelity)
        )

    def dynamic_input_size_helper(self, x):
        # Get the size of the input image x
        print("Input x size: ", x.size())
        size_x = x.size()[2]
        size_y = x.size()[3]
        print(size_x, size_y)

        # check that both dimensions are divisible by self.dynamic_input_multiple in turn, if not then pad the image in that dimension to the nearest multiple of self.dynamic_input_multiple
        if size_x % self.dynamic_input_multiple != 0:
            self.adding_x_pixels = self.dynamic_input_multiple - size_x % self.dynamic_input_multiple
            x = nn.functional.pad(x, (0, 0, 0, self.adding_x_pixels))
            print("Resized x to: ", x.size(), "adding", self.adding_x_pixels, "pixels")
        else:
            self.adding_x_pixels = 0

        if size_y % self.dynamic_input_multiple != 0:
            self.adding_y_pixels = self.dynamic_input_multiple - size_y % self.dynamic_input_multiple
            x = nn.functional.pad(x, (0, self.adding_y_pixels, 0, 0))
            print("Resized x to: ", x.size(), "adding", self.adding_y_pixels, "pixels")
        else:
            self.adding_y_pixels = 0

        return x

    def forward(self, x):
        x = nn.functional.pad(x, (4, 4, 4, 4), mode='constant', value=0)
        if self.encoder_debug == 1:
            print("ENCODER LAYER SIZE DEBUG")
            print("Encoder in", x.size())

        x = self.encoder_cnn(x)                           #Runs convoloutional encoder on x #!!! input data???
        if self.encoder_debug == 1:
            print("Encoder CNN out", x.size())
    
        x = self.flatten(x)                               #Runs flatten  on output of conv encoder #!!! what is flatten?
        if self.encoder_debug == 1:
            print("Encoder Flatten out", x.size())
        
        x = self.encoder_lin(x)                           #Runs linear encoder on flattened output 
        if self.encoder_debug == 1:
            print("Encoder Lin out", x.size(),"\n")
            self.encoder_debug = 0      
        
        return x                                          #Return final result

#%% - Decoder
class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim, decoder_debug):
        super().__init__()

        # Enables decoder layer shape debug printouts
        self.decoder_debug = decoder_debug

        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),                                 #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
            nn.Linear(128, 32 * 16 * 11),
            nn.ReLU(True)                                  #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 16, 11))
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=1),             #Input_channels, Output_channels, Kernal_size, Stride, padding(unused), Output_padding
            nn.BatchNorm2d(16),                                                    #BatchNorm normalises the outputs as a batch? #!!!. argument is 'num_features' (expected input of size)
            nn.ReLU(True),                                                         #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),   #Input_channels, Output_channels, Kernal_size, Stride, padding, Output_padding
            nn.BatchNorm2d(8),                                                     #BatchNorm normalises the outputs as a batch? #!!!. argument is 'num_features' (expected input of size)
            nn.ReLU(True),                                                         #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)     #Input_channels, Output_channels, Kernal_size, Stride, padding, Output_padding
        )

    def encoder_link(self, adding_x_pixels, adding_y_pixels):
        self.adding_x_pixels = adding_x_pixels
        self.adding_y_pixels = adding_y_pixels

    def dynamic_output_size_helper(self, x):
        if self.adding_x_pixels != 0:
            x = x[:, :, :-self.adding_x_pixels, :]
        if self.adding_y_pixels != 0:
            x = x[:,  :, :, :-self.adding_y_pixels]
        return x


    def forward(self, x):
        if self.decoder_debug == 1:            
            print("DECODER LAYER SIZE DEBUG")
            print("Decoder in", x.size())   

        x = self.decoder_lin(x)       #Runs linear decoder on x #!!! is x input data? where does it come from??
        if self.decoder_debug == 1:
            print("Decoder Lin out", x.size()) 
        
        x = self.unflatten(x)         #Runs unflatten on output of linear decoder #!!! what is unflatten?
        if self.decoder_debug == 1:
            print("Decoder Unflatten out", x.size())  
        
        x = self.decoder_conv(x)      #Runs convoloutional decoder on output of unflatten
        if self.decoder_debug == 1:
            print("Decoder CNN out", x.size(),"\n")            
            self.decoder_debug = 0         
        
        x = torch.sigmoid(x)          #THIS IS IMPORTANT PART OF FINAL OUTPUT!: Runs sigmoid function which turns the output data values to range (0-1)#!!! ????    Also can use tanh fucntion if wanting outputs from -1 to 1

        # cropping 10 pixels from each edge of the output
        x = x[:, :, 4:-4, 4:-4]

        return x                      #Retuns the final output
    
    
















