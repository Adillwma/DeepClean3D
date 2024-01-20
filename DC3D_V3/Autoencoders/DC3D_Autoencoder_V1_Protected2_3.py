# -*- coding: utf-8 -*-
"""
DC3D Autoencoder V1 Protected   
Created on Mon Feb 20 2022
authors: Adill Al-Ashgar & Max Carter
University of Bristol   

# Convoloution + Linear Autoencoder

### Possible Improvements
# [TESTING!] Fix activation tracking
# 
# 
"""

#%% - Dependancies
import numpy as np
import torch
from torch import nn

#%% - Encoder
class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim, encoder_debug, fc_layer_size=128):
        super().__init__()

        # Enables encoder layer shape debug printouts
        self.encoder_debug = encoder_debug

        self.fc_layer_size = fc_layer_size

        self.encoder_cnn = nn.Sequential(
            # N.B. input channel dimensions are not the same as output channel dimensions:
            # the images will get smaller into the encoded layer
            #Convolutional encoder layer 1                 
            nn.Conv2d(1, 8, 3, stride=2, padding=0),       #Input_channels, Output_channels, Kernal_size, Stride, Padding
            nn.ReLU(True),                                 #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
            #Convolutional encoder layer 2
            nn.Conv2d(8, 16, 3, stride=2, padding=0),      #Input_channels, Output_channels, Kernal_size, Stride, Padding
            nn.BatchNorm2d(16),                            #BatchNorm normalises the outputs as a batch? #!!!. argument is 'num_features' (expected input of size)
            nn.ReLU(True),                                 #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
            #Convolutional encoder layer 3
            nn.Conv2d(16, 32, 3, stride=2, padding=0),     #Input_channels, Output_channels, Kernal_size, Stride, Padding
            nn.ReLU(True)                                  #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
        )
 
        ###Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
    
        ###Linear Encoder Layers
        self.encoder_lin = nn.Sequential(
            nn.Linear(32 * 16 * 11, 32 * 16 * 11),
            nn.ReLU(True),
            nn.Linear(32 * 16 * 11, 32 * 16 * 11),
            nn.ReLU(True),
            nn.Linear(32 * 16 * 11, 32 * 16 * 11),
            nn.ReLU(True),
            #Linear encoder layer 1  
            nn.Linear(32 * 16 * 11, self.fc_layer_size),                   #!!! linear network layer. arguuments are input dimensions/size, output dimensions/size. Takes in data of dimensions 3* 3 *32 and outputs it in 1 dimension of size 128
            # nn.Linear(L3[0] * L3[1] * 32, 128),
            nn.ReLU(True),                                #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
            #Linear encoder layer 2
            nn.Linear(self.fc_layer_size, encoded_space_dim)             #Takes in data of 1 dimension with size 128 and outputs it in one dimension of size defined by encoded_space_dim (this is the latent space? the smalle rit is the more comression but thte worse the final fidelity)
        )

    def forward(self, x):

        # adding padding
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
            self.encoder_debug = 0                    #Turns off debug printouts after first run   
        
        return x                                          #Return final result

#%% - Decoder
class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim, decoder_debug, fc_layer_size=128):
        super().__init__()

        # Enables decoder layer shape debug printouts
        self.decoder_debug = decoder_debug

        self.fc_layer_size = fc_layer_size

        ###Linear Decoder Layers
        self.decoder_lin = nn.Sequential(
            #Linear decoder layer 1            
            nn.Linear(encoded_space_dim, self.fc_layer_size),
            nn.ReLU(True),                                 #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
            #Linear decoder layer 2
            nn.Linear(self.fc_layer_size, 32 * 16 * 11),
            nn.ReLU(True),                                  #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
            nn.Linear(32 * 16 * 11, 32 * 16 * 11),
            nn.ReLU(True),
            nn.Linear(32 * 16 * 11, 32 * 16 * 11),
            nn.ReLU(True),
            nn.Linear(32 * 16 * 11, 32 * 16 * 11),
            nn.ReLU(True),
       
        )
        ###Unflatten layer
        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 16, 11))
        
        self.decoder_conv = nn.Sequential(
            #Convolutional decoder layer 1
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=1),             #Input_channels, Output_channels, Kernal_size, Stride, padding(unused), Output_padding
            nn.BatchNorm2d(16),                                                    #BatchNorm normalises the outputs as a batch? #!!!. argument is 'num_features' (expected input of size)
            nn.ReLU(True),                                                         #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
            #Convolutional decoder layer 2
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),   #Input_channels, Output_channels, Kernal_size, Stride, padding, Output_padding
            nn.BatchNorm2d(8),                                                     #BatchNorm normalises the outputs as a batch? #!!!. argument is 'num_features' (expected input of size)
            nn.ReLU(True),                                                         #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
            #Convolutional decoder layer 3
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)     #Input_channels, Output_channels, Kernal_size, Stride, padding, Output_padding
        )

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
        
        x = torch.sigmoid(x)          # Runs sigmoid function which turns the output data values to range (0-1)  Also can use tanh fucntion if wanting outputs from -1 to 1

        # cropping 10 pixels from each edge of the output
        x = x[:, :, 4:-4, 4:-4]

        return x                      #Retuns the final output
    
    
















