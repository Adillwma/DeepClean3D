# -*- coding: utf-8 -*-
"""
DC3D Autoencoder V5  NOTE: added four extra lin layers over V1
Created on Friday March 3 2023
Authors: Adill Al-Ashgar & Max Carter
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
    
    def __init__(self, encoded_space_dim,fc2_input_dim, encoder_debug, record_activity):
        super().__init__()

        # Creates lists for tracking network activation for further analysis
        self.enc_input = list()
        self.enc_conv = list()
        self.enc_flatten = list()
        self.enc_lin = list()

        # Enables encoder layer shape debug printouts
        self.encoder_debug = encoder_debug

        # Enables layer activation rcording
        self.record_activity = record_activity
        
        self.encoder_cnn = nn.Sequential(
            #Convolutional encoder layer 1                 
            nn.Conv2d(1, 8, 3, stride=2, padding=1),       #Input_channels, Output_channels, Kernal_size, Stride, Padding
            nn.ReLU(True),                                 #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
            #Convolutional encoder layer 2
            nn.Conv2d(8, 16, 3, stride=2, padding=1),      #Input_channels, Output_channels, Kernal_size, Stride, Padding
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
            #Linear encoder layer 1  
            nn.Linear(4800, 2048),                   #!!! linear network layer. arguuments are input dimensions/size, output dimensions/size. Takes in data of dimensions 3* 3 *32 and outputs it in 1 dimension of size 128
            nn.ReLU(True),                                #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
            
            #Linear encoder layer 2  
            nn.Linear(2048, 1024),                   #!!! linear network layer. arguuments are input dimensions/size, output dimensions/size. Takes in data of dimensions 3* 3 *32 and outputs it in 1 dimension of size 128
            nn.ReLU(True), 

            #Linear encoder layer 3  
            nn.Linear(1024, 512),                   #!!! linear network layer. arguuments are input dimensions/size, output dimensions/size. Takes in data of dimensions 3* 3 *32 and outputs it in 1 dimension of size 128
            nn.ReLU(True), 
            
            #Linear encoder layer 4  
            nn.Linear(512, 256),                   #!!! linear network layer. arguuments are input dimensions/size, output dimensions/size. Takes in data of dimensions 3* 3 *32 and outputs it in 1 dimension of size 128
            nn.ReLU(True), 
        
            #Linear encoder layer 5  
            nn.Linear(256, 128),                   #!!! linear network layer. arguuments are input dimensions/size, output dimensions/size. Takes in data of dimensions 3* 3 *32 and outputs it in 1 dimension of size 128
            nn.ReLU(True), 
            
            #Linear encoder layer 6
            nn.Linear(128, encoded_space_dim)             #Takes in data of 1 dimension with size 128 and outputs it in one dimension of size defined by encoded_space_dim (this is the latent space? the smalle rit is the more comression but thte worse the final fidelity)
        )

    def get_activation_data(self):        # Returns the per node activation values 
        return self.enc_input, self.enc_conv, self.enc_flatten, self.enc_lin

    def forward(self, x):
        if self.record_activity:
            self.enc_input.append(np.abs(x[0].detach().numpy()))
        if self.encoder_debug == 1:
            print("ENCODER LAYER SIZE DEBUG")
            print("Encoder in", x.size())

        x = self.encoder_cnn(x)                           #Runs convoloutional encoder on x #!!! input data???
        if self.record_activity:
            self.enc_conv.append(np.abs(x[0].detach().numpy()))
        if self.encoder_debug == 1:
            print("Encoder CNN out", x.size())
    
        x = self.flatten(x)                               #Runs flatten  on output of conv encoder #!!! what is flatten?
        if self.record_activity:
            self.enc_flatten.append(np.abs(x[0].detach().numpy()))
        if self.encoder_debug == 1:
            print("Encoder Flatten out", x.size())
        
        x = self.encoder_lin(x)                           #Runs linear encoder on flattened output 
        if self.record_activity:
            self.enc_lin.append(np.abs(x[0].detach().numpy()))
        if self.encoder_debug == 1:
            print("Encoder Lin out", x.size(),"\n")
            self.encoder_debug = 0    #????? what is this line for     
        
        return x                                          #Return final result

#%% - Decoder
class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim, decoder_debug, record_activity):
        super().__init__()

        # Creates lists for tracking network activation for further analysis
        self.dec_input = list()
        self.dec_lin = list()
        self.dec_flatten = list()
        self.dec_conv = list()
        self.dec_out = list()
        
        # Enables decoder layer shape debug printouts
        self.decoder_debug = decoder_debug

        # Enables layer activation rcording
        self.record_activity = record_activity

        ###Linear Decoder Layers
        self.decoder_lin = nn.Sequential(
            #Linear decoder layer 1            
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),                                 #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false

            #Linear decoder layer 2            
            nn.Linear(128, 256),
            nn.ReLU(True), 

            #Linear decoder layer 3            
            nn.Linear(256, 512),
            nn.ReLU(True), 

            #Linear decoder layer 4            
            nn.Linear(512, 1024),
            nn.ReLU(True), 

            #Linear decoder layer 5            
            nn.Linear(1024, 2048),
            nn.ReLU(True), 

            #Linear decoder layer 6
            nn.Linear(2048, 4800),
            nn.ReLU(True)                                  #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
        )
        ###Unflatten layer
        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 15, 10))
        
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

    def get_activation_data(self):    # Returns the per node activation values 
        return self.dec_input, self.dec_lin, self.dec_flatten, self.dec_conv, self.dec_out

    def forward(self, x):
        if self.record_activity:
            self.dec_input.append(np.abs(x[0].detach().numpy()))
        if self.decoder_debug == 1:            
            print("DECODER LAYER SIZE DEBUG")
            print("Decoder in", x.size())   

        x = self.decoder_lin(x)       #Runs linear decoder on x #!!! is x input data? where does it come from??
        if self.record_activity:
            self.dec_lin.append(np.abs(x[0].detach().numpy()))
        if self.decoder_debug == 1:
            print("Decoder Lin out", x.size()) 
        
        x = self.unflatten(x)         #Runs unflatten on output of linear decoder #!!! what is unflatten?
        if self.record_activity:
            self.dec_flatten.append(np.abs(x[0].detach().numpy()))
        if self.decoder_debug == 1:
            print("Decoder Unflatten out", x.size())  
        
        x = self.decoder_conv(x)      #Runs convoloutional decoder on output of unflatten
        if self.record_activity:
            self.dec_conv.append(np.abs(x[0].detach().numpy()))
        if self.decoder_debug == 1:
            print("Decoder CNN out", x.size(),"\n")            
            self.decoder_debug = 0         
        
        x = torch.sigmoid(x)          #THIS IS IMPORTANT PART OF FINAL OUTPUT!: Runs sigmoid function which turns the output data values to range (0-1)#!!! ????    Also can use tanh fucntion if wanting outputs from -1 to 1
        if self.record_activity:
            self.dec_out.append(np.abs(x[0].detach().numpy()))
        
        return x                      #Retuns the final output
    
    