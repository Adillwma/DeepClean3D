# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 2022
Autoencoder 2D V1.0.0
@author: Adill Al-Ashgar
"""

import torch
from torch import nn

###Convoloution + Linear Autoencoder
###Encoder
class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim, encoder_debug, record_activity):
        super().__init__()
        


        self.encoder_debug=encoder_debug
        ###Convolutional Encoder Layers
        
        #Conv2d function takes arguments: (input channels, output channels, kernel size/receiptive field (), stride and padding. # here for more info on arguments https://uob-my.sharepoint.com/personal/ex18871_bristol_ac_uk/_layouts/15/Doc.aspx?sourcedoc={109841d1-79cb-45c2-ac39-0fd24300c883}&action=edit&wd=target%28Code%20Companion.one%7C%2FUntitled%20Page%7C55b8dfad-d53c-4d8e-a2ef-de1376232896%2F%29&wdorigin=NavigationUrl
        
        #Input channels is 1 as the image is black and white so only has luminace values no rgb channels, 
        #Output channels which is the amount of seperate output tensors?????????, this defines the number of seperate kernals that will be run across image, all which produce thier own output arrays
        #The kernal size, is the size of the convolving layer. best visualised at the link above. Ours is set at is 3 so i guess a line of 3 pixels? it can be square ie give a tuple (x,x) but out data has been linearlised for some reason? NOTE: Kernal size can be even but is almost always an odd number to there is symetry round a central pixel
        #Stride is how far the kernal moves across each time, the default is across by one pixel a time untill end of line then back to start of line and down by a pixel, followed by the cycle of across etc again. 
        #    setting the stride to larger values makes it move across jumping pixels, i.e (1,3) the filter moves across by 3 pixels, and then back to begining of line and down by 1 pixel, this is used to downsample data, so that is why our stride is larger than default. 
        #Padding adds padding to the edges of the data array  before the convoloution filtering, the padding value are noramlly zero but can be made other things. this is normally used to mamke the input array same size as the output
        #Dilation (not used param) sets the spread of the kernal, by defualt it is one contigous block, dilation spreads it out. best visulaised in the link above
        
        
        #torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        
        self.encoder_cnn = nn.Sequential(
            #Convolutional encoder layer 1                 
            nn.Conv3d(1, 8, 3, stride=2, padding=1),       #Input_channels, Output_channels, Kernal_size, Stride, Padding
            nn.ReLU(True),                                 #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
            #Convolutional encoder layer 2
            nn.Conv3d(8, 16, 3, stride=2, padding=1),      #Input_channels, Output_channels, Kernal_size, Stride, Padding
            nn.BatchNorm3d(16),                            #BatchNorm normalises the outputs as a batch? #!!!. argument is 'num_features' (expected input of size)
            nn.ReLU(True),                                 #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
            #Convolutional encoder layer 3
            nn.Conv3d(16, 32, 3, stride=2, padding=0),     #Input_channels, Output_channels, Kernal_size, Stride, Padding
            nn.ReLU(True)                                  #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
        )
        
        #NEW Encoder nodes: 
        #input data format = [batchsize, 1, 128 88] # [batchsize, channels, pixels_in_x, pixels_in_y]
        #conv layers: input---Conv_L1---> [batchsize, 8, 64, 44]
        #conv layers: input---Conv_L2---> [batchsize, 16, 32, 22]
        #conv layers: input---Conv_L3---> [batchsize, 32, 15, 10]
        #Flatten layer: input --flat_L1-> [10 * 15 * 32, 512]
        #Linear layer: input--Lin_L1----> [512, encoded_space_dim]        
                                        # [batchsize, 32, 15, 10]
                                                
                                                
        #conv layers: input---T Conv_L1---> [batchsize, 16, 31, 21]
        #conv layers: input---T Conv_L2---> [batchsize, 8, 62, 42]        
        #conv layers: input---T Conv_L3---> [batchsize, 1, 124, 84]  

        
        #OLD Encoder nodes: 
        #input data format = [batchsize, 1, 28, 28] # [batchsize, channels, pixels_in_x, pixels_in_y]
        #conv layers: input---Conv_L1---> [batchsize, 8, 14, 14]
        #conv layers: input---Conv_L2---> [batchsize, 16, 7, 7]
        #conv layers: input---Conv_L3---> [batchsize, 32, 3, 3]
        #Flatten layer: input --flat_L1-> [3 * 3 * 32, 128]
        #Linear layer: input--Lin_L1----> [128, encoded_space_dim]
        
        #Linear layer: input--Lin_L1----> [encoded_space_dim, 128]        
        #Flatten layer: input --flat_L1-> [128, 3 * 3 * 32]        
        #conv layers: input---Conv_L1---> [batchsize, 32, 43, 63]
        #conv layers: input---Conv_L2---> [batchsize, 16, 21, 31]
        #conv layers: input---Conv_L3---> [batchsize, 8, 9,  14]        
        #output data format = [batchsize, 1, 28, 28] # [batchsize, channels, pixels_in_x, pixels_in_y]
        
        
        ###Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        ###Linear Encoder Layers
        
        #nn.Linear arguments are: in_features – size of each input sample, out_features – size of each output sample, bias – If set to False, the layer will not learn an additive bias. Default: True
        self.encoder_lin = nn.Sequential(
            #Linear encoder layer 1  
            nn.Linear(12 * 10 * 15 * 32, 512),                   #!!! linear network layer. arguuments are input dimensions/size, output dimensions/size. Takes in data of dimensions 3* 3 *32 and outputs it in 1 dimension of size 128
            nn.ReLU(True),                                #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
            #Linear encoder layer 2
            nn.Linear(512, encoded_space_dim)             #Takes in data of 1 dimension with size 128 and outputs it in one dimension of size defined by encoded_space_dim (this is the latent space? the smalle rit is the more comression but thte worse the final fidelity)
        )
        
    def forward(self, x):
        if self.encoder_debug == 1:
            print("ENCODER LAYER SIZE DEBUG")
            print("x in", x.size())
        x = self.encoder_cnn(x)                           #Runs convoloutional encoder on x #!!! input data???
        if self.encoder_debug == 1:
            print("x CNN out", x.size())
        x = self.flatten(x)                               #Runs flatten  on output of conv encoder #!!! what is flatten?
        if self.encoder_debug == 1:
            print("x Flatten out", x.size())
        x = self.encoder_lin(x)                           #Runs linear encoder on flattened output 
        if self.encoder_debug == 1:
            print("x Lin out", x.size(),"\n")
        return x                                          #Return final result

###Decoder
class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim, decoder_debug, record_activity):
        super().__init__()


        self.decoder_debug=decoder_debug
        
        ###Linear Decoder Layers
        self.decoder_lin = nn.Sequential(
            #Linear decoder layer 1            
            nn.Linear(encoded_space_dim, 512),
            nn.ReLU(True),                                 #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
            #Linear decoder layer 2
            nn.Linear(512, 12 * 10 * 15 * 32),
            nn.ReLU(True)                                  #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
        )
        ###Unflatten layer
        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 15, 10, 12))

        ###Convolutional Decoder Layers
        #NOTE - as this is the decoder and it must perform the reverse operations to the encoder, instead of using conv2d here ConvTranspose2d is used which is the inverse opperation
        #ConvTransopose2d function takes arguments: (input channels, output channels, kernel size/receiptive field (), stride and padding. # here for more info on arguments https://uob-my.sharepoint.com/personal/ex18871_bristol_ac_uk/_layouts/15/Doc.aspx?sourcedoc={109841d1-79cb-45c2-ac39-0fd24300c883}&action=edit&wd=target%28Code%20Companion.one%7C%2FUntitled%20Page%7C55b8dfad-d53c-4d8e-a2ef-de1376232896%2F%29&wdorigin=NavigationUrl
        
        #Input channels 
        #Output channels which is the amount of seperate output tensors?????????, 
        #The kernal size, 
        #Stride 
        #Padding ??? ##!!!!
        #Output_Padding adds padding to the edges of the data array
        #Dilation (not used param) 
        
        self.decoder_conv = nn.Sequential(
            #Convolutional decoder layer 1
            nn.ConvTranspose3d(32, 16, 3, stride=2, padding=0, output_padding=(1,1,0)),             #Input_channels, Output_channels, Kernal_size, Stride, padding(unused), Output_padding
            nn.BatchNorm3d(16),                                                    #BatchNorm normalises the outputs as a batch? #!!!. argument is 'num_features' (expected input of size)
            nn.ReLU(True),                                                         #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
            #Convolutional decoder layer 2
            nn.ConvTranspose3d(16, 8, 3, stride=2, padding=1, output_padding=(0,0,0)),   #Input_channels, Output_channels, Kernal_size, Stride, padding, Output_padding
            nn.BatchNorm3d(8),                                                     #BatchNorm normalises the outputs as a batch? #!!!. argument is 'num_features' (expected input of size)
            nn.ReLU(True),                                                         #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
            #Convolutional decoder layer 3
            nn.ConvTranspose3d(8, 1, 3, stride=2, padding=0, output_padding=1)     #Input_channels, Output_channels, Kernal_size, Stride, padding, Output_padding
        )
        
    def forward(self, x):
        if self.decoder_debug == 1:            
            print("DECODER LAYER SIZE DEBUG")
            print("x in", x.size())
        x = self.decoder_lin(x)       #Runs linear decoder on x #!!! is x input data? where does it come from??
        if self.decoder_debug == 1:
            print("x Lin out", x.size())            
        x = self.unflatten(x)         #Runs unflatten on output of linear decoder #!!! what is unflatten?
        if self.decoder_debug == 1:
            print("x Unflatten out", x.size())            
        x = self.decoder_conv(x)      #Runs convoloutional decoder on output of unflatten
        if self.decoder_debug == 1:
            print("x CNN out", x.size(),"\n")            
        x = torch.sigmoid(x)          #THIS IS IMPORTANT PART OF FINAL OUTPUT!: Runs sigmoid function which turns the output data values to range (0-1)#!!! ????    Also can use tanh fucntion if wanting outputs from -1 to 1
        return x                      #Retuns the final output
    