# -*- coding: utf-8 -*-
"""
DC3D_AE_V2_2L.py
Author: Adill Al-Ashgar
University of Bristol   
"""

import torch

#%% - Encoder
class Encoder(torch.nn.Module):
    
    def __init__(self, encoded_space_dim:int, fc_layer_size:int, activation_function=torch.nn.ReLU(True), encoder_debug=False):  
        super().__init__()
        self.encoder_debug = encoder_debug     # Enables encoder layer shape debug printouts
        self.fc_layer_size = fc_layer_size     # Fully connected layers size
        self.activation = activation_function  # General layer activation function

        self.encoder_cnn = torch.nn.Sequential(
            #Convolutional encoder layer 1                 
            torch.nn.Conv2d(1, 8, 3, stride=2, padding=0, bias=False), # OPTIMISATION: If a nn.Conv2d layer is directly followed by a nn.BatchNorm2d layer, then the bias in the convolution is not needed, because in the first step BatchNorm subtracts the mean, which effectively cancels out the effect of bias.
            torch.nn.BatchNorm2d(8),                            
            self.activation,                               

            #Convolutional encoder layer 2
            torch.nn.Conv2d(8, 16, 3, stride=2, padding=0, bias=False),     
            torch.nn.BatchNorm2d(16),                           
            self.activation,                                

            #Convolutional encoder layer 3
            torch.nn.Conv2d(16, 32, 3, stride=2, padding=0, bias=False),  
            torch.nn.BatchNorm2d(32),                          
            self.activation                                 
        )
 
        ###Flatten layer
        self.flatten = torch.nn.Flatten(start_dim=1)
    
        ###Linear Encoder Layers
        self.encoder_lin = torch.nn.Sequential(
            #Linear encoder layer 1  
            torch.nn.Linear(32 * 16 * 11, self.fc_layer_size),                   
            torch.nn.BatchNorm1d(self.fc_layer_size),
            self.activation,                              

            torch.nn.Linear(self.fc_layer_size, self.fc_layer_size),
            torch.nn.BatchNorm1d(self.fc_layer_size),
            self.activation,     

            #Linear encoder layer 2
            torch.nn.Linear(self.fc_layer_size, encoded_space_dim)          
        )

    def forward(self, x):
        # Adding padding 
        x = torch.nn.functional.pad(x, (4, 4, 4, 4), mode='constant', value=0)
        
        if self.encoder_debug == 1:
            print("ENCODER LAYER SIZE DEBUG")
            print("Encoder in", x.size())

        x = self.encoder_cnn(x)                         
        if self.encoder_debug == 1:
            print("Encoder CNN out", x.size())
    
        x = self.flatten(x)                             
        if self.encoder_debug == 1:
            print("Encoder Flatten out", x.size())
        
        x = self.encoder_lin(x)                       
        if self.encoder_debug == 1:
            print("Encoder Lin out", x.size(),"\n")
            self.encoder_debug = 0                    #Turns off debug printouts after first run   
        
        return x    

#%% - Decoder
class Decoder(torch.nn.Module):
    def __init__(self, encoded_space_dim:int, fc_layer_size:int, activation_function=torch.nn.ReLU(True), decoder_debug=False):
        super().__init__()
        self.decoder_debug = decoder_debug      # Enables decoder layer shape debug printouts
        self.fc_layer_size = fc_layer_size
        self.activation = activation_function    # Determine which activation function to use based on input argument

        ###Linear Decoder Layers
        self.decoder_lin = torch.nn.Sequential(
            #Linear decoder layer 1            
            torch.nn.Linear(encoded_space_dim, self.fc_layer_size),
            torch.nn.BatchNorm1d(self.fc_layer_size),
            self.activation,                                
     
            torch.nn.Linear(self.fc_layer_size, self.fc_layer_size),
            torch.nn.BatchNorm1d(self.fc_layer_size),
            self.activation,    
     
            torch.nn.Linear(self.fc_layer_size, 32 * 16 * 11),
            torch.nn.BatchNorm1d(32 * 16 * 11),
            self.activation                                 
        )

        ###Unflatten layer
        self.unflatten = torch.nn.Unflatten(dim=1, 
        unflattened_size=(32, 16, 11))
        
        self.decoder_cnn = torch.nn.Sequential(
            #Convolutional decoder layer 1
            torch.nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=1, bias=False),           
            torch.nn.BatchNorm2d(16),                                                   
            self.activation,                                                        
            
            #Convolutional decoder layer 2
            torch.nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1, bias=False),  
            torch.nn.BatchNorm2d(8),                                                     
            self.activation,                                                        
            
            #Convolutional decoder layer 3
            torch.nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)    
        )

    def forward(self, x):
        if self.decoder_debug == 1:            
            print("DECODER LAYER SIZE DEBUG")
            print("Decoder in", x.size())   

        x = self.decoder_lin(x)       
        if self.decoder_debug == 1:
            print("Decoder Lin out", x.size()) 
        
        x = self.unflatten(x)        
        if self.decoder_debug == 1:
            print("Decoder Unflatten out", x.size())  
        
        x = self.decoder_cnn(x)    
        if self.decoder_debug == 1:
            print("Decoder CNN out", x.size(),"\n")            
            self.decoder_debug = 0         
        
        x = torch.sigmoid(x)  # Runs sigmoid function which turns the output data values to range (0-1)  Also can use tanh fucntion if wanting outputs from -1 to 1 this could give us more resoloution in the output data time values?

        # cropping 10 pixels from each edge of the output to remove the protective padding added in the encoder
        x = x[:, :, 4:-4, 4:-4]

        return x                 
    
    
















