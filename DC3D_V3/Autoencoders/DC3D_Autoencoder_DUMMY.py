# -*- coding: utf-8 -*-
"""
DC3D_Autoencoder_DUMMY 
Direct passsthrough for testing purposes
Author: Adill Al-Ashgar
University of Bristol   
"""

#%% - Dependancies
from torch import nn

#%% - Encoder
class Encoder(nn.Module):
    def __init__(self, encoded_space_dim, encoder_debug, fc_layer_size=128):
        super().__init__()
        self.encoder_debug = encoder_debug
        self.fc_layer_size = fc_layer_size
        self.dummy_layer = nn.Identity()  # Dummy layer that does nothing

    def forward(self, x):
        x = self.dummy_layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, encoded_space_dim, decoder_debug, fc_layer_size=128):
        super().__init__()
        self.decoder_debug = decoder_debug
        self.fc_layer_size = fc_layer_size
        self.dummy_layer = nn.Identity()  # Dummy layer that does nothing

    def forward(self, x):
        x = self.dummy_layer(x)
        return x















