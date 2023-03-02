# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 2022
Noise Generator Functions V1.0.0
Author: Adill Al-Ashgar
University of Bristol

# File not found error already returns well defined error message so not including reporting for it?
"""

import torch
import numpy as np

#%% - Noise functions
#White Additive Noise :
def add_white_noise(clean_input, time_dimension):
    noise = (torch.normal(time_dimension/2,time_dimension/8, clean_input.shape)).clip(0,time_dimension)
    noisy_images = clean_input + noise
    #print("add_white_noise", torch.max(noisy_images))
    return (noisy_images)

#Masking Noise :
def add_masking_noise(clean_input, time_dimension):         ###Simulates pixels failing to fire when struck by photon
    a = 0.7*torch.ones(clean_input.shape)
    noisy_images = clean_input*torch.bernoulli(a)
    #print("add_masking_noise", torch.max(noisy_images))
    return (noisy_images)

#Poisson Noise :
def add_poisson_noise(clean_input, time_dimension):
    a = time_dimension*torch.ones(clean_input.shape)
    p = torch.poisson(a)
    p_norm = p/p.max()
    noisy_images = (clean_input + p_norm).clip(0,time_dimension)
    #print("add_poisson_noise", torch.max(noisy_images))
    return (noisy_images)

#Gaussian Noise:
def add_gaussian_noise(clean_input, time_dimension):
    noise = torch.normal(time_dimension/2,time_dimension/2, clean_input.shape)
    noisy_images = clean_input + noise
    #print("add_gaussian_noise", torch.max(noisy_images))
    return noisy_images

#Impulse Noise:
def add_impulse_noise(clean_input, time_dimension):         #Salt and pepper type noise
    noise = torch.rand(*clean_input.shape)
    noise[noise <= 0.9] = 0
    noise[noise > 0.1] = np.random.random()
    noisy_images = clean_input + (noise*time_dimension)
    return noisy_images


