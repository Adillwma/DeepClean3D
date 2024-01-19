# -*- coding: utf-8 -*-
"""
Noise Generator Tester
@author: Adill Al-Ashgar
Created on Fri Feb 10 15:32:02 2023
"""
#%% - Dependencies
import torch
import numpy as np
import matplotlib.pyplot as plt

#%% - Noise functions
#White Additive Noise :
def add_white_noise(clean_input, time_dimension):
    noise = (torch.normal(time_dimension/2,time_dimension/2, clean_input.shape)).clip(0,time_dimension)
    noisy_images = clean_input + noise
    return (noisy_images)

#Masking Noise :
def add_masking_noise(clean_input, time_dimension):         ###Simulates pixels failing to fire when struck by photon
    a = 0.7*torch.ones(clean_input.shape)
    noisy_images = clean_input*torch.bernoulli(a)
    return (noisy_images)

#Poisson Noise :
def add_poisson_noise(clean_input, time_dimension):
    """
    Add Poisson noise to an image tensor.
    # The add_poisson_noise() function takes two arguments: the image tensor and the scaling factor 
    # for the Poisson distribution. The torch.poisson() function generates random numbers from a 
    # Poisson distribution.
    Args:
        image (torch.Tensor): The image tensor to which noise will be added.
        time_dimension (float): The value scaling factor for the Poisson distribution.

    Returns:
        torch.Tensor: The noisy image tensor.
    """
    p = torch.poisson(torch.ones(clean_input.shape))
    noise = (p).clip(0,1) * time_dimension
    noised_image = clean_input + noise
    return (noised_image)

#Lorentz noise
def add_lorentzian_noise(image, scale):
    """
    Add Lorentzian noise to an image tensor.

    Args:
        image (torch.Tensor): The image tensor to which noise will be added.
        scale (float): The scaling factor for the Lorentzian distribution.

    Returns:
        torch.Tensor: The noisy image tensor.
    """
    shape = image.shape
    noise = np.random.standard_cauchy(size=shape)
    noisy_image = image + (torch.from_numpy(noise) * time_dimension)
    return noisy_image

#Gaussian Noise:
def add_gaussian_noise(clean_input, time_dimension):
    noise = torch.normal(time_dimension/2,time_dimension/2, clean_input.shape)
    noisy_images = clean_input + noise
    #print("add_gaussian_noise", torch.max(noisy_images))
    return noisy_images

def add_cauchy_noise(image, scale=0.5, loc=44):
    """
    Adds Cauchy/Lorentzian noise to the given image tensor.
    Args:
        image (torch.Tensor): Input image tensor.
        scale (float): Scale parameter for the Cauchy/Lorentzian distribution.
        loc (float): Location parameter for the Cauchy/Lorentzian distribution.
    Returns:
        torch.Tensor: Image tensor with Cauchy/Lorentzian noise added.
    """
    noise = torch.from_numpy(np.random.standard_cauchy(image.shape)).float()
    noise = noise.to(image.device)
    return image + scale * noise + loc





#Brownian noise
def add_brownian_noise(image, scale):
    """
    Add Brownian noise to an image tensor.

    Args:
        image (torch.Tensor): The image tensor to which noise will be added.
        scale (float): The scaling factor for the Brownian noise.

    Returns:
        torch.Tensor: The noisy image tensor.
    """
    noise = torch.randn_like(image) * scale
    noisy_image = image + noise
    return noisy_image

#Impulse Noise:
def add_random_impulse_noise(clean_input, time_dimension):
    shape = clean_input.shape
    noise = torch.zeros(*shape)
    num_of_noise_points = np.random.randint(10,10000)
    for i in range (0, num_of_noise_points):
        noise[np.random.randint(0,shape[0])][np.random.randint(0,shape[1])] = np.random.randint(0, time_dimension)    # Noise points are given values of random integers between 0 and time dim
    noisy_images = clean_input + noise
    return noisy_images

#Impulse Noise:
def add_saltpepper_impulse_noise(clean_input, time_dimension):
    shape = clean_input.shape
    noise = torch.zeros(*shape)
    num_of_noise_points = np.random.randint(10,10000)
    for i in range (0, num_of_noise_points):
        noise[np.random.randint(0,shape[0])][np.random.randint(0,shape[1])] = np.random.randint(0, time_dimension)    # Noise points are given values of random integers between 0 and time dim
    noisy_images = clean_input + noise
    return noisy_images

def add_periodic_impulse_noise(clean_input, time_dimension, period=10):
    shape = clean_input.shape
    noise = torch.zeros(*shape)
    num_of_noise_points = np.random.randint(1000, 10000)
    for i in range(num_of_noise_points):
        t = np.random.randint(0, period) # Choose a random point in the period
        x = np.random.randint(0, shape[0]) # Choose a random x-coordinate
        y = np.random.randint(0, shape[1]) # Choose a random y-coordinate
        # Add the noise as a sine wave with random phase and amplitude
        noise[x][y] += np.random.randint(0, time_dimension) * np.sin(2 * np.pi * t / period + np.random.rand() * 2 * np.pi)
    noisy_images = clean_input + noise
    return noisy_images


#%% Main body
def noise_tester(clean_image):
    image_white_noise = add_white_noise(clean_image, time_dimension)
    image_masking_noise = add_masking_noise(clean_image, time_dimension)
    image_poisson_noise = add_poisson_noise(clean_image, time_dimension)
    image_rnd_impulse_noise = add_random_impulse_noise(clean_image, time_dimension)
    image_periodic_impulse_noise = add_periodic_impulse_noise(clean_image, time_dimension)
    image_gaussian_noise = add_gaussian_noise(clean_image, time_dimension)
    image_lorentzian_noise = add_cauchy_noise(clean_image)
    image_brownian_noise = add_brownian_noise(clean_image, time_dimension)
    images = [clean_image, image_white_noise, image_masking_noise, image_poisson_noise, image_rnd_impulse_noise, image_periodic_impulse_noise, image_gaussian_noise,  image_brownian_noise, image_lorentzian_noise]
    titles = ["Clean Image", "White Noise" ,"Masking Noise", "Poisson Noise","Rnd Impulse Noise", "Periodic Impulse Noise", "Gaussian Noise", "Brownian Noise", "Lorentzian Noise"]
    
    """
    fig, ax = plt.subplots(2, len(images)//2, figsize=(16, 9))
    for i, img in enumerate(images):
        row = i % 2  # calculate the row number
        col = i // 2
        ax[row][col].imshow(img, vmin=0, vmax=time_dimension)
        ax[row][col].set_title(titles[i])
        ax[row][col].axis('off')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.02)  # add some extra vertical space between the rows
    plt.show()
    """                      

    fig, ax = plt.subplots(1, len(images), figsize=(16, 9))
    for i, img in enumerate(images):
        row = i  # calculate the row number
        ax[row].imshow(img, vmin=0, vmax=time_dimension)
        ax[row].set_title(titles[i])
        ax[row].axis('off')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.02)  # add some extra vertical space between the rows
    plt.show()  




EPS = 1e-8  
time_dimension = 100
image_raw = np.zeros((128,88), dtype=float)  
image_raw += 50
#image_raw[:64, :] = 100
#image_raw = np.load("C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/Git Hub Repos/DeepClean Repo/DeepClean-Noise-Suppression-for-LHC-B-Torch-Detector/Datasets/Simple Cross/Data/Simple Cross (flat pixel block data) 5.npy") #np.array([[0,0,9,0,0,0,0],[0,0,5,0,4,0,0],[0,1,4,0,0,6,0],[0,0,0,7,0,0,1],[1,0,1,1,0,7,0]])
#image_raw[10][10] = time_dimension
image_test = torch.tensor(image_raw)


noise_tester(image_test)



