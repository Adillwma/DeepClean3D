import numpy as np 
import torch 
import matplotlib.pyplot as plt
import os
import re
import importlib.util
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


#%% - DC3D Data Degradation Functions

# Function to add n noise points to each image in a tensor batch(must be used AFTER custom norm 
def add_noise_points_to_batch(input_image_batch, noise_points=100, reconstruction_threshold=0.5):
    """
    This function will take a batch of images and add noise_points number of random noise points to each image in the batch. Intended use case is adding noise points to the image after the custom normalisation is applied, the noise points added will be floats between recon_threshold and 1.

    Args:
        input_image_batch (torch tensor): The input image batch to be degraded. Shape [B, C, H, W]  
        noise_points (int): The number of noise points to add to each image in the batch
        reconstruction_threshold (float): The threshold used in the custom normalisation, used to set the lower limit of the noise point values. Default = 0.5
    
    Returns:
        image_batch (torch tensor): The degraded image batch. Shape [B, C, H, W]
    """

    image_batch = input_image_batch.clone()
    if noise_points > 0:
        #Find dimensions of input image 
        x_dim = image_batch.shape[2]
        y_dim = image_batch.shape[3]

        #For each image in the batch
        for image in image_batch:

            # Create a list of unique random x and y coordinates
            num_pixels = x_dim * y_dim
            all_coords = np.arange(num_pixels)
            selected_coords = np.random.choice(all_coords, noise_points, replace=False)
            x_coords, y_coords = np.unravel_index(selected_coords, (x_dim, y_dim))
            
            # Iterate through noise_points number of random pixels to noise
            for i in range(noise_points):

                # Add a random number between recon_threshold and 1 to the pixel 
                image[0][x_coords[i], y_coords[i]] = np.random.uniform(reconstruction_threshold, 1)

    return image_batch

# Function to add n noise points to each image in a tensor batch 
def add_noise_points_to_batch_prenorm(input_image_batch, noise_points=100, time_dimension=100):
    """
    This function will take a batch of images and add noise_points number of random noise points to each image in the batch. Intended use case is adding noise points to the image before the custom normalisation is applied, the noise points added will be integers between 1 and 'time_dimension' which is the max timestep.

    Args:
        input_image_batch (torch tensor): The input image batch to be degraded. Shape [B, C, H, W]  
        noise_points (int): The number of noise points to add to each image in the batch
        time_dimension (int): The number of time steps in the data set, used to set the upper limit of the noise point values. Default = 100

    Returns:
        image_batch (torch tensor): The degraded image batch. Shape [B, C, H, W]
    """

    image_batch = input_image_batch.clone()
    if noise_points > 0:
        #Find dimensions of input image 
        x_dim = image_batch.shape[2]
        y_dim = image_batch.shape[3]

        #For each image in the batch
        for image in image_batch:

            # Create a list of unique random x and y coordinates
            num_pixels = x_dim * y_dim
            all_coords = np.arange(num_pixels)
            selected_coords = np.random.choice(all_coords, noise_points, replace=False)  ##### NOTE: Change to using a torch random function instead of a numpy one
            x_coords, y_coords = np.unravel_index(selected_coords, (x_dim, y_dim))
            
            # Iterate through noise_points number of random pixels to noise
            for i in range(noise_points):

                # Add a random number between recon_threshold and 1 to the pixel 
                image[0][x_coords[i], y_coords[i]] = np.random.uniform(0, time_dimension)   ##### NOTE: Change to using a torch random function instead of a numpy one

    return image_batch

# Function to create sparse signal from a fully dense signal
def create_sparse_signal(input_image_batch, signal_points=2, linear=False):

    """
    This function will take a batch of images and randomly select signal_points number of non-zero values to keep in each image, zeroing out all other values i.e if signal_points = 2 then only 2 non-zero values will be kept in each image and all other non zero values will be zeroed out, effectivly simulating a very sparse signal
    
    Args:
        input_image_batch (torch tensor): The input image batch to be degraded. Shape [B, C, H, W]
        signal_points (int): The number of non-zero values to keep in each image
        linear (bool): If set to true then the signal points are linearly spaced across the signal, otherwise they are randomly selected. Default = False
    """

    # Take as input a torch tensor in form [batch_size, 1, x_dim, y_dim]]
    # Create a copy of the input image batch
    image_batch = input_image_batch.clone()

    # Flatten the image tensor
    flat_batch = image_batch.view(image_batch.size(0), -1)

    # Count the number of non-zero values in each image
    nz_counts = torch.sum(flat_batch != 0, dim=1)

    # Find the indices of the images that have more non-zero values than signal_points
    sparse_indices = torch.where(nz_counts > signal_points)[0]

    # For each sparse image, randomly select signal_points non-zero values to keep
    for idx in sparse_indices:
        # Find the indices of the non-zero values in the flattened image
        nz_indices = torch.nonzero(flat_batch[idx]).squeeze()

        # Randomly select signal_points non-zero values to keep
        if linear:
            kept_indices = torch.linspace(0, nz_indices.numel() - 1, steps=signal_points).long()
        else:
            kept_indices = torch.randperm(nz_indices.numel())[:signal_points]

        # Zero out all non-selected values
        nonkept_indices = nz_indices[~torch.isin(nz_indices, nz_indices[kept_indices])]
        flat_batch[idx, nonkept_indices] = 0

    # Reshape the flat tensor back into the original shape
    output_image_batch = flat_batch.view_as(image_batch)

    return output_image_batch

# Function to add shift in x, y and ToF to a true signal point due to detector resoloution
def simulate_detector_resolution(input_image_batch, x_std_dev, y_std_dev, tof_std_dev, x_scale, y_scale, time_scale, plot=False):
    """
    This function will add a random shift taken from a gaussain std deviation in x, y or ToF to each non-zero pixel in the image tensor to simulate detector resoloution limits

    Args:
        input_image_batch (torch tensor): The input image batch to be degraded. Shape [B, C, H, W]
        x_std_dev (float): The standard deviation of the gaussian distribution to draw the x shift from
        y_std_dev (float): The standard deviation of the gaussian distribution to draw the y shift from
        tof_std_dev (float): The standard deviation of the gaussian distribution to draw the ToF shift from
        x_scale (float): The physical scale of the x axis in mm per pixel to convert pixel values to physical values
        y_scale (float): The physical scale of the y axis in mm per pixel to convert pixel values to physical values
        time_scale (float): The physical scale of the ToF axis in ns per pixel to convert pixel values to physical values

    Returns:
        image_batch_all (torch tensor): The degraded image batch. Shape [B, C, H, W]
    """

    # Convert physical values to pixel values
    x_std_dev_pixels = x_std_dev / x_scale
    y_std_dev_pixels = y_std_dev / y_scale
    tof_std_dev_pixels = tof_std_dev / time_scale

    # Take as input a torch tensor in form [batch_size, 1, x_dim, y_dim]]
    # Create a copy of the input image batch
    image_batch_all = input_image_batch.clone()

    for idx, image_batch_andc in enumerate(image_batch_all):
        image_batch = image_batch_andc.squeeze()
        # Assume that the S2 image is stored in a variable called "image_tensor"
        x, y = image_batch.size()

        # For all the values in the tensor that are non zero (all signal points) adda random value drawn from a gaussian distribution with mean of the original value and std dev of ToF_std_dev so simulate ToF resoloution limiting
        image_batch[image_batch != 0] = image_batch[image_batch != 0] + torch.normal(mean=0, std=tof_std_dev_pixels, size=image_batch[image_batch != 0].shape)

        # Generate random values for shifting the x and y indices
        x_shift = torch.normal(mean=0, std=x_std_dev_pixels, size=(x, y), dtype=torch.float32)
        y_shift = torch.normal(mean=0, std=y_std_dev_pixels, size=(x, y), dtype=torch.float32)

        # Create a mask for selecting non-zero values in the image tensor
        mask = image_batch != 0

        # Apply the x and y shifts to the non-zero pixel locations
        new_x_indices = torch.clamp(torch.round(torch.arange(x).unsqueeze(1) + x_shift), 0, x - 1).long()
        new_y_indices = torch.clamp(torch.round(torch.arange(y).unsqueeze(0) + y_shift), 0, y - 1).long()
        shifted_image_tensor = torch.zeros_like(image_batch)
        shifted_image_tensor[new_x_indices[mask], new_y_indices[mask]] = image_batch[mask]

        if plot:
            plt.imshow(shifted_image_tensor, cmap='gray', vmin=0, vmax=100)
            plt.title('S')
            plt.show()

        image_batch_all[idx,0] = shifted_image_tensor
        
    return image_batch_all

def signal_degredation(signal_settings, image_batch, physical_scale_parameters):
    """
    Sequentially applies the differnt signal degredation functions to the input image batch and returns the output of each stage

    Args:
        signal_settings (list): A list of the signal degredation settings to be applied to the input image batch. Contains [signal_points_r, x_std_dev_r, y_std_dev_r, tof_std_dev_r, noise_points_r]
        image_batch (torch tensor): The input image batch to be degraded. Shape [B, C, H, W]
        physical_scale_parameters (list): A list of the physical scale parameters to be used to convert pixel values to physical values. Contains [x_scale, y_scale, time_scale]
            x_scale (float): The physical scale of the x axis in mm per pixel to convert pixel values to physical distance values
            y_scale (float): The physical scale of the y axis in mm per pixel to convert pixel values to physical distance values
            time_scale (float): The physical scale of the ToF axis in ns per pixel to convert pixel values to physical time values
    """
    x_scale, y_scale, time_scale = physical_scale_parameters
    signal_points_r, x_std_dev_r, y_std_dev_r, tof_std_dev_r, noise_points_r = signal_settings
    sparse_output_batch = create_sparse_signal(image_batch, signal_points_r)
    sparse_and_resolution_limited_batch = simulate_detector_resolution(sparse_output_batch, x_std_dev_r, y_std_dev_r, tof_std_dev_r, x_scale, y_scale, time_scale)
    noised_sparse_reslimited_batch = add_noise_points_to_batch_prenorm(sparse_and_resolution_limited_batch, noise_points_r, time_dimension)
    return sparse_output_batch, sparse_and_resolution_limited_batch, noised_sparse_reslimited_batch   