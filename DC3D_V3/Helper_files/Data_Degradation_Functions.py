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
def add_noise_points_to_batch_prenorm(input_image_batch, noise_points:int, time_dimension:int):
    """
    This function will take a batch of images and add noise_points number of random noise points to each image in the batch.
    Intended use case is adding noise points to the image before the custom normalisation is applied, 
    the noise points added will be integers between 1 and 'time_dimension' which is the max timestep.

    Args:
        input_image_batch (torch tensor): The input image batch to be degraded. Shape [B, C, H, W]  
        noise_points (int): The number of noise points to add to each image in the batch
        time_dimension (int): The number of time steps in the data set, used to set the upper limit of the noise point values. Default = 100

    Returns:
        image_batch (torch tensor): The degraded image batch. Shape [B, C, H, W]
    """
    image_batch = input_image_batch.clone()
    B, _, H, W = image_batch.shape

    if noise_points > 0:
        num_pixels = H * W

        # Generate different coordinates for each image in the batch
        selected_coords = torch.randint(0, num_pixels, (B, noise_points), device=image_batch.device)
        x_coords = selected_coords // W
        y_coords = selected_coords % W

        # Generate noise values
        noise_values = torch.randint(1, time_dimension, (B, noise_points), dtype=image_batch.dtype, device=image_batch.device)

        # Create a batch index for each noise point
        batch_indices = torch.arange(B, device=image_batch.device).unsqueeze(1).expand(-1, noise_points)

        # Simulating occlusion caused by 2D embedding, finding those pixels in the batch set to be replaced by noise that have a value below the noise value and replace them with the noise value
        condition = image_batch[batch_indices, 0, x_coords, y_coords] < noise_values
        image_batch[batch_indices, 0, x_coords, y_coords] = torch.where(condition, noise_values, image_batch[batch_indices, 0, x_coords, y_coords])

    return image_batch


# Function to create sparse signal from a fully dense signal
def create_sparse_signal(input_image_batch, signal_points:int, linear=False):

    """
    This function will take a batch of images and randomly select signal_points number of non-zero values to keep in each image, zeroing out all other values i.e if signal_points = 2 then only 2 non-zero values will be kept in each image and all other non zero values will be zeroed out, effectivly simulating a very sparse signal
    
    Args:
        input_image_batch (torch tensor): The input image batch to be degraded. Shape [B, C, H, W]
        signal_points (int): The number of non-zero values to keep in each image
        linear (bool): If set to true then the signal points are linearly spaced across the signal, otherwise they are randomly selected. Default = False
    """
    image_batch = input_image_batch.clone()
    flat_batch = image_batch.view(image_batch.size(0), -1)
    nz_counts = torch.sum(flat_batch != 0, dim=1)
    sparse_indices = torch.where(nz_counts > signal_points)[0]

    for idx in sparse_indices:
        nz_indices = torch.nonzero(flat_batch[idx]).squeeze()
        if linear:
            kept_indices = torch.linspace(0, nz_indices.numel() - 1, steps=signal_points).long()
        else:
            kept_indices = torch.randperm(nz_indices.numel())[:signal_points]

        nonkept_mask = torch.ones(nz_indices.size(0), dtype=torch.bool, device=nz_indices.device)
        nonkept_mask[kept_indices] = False
        nonkept_indices = nz_indices[nonkept_mask]
        flat_batch[idx, nonkept_indices] = 0

    output_image_batch = flat_batch.view_as(image_batch)
    return output_image_batch

# Function to add shift in x, y and ToF to a true signal point due to detector resoloution
def simulate_detector_resolution(input_image_batch, x_std_dev_pixels: int, y_std_dev_pixels: int, tof_std_dev_pixels: int, time_dimension: int, device: torch.device, clamp_photons_to_slice=False):
    """
    This function will add a random shift taken from a gaussain std deviation in x, y or ToF to each non-zero pixel in the image tensor to simulate detector resoloution limits

    Args:
        input_image_batch (torch tensor): The input image batch to be degraded. Shape [B, C, H, W]
        x_std_dev_pixels (int): The standard deviation of the gaussian distribution to draw the x shift from in pixels
        y_std_dev_pixels (int): The standard deviation of the gaussian distribution to draw the y shift from in pixels
        tof_std_dev_pixels (int): The standard deviation of the gaussian distribution to draw the ToF shift from in pixels
        device (torch device): The device to create tensors on
        plot (bool): If set to true then the function will plot the image after the shifts have been applied. Default = False

    Returns:
        image_batch_all (torch tensor): The degraded image batch. Shape [B, C, H, W]
    """
    dtype = input_image_batch.dtype
    image_batch_all = input_image_batch.clone()

    if clamp_photons_to_slice:
        # perform ToF resolution smearing and dont allow photons to move into other slices (unnatural!)
        image_batch_all[image_batch_all != 0] = image_batch_all[image_batch_all != 0] + torch.normal(mean=0, std=tof_std_dev_pixels, size=image_batch_all[image_batch_all != 0].shape, device=device, dtype=dtype)
        image_batch_all = torch.clamp(image_batch_all, 0, time_dimension)
    else:
        # perfrom ToF resolution smearing and allow photons to move into other slices (i.e thrown away)
        image_batch_all[image_batch_all != 0] = image_batch_all[image_batch_all != 0] + torch.normal(mean=0, std=tof_std_dev_pixels, size=image_batch_all[image_batch_all != 0].shape, device=device, dtype=dtype)
        image_batch_all[image_batch_all > time_dimension] = 0
        image_batch_all[image_batch_all < 1] = 0
        
        
    # Squeeze the batch dimension
    image_batch_all_squeezed = image_batch_all.squeeze(1)
    
    # Get the size of the images
    batch_size, x, y = image_batch_all_squeezed.size()
    
    # Generate random shifts for all points the batch
    x_shift = torch.normal(mean=0, std=x_std_dev_pixels, size=(batch_size, x, y), dtype=dtype, device=device)
    y_shift = torch.normal(mean=0, std=y_std_dev_pixels, size=(batch_size, x, y), dtype=dtype, device=device)
    
    # Generate indices for the entire batch
    x_indices = torch.arange(x, device=device).view(1, x, 1).expand(batch_size, x, y)
    y_indices = torch.arange(y, device=device).view(1, 1, y).expand(batch_size, x, y)
    
    # Create new indices by adding the shifts
    new_x_indices = torch.clamp(torch.round(x_indices + x_shift), 0, x - 1).long()
    new_y_indices = torch.clamp(torch.round(y_indices + y_shift), 0, y - 1).long()
    
    # Create a mask for non-zero values in the batch
    mask = image_batch_all_squeezed != 0
    
    # Initialize the shifted image tensor
    shifted_image_tensor = torch.zeros_like(image_batch_all_squeezed)
    
    # Apply the shifts using advanced indexing
    batch_indices = torch.arange(batch_size, device=device).view(-1, 1, 1).expand(batch_size, x, y)
    shifted_image_tensor[batch_indices[mask], new_x_indices[mask], new_y_indices[mask]] = image_batch_all_squeezed[mask]
    
    # Unsqueeze to add the channel dimension back
    image_batch_all[:, 0, :, :] = shifted_image_tensor
    return image_batch_all

def signal_degredation(signal_settings, image_batch, physical_scale_parameters: list[float], time_dimension: int, device: torch.device, timer=None, clamp_photons_to_slice=False):
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

    # Convert physical values to pixel values
    x_std_dev_pixels = x_std_dev_r / x_scale
    y_std_dev_pixels = y_std_dev_r / y_scale
    tof_std_dev_pixels = tof_std_dev_r / time_scale

    # Create the sparse signal
    if signal_points_r:
        timer.record_time(event_name="Signal Degredation: Sparse Signal Creation", event_type="start")
        sparse_output_batch = create_sparse_signal(image_batch, signal_points_r)
        timer.record_time(event_name="Signal Degredation: Sparse Signal Creation", event_type="stop")
    else:
        sparse_output_batch = image_batch.clone()

    # Apply the detector resolution limits
    if x_std_dev_r != 0 or y_std_dev_r != 0 or tof_std_dev_r != 0:
        timer.record_time(event_name="Signal Degredation: Detector Resolution Limits", event_type="start")
        sparse_and_resolution_limited_batch = simulate_detector_resolution(sparse_output_batch, x_std_dev_pixels, y_std_dev_pixels, tof_std_dev_pixels, time_dimension, device, clamp_photons_to_slice)
        timer.record_time(event_name="Signal Degredation: Detector Resolution Limits", event_type="stop")
    else:
        sparse_and_resolution_limited_batch = sparse_output_batch.clone()
    
    # Add noise points to the image
    if noise_points_r != 0:
        timer.record_time(event_name="Signal Degredation: Noise Points", event_type="start")
        noised_sparse_reslimited_batch = add_noise_points_to_batch_prenorm(sparse_and_resolution_limited_batch, noise_points_r, time_dimension)
        timer.record_time(event_name="Signal Degredation: Noise Points", event_type="stop")
    else:
        noised_sparse_reslimited_batch = sparse_and_resolution_limited_batch.clone()
        
    
    return sparse_output_batch, sparse_and_resolution_limited_batch, noised_sparse_reslimited_batch   