# -*- coding: utf-8 -*-
"""
DeepClean v0.3.8
Build created on Wednesday March 29th 2023
Author: Adill Al-Ashgar
University of Bristol

@Adill: adillwmaa@gmail.co.uk / ex18871@bristol.ac.uk


Possible improvements:
### ~~~~~ [DONE!] Make sure that autoecoder Encoder and Decoder are saved along with model in the models folder 

### ~~~~~~ [DONE!] Allow normalisation/renorm to be bypassed, to check how it affects results 

### ~~~~~~ [DONE!] Find out what is going on with recon threshold scaling issue

### ~~~~~~ [DONE!] fix noise adding to the data, it is not working as intended, need to retain clean images for label data 

### ~~~~~ Possibly set all to double?
dtype (torch.dtype, optional) – the desired data type of returned tensor. 
Default: if None, uses a global default (see torch.set_default_tensor_type()).!!! 

### ~~~~~ Update the completed prints to terminal after saving things to only say that if the task is actually completeted, atm it will do it regardless, as in the error on .py save 

### ~~~~~ [DONE!] MUST SEPERATE 3D recon and flattening from the normalisation and renormalisation

### ~~~~~ #Noticed we need to update and cleanup all the terminal printing during the visulisations, clean up the weird line spaces and make sure to print what plot is currntly generating as some take a while and the progress bar doesn’t let user know what plot we are currently on

### ~~~~~ Update the model and data paths to folders inside the root dir so that they do not need be defined, and so that people can doanload the git repo and just press run without setting new paths etc 

### ~~~~~ fix epoch numbering printouts? they seem to report 1 epoch greater than they should

### ~~~~~ clear up visulisations

### ~~~~~ Functionalise things

### ~~~~~ Clean up custom autoencoder.py file saver terminal prints left over from debugging

### ~~~~~ [DONE!] Fix Val loss save bug

### ~~~~~ [DONE!] Custom MSE loss fucntion with weighting on zero vals to solve class imbalence

### ~~~~~ [DONE!] Move things to other files (AE, Helper funcs, Visulisations etc)

### ~~~~~ [DONE!] Fix reconstruction threshold, use recon threshold to set bottom limit in custom normalisation

### ~~~~~ [DONE!] Turn plot or save into a function 

### ~~~~~ [DONE!] Add in a way to save the model after each epoch, so that if the program crashes we can still use the last saved model

### ~~~~~ [DONE!] Find way to allow user to exit which is non blocking 

### ~~~~~ [DONE!] Train on labeld data which has the fill line paths in labels and then just points on line in the raw data?

### ~~~~~ Add arg to plot or save function that passes a test string to print for the plot generating user notice ratehr than the generic one used each time atm 

### ~~~~~ change telemetry variable name to output_pixel_telemetry

### ~~~~~ [DONE!] Fix this " if plot_higher_dim: AE_visulisation(en...)" break out all seperate plotting functions
    
### ~~~~~ adapt new version for masking - DeepMask3D 

### ~~~~~ sort out val, test and train properly

### ~~~~~ FC2_INPUT_DIM IS NOT USED!! This would be extremely useful. ?? is this for dynamic layer sizing?

### ~~~~~ Update all layer activation tracking from lists and numpy to torch tensors throughout pipeline for speed

### ~~~~~ Add in automatic Enc/Dec layer size calulations

### ~~~~~ [DONE!] Search for and fix errors in custom norm an renorm

### ~~~~~ [DONE!] Seperate and moularise renorm and 3D reconstruction

### ~~~~~ Create flatten module in main body so noise can be added to the 3D cube rather than slicewise

### ~~~~~ [DONE!] Add way to compress the NPZ output as filesize is to large ! ~3Gb+

### ~~~~~ [DONE!] Add all advanced program settings to end of net summary txt file i.e what typ eof normalisation used etc, also add th enam eof the autoencoder file i.e AE_V1 etc from the module name 

### ~~~~~ update custom mse loss fucntion so that user arguments are set in settings page rather than at function def by defualts i.e (zero_weighting=1, nonzero_weighting=5)

### ~~~~~ could investigate programatically setting the non_zero weighting based on the ratio of zero points to non zero points in the data set which would balance out the two classes in the loss functions eyes

### ~~~~~ Add way for program to save the raw data for 3d plots so that they can be replotted after training and reviewed in rotatable 3d 

### ~~~~~ Check if running model in dp (fp64) is causing large slow down???

### ~~~~~ Allow seperate loss fucntion for testing/validation phase?

### ~~~~~ Properly track the choices for split loss funcs in txt output file 

### ~~~~~ Explicitlly pass in split_loss_functions to the split custom weigted func atm is not done to simplify the code but is not ideal

### ~~~~~ Update noise points to take a range as input and randomly select number for each image from the range

### ~~~~~ Add fcuntion next to noise adder that drops out pixels, then can have the labeld image with high signal points and then dropout the points in the input image to network so as to train it to find dense line from sparse points!
"""
import torch


#NOTE to users: Known good parameters so far (changing these either way damages performance): learning_rate = 0.0001, Batch Size = 10, Latent Dim = 10, Reconstruction Threshold = 0.5, loss_function_selection = 0, loss weighting = 0.9 - 1

#%% - User Inputs
dataset_title =  "Dataset 38_X50K Realistic IDEAL SIG2000"# "Dataset 37_X15K Perfect track recovery" #"Dataset 24_X10Ks"           #"Dataset 12_X10K" ###### TRAIN DATASET : NEED TO ADD TEST DATASET?????
model_save_name = "D38 50K DCDV3 REAL LD14 NP10"     #"D27 100K ld8"#"Dataset 18_X_rotshiftlarge"

time_dimension = 100                         # User controll to set the number of time steps in the data
reconstruction_threshold = 0.5               # MUST BE BETWEEN 0-1  #Threshold for 3d reconstruction, values below this confidence level are discounted

#%% - Hyperparameter Settings
num_epochs = 101                             # User controll to set number of epochs (Hyperparameter)
batch_size = 10                              # User controll to set batch size - number of Images to pull per batch (Hyperparameter) 
latent_dim = 14                              # User controll to set number of nodes in the latent space, the bottleneck layer (Hyperparameter)

learning_rate = 0.0001 #!!                       # User controll to set optimiser learning rate (Hyperparameter)
optim_w_decay = 1e-05 #!!!!1e-07 seeems better?? test!                        # User controll to set optimiser weight decay for regularisation (Hyperparameter)
dropout_prob = 0.2                           # [NOTE Not connected yet] User controll to set dropout probability (Hyperparameter)

train_test_split_ratio = 0.8                 # User controll to set the ratio of the dataset to be used for training (Hyperparameter)
val_test_split_ratio = 0.5                   # [NOTE LEAVE AT 0.5, is for future update, not working currently] User controll to set the ratio of the non-training data to be used for validation as opposed to testing (Hyperparameter)

loss_function_selection = 0                  # Select loss function (Hyperparameter): 0 = ada_weighted_mse_loss, 1 = Maxs_Loss_Func, 2 = torch.nn.MSELoss(), 3 = torch.nn.BCELoss(), 4 = torch.nn.L1Loss(), 5 = ada_SSE_loss, 6 ada_weighted_custom_split_loss 

# Below weights only used if loss func set to 0 or 6 aka ada_weighted_mse_loss
zero_weighting = 0.99                           # User controll to set zero weighting for ada_weighted_mse_loss (Hyperparameter)
nonzero_weighting = 1                     # User controll to set non zero weighting for ada_weighted_mse_loss (Hyperparameter)

# Below only used if loss func set to 6 aka ada_weighted_custom_split_loss
zeros_loss_choice = 1                     # Select loss function for zero values (Hyperparameter): 0 = Maxs_Loss_Func, 1 = torch.nn.MSELoss(), 2 = torch.nn.BCELoss(), 3 = torch.nn.L1Loss(), 4 = ada_SSE_loss
nonzero_loss_choice = 1                 # Select loss function for non zero values (Hyperparameter): 0 = Maxs_Loss_Func, 1 = torch.nn.MSELoss(), 2 = torch.nn.BCELoss(), 3 = torch.nn.L1Loss(), 4 = ada_SSE_loss

# Image Preprocessing Settings  (when using perfect track images as labels)
signal_points = 4000                           # User controll to set the number of signal points to add
noise_points = 10                          # User controll to set the number of noise points to add

x_std_dev = 0                              # User controll to set the standard deviation of the detectors error in the x axis
y_std_dev = 0                               # User controll to set the standard deviation of the detectors error in the y axis
tof_std_dev = 0                             # User controll to set the standard deviation of the detectors error in the time of flight 


#%% - Pretraining settings
start_from_pretrained_model = False          # If set to true then the model will load the pretrained model and optimiser state dicts from the path below
load_pretrained_optimser = False             # Only availible if above is set to true
pretrained_model_path = 'N:/Yr 3 Project Results/D20_3 X5k - Training Results/D20_3 X5k - Model + Optimiser State Dicts.pth'      # Specify the path to the saved full state dictionary for pretraining

#%% - Normalisation Settings 
simple_norm_instead_of_custom = False        #[Default is False] # If set to true then the model will use simple normalisation instead of custom normalisation
all_norm_off = False                         #[Default is False] # If set to true then the model will not use any input normalisation
simple_renorm = False                        #[Default is False] # If set to true then the model will use simple output renormalisation instead of custom output renormalisation

#%% - Plotting Control Settings
print_every_other = 2                      #[default = 2] 1 is to save/print all training plots every epoch, 2 is every other epoch, 3 is every 3rd epoch etc
plot_or_save = 1                           #[default = 1] 0 prints plots to terminal (blocking till closed), If set to 1 then saves all end of epoch printouts to disk (non-blocking), if set to 2 then saves outputs whilst also printing for user (blocking till closed)

#%% - Advanced Visulisation Settings
plot_train_loss = True               #[default = True]       
plot_validation_loss = True          #[default = True]               

plot_cutoff_telemetry = True         #[default = False] # Update name to pixel_cuttoff_telemetry    #Very slow, reduces net performance by XXXXXX%

plot_pixel_difference = False        #[default = True]     
plot_latent_generations = True       #[default = True]      
plot_higher_dim = False              #[default = True]    
plot_Graphwiz = False                #[default = True]   

record_activity = False #False  ##Be carefull, the activity file recorded is ~ 2.5Gb  #Very slow, reduces net performance by XXXXXX%
compress_activations_npz_output = False #False   Compresses the activity file above for smaller file size but does increase loading and saving times for the file. (use if low on hdd space)

#%% - Advanced Debugging Settings
print_encoder_debug = False                     # [default = False]  
print_decoder_debug = False                     # [default = False] 
debug_noise_function = False                    # [default = False]  
debug_loader_batch = False                      # SAFELY REMOVE THIS PARAM!!!  #(Default = False) //INPUT 0 or 1//   #Setting debug loader batch will print to user the images taken in by the dataoader in this current batch and print the corresponding labels

full_dataset_integrity_check = False            # [Default = False] V slow  #Checks the integrity of the dataset by checking shape of each item as opposed to when set to false which only checks one single random file in the dataset
full_dataset_distribution_check = False         # [Default = False] V slow  #Checks the distribution of the dataset , false maesn no distributionn check is done
print_network_summary = False                   # [Default = False] Prints the network summary to terminal
seed = 0                                        # [Default = 0] 0 gives no seeeding to RNG, if the value is not zero then this is used for the RNG seeding for numpy, random, and torch libraries

print_partial_training_losses = False           # [Default = True] Prints the training loss for each batch in the epoch
allow_escape = False                            # [Default = True] Allows the user to escape the training loop at end of eaach epoch (blocking till closed)
#response_timeout = 120 # in seconds            # (NOTE BROKEN) [Default = 120]  If the user does not respond within this time then the training loop will continue 

#%% - Program Mode Setting - CLEAN UP THIS SECTION
#mode = 0 ### 0=Data_Gathering, 1=Testing, 2=Speed_Test, 3=Debugging

speed_test = False      # [speed_test=False]Defualt    true sets number of epocs to print to larger than number of epochs to run so no plotting time wasted etc
data_gathering = True



#%% - Data Path Settings
data_path = "N:\Yr 3 Project Datasets\\"
#ADILL - "C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/Yr 3 Project/Circular and Spherical Dummy Datasets/"
#MAX - 

results_output_path = "N:\Yr 3 Project Results\\"
#ADILL - "C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/Git Hub Repos/DeepClean Repo/DeepClean-Noise-Suppression-for-LHC-B-Torch-Detector/Models/"
#MAX - 

#%% - Dependencies
# External Libraries
import numpy as np  
import matplotlib.pyplot as plt     
import torchvision 
from torchvision import transforms  
from torch.utils.data import DataLoader, random_split
from torch import nn    
import random   
import time     # Used to time the training loop
from torchinfo import summary # function to get the summary of the model layers structure, trainable parameters and memory usage
from tqdm import tqdm  # Progress bar 
import os
from functools import partial
import datetime

# Imports from our custom scripts
from Autoencoders.DC3D_Autoencoder_V1 import Encoder, Decoder # This imports the autoencoder classes from the file selected, changig the V# sets the version of the autoencoder

from Helper_files.Robust_model_exporter_V1 import Robust_model_export   # This is a custom function to export the raw .py file that contains the autoencoder class
from Helper_files.System_Information_check import get_system_information # This is a custom function to get the host system performance specs of the training machine
from Helper_files.Dataset_Integrity_Check_V1 import dataset_integrity_check     # This is a custom function to check the integrity of the datasets values
from Helper_files.Dataset_distribution_tester_V1 import dataset_distribution_tester     # This is a custom function to check the distribution of the datasets values
from Helper_files.AE_Visulisations import Generative_Latent_information_Visulisation, Reduced_Dimension_Data_Representations, Graphwiz_visulisation, AE_visual_difference # These are our custom functions to visulise the autoencoders training progression

#%% - Helper functions

# Weighted Custom Split Loss Function
def ada_weighted_custom_split_loss(reconstructed_image, target_image, zero_weighting=zero_weighting, nonzero_weighting=nonzero_weighting):
    """
    Calculates the weighted error loss between target_image and reconstructed_image.
    The loss for zero pixels in the target_image is weighted by zero_weighting, and the loss for non-zero
    pixels is weighted by nonzero_weighting and both have loss functions as passed in by user.

    Args:
    - target_image: a tensor of shape (B, C, H, W) containing the target image
    - reconstructed_image: a tensor of shape (B, C, H, W) containing the reconstructed image
    - zero_weighting: a scalar weighting coefficient for the MSE loss of zero pixels
    - nonzero_weighting: a scalar weighting coefficient for the MSE loss of non-zero pixels

    Returns:
    - weighted_mse_loss: a scalar tensor containing the weighted MSE loss
    """
    
    # Get the indices of 0 and non 0 values in target_image as a mask for speed
    zero_mask = (target_image == 0)
    nonzero_mask = ~zero_mask         # Invert mask
    
    # Get the values in target_image
    values_zero = target_image[zero_mask]
    values_nonzero = target_image[nonzero_mask]
    
    # Get the corresponding values in reconstructed_image
    corresponding_values_zero = reconstructed_image[zero_mask]
    corresponding_values_nonzero = reconstructed_image[nonzero_mask]
    
    # Get the loss functions
    loss_func_zeros = split_loss_functions[0]
    loss_func_nonzeros = split_loss_functions[1]
    
    # Compute the MSE losses
    zero_loss = loss_func_zeros(corresponding_values_zero, values_zero)
    nonzero_loss = loss_func_nonzeros(corresponding_values_nonzero, values_nonzero)

    # Protection from there being no 0 vals or no non zero vals, which then retunrs nan for MSE and creates a nan overall MSE return (which is error)
    if torch.isnan(zero_loss):
        zero_loss = 0
    if torch.isnan(nonzero_loss):
        nonzero_loss = 0
    
    # Sum losses with weighting coefficiants 
    weighted_mse_loss = (zero_weighting * zero_loss) + (nonzero_weighting * nonzero_loss) 
    
    return weighted_mse_loss



# Custom weighted signal/noise MSE loss function
def ada_weighted_mse_loss(reconstructed_image, target_image, zero_weighting=zero_weighting, nonzero_weighting=nonzero_weighting):
    """
    Calculates the weighted mean squared error (MSE) loss between target_image and reconstructed_image.
    The loss for zero pixels in the target_image is weighted by zero_weighting, and the loss for non-zero
    pixels is weighted by nonzero_weighting.

    Args:
    - target_image: a tensor of shape (B, C, H, W) containing the target image
    - reconstructed_image: a tensor of shape (B, C, H, W) containing the reconstructed image
    - zero_weighting: a scalar weighting coefficient for the MSE loss of zero pixels
    - nonzero_weighting: a scalar weighting coefficient for the MSE loss of non-zero pixels

    Returns:
    - weighted_mse_loss: a scalar tensor containing the weighted MSE loss
    """
    
    # Get the indices of 0 and non 0 values in target_image as a mask for speed
    zero_mask = (target_image == 0)
    nonzero_mask = ~zero_mask         # Invert mask
    
    # Get the values in target_image
    values_zero = target_image[zero_mask]
    values_nonzero = target_image[nonzero_mask]
    
    # Get the corresponding values in reconstructed_image
    corresponding_values_zero = reconstructed_image[zero_mask]
    corresponding_values_nonzero = reconstructed_image[nonzero_mask]
    
    # Create an instance of MSELoss class
    mse_loss = torch.nn.MSELoss(reduction='mean')
    
    # Compute the MSE losses
    zero_loss = mse_loss(corresponding_values_zero, values_zero)
    nonzero_loss = mse_loss(corresponding_values_nonzero, values_nonzero)

    # Protection from there being no 0 vals or no non zero vals, which then retunrs nan for MSE and creates a nan overall MSE return (which is error)
    if torch.isnan(zero_loss):
        zero_loss = 0
    if torch.isnan(nonzero_loss):
        nonzero_loss = 0
    
    # Sum losses with weighting coefficiants 
    weighted_mse_loss = (zero_weighting * zero_loss) + (nonzero_weighting * nonzero_loss) 
    
    return weighted_mse_loss

#  Adaptive Sum of Squared Errors loss function
def ada_SSE_loss(target, input):
    """Adaptive Sum of Squared Errors Loss Function"""
    loss = ((input-target)**2).sum()
    return(loss)

# Custom normalisation function
def custom_normalisation(data, reconstruction_threshold, time_dimension=100):
    data = np.where(data > 0, (((data / time_dimension) / (1/(1-reconstruction_threshold))) + reconstruction_threshold), 0 )
    return data

def custom_normalisation_torch(data, reconstruction_threshold, time_dimension=100):
    data = torch.where(data > 0, (((data / time_dimension) / (1/(1-reconstruction_threshold))) + reconstruction_threshold), 0 )
    return data

# Custom renormalisation function
def custom_renormalisation(data, reconstruction_threshold, time_dimension=100):
    data = np.where(data > reconstruction_threshold, ((data - reconstruction_threshold)*(1/(1-reconstruction_threshold)))*(time_dimension), 0)
    return data

# 3D Reconstruction function
def reconstruct_3D(data, reconstruction_threshold):
    data_output = []
    for cdx, row in enumerate(data):
        for idx, num in enumerate(row):
            if num > 0:  
                data_output.append([cdx,idx,num])
    return np.array(data_output)

# Helper function to clean up repeated plot save/show code
def plot_save_choice(plot_or_save, output_file_path):
    if plot_or_save == 0:
        plt.show()
    else:
        plt.savefig(output_file_path, format='png')    
        if plot_or_save == 1:    
            plt.close()
        else:
            plt.show()

# Helper function to return the batch learning method string to user
def batch_learning(training_dataset_size, batch_size):
    if batch_size == 1: 
        output = "Stochastic Gradient Descent"
    elif batch_size == training_dataset_size:
        output = "Batch Gradient Descent"        
    else:
        output = "Mini-Batch Gradient Descent"
    return(output)

# Tracks confidence of each pixel as histogram per epoch with line showing the detection threshold
def belief_telemetry(data, reconstruction_threshold, epoch, settings, plot_or_save=0):
    data2 = data.flatten()

    #Plots histogram showing the confidence level of each pixel being a signal point
    _, _, bars = plt.hist(data2, 10, histtype='bar')
    plt.axvline(x= reconstruction_threshold, color='red', marker='|', linestyle='dashed', linewidth=2, markersize=12)
    plt.title("Epoch %s" %epoch)
    plt.bar_label(bars, fontsize=10, color='navy') 

    Out_Label = graphics_dir + f'{model_save_name} - Reconstruction Telemetry Histogram - Epoch {epoch}.png'
    plot_save_choice(plot_or_save, Out_Label)

    above_threshold = (data2 >= reconstruction_threshold).sum()
    below_threshold = (data2 < reconstruction_threshold).sum()
    return (above_threshold, below_threshold)

# Plots the confidence telemetry data
def plot_telemetry(telemetry, plot_or_save=0):
    tele = np.array(telemetry)
    #!!! Add labels to lines
    plt.plot(tele[:,0],tele[:,1], color='r', label="Points above threshold") #red = num of points above threshold
    plt.plot(tele[:,0],tele[:,2], color='b', label="Points below threshold") #blue = num of points below threshold
    plt.title("Telemetry over epochs")
    plt.xlabel("Epoch number")
    plt.ylabel("Number of Signal Points")
    plt.legend()
    Out_Label = graphics_dir + f'{model_save_name} - Reconstruction Telemetry Histogram - Epoch {epoch}.png'
    plot_save_choice(plot_or_save, Out_Label)

# Helper fucntion that sets RNG Seeding for determinism in debugging
def Determinism_Seeding(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

# Function to add n noise points to an image 
def add_noise_points(image, noise_points=100, reconstruction_threshold=0.5):

    if noise_points > 0:
        #Find dimensions of input image 
        x_dim = image.shape[0]
        y_dim = image.shape[1]

        #Create a list of random x and y coordinates
        x_coords = np.random.randint(0, x_dim, noise_points)
        y_coords = np.random.randint(0, y_dim, noise_points)

        # Iterate through noise_points number of random pixels to noise
        for i in range(noise_points):

            # Add a random number between recon_threshold and 1 to the pixel 
            image[x_coords[i], y_coords[i]] = np.random.uniform(reconstruction_threshold, 1)

    return image

# Function to add n noise points to each image in a tensor batch(must be used AFTER custom norm 
def add_noise_points_to_batch(input_image_batch, noise_points=100, reconstruction_threshold=0.5):
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
                image[0][x_coords[i], y_coords[i]] = np.random.uniform(0, time_dimension)

    return image_batch

# Function to create sparse signal from a fully dense signal
def create_sparse_signal(input_image_batch, signal_points=2, linear=False):
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
def simulate_detector_resolution(input_image_batch, x_std_dev, y_std_dev, tof_std_dev, plot=False):
    """
    Improve this so it takes pysical values rather than indicies and also so that it uses gaussian to draw randoms from 
    
    """
    # Take as input a torch tensor in form [batch_size, 1, x_dim, y_dim]]
    # Create a copy of the input image batch
    image_batch_all = input_image_batch.clone()

    for idx, image_batch_andc in enumerate(image_batch_all):
        image_batch = image_batch_andc.squeeze()
        # Assume that the S2 image is stored in a variable called "image_tensor"
        x, y = image_batch.size()

        # For all the values in the tensor that are non zero (all signal points) adda random value drawn from a gaussian distribution with mean of the original value and std dev of ToF_std_dev so simulate ToF resoloution limiting
        image_batch[image_batch != 0] = image_batch[image_batch != 0] + torch.normal(mean=0, std=tof_std_dev, size=image_batch[image_batch != 0].shape)

        # Generate random values for shifting the x and y indices
        x_shift = torch.normal(mean=0, std=x_std_dev, size=(x, y), dtype=torch.float32)
        y_shift = torch.normal(mean=0, std=y_std_dev, size=(x, y), dtype=torch.float32)

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

#%% - Classes
class Max_loss(torch.nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean', furthest_line = True, furthest = 1, sig_weight = 30, close_min = 0.05):
        """
        Inputs:
        furthest_line = True - This is set to true to save compute. This will add special loss effect to a LINE of pixels in y-axis
        from the signal point. If set to false will do a square instead of line. (This is much better, but takes more compute)
        furthest - This is how far to extend this special loss function box.
        sig_weight - How much the signal is weighted over the empty points when calculating loss.
        close_min - This is to set the minimum loss in the local minima around the signal points. This is here so that continuing to
        guess close to the signal points is not optimal in the long run.
        """
        super(Max_loss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction
        self.furthest = furthest
        self.sig_weight = sig_weight
        self.close_min = close_min
        self.furthest_line = furthest_line

    def forward(self, reconstruction, original):
        reduction = self.reduction
        furthest = self.furthest
        sig_weight = self.sig_weight
        close_min = self.close_min
        furthest_line = self.furthest_line

        # mse original:
        orig_mseloss = (reconstruction - original)**2

        # first make a list of all the indices of the non-zero original points:
        non_zero = torch.nonzero(original)

        # fix cos im getting the error:
        original_mod = original.clone()

        # set pixels around signal in x to the same as them:
        for img, chan, x, y in non_zero:

            # set all within furthest in x to the signal height:
            if furthest_line:
                original_mod[img, 0, x, y-furthest:y+furthest+1] = original[img,0,x,y]

            # if you want a square around them, not lines (this is much much better for sparse data, not that youd want
            # to use it on that anyway)
            else:
                original_mod[img, 0, x-furthest:x+furthest+1, y-furthest:y+furthest+1] = original[img,0,x,y]
        

        # now we find the minimum of either mse to 0 or to the altered one:
        # cubed for altered (so that it increases faster than 0 MSE):
        alt_mseloss = (reconstruction - original_mod)**3

        # this is where we add the minimum loss the alt_mseloss can get to:
        alt_mseloss += close_min

        # now find the minimum of the two:
        mseloss = torch.min(orig_mseloss, alt_mseloss)

        signals = tuple(torch.nonzero(original).t())

        mseloss[signals] *= sig_weight
        
        # now we just find the mean of the matrix:
        loss = torch.mean(mseloss)

        return loss

#%% - # Input / Output Path Initialisation

# Create output directory if it doesn't exist
dir = results_output_path + model_save_name + " - Training Results/"
os.makedirs(dir, exist_ok=True)

# Create output directory for images if it doesn't exist
graphics_dir = results_output_path + model_save_name + " - Training Results/Output_Graphics/"
os.makedirs(graphics_dir, exist_ok=True)

# Joins up the parts of the differnt output files save paths
full_model_path = dir + model_save_name + " - Model.pth"
full_activity_filepath = dir + model_save_name + " - Activity.npz"
full_netsum_filepath = dir + model_save_name + " - Network Summary.txt"
full_statedict_path = dir + model_save_name + " - Model + Optimiser State Dicts.pth"

# Joins up the parts of the differnt input dataset load paths
train_dir = data_path + dataset_title
test_dir = data_path + dataset_title   #??????????????????????????????????????????


#%% - Parameters Initialisation
# Sets program into speed test mode
if speed_test:                           # If speed test is set to true
    print_every_other = num_epochs + 5   # Makes sure print is larger than total num of epochs to avoid delays in execution for testing

# Initialises pixel belief telemetry
telemetry = [[0,0.5,0.5]]                # Initalises the telemetry memory, starting values are 0, 0.5, 0.5 which corrspond to epoch(0), above_threshold(0.5), below_threshold(0.5)

# Initialises seeding values to RNGs
if seed != 0:                             # If seed is not set to 0
    Determinism_Seeding(seed)             # Set the seed for the RNGs

#%% - Set loss function choice
availible_loss_functions = [ada_weighted_mse_loss, Max_loss(), torch.nn.MSELoss(), torch.nn.BCELoss(), torch.nn.L1Loss(), ada_SSE_loss, ada_weighted_custom_split_loss]    # List of all availible loss functions
loss_fn = availible_loss_functions[loss_function_selection]            # Sets loss function based on user input of parameter loss_function_selection

# Set loss function choice for split loss (if loss function choice is set to ada weighted custom split loss)
availible_split_loss_functions = [Max_loss(), torch.nn.MSELoss(), torch.nn.BCELoss(), torch.nn.L1Loss(), ada_SSE_loss]    # List of all availible loss functions is set to ada_weighted_custom_split_loss
split_loss_functions = [availible_split_loss_functions[zeros_loss_choice], availible_split_loss_functions[nonzero_loss_choice]] # Sets loss functions based on user input


#%% - Create record of all user input settings, to add to output data for testing and keeping track of settings
settings = {}  # Creates empty dictionary to store settings 
settings["Epochs"] = num_epochs     # Adds the number of epochs to the settings dictionary
settings["Batch Size"] = batch_size # Adds the batch size to the settings dictionary
settings["Learning Rate"] = learning_rate # Adds the learning rate to the settings dictionary
settings["Optimiser Decay"] = optim_w_decay # Adds the optimiser decay to the settings dictionary
settings["Dropout Probability"] = dropout_prob # Adds the dropout probability to the settings dictionary
settings["Latent Dimension"] = latent_dim # Adds the latent dimension to the settings dictionary
settings["Noise Points"] = noise_points # Adds the noise points to the settings dictionary
settings["Train Split"] = train_test_split_ratio # Adds the train split to the settings dictionary
settings["Val/Test Split"] = val_test_split_ratio   # Adds the val/test split to the settings dictionary
settings["Dataset"] = dataset_title # Adds the dataset title to the settings dictionary
settings["Time Dimension"] = time_dimension # Adds the time dimension to the settings dictionary
settings["Seed Val"] = seed # Adds the seed value to the settings dictionary
settings["Reconstruction Threshold"] = reconstruction_threshold # Adds the reconstruction threshold to the settings dictionary

#%% - Train Test and Plot Functions

### Training Function
def train_epoch_den(encoder, decoder, device, dataloader, loss_fn, optimizer, signal_points, noise_points=0, x_std_dev=0, y_std_dev=0, tof_std_dev=0, time_dimension=100, reconstruction_threshold=0.5, print_partial_training_losses=False):
    # Set train mode for both the encoder and the decoder
    # train mode makes the autoencoder know the parameters can change
    encoder.train()   
    decoder.train()   
    train_loss = [] # List to store the loss values for each batch
    
    if print_partial_training_losses:  # Prints partial train losses per batch
        image_loop  = (dataloader)     # No progress bar for the batches
    else:                              # Rather than print partial train losses per batch, instead create progress bar
        image_loop  = tqdm(dataloader, desc='Batches', leave=False) # Creates a progress bar for the batches

    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _ in image_loop: # with "_" we just ignore the labels (the second element of the dataloader tuple
        # Move tensor to the proper device

        with torch.no_grad(): # No need to track the gradients
            sparse_output_batch = create_sparse_signal(image_batch, signal_points)
            sparse_and_resolution_limited_batch = simulate_detector_resolution(sparse_output_batch, x_std_dev, y_std_dev, tof_std_dev)
            noised_sparse_reslimited_batch = add_noise_points_to_batch_prenorm(sparse_and_resolution_limited_batch, noise_points, time_dimension)
            normalised_batch = custom_normalisation_torch(noised_sparse_reslimited_batch, reconstruction_threshold, time_dimension)

            normalised_inputs = custom_normalisation_torch(image_batch, reconstruction_threshold, time_dimension)
        image_clean = normalised_inputs.to(device) # Move the clean image batch to the device
        image_noisy = normalised_batch.to(device) # Move the noised image batch to the device
        
        # Encode data
        encoded_data = encoder(image_noisy) # Encode the noised image batch
        # Decode data
        decoded_data = decoder(encoded_data) # Decode the encoded image batch
        
        # Evaluate loss
        loss = loss_fn(decoded_data, image_clean)  # Compute the loss between the decoded image batch and the clean image batch
        
        # Backward pass
        optimizer.zero_grad() # Reset the gradients
        loss.backward() # Compute the gradients
        optimizer.step() # Update the parameters
        train_loss.append(loss.detach().cpu().numpy()) # Store the loss value for the batch

        if print_partial_training_losses:         # Prints partial train losses per batch
            
            print('\t partial train loss (single batch): %f' % (loss.data))  # Print batch loss value
 
    return np.mean(train_loss) # Return the mean loss value for the epoch???

### Testing Function
def test_epoch_den(encoder, decoder, device, dataloader, loss_fn, noise_points=0, reconstruction_threshold=0.5):
    # Set evaluation mode for encoder and decoder
    encoder.eval() # Evaluation mode for the encoder
    decoder.eval() # Evaluation mode for the decoder
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = [] # List to store the output of the network
        conc_label = []  # List to store the original image
        for image_batch, _ in dataloader:  
            # Move tensor to the proper device
            sparse_output_batch = create_sparse_signal(image_batch, signal_points)
            sparse_and_resolution_limited_batch = simulate_detector_resolution(sparse_output_batch, x_std_dev, y_std_dev, tof_std_dev)
            noised_sparse_reslimited_batch = add_noise_points_to_batch_prenorm(sparse_and_resolution_limited_batch, noise_points, time_dimension)
            normalised_batch = custom_normalisation_torch(noised_sparse_reslimited_batch, reconstruction_threshold, time_dimension)

            image_batch_norm = custom_normalisation_torch(image_batch, reconstruction_threshold, time_dimension)

            image_noisy = normalised_batch.to(device) # Move the noised image batch to the device

            # Encode data
            encoded_data = encoder(image_noisy) # Encode the noised image batch
            # Decode data
            decoded_data = decoder(encoded_data) # Decode the encoded image batch

            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu()) # Append the decoded image batch to the list
            conc_label.append(image_batch_norm.cpu()) # Append the clean image batch to the list

        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)  
        conc_label = torch.cat(conc_label) 

        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label) # Compute the loss between the decoded image batch and the clean image batch

    return float(val_loss.data) # Return the loss value for the epoch

### Plotting Function
def plot_ae_outputs_den(encoder, decoder, epoch, model_save_name, time_dimension, reconstruction_threshold, n=10):       #Defines a function for plotting the output of the autoencoder. And also the input + clean training data? Function takes inputs, 'encoder' and 'decoder' which are expected to be classes (defining the encode and decode nets), 'n' which is the number of ?????Images in the batch????, and 'noise_factor' which is a multiplier for the magnitude of the added noise allowing it to be scaled in intensity.  
    """
    n is the number of images to plot
    """
    ### 2D Input/Output Comparison Plots 

    #Initialise lists for true and recovered signal point values 
    number_of_true_signal_points = []
    number_of_recovered_signal_points = []

    plt.figure(figsize=(16,4.5))                                      #Sets the figure size

    for i in range(n):                                                #Runs for loop where 'i' itterates over 'n' total values which range from 0 to n-1
        
      #Following section creates the noised image data drom the original clean labels (images)   
      ax = plt.subplot(3,n,i+1)                                       #Creates a number of subplots for the 'Original images??????' i.e the labels. the position of the subplot is i+1 as it falls in the first row
      
      # Load input image
      img = test_dataset[i][0].unsqueeze(0) # [t_idx[i]][0].unsqueeze(0)                    #!!! ????
      
      #Determine the number of signal points on the input image (have to change this to take it directly from the embeded val in the datsaset as when addig noise this method will break)   
      int_sig_points = (img >= reconstruction_threshold).sum()
      number_of_true_signal_points.append(int(int_sig_points.numpy()))
      
      #if epoch <= print_every_other:                                                  #CHECKS TO SEE IF THE EPOCH IS LESS THAN ZERO , I ADDED THIS TO GET THE SAME NOISED IMAGES EACH EPOCH THOUGH THIS COULD BE WRONG TO DO?
      global image_noisy                                          #'global' means the variable (image_noisy) set inside a function is globally defined, i.e defined also outside the function
      sparse_output_batch = create_sparse_signal(img, signal_points)
      sparse_and_resolution_limited_batch = simulate_detector_resolution(sparse_output_batch, x_std_dev, y_std_dev, tof_std_dev)
      noised_sparse_reslimited_batch = add_noise_points_to_batch_prenorm(sparse_and_resolution_limited_batch, noise_points, time_dimension)
      normalised_batch = custom_normalisation_torch(noised_sparse_reslimited_batch, reconstruction_threshold, time_dimension)
      #image_noisy_list.append(image_noisy)                        #Adds the just generated noise image to the list of all the noisy images
      #image_noisy = image_noisy_list[i].to(device)                    #moves the list (i think of tensors?) to the device that will process it i.e either cpu or gpu, we have a check elsewhere in the code that detects if gpu is availible and sets the value of 'device' to gpu or cpu depending on availibility (look for the line that says "device = 'cuda' if torch.cuda.is_available() else 'cpu'"). NOTE: this moves the noised images to device, i think that the original images are already moved to device in previous code
    
      #Following section sets the autoencoder to evaluation mode rather than training (up till line 'with torch.no_grad()')
      encoder.eval()                                   #.eval() is a kind of switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time. For example, Dropouts Layers, BatchNorm Layers etc. You need to turn off them during model evaluation, and .eval() will do it for you. In addition, the common practice for evaluating/validation is using torch.no_grad() in pair with model.eval() to turn off gradients computation
      decoder.eval()                                   #Simarlary as above

      with torch.no_grad():                                               #As mentioned in .eval() comment, the common practice for evaluating/validation is using torch.no_grad() which turns off gradients computation whilst evaluating the model (the opposite of training the model)     
      #Following line runs the autoencoder on the noised data
         rec_img = decoder(encoder(normalised_batch))                         #Creates a recovered image (denoised image), by running a noisy image through the encoder and then the output of that through the decoder.

      #Determine the number of signal points on the recovered image 
      int_rec_sig_points = (rec_img >= reconstruction_threshold).sum()      
      number_of_recovered_signal_points.append(int(int_rec_sig_points.numpy()))
      
      test_image = img.squeeze() #.cpu().squeeze().numpy()???????????????????????
      noised_test_image = normalised_batch.squeeze() #.cpu().squeeze().numpy()??????????????????
      recovered_test_image = rec_img.cpu().squeeze().numpy()

      #Following section generates the img plots for the original(labels), noised, and denoised data)
      plt.imshow(test_image, cmap='gist_gray')           #plt.imshow plots an image. The arguments for imshow are, 'image data array' and cmap= which is the colour map. #.squeeze() acts on a tensor and returns a tensor, it removes all dimensions of the tensor that are of length 1, (A×1×B) becomes (AxB) where A and B are greater than 1 #.numpy() creates a numpy array from a tensor #!!! is the .cpu part becuase the code was not made to accept the gpu/cpu check i made????
      ax.get_xaxis().set_visible(False)                                   #Hides the x axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
      ax.get_yaxis().set_visible(False)                                   #Hides the y axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
      if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
        ax.set_title('EPOCH %s \nOriginal images' %(epoch))               #When above condition is reached, the plots title is set                                   #When above condition is reached, the plots title is set

      ax = plt.subplot(3, n, i + 1 + n)                                   #Creates a number of subplots for the 'Corrupted images??????' i.e the labels. the position of the subplot is i+1+n as it falls in the second row
      plt.imshow(noised_test_image, cmap='gist_gray')   #plt.imshow plots an image. The arguments for imshow are, 'image data array' and cmap= which is the colour map. #.squeeze() acts on a tensor and returns a tensor, it removes all dimensions of the tensor that are of length 1, (A×1×B) becomes (AxB) where A and B are greater than 1 #.numpy() creates a numpy array from a tensor #!!! is the .cpu part becuase the code was not made to accept the gpu/cpu check i made????
      ax.get_xaxis().set_visible(False)                                   #Hides the x axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
      ax.get_yaxis().set_visible(False)                                   #Hides the y axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
      if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
        ax.set_title('Corrupted images')                                  #When above condition is reached, the plots title is set

      ax = plt.subplot(3, n, i + 1 + n + n)                               #Creates a number of subplots for the 'Reconstructed images??????' i.e the labels. the position of the subplot is i+1+n+n as it falls in the third row
      plt.imshow(recovered_test_image, cmap='gist_gray')       #plt.imshow plots an image. The arguments for imshow are, 'image data array' and cmap= which is the colour map. #.squeeze() acts on a tensor and returns a tensor, it removes all dimensions of the tensor that are of length 1, (A×1×B) becomes (AxB) where A and B are greater than 1 #.numpy() creates a numpy array from a tensor #!!! is the .cpu part becuase the code was not made to accept the gpu/cpu check i made????
      ax.get_xaxis().set_visible(False)                                   #Hides the x axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
      ax.get_yaxis().set_visible(False)                                   #Hides the y axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
      if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
         ax.set_title('Reconstructed images')                             #When above condition is reached, the plots title is set 
    
    plt.subplots_adjust(left=0.1,              #Adjusts the exact layout of the plots including whwite space round edges
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.1, 
                    hspace=0.3)     
    
    ### NEW PLOT SAVING SIMPLIFIED (TESTING CURRENTLY)
    if (epoch) % print_every_other == 0:     #if the epoch is a multiple of print_every_other, then the plot is saved
        Out_Label = graphics_dir + f'{model_save_name} - Epoch {epoch}.png' #creates the name of the file to be saved
        plot_save_choice(plot_or_save, Out_Label) #saves the plot if plot_or_save is set to 1, if 0 it displays, if 2 it displays and saves
        plt.close()
    else:
        plt.close()

    ### 3D Reconstruction Plots 

    in_im = test_image   
    noise_im = noised_test_image
    rec_im = recovered_test_image

    ###################WHY IS NORM HERE IN THE CODE??? shouldent it be directly after output and speed time is saved? (i think this is actually for the plots generated during the testig ratehr than afetr so okay maybe not?)
    # RENORMALISATIONN
    if simple_norm_instead_of_custom or all_norm_off: #if simple_norm_instead_of_custom is set to 1, then the normalisation is done using the simple_renormalisation function, if all_norm_off is set to 1, then no normalisation is done
        rec_im = rec_im * time_dimension #multiplies the reconstructed image by the time dimension   
    else: 

        #REMOVE - used for debugging
        if simple_renorm: 
            noise_im  = noise_im * time_dimension
            rec_im  = rec_im * time_dimension

        else:
            noise_im = custom_renormalisation(noise_im, reconstruction_threshold, time_dimension)
            rec_im = custom_renormalisation(rec_im, reconstruction_threshold, time_dimension)







    # 3D Reconstruction
    in_im = reconstruct_3D(in_im, reconstruction_threshold) #reconstructs the 3D image using the reconstruct_3D function
    noise_im = reconstruct_3D(noise_im, reconstruction_threshold) 
    rec_im = reconstruct_3D(rec_im, reconstruction_threshold)
    
    #3D Plottting
    if rec_im.ndim != 1:                       # Checks if there are actually values in the reconstructed image, if not no image is aseved/plotted
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, subplot_kw={'projection': '3d'})
        #ax1 = plt.axes(projection='3d')
        ax1.scatter(in_im[:,0], in_im[:,1], in_im[:,2]) #plots the 3D scatter plot for input image
        ax1.set_zlim(0, time_dimension)
        ax2.scatter(noise_im[:,0], noise_im[:,1], noise_im[:,2]) #plots the 3D scatter plot for noised image
        ax2.set_zlim(0, time_dimension)
        ax3.scatter(rec_im[:,0], rec_im[:,1], rec_im[:,2]) #plots the 3D scatter plot for reconstructed image
        ax3.set_zlim(0, time_dimension)
        fig.suptitle(f"3D Reconstruction - Epoch {epoch}")
        
        if (epoch) % print_every_other == 0:    #if the epoch is a multiple of print_every_other, then the plot is saved
            Out_Label = graphics_dir + f'{model_save_name} 3D Reconstruction - Epoch {epoch}.png' #creates the name of the file to be saved
            plot_save_choice(plot_or_save, Out_Label) #saves the plot if plot_or_save is set to 1, if 0 it displays, if 2 it displays and saves
        else:
            plt.close()

    if (epoch) % print_every_other == 0:         #if the epoch is a multiple of print_every_other
        #Telemetry plots
        if plot_cutoff_telemetry == 1:      #if plot_cutoff_telemetry is set to 1, then the telemetry plots are generated
            above_threshold, below_threshold = belief_telemetry(recovered_test_image, reconstruction_threshold, epoch+1, settings, plot_or_save)    #calls the belief_telemetry function to generate the telemetry plots
            telemetry.append([epoch, above_threshold, below_threshold]) #appends the telemetry data to the telemetry list


    return(number_of_true_signal_points, number_of_recovered_signal_points, in_im, noise_im, rec_im)        #returns the number of true signal points, number of recovered signal points, input image, noised image and reconstructed image
    
#%% - Program begins
print("\n \nProgram Initalised - Welcome to DC3D Trainer\n")  #prints the welcome message
#%% - Dataset Pre-tests
# Dataset Integrity Check    #???????? aslso perform on train data dir if ther is one?????? 
scantype = "quick" #sets the scan type to quick
if full_dataset_integrity_check: #if full_dataset_integrity_check is set to True, then the scan type is set to full (slow)
    scantype = "full"

print(f"Testing training dataset integrity, with {scantype} scan")  #prints the scan type
dataset_integrity_check(train_dir, full_test=full_dataset_integrity_check, print_output=True) #checks the integrity of the training dataset
print("Test completed\n")

if train_dir != test_dir: #if the training dataset directory is not the same as the test dataset directory, then the test dataset is checked
    print("Testing test dataset signal distribution") #checks the integrity of the test dataset
    dataset_integrity_check(test_dir, full_test=full_dataset_integrity_check, print_output=True) 
    print("Test completed\n")

# Dataset Distribution Check
if full_dataset_distribution_check: #if full_dataset_distribution_check is set to True, then the dataset distribution is checked
    print("\nTesting training dataset signal distribution")
    dataset_distribution_tester(train_dir, time_dimension, ignore_zero_vals_on_plot=True, output_image_dir=graphics_dir) #checks the distribution of the training dataset
    print("Test completed\n")

    if train_dir != test_dir: #if the training dataset directory is not the same as the test dataset directory, then the test dataset is checked
        print("Testing test dataset signal distribution") 
        dataset_distribution_tester(test_dir, time_dimension, ignore_zero_vals_on_plot=True) #checks the distribution of the test dataset
        print("Test completed\n")
    

#%% - Data Loader
"""
The DatasetFolder is a generic DATALOADER. It takes arguments:
root - Root directory path
loader - a function to load a sample given its path
others that arent so relevant....
"""

def train_loader2d(path): #loads the 2D image from the path
    sample = (np.load(path))     
    return (sample) #[0]

def test_loader2d(path): 
    sample = (np.load(path))    
    return (sample) #[0]

def val_loader2d(path):
    sample = (np.load(path))            
    return (sample)


####check for file count in folder####
files_in_path = os.listdir(data_path + dataset_title + '/Data/')  #list of files in path
num_of_files_in_path = len(files_in_path) #number of files in path

# Report type of gradient descent
learning = batch_learning(num_of_files_in_path, batch_size)  #calculates which type of batch learning is being used
print("%s files in path." %num_of_files_in_path ,"// Batch size =",batch_size, "\nLearning via: " + learning,"\n") #prints the number of files in the path and the batch size and the resultant type of batch learning

# - Path images, greater than batch choice? CHECK
if num_of_files_in_path < batch_size: #if the number of files in the path is less than the batch size, user is promted to input a new batch size
    print("Error, the path selected has", num_of_files_in_path, "image files, which is", (batch_size - num_of_files_in_path) , "less than the chosen batch size. Please select a batch size less than the total number of images in the directory")
    batch_err_message = "Choose new batch size, must be less than total amount of images in directory", (num_of_files_in_path) #creates the error message
    batch_size = int(input(batch_err_message))  #!!! not sure why input message is printing with wierd brakets and speech marks in the terminal? Investigate


#train_dir = r'C:\Users\maxsc\OneDrive - University of Bristol\3rd Year Physics\Project\Autoencoder\2D 3D simple version\Circular and Spherical Dummy Datasets\New big simp\Rectangle\\'
train_dataset = torchvision.datasets.DatasetFolder(train_dir, train_loader2d, extensions='.npy') #creates the training dataset using the DatasetFolder function from torchvision.datasets

#test_dir = r'C:\Users\maxsc\OneDrive - University of Bristol\3rd Year Physics\Project\Autoencoder\2D 3D simple version\Circular and Spherical Dummy Datasets\New big simp test\Rectangle\\'
test_dataset = torchvision.datasets.DatasetFolder(test_dir, train_loader2d, extensions='.npy') #creates the test dataset using the DatasetFolder function from torchvision.datasets


#%% - Data Preparation

### Normalisation
####CLEAN UP!!!!!
if simple_norm_instead_of_custom:
    custom_normalisation_with_args = lambda x: x/time_dimension #if simple_norm_instead_of_custom is set to True, then the custom_normalisation_with_args is set to a lambda function that returns the input data divided by the time dimension
else:
    custom_normalisation_with_args = partial(custom_normalisation, reconstruction_threshold=reconstruction_threshold, time_dimension=time_dimension)   #using functools partial to bundle the args into custom norm to use in custom torch transform using lambda function

if all_norm_off:
    custom_normalisation_with_args = lambda x: x #if all_norm_off is set to True, then the custom_normalisation_with_args is set to a lambda function that returns the input data unchanged

add_noise_with_Args = partial(add_noise_points, noise_points=noise_points, reconstruction_threshold=reconstruction_threshold)   #using functools partial to bundle the args into custom norm to use in custom torch transform using lambda function
######!!!!!!!!

### Transormations
train_transform = transforms.Compose([                                         #train_transform variable holds the tensor tranformations to be performed on the training data.  transforms.Compose([ ,  , ]) allows multiple transforms to be chained together (in serial?) (#!!! does it do more than this??)
                                       #transforms.RandomRotation(30),         #transforms.RandomRotation(angle (degrees?) ) rotates the tensor randomly up to max value of angle argument
                                       #transforms.RandomResizedCrop(224),     #transforms.RandomResizedCrop(pixels) crops the data to 'pixels' in height and width (#!!! and (maybe) chooses a random centre point????)
                                       #transforms.RandomHorizontalFlip(),     #transforms.RandomHorizontalFlip() flips the image data horizontally 
                                       #transforms.Normalize((0.5), (0.5)),    #transforms.Normalize can be used to normalise the values in the array
                                       #transforms.Lambda(custom_normalisation_with_args),
                                       #transforms.Lambda(add_noise_with_Args),        ####USed during debugging, noise adding should be moved to later? or maybe not tbf as this is place to add it if wanting it trained on??
                                       transforms.ToTensor(),
                                       #transforms.RandomRotation(180)
                                       ])                 #other transforms can be dissabled but to tensor must be left enabled ! it creates a tensor from a numpy array #!!! ?

test_transform = transforms.Compose([                                          #test_transform variable holds the tensor tranformations to be performed on the evaluation data.  transforms.Compose([ ,  , ]) allows multiple transforms to be chained together (in serial?) (#!!! does it do more than this??)
                                      #transforms.Resize(255),                 #transforms.Resize(pixels? #!!!) ??
                                      #transforms.CenterCrop(224),             #transforms.CenterCrop(pixels? #!!!) ?? Crops the given image at the center. If the image is torch Tensor, it is expected to have […, H, W] shape, where … means an arbitrary number of leading dimensions. If image size is smaller than output size along any edge, image is padded with 0 and then center cropp
                                      #transforms.Normalize((0.5), (0.5)),     #transforms.Normalize can be used to normalise the values in the array
                                      #transforms.Lambda(custom_normalisation_with_args), #transforms.Lambda(function) applies a custom transform function to the data
                                      #transforms.Lambda(add_noise_with_Args),
                                      transforms.ToTensor(),                           #transforms.ToTensor() converts a numpy array to a tensor
                                      #transforms.RandomRotation(180)               #transforms.RandomRotation(angle (degrees?) ) rotates the tensor randomly up to max value of angle argument (if arg is int then it is used as both limits i.e form -180 to +180 deg is the range)
                                      ])                  #other transforms can be dissabled but to tensor must be left enabled !

# this applies above transforms to dataset (dataset transform = transform above)
train_dataset.transform = train_transform       #!!! train_dataset is the class? object 'dataset' it has a subclass called transforms which is the list of transofrms to perform on the dataset when loading it. train_tranforms is the set of chained transofrms we created, this is set to the dataset transforms subclass 
test_dataset.transform = test_transform         #!!! similar to the above but for the test(eval) dataset, check into this for the exact reason for using it, have seen it deone in other ways i.e as in the dataloader.py it is performed differntly. this way seems to be easier to follow
#####For info on all transforms check out: https://pytorch.org/vision/0.9/transforms.html

### Dataset Partitioning


###Following section splits the training dataset into two, train_data (to be noised) and valid data (to use in eval)
m = len(train_dataset)  #m is the length of the train_dataset, i.e the number of images in the dataset
train_split = int(m * train_test_split_ratio) #train_split is the ratio of train images to be used in the training set as opposed to non_training set
train_data, non_training_data = random_split(train_dataset, [train_split, m-train_split])    #random_split(data_to_split, [size of output1, size of output2]) just splits the train_dataset into two parts, 4/5 goes to train_data and 1/5 goes to val_data , validation?

m2 = len(non_training_data)  #m2 is the length of the non_training_data, i.e the number of images in the dataset
val_split = int(m2 * val_test_split_ratio) #val_split is the ratio of npon train images to be used in the validation set as opposed to test set
test_data, val_data = random_split(non_training_data, [m2 - val_split, val_split])  

###Following section for Dataloaders, they just pull a random sample of images from each of the datasets we now have, train_data, valid_data, and test_data. the batch size defines how many are taken from each set, shuffle argument shuffles them each time?? #!!!
# required to load the data into the endoder/decoder. Combines a dataset and a sampler, and provides an iterable over the given dataset.
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)                 #Training data loader, can be run to pull training data as configured  Also is shuffled using parameter shuffle
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)                                 #Testing data loader, can be run to pull training data as configured. 
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)                                 #Validation data loader, can be run to pull training data as configured


#%% - Setup model, loss criteria and optimiser    
### Initialize the encoder and decoder
encoder = Encoder(encoded_space_dim=latent_dim,fc2_input_dim=128, encoder_debug=print_encoder_debug, record_activity=record_activity)
decoder = Decoder(encoded_space_dim=latent_dim,fc2_input_dim=128, decoder_debug=print_decoder_debug, record_activity=record_activity)

# Sets the encoder and decoder to double precision floating point arithmetic (fp64)
encoder.double()   
decoder.double()

### Define the optimizer
params_to_optimize = [{'params': encoder.parameters()} ,{'params': decoder.parameters()}] #Selects what to optimise, 
optim = torch.optim.Adam(params_to_optimize, lr=learning_rate, weight_decay=optim_w_decay)


#%% - Load in pretrained network if needed 
if start_from_pretrained_model:
    # load the full state dictionary into memory
    full_state_dict = torch.load(pretrained_model_path)

    # load the state dictionaries into the models
    encoder.load_state_dict(full_state_dict['encoder_state_dict'])
    decoder.load_state_dict(full_state_dict['decoder_state_dict'])

    if load_pretrained_optimser:
        # load the optimizer state dictionary, if necessary
        optim.load_state_dict(full_state_dict['optimizer_state_dict'])

#%% - Initalise Model on compute device
# Following section checks if a CUDA enabled GPU is available. If found it is selected as the 'device' to perform the tensor opperations. If no CUDA GPU is found the 'device' is set to CPU (much slower) 
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected compute device: {device}\n')  #Informs user if running on CPU or GPU

# Following section moves both the encoder and the decoder to the selected device i.e detected CUDA enabled GPU or to CPU
encoder.to(device)   #Moves encoder to selected device, CPU/GPU
decoder.to(device)   #Moves decoder to selected device, CPU/GPU

#%% - Prepare Network Summary
# Set evaluation mode for encoder and decoder

with torch.no_grad(): # No need to track the gradients
    # Create dummy input tensor
    enc_input_size = (batch_size, 1, 128, 88)
    enc_input_tensor = torch.randn(enc_input_size).double()  # Cast input tensor to double precision

    # Join the encoder and decoder models
    full_network_model = torch.nn.Sequential(encoder, decoder)   #should this be done with no_grad?

    # Generate network summary and then convert to string
    model_stats = summary(full_network_model, input_data=enc_input_tensor, device=device, verbose=0)
    summary_str = str(model_stats)             

    # Print Encoder/Decoder Network Summary
    if print_network_summary:
        print(summary_str)



#%% - Running Training Loop
# this is a dictionary ledger of train val loss history
history_da={'train_loss':[],'val_loss':[]}                   #Just creates a variable called history_da which contains two lists, 'train_loss' and 'val_loss' which are both empty to start with. value are latter appeneded to the two lists by way of history_da['val_loss'].append(x)

print("\nTraining Initiated")

# Begin the training timer
start_time = time.time()

#for j in tqdm(range(5), desc='Inner loop', leave=False
if print_partial_training_losses:  # Prints partial train losses per batch
    loop_range = range(num_epochs)
else:                              # No print partial train losses per batch, instead create progress bar
    loop_range = tqdm(range(num_epochs), desc='Epochs', colour='red')


# bringing everything together to train model
for epoch in loop_range:                              #For loop that iterates over the number of epochs where 'epoch' takes the values (0) to (num_epochs - 1)
    if print_partial_training_losses:
        print('\nStart of EPOCH %d/%d' % (epoch + 1, num_epochs))
    ### Training (use the training function)
    # N.B. train_epoch_den does training phase with encoder/decoder, but only returns the trainloss to show for it. Same w valloss.
    # this has batches built in from dataloader part. Does all train batches.
    # loss for each batch is averaged and single loss produced as output.
    train_loss=train_epoch_den(
                               encoder, 
                               decoder, 
                               device, 
                               dataloader=train_loader, 
                               loss_fn=loss_fn, 
                               optimizer=optim,
                               signal_points=signal_points,
                               noise_points=noise_points,
                               x_std_dev = x_std_dev, 
                               y_std_dev = y_std_dev,
                               tof_std_dev = tof_std_dev,
                               time_dimension=time_dimension,
                               reconstruction_threshold=reconstruction_threshold,
                               print_partial_training_losses = print_partial_training_losses
                               )

    ### Validation (use the testing function)
    # does all validation batches. single average loss produced.
    val_loss = test_epoch_den(
                              encoder, 
                              decoder, 
                              device, 
                              dataloader=valid_loader, 
                              loss_fn=loss_fn,
                              noise_points=noise_points,
                              reconstruction_threshold=reconstruction_threshold)
    
    # Print Validation_loss and plots at end of each epoch
    history_da['train_loss'].append(train_loss)
    history_da['val_loss'].append(val_loss)
    
    #Updates the epoch reached counter  
    max_epoch_reached = epoch    

    if print_partial_training_losses:
        print('\n End of EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))     #epoch +1 is to make up for the fact the range spans 0 to epoch-1 but we want to numerate things from 1 upwards for sanity
    
    if epoch % print_every_other == 0 and epoch != 0:
        
        print("\n## EPOCH {} PLOTS DRAWN ## \n  \n".format(epoch))
        
        # Run plotting function for training feedback and telemetry.
        encoder.eval()
        decoder.eval()

        returned_data_from_plotting_function = plot_ae_outputs_den(encoder, 
                                                                   decoder, 
                                                                   epoch, 
                                                                   model_save_name, 
                                                                   time_dimension, 
                                                                   reconstruction_threshold)
        
        number_of_true_signal_points, number_of_recovered_signal_points, in_data, noisy_data, rec_data = returned_data_from_plotting_function
        encoder.train()
        decoder.train()
        # Allow user to exit training loop    

        if allow_escape:
            user_input = input("Press q to end training, or any other key to continue: \n")
            if user_input == 'q' or user_input == 'Q':
                break

#%% - After Training
# Stop timing the training process and calculate the total training time
end_time = time.time()
training_time = end_time - start_time

# Report the training time
print(f"\nTotal Training Cycle Took {training_time:.2f} seconds")

# Warn user of inaccurate timing if they are not in speed test mode
if speed_test is False:
    ("This timer includes time takes to close plots and respond to inputs, for testing time, set speed_test=True so that no plots are created)\n")

# Build Dictionary to collect all output data for .txt file
full_data_output = {}
full_data_output["Train Loss"] = round(history_da['train_loss'][-1], 3)
full_data_output["Val Loss"] = round(history_da['val_loss'][-1], 3)   #Val loss calulaton is broken? check it above

encoder.eval()
decoder.eval()

#%% NOTE FIX ME!!! 
#!!! NOTE HERE BEFORE THE VISULISATIONS WE SHOULD RUN THE RENORAMLISATION TO RETURN THE CORRECT VALUES FOR VISUALS:

#THEN RECONSTRUCTION CAN HAPPEN LATER???

#%% - Output Visulisations
###Loss function plots
epochs_range = range(1,max_epoch_reached+2) 
if plot_train_loss:
    plt.plot(epochs_range, history_da['train_loss']) 
    plt.title("Training loss")   
    plt.xlabel("Epoch number")
    plt.ylabel("Train loss (MSE)")
    Out_Label =  graphics_dir + f'{model_save_name} - Train loss - Epoch {epoch}.png'
    plot_save_choice(plot_or_save, Out_Label)  

if plot_validation_loss:
    plt.plot(epochs_range, history_da['train_loss'])   #ERROR SHOULD BE VAL LOSS!
    plt.title("Validation loss") 
    plt.xlabel("Epoch number")
    plt.ylabel("Val loss (MSE)")
    Out_Label =  graphics_dir + f'{model_save_name} - Val loss - Epoch {epoch}.png'
    plot_save_choice(plot_or_save, Out_Label)    

if plot_cutoff_telemetry:
    plot_telemetry(telemetry, plot_or_save=plot_or_save)

if plot_pixel_difference:
    num_diff_noised, num_same_noised, num_diff_cleaned, num_same_cleaned, im_diff_noised, im_diff_cleaned = AE_visual_difference(in_data, noisy_data, rec_data)
    full_data_output["num_diff_noised"] = num_diff_noised
    full_data_output["num_same_noised"] = num_same_noised
    full_data_output["num_diff_cleaned"] = num_diff_cleaned
    full_data_output["num_same_cleaned"] = num_same_cleaned

    # Create a new figure
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    # Plot the im_diff_noised subplot
    ax1.imshow(im_diff_noised, cmap='gray')
    ax1.set_title('Original -> Noised Img')

    # Plot the im_diff_cleaned subplot
    ax2.imshow(im_diff_cleaned, cmap='gray')
    ax2.set_title('Original -> Cleaned Img')

    # Add title for the whole figure
    fig.suptitle('Pixel-wise Difference Comparisons')
    
    Out_Label = graphics_dir + f'{model_save_name} - Pixel Difference Epoch {epoch}.png'
    plot_save_choice(plot_or_save, Out_Label)  

if plot_latent_generations:
    def show_image(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    img_recon = Generative_Latent_information_Visulisation(encoder, decoder, latent_dim, device, test_loader)
    
    fig, ax = plt.subplots(figsize=(20, 8.5))
    show_image(torchvision.utils.make_grid(img_recon[:100],10,10, pad_value=100))

    Out_Label = graphics_dir + f'{model_save_name} - Latent Generation Epoch {epoch}.png'
    plot_save_choice(plot_or_save, Out_Label)  

if plot_higher_dim:
    encoded_samples, tsne_results = Reduced_Dimension_Data_Representations(encoder, device, test_dataset, plot_or_save=plot_or_save)
    
    
    # Higher dim
    plt.scatter(encoded_samples['Enc. Variable 0'], encoded_samples['Enc. Variable 1'],
            c=encoded_samples['label'], alpha=0.7)
    plt.grid()
    Out_Label = graphics_dir + f'{model_save_name} - Higher Dimensisions Epoch {epoch}.png'
    plot_save_choice(plot_or_save, Out_Label)  

    # TSNE of Higher dim
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=encoded_samples['label'])
    plt.xlabel('tsne-2d-one')
    plt.ylabel('tsne-2d-two')
    plt.grid()
    Out_Label = graphics_dir + f'{model_save_name} - TSNE Epoch {epoch}.png'
    plot_save_choice(plot_or_save, Out_Label)  
        
if plot_Graphwiz:
    print("Graphwiz Plot Currently Unavailable\n")
    #Out_Label = graphics_dir + f'{model_save_name} - Graphwiz Epoch {epoch}.png'    
    #plot_save_choice(plot_or_save, Out_Label)  
#%% - Export all data logged to disk in form of .txt file in the output dir
if data_gathering:

    print("\nSaving model to .pth file in output dir...")
    # Save and export trained model to user output dir
    torch.save((encoder, decoder), full_model_path)
    print("- Completed -")

    print("\nSaving Autoencoder python file to output dir...")
    # Get the directory name of the current Python file for the autoencoder export
    search_dir = os.path.abspath(__file__)
    search_dir = os.path.dirname(search_dir)

    # Locate .py file that defines the Encoder and Decoder and copies it to the model save dir, due to torch.save model save issues
    AE_file_name = Robust_model_export(Encoder, search_dir, dir) #Only need to run on encoder as encoder and decoder are both in the same file so both get saved this way
    print("- Completed -")

    print("\nSaving model state dictionary to output dir...")
    # Save the state dictionary
    torch.save({'encoder_state_dict': encoder.state_dict(),   
                'decoder_state_dict': decoder.state_dict(),   
                'optimizer_state_dict': optim.state_dict()},  
                full_statedict_path)  
    print("- Completed -") 
    
    print("\nSaving network activations to .npz file in output dir...")

    try:
        # Retriveve network activity and convert to numpy from torch tensors !!! # Simplify the conversion code 
        enc_input, enc_conv, enc_flatten, enc_lin = encoder.get_activation_data()       
        enc_input = np.array(enc_input)     
        enc_conv = np.array(enc_conv)
        enc_flatten = np.array(enc_flatten)
        enc_lin = np.array(enc_lin)

        dec_input, dec_lin, dec_flatten, dec_conv, dec_out = decoder.get_activation_data()
        dec_input = np.array(dec_input)
        dec_lin = np.array(dec_lin)
        dec_flatten = np.array(dec_flatten)
        dec_conv = np.array(dec_conv)
        dec_out = np.array(dec_out)


        # Save network activity for analysis
        if compress_activations_npz_output:    # Saves as a compressed (Zipped) NPZ file (Smaller file size for activations file but takes longer to save and load due to comp/decomp cycles)
            np.savez_compressed((full_activity_filepath), enc_input=enc_input, enc_conv=enc_conv, enc_flatten=enc_flatten, enc_lin=enc_lin, dec_input=dec_input, dec_lin=dec_lin, dec_flatten=dec_flatten, dec_conv=dec_conv, dec_out=dec_out)
        
        else:                                 # Saves as an uncompressed NPZ file (large file size ~3Gb+ but faster to load and save files, prefered if hdd space no issue)
            np.savez((full_activity_filepath), enc_input=enc_input, enc_conv=enc_conv, enc_flatten=enc_flatten, enc_lin=enc_lin, dec_input=dec_input, dec_lin=dec_lin, dec_flatten=dec_flatten, dec_conv=dec_conv, dec_out=dec_out)
        
        print("- Completed -")
    
    except:
        print("- NPZ save failed (Check output disk has enough free space ~3Gb+) -")

    print("\nSaving inputs, model stats and results to .txt file in output dir...")
    #Comparison of true signal points to recovered signal points
    try: 
        #print("True signal points",number_of_true_signal_points)
        #print("Recovered signal points: ",number_of_recovered_signal_points, "\n")
        full_data_output["true_signal_points"] = number_of_true_signal_points  # Save the number of true signal points to the full_data_output dictionary
        full_data_output["recovered_signal_points"] = number_of_recovered_signal_points  # Save the number of recovered signal points to the full_data_output dictionary
    except:
        pass
    
    # Save .txt Encoder/Decoder Network Summary
    with open(full_netsum_filepath, 'w', encoding='utf-8') as output_file:    #utf_8 encoding needed as default (cp1252) unable to write special charecters present in the summary
        # Write the local date and time to the file
        TD_now = datetime.datetime.now()         # Get the current local date and time
        output_file.write(f"Date data taken: {TD_now.strftime('%Y-%m-%d %H:%M:%S')}\n")     # Write the current local date and time to the file

        output_file.write(("Model ID: " + model_save_name + f"\nTrained on device: {device}\n"))   # Write the model ID and device used to train the model to the file
        output_file.write((f"\nMax Epoch Reached: {max_epoch_reached}\n"))  # Write the max epoch reached during training to the file
        
        timer_warning = "(Not accurate - not recorded in speed_test mode)\n" # Warning message to be written to the file if the timer is not accurate
        if speed_test:
            timer_warning = "\n"
        output_file.write((f"Training Time: {training_time:.2f} seconds\n{timer_warning}\n")) # Write the training time to the file
        
        output_file.write("Input Settings:\n")  # Write the input settings to the file
        for key, value in settings.items():
            output_file.write(f"{key}: {value}\n")
        
        output_file.write("\Loss Function:\n")    # Write the loss function settings to the file
        output_file.write(f"Loss Function Choice: {loss_fn}\n")
        if loss_function_selection == 0:
            output_file.write(f"zero_weighting: {zero_weighting}\n")
            output_file.write(f"nonzero_weighting: {nonzero_weighting}\n")    

        output_file.write("\nNormalisation:\n")     # Write the normalisation settings to the file
        output_file.write(f"simple_norm_instead_of_custom: {simple_norm_instead_of_custom}\n")    
        output_file.write(f"all_norm_off: {all_norm_off}\n") 

        output_file.write("\nPre Training:\n")   # Write the pre training settings to the file
        if start_from_pretrained_model:
            output_file.write(f"full_statedict_path: {full_statedict_path}\n")    
        output_file.write(f"start_from_pretrained_model: {start_from_pretrained_model}\n")
        output_file.write(f"load_pretrained_optimser: {load_pretrained_optimser}\n")  
        
        output_file.write("\nAutoencoder Network:\n")  # Write the autoencoder network settings to the file
        output_file.write((f"AE File ID: {AE_file_name}\n"))    # Write the autoencoder network file ID to the file
        output_file.write("\n" + summary_str)   # Write the autoencoder network summary to the file
        
        output_file.write("\n \nFull Data Readouts:\n") 
        for key, value in full_data_output.items(): 
            output_file.write(f"{key}: {value}\n") 

        output_file.write("\nPython Lib Versions:\n") # Write the python library versions to the file
        output_file.write((f"PyTorch: {torch.__version__}\n"))  
        output_file.write((f"Torchvision: {torchvision.__version__}\n"))     
        output_file.write((f"Numpy: {np.__version__}\n")) 

        system_information = get_system_information()  # Get the system information using helper function
        output_file.write("\n" + system_information)  # Write the system information to the file
    print("- Completed -")
#%% - End of Program - Printing message to notify user!
print("\nProgram Complete - Shutting down...\n")    
    
