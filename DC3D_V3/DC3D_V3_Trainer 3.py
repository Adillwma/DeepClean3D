# DeepClean Trainer v1.1.0
# Build created on Wednesday May 6th 2023
# Author: Adill Al-Ashgar
# University of Bristol
# adillwmaa@gmail.co.uk

"""
Possible improvements:

### ~~~~~ NEEDS TESTING [DONE!] Add user controll to overide double precision processing

### ~~~~~ NEEDS TESTING [DONE!] Improve the pixel telemtry per epoch by adding a dotted green li9ne indicating the true number of signal points\

### ~~~~~ Attach the new masking optimised normalisation check if it need a corresponding renorm

### ~~~~~ [DONE!] Make sure that autoecoder Encoder and Decoder are saved along with model in the models folder 

### ~~~~~ Add the new performance metrics per epoch tot he history da dictionary to clean up??

### ~~~~~ clean up the perforance loss plotting metircs calulation section, move to external script?

### ~~~~~ Clean up the performance loss plottsing code, it is too long and unwieldy, move it to an external file and load as a function

### ~~~~~~ [DONE!] Allow normalisation/renorm to be bypassed, to check how it affects results 

### ~~~~~~ [DONE!] Find out what is going on with recon threshold scaling issue

### ~~~~~~ [DONE!] fix noise adding to the data, it is not working as intended, need to retain clean images for label data 

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

### ~~~~~ [DONE!] could investigate programatically setting the non_zero weighting based on the ratio of zero points to non zero points in the data set which would balance out the two classes in the loss functions eyes

### ~~~~~ Add way for program to save the raw data for 3d plots so that they can be replotted after training and reviewed in rotatable 3d 

### ~~~~~ Check if running model in dp (fp64) is causing large slow down???

### ~~~~~ Allow seperate loss fucntion for testing/validation phase?

### ~~~~~ Properly track the choices for split loss funcs in txt output file 

### ~~~~~ Explicitlly pass in split_loss_functions to the split custom weigted func atm is not done to simplify the code but is not ideal

### ~~~~~ [DONE!] Update noise points to take a range as input and randomly select number for each image from the range

### ~~~~~ [DONE!] Add fcuntion next to noise adder that drops out pixels, then can have the labeld image with high signal points and then dropout the points in the input image to network so as to train it to find dense line from sparse points!

### ~~~~~ Add plots of each individual degradation step rathert than just all shown on one (this could be done instead of the current end of epoch 10 plots or alongside)

### ~~~~~ colour true signal points red in the input distroted image so that viewer can see the true signal points and the noise added

### ~~~~~ add masking directly to the trainer so we can see masked output too 
"""

#NOTE to users: Known good parameters so far (changing these either way damages performance): learning_rate = 0.0001, Batch Size = 10, Latent Dim = 10, Reconstruction Threshold = 0.5, loss_function_selection = 0, loss weighting = 0.9 - 1

#%% - User Inputs
dataset_title =  'RDT 100K 1000ToF' #"RDT 10K MOVE"#"RDT 50KM"# "Dataset 37_X15K Perfect track recovery" #"Dataset 24_X10Ks"           #"Dataset 12_X10K" ###### TRAIN DATASET : NEED TO ADD TEST DATASET?????
model_save_name = "RDT 100K 30s 100n"#"RDT 50KM tdim1000 AE2PROTECT 30 sig 200NP LD10"     #"D27 100K ld8"#"Dataset 18_X_rotshiftlarge"

time_dimension = 1000                         # User controll to set the number of time steps in the data
reconstruction_threshold = 0.5               # MUST BE BETWEEN 0-1  #Threshold for 3d reconstruction, values below this confidence level are discounted

#%% - Training Hyperparameter Settings
num_epochs = 2001                             # User controll to set number of epochs (Hyperparameter)
batch_size = 10                              # User controll to set batch size - number of Images to pull per batch (Hyperparameter) 
latent_dim = 10                              # User controll to set number of nodes in the latent space, the bottleneck layer (Hyperparameter)

learning_rate = 0.0001 #!!                       # User controll to set optimiser learning rate (Hyperparameter)
optim_w_decay = 1e-05 #!!!!1e-07 seeems better?? test!                        # User controll to set optimiser weight decay for regularisation (Hyperparameter)
dropout_prob = 0.2                           # [NOTE Not connected yet] User controll to set dropout probability (Hyperparameter)

train_test_split_ratio = 0.8                 # User controll to set the ratio of the dataset to be used for training (Hyperparameter)

val_set_on = False                         # User controll to set if a validation set is used
val_test_split_ratio = 0.9              #This needs to be better explained its actually test_val ration ratehr than oterh way round     # [NOTE LEAVE AT 0.5, is for future update, not working currently] User controll to set the ratio of the non-training data to be used for validation as opposed to testing (Hyperparameter)

loss_vs_sparse_img = False                # User controll to set if the loss is calculated against the sparse image or the full image (Hyperparameter)
loss_function_selection = 0                  # Select loss function (Hyperparameter): 0 = ada_weighted_mse_loss, 1 = torch.nn.MSELoss(), 2 = torch.nn.BCELoss(), 3 = torch.nn.L1Loss(), 4 = ada_SSE_loss, 5 = ada_weighted_custom_split_loss, 6 = weighted_perfect_reconstruction_loss

# Below weights only used if loss func set to 0 or 6 aka ada_weighted_mse_loss
zero_weighting = 0.99                           # User controll to set zero weighting for ada_weighted_mse_loss (Hyperparameter)
nonzero_weighting = 1                     # User controll to set non zero weighting for ada_weighted_mse_loss (Hyperparameter)

# Below only used if loss func set to 6 aka ada_weighted_custom_split_loss
zeros_loss_choice = 1                     # Select loss function for zero values (Hyperparameter): 0 = Maxs_Loss_Func, 1 = torch.nn.MSELoss(), 2 = torch.nn.BCELoss(), 3 = torch.nn.L1Loss(), 4 = ada_SSE_loss
nonzero_loss_choice = 1                # Select loss function for non zero values (Hyperparameter): 0 = Maxs_Loss_Func, 1 = torch.nn.MSELoss(), 2 = torch.nn.BCELoss(), 3 = torch.nn.L1Loss(), 4 = ada_SSE_loss

#%% - Image Preprocessing Settings  (when using perfect track images as labels)
signal_points = 30                           # User controll to set the number of signal points to add
noise_points = 100                         # User controll to set the number of noise points to add

x_std_dev = 0                              # User controll to set the standard deviation of the detectors error in the x axis
y_std_dev = 0                               # User controll to set the standard deviation of the detectors error in the y axis
tof_std_dev = 0                             # User controll to set the standard deviation of the detectors error in the time of flight 

#%% - Pretraining settings
start_from_pretrained_model = False          # If set to true then the model will load the pretrained model and optimiser state dicts from the path below
load_pretrained_optimser = True             # Only availible if above is set to true - (pretrain seems to perform better if this is set to true)
pretrained_model_path = 'N:/Yr 3 Project Results/RDT 50KMF Base Model 2 - Training Results/RDT 50KMF Base Model 2 - Model + Optimiser State Dicts.pth'      # Specify the path to the saved full state dictionary for pretraining

#%% - Normalisation Settings 
simple_norm_instead_of_custom = False        #[Default is False] # If set to true then the model will use simple normalisation instead of custom normalisation
all_norm_off = False                         #[Default is False] # If set to true then the model will not use any input normalisation
simple_renorm = False                        #[Default is False] # If set to true then the model will use simple output renormalisation instead of custom output renormalisation

#%% - Plotting Control Settings
print_every_other = 2                      #[default = 2] 1 is to save/print all training plots every epoch, 2 is every other epoch, 3 is every 3rd epoch etc
plot_or_save = 1                           #[default = 1] 0 prints plots to terminal (blocking till closed), If set to 1 then saves all end of epoch printouts to disk (non-blocking), if set to 2 then saves outputs whilst also printing for user (blocking till closed)
num_to_plot = 10
save_all_raw_plot_data = True              #[default = False] If set to true then all raw data for plots is saved to disk for replotting and analysis later

#%% - New beta feature settings 
double_precision = False
test_just_masking_mode = False                 # [Default = False] Sets the model to just masking mode for testing purposes, this optimises the network for creating the best mask possible, by encoding hits and no hits as binary values (0, or 1's) ### TEST ALSO FORCING BCE LOSS OR SPLIT BCE LOSS


record_weights = False
record_biases = False
record_activity = False #False  ##Be carefull, the activity file recorded is ~ 2.5Gb  #Very slow, reduces net performance by XXXXXX%
compress_activations_npz_output = False #False   Compresses the activity file above for smaller file size but does increase loading and saving times for the file. (use if low on hdd space)



#%% - Advanced Visulisation Settings
plot_train_loss = True               #[default = True]       
plot_validation_loss = True          #[default = True]               
plot_detailed_performance_loss = True   #plots ssim nmi etc for each epoch 

plot_cutoff_telemetry = True         #[default = False] # Update name to pixel_cuttoff_telemetry    #Very slow, reduces net performance by XXXXXX%

plot_live_training_loss = True       #[default = True]                Generate plot of live training loss during trainig which is overwritten each epoch, this is useful for seeing how the training is progressing
comparative_live_loss = True         #[default = True]              Generate plot of live training loss during trainig which is overwritten each epoch, this is useful for seeing how the training is progressing
path_to_control_loss = 'N:\Yr 3 Project Results\RDT 10KM Retest 2 100 noise - Training Results'
path_to_compared_loss = 'N:\Yr 3 Project Results\RDT 10KM Retest 2 30 noise - Training Results'

plot_pixel_difference = False #BROKEN        #[default = True]          
plot_latent_generations = True       #[default = True]              
plot_higher_dim = False              #[default = True]  
plot_Graphwiz = False                #[default = True]       


#%% - Advanced Debugging Settings
print_encoder_debug = False                     # [default = False]  
print_decoder_debug = False                     # [default = False] 
print_network_summary = False                   # [Default = False] Prints the network summary to terminal
print_partial_training_losses = False           # [Default = True] Prints the training loss for each batch in the epoch

debug_noise_function = False                    # [default = False]  
debug_loader_batch = False                      # SAFELY REMOVE THIS PARAM!!!  #(Default = False) //INPUT 0 or 1//   #Setting debug loader batch will print to user the images taken in by the dataoader in this current batch and print the corresponding labels

full_dataset_integrity_check = False            # [Default = False] V slow  #Checks the integrity of the dataset by checking shape of each item as opposed to when set to false which only checks one single random file in the dataset
full_dataset_distribution_check = False         # [Default = False] V slow  #Checks the distribution of the dataset , false maesn no distributionn check is done

seed = 0                                        # [Default = 0] 0 gives no seeeding to RNG, if the value is not zero then this is used for the RNG seeding for numpy, random, and torch libraries

#%% - Program Mode Setting - CLEAN UP THIS SECTION
#mode = 0 ### 0=Data_Gathering, 1=Testing, 2=Speed_Test, 3=Debugging

speed_test = False      # [speed_test=False]Defualt    true sets number of epocs to print to larger than number of epochs to run so no plotting time wasted etc
data_gathering = True

#%% - HACKS NEED FIXING!!!
if print_every_other > num_epochs:   #protection from audio data not being there to save if plot not genrated - Can fix by moving audio genration out of the plotting function entirely and only perform it once at the end wher eit is actually saved.
    print_every_other = num_epochs

### IMPLEMENT !!!!!!!!!!!!!!!!!
###history_da = {'train_loss':[], 'test_loss':[], 'val_loss':[], 'HTO_val':[], 'training_time':[]} # needs better placement???

max_epoch_reached = 0 # in case user exits before end of first epoch 

# Create a dictionary to store the activations
activations = {}
weights_data = {}
biases_data = {}

#%% - Data Path Settings
data_path = "N:\Yr 3 Project Datasets\\"
results_output_path = "N:\Yr 3 Project Results\\"


#%% - Dependencies
# External Libraries
import os
import time     # Used to time the training loop
import torch
import random  
import datetime 
import torchvision 
import numpy as np  
from tqdm import tqdm  # Progress bar
import matplotlib.cm as cm
from functools import partial
from torchinfo import summary # function to get the summary of the model layers structure, trainable parameters and memory usage
import matplotlib.pyplot as plt     
from torchvision import transforms  
from torch.utils.data import DataLoader, random_split
from skimage.metrics import normalized_mutual_information, structural_similarity
import pickle
import pandas as pd

# Imports from our custom scripts
from Autoencoders.DC3D_Autoencoder_V1_Protected2_2 import Encoder, Decoder # This imports the autoencoder classes from the file selected, changig the V# sets the version of the autoencoder

from Helper_files.Robust_model_exporter_V1 import Robust_model_export   # This is a custom function to export the raw .py file that contains the autoencoder class
from Helper_files.System_Information_check import get_system_information # This is a custom function to get the host system performance specs of the training machine
from Helper_files.Dataset_Integrity_Check_V1 import dataset_integrity_check     # This is a custom function to check the integrity of the datasets values
from Helper_files.Dataset_distribution_tester_V1 import dataset_distribution_tester     # This is a custom function to check the distribution of the datasets values
from Helper_files.AE_Visulisations import Generative_Latent_information_Visulisation, Reduced_Dimension_Data_Representations, Graphwiz_visulisation, AE_visual_difference # These are our custom functions to visulise the autoencoders training progression

#%% - Comparative Live Loss Plotting
if comparative_live_loss:
    #load pkl file into dictionary
    with open(path_to_control_loss + '\\Raw_Data_Output\\history_da_dict.pkl', 'rb') as f:
        control_history_da = pickle.load(f)
    with open(path_to_compared_loss + '\\Raw_Data_Output\\history_da_dict.pkl', 'rb') as f:
        comparative_history_da = pickle.load(f)


#%% - Helper functions

def weighted_perfect_recovery_lossOLD(reconstructed_image, target_image, zero_weighting=1, nonzero_weighting=1):

    # Get the indices of 0 and non 0 values in target_image as a mask for speed
    zero_mask = (target_image == 0)
    nonzero_mask = ~zero_mask         # Invert mask
    
    # Get the values in target_image
    values_zero = target_image[zero_mask]
    values_nonzero = target_image[nonzero_mask]

    #Calualte the number of value sin each of values_zero and values_nonzero for use in the class balancing
    zero_n = len(values_zero)
    nonzero_n = len(values_nonzero)
    
    # Get the corresponding values in reconstructed_image
    corresponding_values_zero = reconstructed_image[zero_mask]
    corresponding_values_nonzero = reconstructed_image[nonzero_mask]

    if zero_n == 0:
        zero_loss = 0
    else:
        # Calculate the loss for zero values
        loss_value_zero = (values_zero != corresponding_values_zero).float().sum() 
        zero_loss = zero_weighting*( (1/zero_n) * loss_value_zero)

    if nonzero_n == 0:
        nonzero_loss = 0
    else:
        # Calculate the loss for non-zero values
        loss_value_nonzero = (values_nonzero != corresponding_values_nonzero).float().sum() 
        nonzero_loss = nonzero_weighting*( (1/nonzero_n) * loss_value_nonzero) 

    # Calculate the total loss with automatic class balancing and user class weighting
    loss_value = zero_loss + nonzero_loss

    return loss_value

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

# Special normalisation for pure masking
def custom_mask_optimised_normalisation_torch(data):
    data = torch.where(data > 0, 1, 0)
    return data

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
def reconstruct_3D(data):
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
    plt.xlabel("Output Values")
    plt.ylabel("Number of Pixels")
    plt.grid(alpha=0.2)
    Out_Label = graphics_dir + f'{model_save_name} - Reconstruction Telemetry Histogram - Epoch {epoch}.png'
    plot_save_choice(plot_or_save, Out_Label)

    above_threshold = (data2 >= reconstruction_threshold).sum()
    below_threshold = (data2 < reconstruction_threshold).sum()
    return (above_threshold, below_threshold)

# Masking technique
def masking_recovery(input_image, recovered_image, time_dimension, print_result=False):
    raw_input_image = input_image.copy()
    net_recovered_image = recovered_image.copy()
    #Evaluate usefullness 
    # count the number of non-zero values
    masking_pixels = np.count_nonzero(net_recovered_image)
    image_shape = net_recovered_image.shape
    total_pixels = image_shape[0] * image_shape[1] * time_dimension
    # print the count
    if print_result:
        print(f"Total number of pixels in the timescan: {format(total_pixels, ',')}\nNumber of pixels returned by the masking: {format(masking_pixels, ',')}\nNumber of pixels removed from reconstruction by masking: {format(total_pixels - masking_pixels, ',')}")

    # use np.where and boolean indexing to update values in a
    mask_indexs = np.where(net_recovered_image != 0)
    net_recovered_image[mask_indexs] = raw_input_image[mask_indexs]
    result = net_recovered_image
    return result
                
# Plots the confidence telemetry data
def plot_telemetry(telemetry, true_num_of_signal_points, plot_or_save=0):
    tele = np.array(telemetry)
    #!!! Add labels to lines
    plt.plot(tele[:,0],tele[:,1], color='r', label="Points above threshold") #red = num of points above threshold
    plt.plot(tele[:,0],tele[:,2], color='b', label="Points below threshold") #blue = num of points below threshold
    plt.axhline(y=true_num_of_signal_points, color='g', linestyle='dashed', label="True number of signal points")
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

# Helper function to allow values to be input as a range from which random values are chosen each time unless the input is a single value in which case it is used as the constant value
def input_range_to_random_value(input_range):
    # check if input is a single value (int or float) or a range (list or tuple)
    if isinstance(input_range, (int, float)):
        return input_range  
    elif isinstance(input_range, (list, tuple)):
        return random.randint(input_range[0], input_range[1])
    else:
        print("Error: input_range_to_random_value() input is not an value or pair of values")
        return None

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

def save_variable(variable, variable_name, path, force_pickle=False):

    if force_pickle:
        with open(path + variable_name + "_forcepkl.pkl", 'wb') as file:
            pickle.dump(variable, file)
    else:
        if isinstance(variable, dict):
            with open(path + variable_name + "_dict.pkl", 'wb') as file:
                pickle.dump(variable, file)

        elif isinstance(variable, np.ndarray):
            np.save(path + variable_name + "_array.npy", variable)

        elif isinstance(variable, torch.Tensor):
            torch.save(variable, path + variable_name + "_tensor.pt")

        elif isinstance(variable, list):
            df = pd.DataFrame(variable)
            df.to_csv(path + variable_name + "_list.csv", index=False)

        elif isinstance(variable, int):
            with open(path + variable_name + "_int.pkl", 'wb') as file:
                pickle.dump(variable, file)

        elif isinstance(variable, float):
            with open(path + variable_name + "_float.pkl", 'wb') as file:
                pickle.dump(variable, file)

        elif isinstance(variable, str):
            with open(path + variable_name + "_str.pkl", 'wb') as file:
                pickle.dump(variable, file)

        else:
            raise ValueError("Unsupported variable type.")
            
def comparitive_loss_plot(x_list, y_list, legend_label_list, x_label, y_label, title, save_path, plot_or_save):
    for x, y, legend_label in zip(x_list, y_list, legend_label_list):
        plt.plot(x, y, label=legend_label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(alpha=0.2)
    plt.legend()
    plot_save_choice(plot_or_save, save_path) 

def np_to_tensor(np_array, double_precision=False):
    """
    Convert np array to torch tensor of user selected precision. 
    Takes in np array of shape [H, W] and returns torch tensor of shape [C, H, W]
    """
    dtype = torch.float64 if double_precision else torch.float32
    tensor = torch.tensor(np_array, dtype=dtype)
    tensor = tensor.unsqueeze(0)        # Append channel dimension to begining of tensor
    return(tensor)

def activation_hook_fn(module, input, output, layer_index):
    """
    This function will be called whenever a layer is called during the forward pass.
    It records the activations and saves them in the specified dictionary.
    """
    activations[layer_index] = output.detach()

def weights_hook_fn(module, input, output, layer_index):
    """
    This function will be called whenever a layer is called during the forward pass.
    It records the activations and saves them in the specified dictionary.
    """
    weights_data[layer_index] = module.weight.data.clone().detach()

def biases_hook_fn(module, input, output, layer_index):
    """
    This function will be called whenever a layer is called during the forward pass.
    It records the activations and saves them in the specified dictionary.
    """
    biases_data[layer_index] = module.bias.data.clone().detach()

def write_hook_data_to_disk_and_clear(activations, weights_data, biases_data, epoch, output_dir):

    if len(activations) != 0:
        activ_path = output_dir + "Activation Data/"
        os.makedirs(activ_path, exist_ok=True)
        # Save the activations to a file named 'activations_epoch_{epoch}.pt'
        torch.save(activations, activ_path + f'activations_epoch_{epoch}.pt')

        # Clear the activations dictionary to free up memory
        activations.clear()

    if len(weights_data) != 0:
        weight_path = output_dir + "Weights Data/"
        os.makedirs(weight_path, exist_ok=True)
        # Save the weights to a file named 'weights_epoch_{epoch}.pt'
        torch.save(weights_data, weight_path + f'weights_epoch_{epoch}.pt')

        # Clear the weights dictionary to free up memory
        weights_data.clear()

    if len(biases_data) != 0:
        bias_path = output_dir + "Biases Data/"
        os.makedirs(bias_path, exist_ok=True)
        # Save the weights to a file named 'weights_epoch_{epoch}.pt'
        torch.save(biases_data, bias_path + f'biases_epoch_{epoch}.pt')

        # Clear the weights dictionary to free up memory
        biases_data.clear()


#%% NEW!! IMAGE METRICS - NEEDS CLEANING UP!!!

#Signal to Noise Ratio (SNR)
def SNR(clean_input, noised_target):
    """
    Calculates the Signal to Noise Ratio (SNR) of a given signal and noise.
    SNR is defined as the ratio of the magnitude of the signal and the magnitude of the noise.
    
    Args:
    clean_input (torch.Tensor): The original signal.
    noised_target (torch.Tensor): The signal with added noise.
    
    Returns:
    The calculated SNR value.    
    """
    signal_power = torch.mean(torch.pow(clean_input, 2))

    noise = clean_input - noised_target 
    noise_power = torch.mean(torch.pow(noise, 2))

    snr = 10 * torch.log10(signal_power / noise_power)
       
    return (float(snr.numpy()))

#Peak Signal-to-Noise Ratio (PSNR):
def PSNR(clean_input, noised_target, time_dimension):
    """
    Calculates the Peak Signal to Noise Ratio (PSNR) of a given image and its recovered version. PSNR is defined as the ratio of 
    the maximum possible power of a signal and the power of corrupting noise. The measure focuses on how well high-intensity 
    regions of the image come through the noise, and pays much less attention to low intensity regions.

    Args:
    clean_input (torch.Tensor): The original image.
    noised_target (torch.Tensor): The recovered image.
    
    Returns:
    The calculated PSNR value.
    """
    mse = torch.mean(torch.pow(clean_input - noised_target, 2))   #Finds the mean square error
    max_value = time_dimension
    psnr = 10 * torch.log10((max_value**2) / mse)
    return (float(psnr.numpy()))

#Mean Squared Error (MSE):
def MSE(clean_input, noised_target):
    """
    Mean Squared Error (MSE)

    Args:
    clean_input (torch.Tensor): The original image.
    noised_target (torch.Tensor): The recovered image.
    
    Returns:
    The calculated Mean Squared Error value.
    """
    mse = torch.mean((torch.pow(clean_input - noised_target, 2)))
    return (float(mse.numpy()))

#Mean Absolute Error (MAE):
def MAE(clean_input, noised_target):
    """
    Mean Absolute Error (MAE)

    Args:
    clean_input (torch.Tensor): The original image.
    noised_target (torch.Tensor): The recovered image.
    
    Returns:
    The calculated Mean Absolute Error value.
    """
    return float((torch.mean(torch.abs(clean_input - noised_target))).numpy())

#Structural Similarity Index (SSIM):
def SSIM(clean_input, noised_target):
    """
    Structural Similarity Index Measure (SSIM), is a perceptual quality index that measures the structural similarity between 
    two images. SSIM takes into account the structural information of an image, such as luminance, contrast, and structure, 
    and compares the two images based on these factors. SSIM is based on a three-part similarity metric that considers the 
    structural information in the image, the dynamic range of the image, and the luminance information of the image. SSIM is 
    designed to provide a more perceptually relevant measure of image similarity than traditional metrics such as Mean Squared 
    Error or Peak Signal-to-Noise Ratio.

    Args:
    clean_input (torch.Tensor): The original image.
    noised_target (torch.Tensor): The recovered image.
    
    Returns:
    The calculated Structural Similarity Index Measure value.
    """
    clean_image = clean_input.detach().cpu().numpy()
    recovered_image = noised_target.detach().cpu().numpy()
    return structural_similarity(clean_image, recovered_image, data_range=float(time_dimension))

#Correlation Coefficent
def correlation_coeff(clean_input, noised_target):
    
    """
    Correlation coefficient is a scalar value that measures the linear relationship between two signals. The correlation 
    coefficient ranges from -1 to 1, where a value of 1 indicates a perfect positive linear relationship, a value of -1 indicates 
    a perfect negative linear relationship, and a value of 0 indicates no linear relationship between the two signals. Correlation 
    coefficient only measures the linear relationship between two signals, and does not take into account the structure of the signals.

    ρ = cov(x,y) / (stddev(x) * stddev(y))

    The function first computes the mean and standard deviation of each tensor, and then subtracts the mean from each element 
    to get the centered tensors x_center and y_center. The numerator is the sum of the element-wise product of x_center 
    and y_center, and the denominator is the product of the standard deviations of the two centered tensors multiplied by the 
    number of elements in the tensor. The function returns the value of the correlation coefficient ρ as the ratio of the numerator 
    and denominator.

    Args:
    clean_input (torch.Tensor): The original image.
    noised_target (torch.Tensor): The recovered image.
    
    Returns:
    The calculated correlation coefficient value.
    """
    clean_mean = clean_input.mean()
    noised_mean = noised_target.mean()
    clean_std = clean_input.std()
    noised_std = noised_target.std()
    clean_center = clean_input - clean_mean
    noised_center = noised_target - noised_mean
    numerator = (clean_center * noised_center).sum()
    denominator = clean_std * noised_std * clean_input.numel()
    return float((numerator / denominator).numpy())

#Mutual Information:
def NomalisedMutualInformation(clean_input, noised_target):
    clean_image = clean_input.detach().cpu().numpy()
    recovered_image = noised_target.detach().cpu().numpy()
    return normalized_mutual_information(clean_image, recovered_image)-1

def compare_images_pixels(clean_img, denoised_img, terminal_print=False):   ###!!!INVESTIGATE USING PRINT = TRUE !!!!
    clean_img = clean_img.detach().cpu().numpy()
    denoised_img = denoised_img.detach().cpu().numpy()
    ###TRUE HITS STATS###
    if terminal_print:
        print("###TRUE HITS STATS###")
    
    ##X,Y##
    true_hits_indexs = np.nonzero(clean_img)     # Find the indexs of the non zero pixels in clean_img
    numof_true_hits = len(true_hits_indexs[0])   # Find the number of lit pixels in clean_img
    if terminal_print:
        print("numof_true_hits:", numof_true_hits)
    
    # Check the values in corresponding indexs in denoised_img, retunr the index's and number of them that are also non zero
    true_positive_xy_indexs = np.nonzero(denoised_img[true_hits_indexs]) 
    numof_true_positive_xy = len(true_positive_xy_indexs[0])                     # Calculate the number of pixels in clean_img that are also in denoised_img ###NUMBER OF SUCSESSFUL X,Y RECON PIXELS
    if terminal_print:
        print("numof_true_positive_xy:", numof_true_positive_xy)

    # Calculate the number of true hit pixels in clean_img that are not lit at all in denoised_img  ###NUMBER OF LOST TRUE PIXELS
    false_negative_xy = numof_true_hits - numof_true_positive_xy
    if terminal_print:
        print("false_negative_xy:", false_negative_xy)
    
    # Calculate the percentage of non zero pixels in clean_img that are also non zero in denoised_img   ###PERCENTAGE OF SUCSESSFUL X,Y RECON PIXELS
    if numof_true_hits == 0:
        percentage_of_true_positive_xy = 0
    else:
        percentage_of_true_positive_xy = (numof_true_positive_xy / numof_true_hits) * 100
    
    if terminal_print:
        print(f"percentage_of_true_positive_xy: {percentage_of_true_positive_xy}%")
    

    ##TOF##
    # Calculate the number of pixels in clean_img that are also in denoised_img and have the same TOF value  ###NUMBER OF SUCSESSFUL X,Y,TOF RECON PIXELS
    num_of_true_positive_tof = np.count_nonzero(np.isclose(clean_img[true_hits_indexs], denoised_img[true_hits_indexs], rtol=1e-6))
    if terminal_print:
        print("num_of_true_positive_tof:", num_of_true_positive_tof)
    
    # Calculate the percentage of pixels in clean_img that are also in denoised_img and have the same value   ###PERCENTAGE OF SUCSESSFUL X,Y,TOF RECON PIXELS
    if numof_true_hits == 0:
        percentage_of_true_positive_tof = 0
    else:
        percentage_of_true_positive_tof = (num_of_true_positive_tof / numof_true_hits) * 100
    if terminal_print:
        print(f"percentage_of_true_positive_tof: {percentage_of_true_positive_tof}%")    
    

    ###FALSE HIT STATS###
    if terminal_print:
        print("\n###FALSE HIT STATS###")        
    clean_img_zero_indexs = np.where(clean_img == 0)   # find the index of the 0 valued pixels in clean image 
    number_of_zero_pixels = np.sum(clean_img_zero_indexs[0])   # Find the number of pixels in clean image that are zero
    if terminal_print:
        print("number_of_true_zero_pixels:",number_of_zero_pixels)

    #check the values in corresponding indexs in denoised_img, return the number of them that are non zero
    denoised_img_false_lit_pixels = np.nonzero(denoised_img[clean_img_zero_indexs])
    numof_false_positives_xy = len(denoised_img_false_lit_pixels[0])
    if terminal_print:
        print("numof_false_positives_xy:",numof_false_positives_xy)

    # Calculate the percentage of pixels in clean_img that are zero and are also non zero in denoised_img   ###PERCENTAGE OF FALSE LIT PIXELS

    if number_of_zero_pixels == 0:
        percentage_of_false_lit_pixels = 0
    else:
        percentage_of_false_lit_pixels = (numof_false_positives_xy / number_of_zero_pixels) * 100
    
    
    if terminal_print:
        print(f"percentage_of_false_positives_xy: {percentage_of_false_lit_pixels}%")
    
    return percentage_of_true_positive_xy, percentage_of_true_positive_tof, numof_false_positives_xy

avg_loss_mse = []
avg_loss_mae = []
avg_loss_snr = []
avg_loss_psnr = []
avg_loss_ssim = []
avg_loss_nmi = []
avg_loss_cc = []
avg_loss_true_positive_xy = []
avg_loss_true_positive_tof = []
avg_loss_false_positive_xy = []

#Combine all performance metrics into simple test script
def quantify_loss_performance(clean_input_batch, noised_target_batch, time_dimension):
    loss_mse = []
    loss_mae = []
    loss_snr = []
    loss_psnr = []
    loss_ssim = []
    loss_nmi = []
    loss_cc = []
    loss_true_positive_xy = []
    loss_true_positive_tof = []
    loss_false_positive_xy = [] 

    for i in range(len(clean_input_batch)):
        clean_input = clean_input_batch[i][0]
        noised_target = noised_target_batch[i][0]

        loss_mse.append(MSE(clean_input, noised_target))
        loss_mae.append(MAE(clean_input, noised_target))
        loss_snr.append(SNR(clean_input, noised_target))
        loss_psnr.append(PSNR(clean_input, noised_target, time_dimension))
        loss_ssim.append(SSIM(clean_input, noised_target))
        loss_nmi.append(NomalisedMutualInformation(clean_input, noised_target))
        loss_cc.append(correlation_coeff(clean_input, noised_target))
        percentage_of_true_positive_xy, percentage_of_true_positive_tof, numof_false_positives_xy = compare_images_pixels(clean_input, noised_target)
        loss_true_positive_xy.append(percentage_of_true_positive_xy)
        loss_true_positive_tof.append(percentage_of_true_positive_tof)
        loss_false_positive_xy.append(numof_false_positives_xy)

    avg_loss_mse.append(np.mean(loss_mse))
    avg_loss_mae.append(np.mean(loss_mae))
    avg_loss_snr.append(np.mean(loss_snr))
    avg_loss_psnr.append(np.mean(loss_psnr))
    avg_loss_ssim.append(np.mean(loss_ssim))
    avg_loss_nmi.append(np.mean(loss_nmi))
    avg_loss_cc.append(np.mean(loss_cc))
    avg_loss_true_positive_xy.append(np.mean(loss_true_positive_xy))
    avg_loss_true_positive_tof.append(np.mean(loss_true_positive_tof))
    avg_loss_false_positive_xy.append(np.mean(loss_false_positive_xy))

"""
time_dimension = 100 
clean_input = np.random.rand(1, 1, 128, 88)
noised_target = np.random.rand(1, 1, 128, 88)
clean_input = torch.tensor(clean_input)
noised_target = torch.tensor(noised_target)
quantify_loss_performance(clean_input, noised_target, time_dimension)
"""

#%% - Classes
class WeightedPerfectRecoveryLoss(torch.nn.Module):
    def __init__(self, zero_weighting=1, nonzero_weighting=1):
        super(WeightedPerfectRecoveryLoss, self).__init__()
        self.zero_weighting = zero_weighting
        self.nonzero_weighting = nonzero_weighting

    def backward(self, grad_output):
        # Retrieve the tensors saved in the forward method
        reconstructed_image, target_image = self.saved_tensors  # <----- Remove this line

        # Get the indices of 0 and non 0 values in target_image as a mask for speed
        zero_mask = (target_image == 0)
        nonzero_mask = ~zero_mask         # Invert mask

        # Get the values in target_image
        values_zero = target_image[zero_mask]
        values_nonzero = target_image[nonzero_mask]

        #Calualte the number of value sin each of values_zero and values_nonzero for use in the class balancing
        zero_n = len(values_zero)
        nonzero_n = len(values_nonzero)

        # Get the corresponding values in reconstructed_image
        corresponding_values_zero = reconstructed_image[zero_mask]
        corresponding_values_nonzero = reconstructed_image[nonzero_mask]

        # Calculate the gradients
        grad_reconstructed_image = torch.zeros_like(reconstructed_image)
        grad_reconstructed_image[zero_mask] += self.zero_weighting*(1/zero_n)*(corresponding_values_zero != values_zero).float()
        grad_reconstructed_image[nonzero_mask] += self.nonzero_weighting*(1/nonzero_n)*(corresponding_values_nonzero != values_nonzero).float()

        return grad_reconstructed_image * grad_output.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    
    def forward(self, reconstructed_image_in, target_image_in):
        reconstructed_image = reconstructed_image_in.clone()
        target_image = target_image_in.clone()

        # Get the indices of 0 and non 0 values in target_image as a mask for speed
        zero_mask = (target_image == 0)
        nonzero_mask = ~zero_mask         # Invert mask
        
        # Get the values in target_image
        values_zero = target_image[zero_mask]
        values_nonzero = target_image[nonzero_mask]

        #Calualte the number of value sin each of values_zero and values_nonzero for use in the class balancing
        zero_n = len(values_zero)
        nonzero_n = len(values_nonzero)

        # Get the corresponding values in reconstructed_image
        corresponding_values_zero = reconstructed_image[zero_mask]
        corresponding_values_nonzero = reconstructed_image[nonzero_mask]

        if zero_n == 0:
            zero_loss = 0
        else:
            # Calculate the loss for zero values
            loss_value_zero = (values_zero != corresponding_values_zero).float().sum() 
            zero_loss = self.zero_weighting*( (1/zero_n) * loss_value_zero)

        if nonzero_n == 0:
            nonzero_loss = 0
        else:
            # Calculate the loss for non-zero values
            loss_value_nonzero = (values_nonzero != corresponding_values_nonzero).float().sum() 
            nonzero_loss = self.nonzero_weighting*( (1/nonzero_n) * loss_value_nonzero) 

        # Calculate the total loss with automatic class balancing and user class weighting
        loss_value = zero_loss + nonzero_loss


        return loss_value

#%% - # Input / Output Path Initialisation

# Create output directory if it doesn't exist
dir = results_output_path + model_save_name + " - Training Results/"
os.makedirs(dir, exist_ok=True)

# Create output directory for images if it doesn't exist
graphics_dir = dir + "Output_Graphics/"
os.makedirs(graphics_dir, exist_ok=True)

raw_plotdata_output_dir = dir + "Raw_Data_Output/"
os.makedirs(raw_plotdata_output_dir, exist_ok=True)

model_output_dir = dir + "Model_Deployment/"
os.makedirs(model_output_dir, exist_ok=True)

# Joins up the parts of the differnt output files save paths
full_model_path = model_output_dir + model_save_name + " - Model.pth"
full_statedict_path = model_output_dir + model_save_name + " - Model + Optimiser State Dicts.pth"

full_activity_filepath = dir + model_save_name + " - Activity.npz"
full_netsum_filepath = dir + model_save_name + " - Network Summary.txt"


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
availible_loss_functions = [ada_weighted_mse_loss, torch.nn.MSELoss(), torch.nn.BCELoss(), torch.nn.L1Loss(), ada_SSE_loss, ada_weighted_custom_split_loss, WeightedPerfectRecoveryLoss()]    # List of all availible loss functions
loss_fn = availible_loss_functions[loss_function_selection]            # Sets loss function based on user input of parameter loss_function_selection

# Set loss function choice for split loss (if loss function choice is set to ada weighted custom split loss)
availible_split_loss_functions = [torch.nn.MSELoss(), torch.nn.BCELoss(), torch.nn.L1Loss(), ada_SSE_loss]    # List of all availible loss functions is set to ada_weighted_custom_split_loss
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

        #Select settings randomly from user range
        signal_points_r = input_range_to_random_value(signal_points)
        x_std_dev_r = input_range_to_random_value(x_std_dev) 
        y_std_dev_r = input_range_to_random_value(y_std_dev) 
        tof_std_dev_r = input_range_to_random_value(tof_std_dev)
        noise_points_r = input_range_to_random_value(noise_points)

        # DATA PREPROCESSING
        with torch.no_grad(): # No need to track the gradients
            sparse_output_batch = create_sparse_signal(image_batch, signal_points_r)
            sparse_and_resolution_limited_batch = simulate_detector_resolution(sparse_output_batch, x_std_dev_r, y_std_dev_r, tof_std_dev_r)
            noised_sparse_reslimited_batch = add_noise_points_to_batch_prenorm(sparse_and_resolution_limited_batch, noise_points_r, time_dimension)
            
            if test_just_masking_mode:
                normalised_batch = custom_mask_optimised_normalisation_torch(noised_sparse_reslimited_batch)
                normalised_inputs = custom_mask_optimised_normalisation_torch(image_batch)
            else:
                normalised_batch = custom_normalisation_torch(noised_sparse_reslimited_batch, reconstruction_threshold, time_dimension)
                normalised_inputs = custom_normalisation_torch(image_batch, reconstruction_threshold, time_dimension)
            
        # Move tensor to the proper device
        image_clean = normalised_inputs.to(device) # Move the clean image batch to the device
        image_noisy = normalised_batch.to(device) # Move the noised image batch to the device
        
        # Encode data
        encoded_data = encoder(image_noisy) # Encode the noised image batch
        # Decode data
        decoded_data = decoder(encoded_data) # Decode the encoded image batch
        
        if loss_vs_sparse_img:
            loss_comparator = sparse_output_batch
        else:
            loss_comparator = image_clean

        # Evaluate loss
        loss = loss_fn(decoded_data, loss_comparator)  # Compute the loss between the decoded image batch and the clean image batch
        
        # Backward pass
        optimizer.zero_grad() # Reset the gradients
        loss.backward() # Compute the gradients
        optimizer.step() # Update the parameters
        train_loss.append(loss.detach().cpu().numpy()) # Store the loss value for the batch

        if print_partial_training_losses:         # Prints partial train losses per batch
            
            print('\t partial train loss (single batch): %f' % (loss.data))  # Print batch loss value
 
    return np.mean(train_loss) # Return the mean loss value for the epoch???

### Testing Function
def test_epoch_den(encoder, decoder, device, dataloader, loss_fn, signal_points, noise_points=0, x_std_dev=0, y_std_dev=0, tof_std_dev=0, time_dimension=100, reconstruction_threshold=0.5, print_partial_training_losses=False):
    # Set evaluation mode for encoder and decoder
    encoder.eval() # Evaluation mode for the encoder
    decoder.eval() # Evaluation mode for the decoder
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = [] # List to store the output of the network
        conc_label = []  # List to store the original image


        if print_partial_training_losses:  # Prints partial train losses per batch
            image_loop  = (dataloader)     # No progress bar for the batches
        else:                              # Rather than print partial train losses per batch, instead create progress bar
            image_loop  = tqdm(dataloader, desc='Validation', leave=False, colour="yellow") # Creates a progress bar for the batches


        for image_batch, _ in image_loop:  

            #Select settings randomly from user range
            signal_points_r = input_range_to_random_value(signal_points)
            x_std_dev_r = input_range_to_random_value(x_std_dev) 
            y_std_dev_r =input_range_to_random_value(y_std_dev) 
            tof_std_dev_r = input_range_to_random_value(tof_std_dev)
            noise_points_r = input_range_to_random_value(noise_points)

            # Move tensor to the proper device
            sparse_output_batch = create_sparse_signal(image_batch, signal_points_r)
            sparse_and_resolution_limited_batch = simulate_detector_resolution(sparse_output_batch, x_std_dev_r, y_std_dev_r, tof_std_dev_r)
            noised_sparse_reslimited_batch = add_noise_points_to_batch_prenorm(sparse_and_resolution_limited_batch, noise_points_r, time_dimension)
            
            if test_just_masking_mode:
                normalised_batch = custom_mask_optimised_normalisation_torch(noised_sparse_reslimited_batch)
                image_batch_norm = custom_mask_optimised_normalisation_torch(image_batch)
            else:
                normalised_batch = custom_normalisation_torch(noised_sparse_reslimited_batch, reconstruction_threshold, time_dimension)
                image_batch_norm = custom_normalisation_torch(image_batch, reconstruction_threshold, time_dimension)
            
            image_noisy = normalised_batch.to(device) # Move the noised image batch to the device

            # Encode data
            encoded_data = encoder(image_noisy) # Encode the noised image batch
            # Decode data
            decoded_data = decoder(encoded_data) # Decode the encoded image batch

            if loss_vs_sparse_img:
                lables = sparse_output_batch.cpu()
            else:
                lables = image_batch_norm.cpu()

            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu()) # Append the decoded image batch to the list
            conc_label.append(lables) # Append the clean image batch to the list


        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)  
        conc_label = torch.cat(conc_label) 

        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label) # Compute the loss between the decoded image batch and the clean image batch

        #Run additional perfomrnace metric loss functions for final plots, this needs cleaning up!!!!!
        quantify_loss_performance(conc_label, conc_out, time_dimension)

    return float(val_loss.data) # Return the loss value for the epoch

### Plotting Function
def plot_ae_outputs_den(encoder, decoder, epoch, model_save_name, time_dimension, reconstruction_threshold, signal_points, noise_points=0, x_std_dev=0, y_std_dev=0, tof_std_dev=0, n=10):       #Defines a function for plotting the output of the autoencoder. And also the input + clean training data? Function takes inputs, 'encoder' and 'decoder' which are expected to be classes (defining the encode and decode nets), 'n' which is the number of ?????Images in the batch????, and 'noise_factor' which is a multiplier for the magnitude of the added noise allowing it to be scaled in intensity.  
    """
    n is the number of images to plot
    """
    ### 2D Input/Output Comparison Plots 

    #Initialise lists for true and recovered signal point values 
    number_of_true_signal_points = []
    number_of_recovered_signal_points = []

    plt.figure(figsize=(16,4.5))                                      #Sets the figure size

    for i in range(n):                                                #Runs for loop where 'i' itterates over 'n' total values which range from 0 to n-1

      #Select settings randomly from user range
      signal_points_r = input_range_to_random_value(signal_points)
      x_std_dev_r = input_range_to_random_value(x_std_dev) 
      y_std_dev_r = input_range_to_random_value(y_std_dev) 
      tof_std_dev_r = input_range_to_random_value(tof_std_dev)
      noise_points_r = input_range_to_random_value(noise_points)
      
      #Following section creates the noised image data drom the original clean labels (images)   
      ax = plt.subplot(3,n,i+1)                                       #Creates a number of subplots for the 'Original images??????' i.e the labels. the position of the subplot is i+1 as it falls in the first row
      
      # Load input image
      img = test_dataset[i][0].unsqueeze(0) # [t_idx[i]][0].unsqueeze(0)                    #!!! ????
      
      #Determine the number of signal points on the input image (have to change this to take it directly from the embeded val in the datsaset as when addig noise this method will break)   
      int_sig_points = (img >= reconstruction_threshold).sum()
      number_of_true_signal_points.append(int(int_sig_points.numpy()))
      
      #if epoch <= print_every_other:                                                  #CHECKS TO SEE IF THE EPOCH IS LESS THAN ZERO , I ADDED THIS TO GET THE SAME NOISED IMAGES EACH EPOCH THOUGH THIS COULD BE WRONG TO DO?
      global image_noisy                                          #'global' means the variable (image_noisy) set inside a function is globally defined, i.e defined also outside the function
      sparse_output_batch = create_sparse_signal(img, signal_points_r)
      sparse_and_resolution_limited_batch = simulate_detector_resolution(sparse_output_batch, x_std_dev_r, y_std_dev_r, tof_std_dev_r)
      noised_sparse_reslimited_batch = add_noise_points_to_batch_prenorm(sparse_and_resolution_limited_batch, noise_points_r, time_dimension)

      if test_just_masking_mode:
          normalised_batch = custom_mask_optimised_normalisation_torch(noised_sparse_reslimited_batch)
      else:
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

      #ADD IN MASKING BELOW!!!!
      #masked_rec_image = masking_recovery(noise_im, rec_im)



      #Following section generates the img plots for the original(labels), noised, and denoised data)
      plt.imshow(in_im, cmap='gist_gray', vmin=0, vmax=time_dimension)           #plt.imshow plots an image. The arguments for imshow are, 'image data array' and cmap= which is the colour map. #.squeeze() acts on a tensor and returns a tensor, it removes all dimensions of the tensor that are of length 1, (A×1×B) becomes (AxB) where A and B are greater than 1 #.numpy() creates a numpy array from a tensor #!!! is the .cpu part becuase the code was not made to accept the gpu/cpu check i made????
      ax.get_xaxis().set_visible(False)                                   #Hides the x axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
      ax.get_yaxis().set_visible(False)                                   #Hides the y axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
      if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
        ax.set_title('EPOCH %s \nOriginal images' %(epoch))               #When above condition is reached, the plots title is set                                   #When above condition is reached, the plots title is set

      ax = plt.subplot(3, n, i + 1 + n)                                   #Creates a number of subplots for the 'Corrupted images??????' i.e the labels. the position of the subplot is i+1+n as it falls in the second row
      plt.imshow(noise_im, cmap='gist_gray', vmin=0, vmax=time_dimension)   #plt.imshow plots an image. The arguments for imshow are, 'image data array' and cmap= which is the colour map. #.squeeze() acts on a tensor and returns a tensor, it removes all dimensions of the tensor that are of length 1, (A×1×B) becomes (AxB) where A and B are greater than 1 #.numpy() creates a numpy array from a tensor #!!! is the .cpu part becuase the code was not made to accept the gpu/cpu check i made????
      ax.get_xaxis().set_visible(False)                                   #Hides the x axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
      ax.get_yaxis().set_visible(False)                                   #Hides the y axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
      if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
        ax.set_title('Corrupted images')                                  #When above condition is reached, the plots title is set

      ax = plt.subplot(3, n, i + 1 + n + n)                               #Creates a number of subplots for the 'Reconstructed images??????' i.e the labels. the position of the subplot is i+1+n+n as it falls in the third row
      plt.imshow(rec_im, cmap='gist_gray', vmin=0, vmax=time_dimension)       #plt.imshow plots an image. The arguments for imshow are, 'image data array' and cmap= which is the colour map. #.squeeze() acts on a tensor and returns a tensor, it removes all dimensions of the tensor that are of length 1, (A×1×B) becomes (AxB) where A and B are greater than 1 #.numpy() creates a numpy array from a tensor #!!! is the .cpu part becuase the code was not made to accept the gpu/cpu check i made????
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






    # 3D Reconstruction
    in_im = reconstruct_3D(in_im) #reconstructs the 3D image using the reconstruct_3D function
    noise_im = reconstruct_3D(noise_im) 
    rec_im = reconstruct_3D(rec_im)
    
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

## PLOT FUNC V2
def plot_ae_outputs_den2(encoder, decoder, epoch, model_save_name, time_dimension, reconstruction_threshold, signal_points, noise_points=0, x_std_dev=0, y_std_dev=0, tof_std_dev=0, n=10):       #Defines a function for plotting the output of the autoencoder. And also the input + clean training data? Function takes inputs, 'encoder' and 'decoder' which are expected to be classes (defining the encode and decode nets), 'n' which is the number of ?????Images in the batch????, and 'noise_factor' which is a multiplier for the magnitude of the added noise allowing it to be scaled in intensity.  
    """
    n is the number of images to plot
    """
    ### 2D Input/Output Comparison Plots 

    #Initialise lists for true and recovered signal point values 
    number_of_true_signal_points = []
    number_of_recovered_signal_points = []

    plt.figure(figsize=(16,9))                                      #Sets the figure size

    for i in range(n):                                                #Runs for loop where 'i' itterates over 'n' total values which range from 0 to n-1

        #Select settings randomly from user range
        signal_points_r = input_range_to_random_value(signal_points)
        x_std_dev_r = input_range_to_random_value(x_std_dev) 
        y_std_dev_r = input_range_to_random_value(y_std_dev) 
        tof_std_dev_r = input_range_to_random_value(tof_std_dev)
        noise_points_r = input_range_to_random_value(noise_points)

        # Load input image
        img = test_dataset[i][0].unsqueeze(0) # [t_idx[i]][0].unsqueeze(0)                    #!!! ????
        
        #Determine the number of signal points on the input image (have to change this to take it directly from the embeded val in the datsaset as when addig noise this method will break)   
        int_sig_points = (img >= reconstruction_threshold).sum()
        number_of_true_signal_points.append(int(int_sig_points.numpy()))
        
        #if epoch <= print_every_other:                                                  #CHECKS TO SEE IF THE EPOCH IS LESS THAN ZERO , I ADDED THIS TO GET THE SAME NOISED IMAGES EACH EPOCH THOUGH THIS COULD BE WRONG TO DO?
        global image_noisy                                          #'global' means the variable (image_noisy) set inside a function is globally defined, i.e defined also outside the function
        sparse_output_batch = create_sparse_signal(img, signal_points_r)
        sparse_and_resolution_limited_batch = simulate_detector_resolution(sparse_output_batch, x_std_dev_r, y_std_dev_r, tof_std_dev_r)
        noised_sparse_reslimited_batch = add_noise_points_to_batch_prenorm(sparse_and_resolution_limited_batch, noise_points_r, time_dimension)
    
        if test_just_masking_mode:
            normalised_batch = custom_mask_optimised_normalisation_torch(noised_sparse_reslimited_batch)
        else:
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

        sparse_im = sparse_output_batch.squeeze()
        reslim_im = sparse_and_resolution_limited_batch.squeeze()

        network_input_image = normalised_batch.squeeze() #.cpu().squeeze().numpy()??????????????????
        recovered_test_image = rec_img.cpu().squeeze().numpy()



        #clean up lines
        in_im = test_image   
        noise_im = network_input_image
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

        #ADD IN MASKING BELOW!!!!
        masked_im = masking_recovery(noise_im, rec_im, time_dimension)



        #cmap = cm.get_cmap('viridis')
        #cmap.set_under('k') # set the color for 0 to black ('k')

        #Following section generates the img plots for the original(labels), noised, and denoised data)
        

        ax = plt.subplot(6,n,i+1)                                       #Creates a number of subplots for the 'Original images??????' i.e the labels. the position of the subplot is i+1 as it falls in the first row
        plt.imshow(in_im.T, cmap='gist_gray', vmin=0, vmax=time_dimension)           #plt.imshow plots an image. The arguments for imshow are, 'image data array' and cmap= which is the colour map. #.squeeze() acts on a tensor and returns a tensor, it removes all dimensions of the tensor that are of length 1, (A×1×B) becomes (AxB) where A and B are greater than 1 #.numpy() creates a numpy array from a tensor #!!! is the .cpu part becuase the code was not made to accept the gpu/cpu check i made????
        ax.get_xaxis().set_visible(False)                                   #Hides the x axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
        ax.get_yaxis().set_visible(False)                                   #Hides the y axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
        if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
            ax.set_title('EPOCH %s \nOriginal images' %(epoch))               #When above condition is reached, the plots title is set                                   #When above condition is reached, the plots title is set

        ax = plt.subplot(6, n, i + 1 + n)                                   #Creates a number of subplots for the 'Corrupted images??????' i.e the labels. the position of the subplot is i+1+n as it falls in the second row
        plt.imshow(sparse_im.T, cmap='gist_gray', vmin=0, vmax=time_dimension)   #plt.imshow plots an image. The arguments for imshow are, 'image data array' and cmap= which is the colour map. #.squeeze() acts on a tensor and returns a tensor, it removes all dimensions of the tensor that are of length 1, (A×1×B) becomes (AxB) where A and B are greater than 1 #.numpy() creates a numpy array from a tensor #!!! is the .cpu part becuase the code was not made to accept the gpu/cpu check i made????
        ax.get_xaxis().set_visible(False)                                   #Hides the x axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
        ax.get_yaxis().set_visible(False)                                   #Hides the y axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
        if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
            ax.set_title('Sparsity Applied')                                  #When above condition is reached, the plots title is set

        ax = plt.subplot(6, n, i + 1 + n + n)                               #Creates a number of subplots for the 'Reconstructed images??????' i.e the labels. the position of the subplot is i+1+n+n as it falls in the third row
        plt.imshow(reslim_im.T, cmap='gist_gray', vmin=0, vmax=time_dimension)       #plt.imshow plots an image. The arguments for imshow are, 'image data array' and cmap= which is the colour map. #.squeeze() acts on a tensor and returns a tensor, it removes all dimensions of the tensor that are of length 1, (A×1×B) becomes (AxB) where A and B are greater than 1 #.numpy() creates a numpy array from a tensor #!!! is the .cpu part becuase the code was not made to accept the gpu/cpu check i made????
        ax.get_xaxis().set_visible(False)                                   #Hides the x axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
        ax.get_yaxis().set_visible(False)                                   #Hides the y axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
        if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
            ax.set_title('Resoloution Limited')                             #When above condition is reached, the plots title is set 

        ax = plt.subplot(6, n, i + 1 + n + n + n)                                   #Creates a number of subplots for the 'Corrupted images??????' i.e the labels. the position of the subplot is i+1+n as it falls in the second row
        plt.imshow(noise_im.T, cmap='gist_gray', vmin=0, vmax=time_dimension)   #plt.imshow plots an image. The arguments for imshow are, 'image data array' and cmap= which is the colour map. #.squeeze() acts on a tensor and returns a tensor, it removes all dimensions of the tensor that are of length 1, (A×1×B) becomes (AxB) where A and B are greater than 1 #.numpy() creates a numpy array from a tensor #!!! is the .cpu part becuase the code was not made to accept the gpu/cpu check i made????
        ax.get_xaxis().set_visible(False)                                   #Hides the x axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
        ax.get_yaxis().set_visible(False)                                   #Hides the y axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
        if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
            ax.set_title('Corrupted images')                                  #When above condition is reached, the plots title is set

        ax = plt.subplot(6, n, i + 1 + n + n + n + n)                               #Creates a number of subplots for the 'Reconstructed images??????' i.e the labels. the position of the subplot is i+1+n+n as it falls in the third row
        plt.imshow(rec_im.T, cmap='gist_gray', vmin=0, vmax=time_dimension)       #plt.imshow plots an image. The arguments for imshow are, 'image data array' and cmap= which is the colour map. #.squeeze() acts on a tensor and returns a tensor, it removes all dimensions of the tensor that are of length 1, (A×1×B) becomes (AxB) where A and B are greater than 1 #.numpy() creates a numpy array from a tensor #!!! is the .cpu part becuase the code was not made to accept the gpu/cpu check i made????
        ax.get_xaxis().set_visible(False)                                   #Hides the x axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
        ax.get_yaxis().set_visible(False)                                   #Hides the y axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
        if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
            ax.set_title('Reconstructed images')                             #When above condition is reached, the plots title is set 

        ax = plt.subplot(6, n, i + 1 + n + n + n + n + n)                               #Creates a number of subplots for the 'Reconstructed images??????' i.e the labels. the position of the subplot is i+1+n+n as it falls in the third row
        plt.imshow(masked_im.T, cmap='gist_gray', vmin=0, vmax=time_dimension)       #plt.imshow plots an image. The arguments for imshow are, 'image data array' and cmap= which is the colour map. #.squeeze() acts on a tensor and returns a tensor, it removes all dimensions of the tensor that are of length 1, (A×1×B) becomes (AxB) where A and B are greater than 1 #.numpy() creates a numpy array from a tensor #!!! is the .cpu part becuase the code was not made to accept the gpu/cpu check i made????
        ax.get_xaxis().set_visible(False)                                   #Hides the x axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
        ax.get_yaxis().set_visible(False)                                   #Hides the y axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
        if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
            ax.set_title('Masked Reconstruction')                             #When above condition is reached, the plots title is set 


    plt.subplots_adjust(left=0.01,              #Adjusts the exact layout of the plots including whwite space round edges
                    bottom=0.02, 
                    right=0.99, 
                    top=0.95, 
                    #wspace=0.001,  
                    hspace=0.3
                    )     
    #plt.tight_layout()

    ### NEW PLOT SAVING SIMPLIFIED (TESTING CURRENTLY)
    if (epoch) % print_every_other == 0:     #if the epoch is a multiple of print_every_other, then the plot is saved
        Out_Label = graphics_dir + f'{model_save_name} - Epoch {epoch}.png' #creates the name of the file to be saved
        plot_save_choice(plot_or_save, Out_Label) #saves the plot if plot_or_save is set to 1, if 0 it displays, if 2 it displays and saves
        plt.close()
    else: #??
        plt.close()


    ### 3D Reconstruction Plots 
    # 3D Reconstruction
    in_im = reconstruct_3D(in_im) #reconstructs the 3D image using the reconstruct_3D function
    sparse_im = reconstruct_3D(sparse_im)
    reslim_im = reconstruct_3D(reslim_im)
    noise_im = reconstruct_3D(noise_im) 
    rec_im = reconstruct_3D(rec_im)
    masked_im = reconstruct_3D(masked_im)

    #3D Plottting
    if rec_im.ndim != 1:                       # Checks if there are actually values in the reconstructed image, if not no image is aseved/plotted
        fig, axs = plt.subplots(2, 3, subplot_kw={'projection': '3d'})
        ax1, ax2, ax3, ax4, ax5, ax6 = axs.flatten()
        fig.suptitle(f"3D Reconstruction - Epoch {epoch}") #sets the title of the plot

        ax1.scatter(in_im[:,0], in_im[:,1], in_im[:,2]) #plots the 3D scatter plot for input 
        ax1.set_xlim(0, 128)
        ax1.set_ylim(0, 88)
        ax1.set_zlim(0, time_dimension)

        ax2.scatter(sparse_im[:,0], sparse_im[:,1], sparse_im[:,2]) #plots the 3D scatter plot for sparse image
        ax2.set_zlim(0, time_dimension)
        ax2.set_xlim(0, 128)
        ax2.set_ylim(0, 88)
    
        ax3.scatter(reslim_im[:,0], reslim_im[:,1], reslim_im[:,2]) #plots the 3D scatter plot for reslim image
        ax3.set_zlim(0, time_dimension)
        ax3.set_xlim(0, 128)
        ax3.set_ylim(0, 88)

        ax4.scatter(noise_im[:,0], noise_im[:,1], noise_im[:,2]) #plots the 3D scatter plot for noised image
        ax4.set_zlim(0, time_dimension)
        ax4.set_xlim(0, 128)
        ax4.set_ylim(0, 88)

        ax5.scatter(rec_im[:,0], rec_im[:,1], rec_im[:,2]) #plots the 3D scatter plot for reconstructed image
        ax5.set_zlim(0, time_dimension)
        ax5.set_xlim(0, 128)
        ax5.set_ylim(0, 88)

        ax6.scatter(masked_im[:,0], masked_im[:,1], masked_im[:,2]) #plots the 3D scatter plot for masked image
        ax6.set_zlim(0, time_dimension)
        ax6.set_xlim(0, 128)
        ax6.set_ylim(0, 88)

        #plt.tight_layout() #Tight layout is used to make sure the plots do not overlap

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
"""
if simple_norm_instead_of_custom:
    custom_normalisation_with_args = lambda x: x/time_dimension #if simple_norm_instead_of_custom is set to True, then the custom_normalisation_with_args is set to a lambda function that returns the input data divided by the time dimension
else:
    custom_normalisation_with_args = partial(custom_normalisation, reconstruction_threshold=reconstruction_threshold, time_dimension=time_dimension)   #using functools partial to bundle the args into custom norm to use in custom torch transform using lambda function

if all_norm_off:
    custom_normalisation_with_args = lambda x: x #if all_norm_off is set to True, then the custom_normalisation_with_args is set to a lambda function that returns the input data unchanged

add_noise_with_Args = partial(add_noise_points, noise_points=noise_points, reconstruction_threshold=reconstruction_threshold)   #using functools partial to bundle the args into custom norm to use in custom torch transform using lambda function
"""
######!!!!!!!!

tensor_transform = partial(np_to_tensor, double_precision=double_precision) #using functools partial to bundle the args into np_to_tensor to use in custom torch transform using lambda function


### Transormations
train_transform = transforms.Compose([                                         #train_transform variable holds the tensor tranformations to be performed on the training data.  transforms.Compose([ ,  , ]) allows multiple transforms to be chained together (in serial?) (#!!! does it do more than this??)
                                       #transforms.RandomRotation(30),         #transforms.RandomRotation(angle (degrees?) ) rotates the tensor randomly up to max value of angle argument
                                       #transforms.RandomResizedCrop(224),     #transforms.RandomResizedCrop(pixels) crops the data to 'pixels' in height and width (#!!! and (maybe) chooses a random centre point????)
                                       #transforms.RandomHorizontalFlip(),     #transforms.RandomHorizontalFlip() flips the image data horizontally 
                                       #transforms.Normalize((0.5), (0.5)),    #transforms.Normalize can be used to normalise the values in the array
                                       #transforms.Lambda(custom_normalisation_with_args),
                                       #transforms.Lambda(add_noise_with_Args),        ####USed during debugging, noise adding should be moved to later? or maybe not tbf as this is place to add it if wanting it trained on??
                                       transforms.Lambda(tensor_transform),
                                       #transforms.ToTensor(),
                                       #transforms.RandomRotation(180)
                                       ])                 #other transforms can be dissabled but to tensor must be left enabled ! it creates a tensor from a numpy array #!!! ?

test_transform = transforms.Compose([                                          #test_transform variable holds the tensor tranformations to be performed on the evaluation data.  transforms.Compose([ ,  , ]) allows multiple transforms to be chained together (in serial?) (#!!! does it do more than this??)
                                      #transforms.Resize(255),                 #transforms.Resize(pixels? #!!!) ??
                                      #transforms.CenterCrop(224),             #transforms.CenterCrop(pixels? #!!!) ?? Crops the given image at the center. If the image is torch Tensor, it is expected to have […, H, W] shape, where … means an arbitrary number of leading dimensions. If image size is smaller than output size along any edge, image is padded with 0 and then center cropp
                                      #transforms.Normalize((0.5), (0.5)),     #transforms.Normalize can be used to normalise the values in the array
                                      #transforms.Lambda(custom_normalisation_with_args), #transforms.Lambda(function) applies a custom transform function to the data
                                      #transforms.Lambda(add_noise_with_Args),
                                      #transforms.ToTensor(),                           #transforms.ToTensor() converts a numpy array to a tensor
                                      transforms.Lambda(tensor_transform),
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


if val_set_on:
    val_split = int(m2 * val_test_split_ratio) #val_split is the ratio of npon train images to be used in the validation set as opposed to test set
    test_data, val_data = random_split(non_training_data, [m2 - val_split, val_split])  
else:   #this all needs cleaning up, dont need val set in this case bu tnot having one does break other lines
    test_data = non_training_data  
    val_data = non_training_data

###Following section for Dataloaders, they just pull a random sample of images from each of the datasets we now have, train_data, valid_data, and test_data. the batch size defines how many are taken from each set, shuffle argument shuffles them each time?? #!!!
# required to load the data into the endoder/decoder. Combines a dataset and a sampler, and provides an iterable over the given dataset.
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)                 #Training data loader, can be run to pull training data as configured  Also is shuffled using parameter shuffle
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)                                 #Testing data loader, can be run to pull training data as configured. 
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)                                 #Validation data loader, can be run to pull training data as configured


#%% - Setup model, loss criteria and optimiser    
### Initialize the encoder and decoder
encoder = Encoder(latent_dim, print_encoder_debug)
decoder = Decoder(latent_dim, print_decoder_debug)

# Sets the encoder and decoder to double precision floating point arithmetic (fp64)
if double_precision:
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

#%% - Add Activation data hooks to encoder and decoder layers
if record_activity or record_weights or record_biases:

    # Create a list to store the activations of each layer
    all_activations = []
    recorded_weights = [] 
    recorded_biases = [] 

    # Loop through all the modules (layers) in the encoder and register the hooks
    for idx, module in enumerate(encoder.encoder_lin.modules()):
        if isinstance(module, torch.nn.Linear):
            print("Registering hooks for encoder layer: ", idx)
            if record_activity:
                # Register the hook with the layer_index as the key to identify activations for this layer
                module.register_forward_hook(partial(activation_hook_fn, layer_index=idx))
            if record_weights:
                # Register the hook with the layer_index as the key to identify activations for this layer
                module.register_forward_hook(partial(weights_hook_fn, layer_index=idx))
            if record_biases:
                # Register the hook with the layer_index as the key to identify activations for this layer
                module.register_forward_hook(partial(biases_hook_fn, layer_index=idx))

    enc_max_idx = idx
    # Loop through all the modules (layers) in the decoder and register the hooks
    for idx, module in enumerate(decoder.decoder_lin.modules()):
        if isinstance(module, torch.nn.Linear):
            print("Registering hooks for decoder layer: ", enc_max_idx + idx)
            if record_activity:
                # Register the hook with the layer_index as the key to identify activations for this layer
                module.register_forward_hook(partial(activation_hook_fn, layer_index = enc_max_idx + idx))
            if record_weights:
                # Register the hook with the layer_index as the key to identify activations for this layer
                module.register_forward_hook(partial(weights_hook_fn, layer_index = enc_max_idx + idx))
            if record_biases:
                # Register the hook with the layer_index as the key to identify activations for this layer
                module.register_forward_hook(partial(biases_hook_fn, layer_index = enc_max_idx + idx))

    print("All hooks registered\n")




        



#%% - Prepare Network Summary
# Set evaluation mode for encoder and decoder

with torch.no_grad(): # No need to track the gradients
    # Create dummy input tensor
    enc_input_size = (batch_size, 1, 128, 88)
    enc_input_tensor = torch.randn(enc_input_size)  # Cast input tensor to double precision

    if double_precision:
        enc_input_tensor = enc_input_tensor.double()
        
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

# Create a progress bar if print_partial_training_losses is False                #####FIX THE RANGE OF EPOCHS NOT TO INLCUDE 0! makse sure this dosent effect any of the rest of the code!!!
if print_partial_training_losses:  # Prints partial train losses per batch
    loop_range = range(1, num_epochs+1)
else:                              # No print partial train losses per batch, instead create progress bar
    loop_range = tqdm(range(1, num_epochs+1), desc='Epochs', colour='red')




try:    # Try except clause allows user to exit trainig gracefully whilst still retaiing a saved model and ouput plots
    # bringing everything together to train model
    for epoch in loop_range:                              #For loop that iterates over the number of epochs where 'epoch' takes the values (0) to (num_epochs - 1)
        if print_partial_training_losses:
            print(f'\nStart of EPOCH {epoch + 1}/{num_epochs}')
        
        ### Training (use the training function)
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
                                print_partial_training_losses = print_partial_training_losses)

        ### Validation (use the testing function)
        val_loss = test_epoch_den(
                                encoder, 
                                decoder, 
                                device, 
                                dataloader=valid_loader, 
                                loss_fn=loss_fn,
                                signal_points=signal_points,
                                noise_points=noise_points,
                                x_std_dev = x_std_dev, 
                                y_std_dev = y_std_dev,
                                tof_std_dev = tof_std_dev,
                                time_dimension=time_dimension,
                                reconstruction_threshold=reconstruction_threshold,
                                print_partial_training_losses = print_partial_training_losses)
        
        # Print Validation_loss and plots at end of each epoch
        history_da['train_loss'].append(train_loss)
        history_da['val_loss'].append(val_loss)

        #Updates the epoch reached counter  
        max_epoch_reached = epoch    

        if print_partial_training_losses:
            print('\n End of EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))     #epoch +1 is to make up for the fact the range spans 0 to epoch-1 but we want to numerate things from 1 upwards for sanity
        
        if epoch % print_every_other == 0 and epoch != 0:
            
            print("\n## EPOCH {} PLOTS DRAWN ##\nPress Ctr + c to exit and save the model from this epoch. \n  \n".format(epoch))
            
            # Run plotting function for training feedback and telemetry.
            encoder.eval()
            decoder.eval()

            returned_data_from_plotting_function = plot_ae_outputs_den2(encoder, 
                                                                    decoder, 
                                                                    epoch, 
                                                                    model_save_name, 
                                                                    time_dimension, 
                                                                    reconstruction_threshold,
                                                                    signal_points=signal_points,
                                                                    noise_points=noise_points,
                                                                    x_std_dev = x_std_dev, 
                                                                    y_std_dev = y_std_dev,
                                                                    tof_std_dev = tof_std_dev,
                                                                    n = num_to_plot)
            
            number_of_true_signal_points, number_of_recovered_signal_points, in_data, noisy_data, rec_data = returned_data_from_plotting_function
            
            if save_all_raw_plot_data:
                save_variable(number_of_true_signal_points, f'Epoch {epoch}_number_of_true_signal_points', raw_plotdata_output_dir, force_pickle=True)
                save_variable(number_of_recovered_signal_points, f'Epoch {epoch}_number_of_recovered_signal_points', raw_plotdata_output_dir, force_pickle=True)
                save_variable(in_data, f'Epoch {epoch}_input_list', raw_plotdata_output_dir, force_pickle=True)
                save_variable(noisy_data, f'Epoch {epoch}_noised_list', raw_plotdata_output_dir, force_pickle=True)
                save_variable(rec_data, f'Epoch {epoch}_recovered_list', raw_plotdata_output_dir, force_pickle=True)
        
            
            
            
            encoder.train()
            decoder.train()

            if plot_live_training_loss:
                Out_Label = graphics_dir + f'{model_save_name} - Live Train loss.png'

                if comparative_live_loss:
                    # protection from surpassing the comaprison data during training. 
                    if epoch > len(control_history_da['train_loss']):
                        control_history_da['train_loss'].append(np.nan)
                    if epoch > len(comparative_history_da['train_loss']):
                        comparative_history_da['train_loss'].append(np.nan)


                    x_list = [range(0,max_epoch_reached), range(0,max_epoch_reached), range(0,max_epoch_reached)]
                    y_list = [history_da['train_loss'][0:max_epoch_reached], control_history_da['train_loss'][0:max_epoch_reached], comparative_history_da['train_loss'][0:max_epoch_reached]]
                    legend_label_list = ["Training loss", "Control Training loss", "Comparative Training loss"]
                else:
                    x_list = [range(0,max_epoch_reached)]
                    y_list = [history_da['train_loss']]
                    legend_label_list = ["Training loss"]
                comparitive_loss_plot(x_list, y_list, legend_label_list, "Epoch number", "Train loss (MSE)", "Live Training loss", Out_Label, plot_or_save)

        pass # end of try clause, if all goes well and user doesen't request an early exit then the training loop will end here

# If user presses Ctr + c to exit training loop, this handles the exception and allows the code to run its final data and model saving etc before exiting        
except KeyboardInterrupt:
    print("Keyboard interrupt detected. Exiting training gracefully...")


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
epochs_range = range(1,max_epoch_reached+1) 
if save_all_raw_plot_data:
    save_variable(np.array(epochs_range), "epochs_range", raw_plotdata_output_dir)  #!!!!!! testing raw plot outputter!!!!

if plot_train_loss:
    plt.plot(epochs_range, history_da['train_loss']) 
    plt.title("Training loss")   
    plt.xlabel("Epoch number")
    plt.ylabel("Train loss (ACB-MSE)")
    plt.grid(alpha=0.2)
    Out_Label =  graphics_dir + f'{model_save_name} - Train loss - Epoch {epoch}.png'
    plot_save_choice(plot_or_save, Out_Label)  

if plot_validation_loss:
    plt.plot(epochs_range, history_da['train_loss'])   #ERROR SHOULD BE VAL LOSS!
    plt.title("Validation loss") 
    plt.xlabel("Epoch number")
    plt.ylabel("Val loss (ACB-MSE)")
    plt.grid(alpha=0.2)
    Out_Label =  graphics_dir + f'{model_save_name} - Val loss - Epoch {epoch}.png'
    plot_save_choice(plot_or_save, Out_Label)    

if plot_detailed_performance_loss:   #CLEAN UP!!!!!!!
        
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

    axs[0, 0].plot(epochs_range, avg_loss_mae)
    axs[0, 0].set_title("MAE loss")
    axs[0, 0].set_xlabel("Epoch number")
    axs[0, 0].set_ylabel("Loss (MAE)")
    axs[0, 0].grid(alpha=0.2) 

    axs[0, 1].plot(epochs_range, avg_loss_snr)
    axs[0, 1].set_title("SNR loss")
    axs[0, 1].set_xlabel("Epoch number")
    axs[0, 1].set_ylabel("Loss (SNR)")
    axs[0, 1].grid(alpha=0.2) 

    axs[0, 2].plot(epochs_range, avg_loss_psnr)
    axs[0, 2].set_title("PSNR loss")
    axs[0, 2].set_xlabel("Epoch number")
    axs[0, 2].set_ylabel("Loss (PSNR)")
    axs[0, 2].grid(alpha=0.2) 

    axs[1, 0].plot(epochs_range, avg_loss_ssim)
    axs[1, 0].set_title("SSIM loss")
    axs[1, 0].set_xlabel("Epoch number")
    axs[1, 0].set_ylabel("Loss (SSIM)")
    axs[1, 0].grid(alpha=0.2) 

    axs[1, 1].plot(epochs_range, avg_loss_nmi)
    axs[1, 1].set_title("NMI loss")
    axs[1, 1].set_xlabel("Epoch number")
    axs[1, 1].set_ylabel("Loss (NMI)")
    axs[1, 1].grid(alpha=0.2) 

    axs[1, 2].plot(epochs_range, avg_loss_cc)
    axs[1, 2].set_title("Coreelation Coefficent? loss")
    axs[1, 2].set_xlabel("Epoch number")
    axs[1, 2].set_ylabel("Loss (CC)")
    axs[1, 2].grid(alpha=0.2) 

    axs[2, 0].plot(epochs_range, avg_loss_true_positive_xy)
    axs[2, 0].set_title("True Positive XY loss")
    axs[2, 0].set_xlabel("Epoch number")
    axs[2, 0].set_ylabel("Loss (True Positive XY %)")
    axs[2, 0].set_ylim(-5 ,105)
    axs[2, 0].grid(alpha=0.2) 

    axs[2, 1].plot(epochs_range, avg_loss_true_positive_tof)
    axs[2, 1].set_title("True Positive TOF loss")
    axs[2, 1].set_xlabel("Epoch number")
    axs[2, 1].set_ylabel("Loss (True Positive TOF %)")
    axs[2, 1].set_ylim(-5 ,105)
    axs[2, 1].grid(alpha=0.2) 

    axs[2, 2].plot(epochs_range, avg_loss_false_positive_xy)
    axs[2, 2].set_title("False Positive XY loss BROKEN?")
    axs[2, 2].set_xlabel("Epoch number")
    axs[2, 2].set_ylabel("Loss (False Positive XY)")
    axs[2, 2].grid(alpha=0.2)

    Out_Label =  graphics_dir + f'{model_save_name} - Detailed Performance loss - Epoch {epoch}.png'
    plot_save_choice(plot_or_save, Out_Label)

if plot_cutoff_telemetry:
    plot_telemetry(telemetry, signal_points, plot_or_save=plot_or_save)

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
    plt.axis("off")
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

    print("\nSaving model state dictionary to output dir...")
    # Save the state dictionary
    torch.save({'encoder_state_dict': encoder.state_dict(),   
                'decoder_state_dict': decoder.state_dict(),   
                'optimizer_state_dict': optim.state_dict()},  
                full_statedict_path)  
    print("- Completed -") 



    print("\nSaving Autoencoder python file to output dir...")

    # Save the latent dimesion value for automatic setting when deploying the models
    save_variable(latent_dim, "deployment_variables_ld", model_output_dir)

    # Save the processing precision value for automatic setting when deploying the models
    save_variable(double_precision, "deployment_variables_double_precision", model_output_dir)

    # Get the directory name of the current Python file for the autoencoder export
    search_dir = os.path.abspath(__file__)
    search_dir = os.path.dirname(search_dir)

    # Locate .py file that defines the Encoder and Decoder and copies it to the model save dir, due to torch.save model save issues
    AE_file_name = Robust_model_export(Encoder, search_dir, dir) #Only need to run on encoder as encoder and decoder are both in the same file so both get saved this way
    print("- Completed -")


    #%% - Saving network activations 
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

    #%% - Export all data logged to disk in form of .txt file in the output dir
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
        
        output_file.write("\nLoss Function Settings:\n")    # Write the loss function settings to the file
        output_file.write(f"Loss Function Choice: {loss_fn}\n")
        if loss_function_selection == 0:
            output_file.write(f"zero_weighting: {zero_weighting}\n")
            output_file.write(f"nonzero_weighting: {nonzero_weighting}\n")    
        if loss_function_selection == 6:
            output_file.write(f"Split Loss Functions Selected (Hits, Non-Hits): {split_loss_functions}\n")

        output_file.write("\nNormalisation:\n")     # Write the normalisation settings to the file
        output_file.write(f"simple_norm_instead_of_custom: {simple_norm_instead_of_custom}\n")    
        output_file.write(f"all_norm_off: {all_norm_off}\n") 

        output_file.write("\nPre Training:\n")   # Write the pre training settings to the file
        if start_from_pretrained_model:
            output_file.write(f"pretrained_model_path: {pretrained_model_path}\n")
            output_file.write(f"full_statedict_path: {full_statedict_path}\n")    
        output_file.write(f"start_from_pretrained_model: {start_from_pretrained_model}\n")
        output_file.write(f"load_pretrained_optimser: {load_pretrained_optimser}\n")  
        
        output_file.write("\n \nFull Data Readouts:\n") 
        for key, value in full_data_output.items(): 
            output_file.write(f"{key}: {value}\n") 

        output_file.write("\nPython Lib Versions:\n") # Write the python library versions to the file
        output_file.write((f"PyTorch: {torch.__version__}\n"))  
        output_file.write((f"Torchvision: {torchvision.__version__}\n"))     
        output_file.write((f"Numpy: {np.__version__}\n")) 

        system_information = get_system_information()  # Get the system information using helper function
        output_file.write("\n" + system_information)  # Write the system information to the file

        output_file.write("\nAutoencoder Network:\n")  # Write the autoencoder network settings to the file
        output_file.write((f"AE File ID: {AE_file_name}\n"))    # Write the autoencoder network file ID to the file
        output_file.write("\n" + summary_str)   # Write the autoencoder network summary to the file
    print("- Completed -")


#%% - Save any remaining unsaved raw plot data
if save_all_raw_plot_data:
    save_variable(history_da, "history_da", raw_plotdata_output_dir) 

    save_variable(settings, "settings", raw_plotdata_output_dir)


#%% - End of Program - Printing message to notify user!
print("\nProgram Complete - Shutting down...\n")    
    
