# -*- coding: utf-8 -*-
"""
DeepClean v0.3.4
Build created on Sat Feb 1 2022
Authors: Adill Al-Ashgar & Max Carter
University of Bristol

@Adill: adillwmaa@gmail.co.uk - ex18871@bristol.ac.uk
@Max: qa19105@bristol.ac.uk



Possible improvements:
### ~~~~~ [DONE!] Make sure that autoecoder Encoder and Decoder are saved along with model in the models folder 

### ~~~~~~ [TESTING!] Allow normalisation/renorm to be bypassed, to check how it affects results 

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

### ~~~~~ [DONE!] Fix Val loss save bug

### ~~~~~ [DONE!] Move things to other files (AE, Helper funcs, Visulisations etc)

### ~~~~~ [DONE!] Fix reconstruction threshold, use recon threshold to set bottom limit in custom normalisation

### ~~~~~ [DONE!] Turn plot or save into a function 

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

### ~~~~~ [TESTING!]Add way to compress the NPZ output as filesize is to large ! ~3Gb+

### ~~~~~ Add all advanced program settings to end of net summary txt file i.e what typ eof normalisation used etc, also add th enam eof the autoencoder file i.e AE_V1 etc from the module name 
"""
import torch

#%% - User Inputs
#mode = 0 ### 0=Data_Gathering, 1=Testing, 2=Speed_Test, 3=Debugging
dataset_title = "Dataset 24_X10ks"#"Dataset 27_X100K" #"Dataset 12_X10K" ###### TRAIN DATASET : NEED TO ADD TEST DATASET?????
model_save_name = "D24 10K 120epochs simple renorm"#"D27 100K ld8"#"Dataset 18_X_rotshiftlarge"

num_epochs = 120                                          #User controll to set number of epochs (Hyperparameter)
batch_size = 10                                 #User controll to set batch size (Hyperparameter) - #Data Loader, number of Images to pull per batch 
latent_dim = 6                    #User controll to set number of nodes in the latent space, the bottleneck layer (Hyperparameter)

learning_rate = 0.001  #User controll to set optimiser learning rate(Hyperparameter)
optim_w_decay = 1e-5  #User controll to set optimiser weight decay (Hyperparameter)
loss_fn = torch.nn.MSELoss()  # torch.nn.BCELoss(reduction='none') #torch.nn.MSELoss()   #!!!!!!   #MSELoss()          #(mean square error) User controll to set loss function (Hyperparameter)

time_dimension = 100
noise_factor = 0                                          #User controll to set the noise factor, a multiplier for the magnitude of noise added. 0 means no noise added, 1 is defualt level of noise added, 10 is 10x default level added (Hyperparameter)

reconstruction_threshold = 0.5      #MUST BE BETWEEN 0-1        #Threshold for 3d reconstruction, values below this confidence level are discounted

#%% - Pretraining settings
start_from_pretrained_model = False
load_pretrained_optimser = False      #only availible if above is set to true
pretrained_model_path = 'N:/Yr 3 Project Results/D20_3 X5k - Training Results/D20_3 X5k - Model + Optimiser State Dicts.pth'      # specify the path to the saved full state dictionary


"""
#### NEW MULTI-LOSS FUCN WITH WEIGHTS
loss_functions = [torch.nn.L1Loss(), torch.nn.MSELoss()] 
loss_fn_weightings = [0.5, 0.5]

# Protection from weights and loss_fns not being same len
if len(loss_functions) != len(loss_fn_weightings):
    raise ValueError("Number of Loss Fucntions does not match number of weights for loss functions")

#### NEW MULTI-LOSS FUNC CONSTRUCTION - THIS NEEDS TO BE MOVED DOWN TO THE MODEL SETUP ETC SECTION (NEEDS TESTING FIRST)
lf_total = 0
for i in range (0, len(loss_fn_weightings)):
    lf_contribution = loss_fn_weightings[i] * loss_functions[i]
    lf_total = lf_total + lf_contribution
"""

#%% - Advanced Debugging Settings
print_encoder_debug = False                     #[default = False]
print_decoder_debug = False                     #[default = False]
debug_noise_function = False                    #[default = False]
debug_loader_batch = False   #REMOVE THIS PARAM!!!  #(Default = False) //INPUT 0 or 1//   #Setting debug loader batch will print to user the images taken in by the dataoader in this current batch and print the corresponding labels

full_dataset_integrity_check = False       #[Default = False] V slow   
full_dataset_distribution_check = False    #[Default = False]

print_network_summary = False              #[Default = False]
seed = 0                                   #[Default = 0] which gives no seeeding to RNG, if the value is not zero then this is used for the RNG seeding for numpy, random, and torch libraries

#Normalisation
simple_norm_instead_of_custom = False      #[Default is False]
all_norm_off = False                       #[Default is False]
simple_renorm = False                       #[Default is False]

#%% - Plotting Control Settings
print_every_other = 2
plot_or_save = 1                            #[default = 1] 0 is normal behavior, If set to 1 then saves all end of epoch printouts to disk, if set to 2 then saves outputs whilst also printing for user

#%% - Advanced Visulisation Settings
plot_train_loss = True
plot_validation_loss = True

plot_cutoff_telemetry = True          #[default = False] # Update name to pixel_cuttoff_telemetry    #Very slow, reduces net performance by XXXXXX%

plot_pixel_difference = False 
plot_latent_generations = True
plot_higher_dim = False
plot_Graphwiz = False

record_activity = False #False  ##Be carefull, the activity file recorded is ~ 2.5Gb  #Very slow, reduces net performance by XXXXXX%
compress_activations_npz_output = False #False   Compresses the activity file above for smaller file size but does increase loading and saving times for the file. (use if low on hdd space)

#%% - Program Settings - CLEAN UP THIS SECTION
speed_test = False      # [speed_test=False]Defualt    true sets number of epocs to print to larger than number of epochs to run so no plotting time wasted etc
data_gathering = True

print_partial_training_losses = False            #[default = True]
allow_escape = False # Default = True
#response_timeout = 120 # in seconds

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
import time
from torchinfo import summary
from tqdm import tqdm
import os
from functools import partial
import datetime

# Imports from our other custom scripts
from Autoencoders.DC3D_Autoencoder_V1 import Encoder, Decoder

from Helper_files.Robust_model_exporter_V1 import Robust_model_export
from Helper_files.System_Information_check import get_system_information
from Helper_files.Dataset_Integrity_Check_V1 import dataset_integrity_check
from Helper_files.Dataset_distribution_tester_V1 import dataset_distribution_tester
from Helper_files.AE_Visulisations import Generative_Latent_information_Visulisation, Reduced_Dimension_Data_Representations, Graphwiz_visulisation, AE_visual_difference

#%% - Helper functions
def custom_normalisation(data, reconstruction_threshold, time_dimension=100):
    data = ((data / time_dimension) / (1/(1-reconstruction_threshold))) + reconstruction_threshold
    for row in data:
        for i, ipt in enumerate(row):
            if ipt == reconstruction_threshold:
                row[i] = 0
    return data

def custom_renormalisation(data, reconstruction_threshold, time_dimension=100):
    data = np.where(data > reconstruction_threshold, ((data - reconstruction_threshold)*(1/(1-reconstruction_threshold)))*(time_dimension), 0)
    return data

def reconstruct_3D(data, reconstruction_threshold):
    data_output = []
    for cdx, row in enumerate(data):
        for idx, num in enumerate(row):
            if num > 0:  #should this be larger than or equal to??? depends how we deal with the 0 slice problem
                data_output.append([cdx,idx,num])
    return np.array(data_output)

def reconstruct_3D2(data, reconstruction_threshold):
    data_output = []
    for cdx, row in enumerate(data):
        for idx, num in enumerate(row):
            if num > reconstruction_threshold:  #should this be larger than or equal to??? depends how we deal with the 0 slice problem
                num_renorm = custom_renormalisation(num)
                data_output.append([cdx,idx,num_renorm])
    return np.array(data_output)

def plot_save_choice(plot_or_save, output_file_path):
    if plot_or_save == 0:
        plt.show()
    else:
        plt.savefig(output_file_path, format='png')    
        if plot_or_save == 1:    
            plt.close()
        else:
            plt.show()

def batch_learning(training_dataset_size, batch_size):
    if batch_size == 1: 
        output = "Stochastic Gradient Descent"
    elif batch_size == training_dataset_size:
        output = "Batch Gradient Descent"        
    else:
        output = "Mini-Batch Gradient Descent"
    return(output)

###Ploting confidence of each pixel as histogram per epoch with line showing the detection threshold
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

###RNG Seeding for Determinism Function
def Determinism_Seeding(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)



#%% - Classes
### Gaussian Noise Generator Class
class AddGaussianNoise(object):                   #Class generates noise based on the mean 0 and std deviation 1, (gaussian)
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

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
if speed_test:
    print_every_other = num_epochs + 5   # Makes sure print is larger than total num of epochs to avoid delays in execution for testing

# Initialises pixel belief telemetry
telemetry = [[0,0.5,0.5]]                # Initalises the telemetry memory, starting values are 0, 0.5, 0.5 which corrspond to epoch(0), above_threshold(0.5), below_threshold(0.5)

# Initialises list for noised image storage
#image_noisy_list = []

# Initialises seeding values to RNGs
if seed != 0: 
    Determinism_Seeding(seed)

#%% - Create record of all user input settings, to add to output data for testing and keeping track of settings
settings = {} 
settings["Epochs"] = num_epochs
settings["Batch Size"] = batch_size
settings["Learning Rate"] = learning_rate
settings["Optimiser Decay"] = optim_w_decay
settings["Loss Function"] = loss_fn
settings["Latent Dimension"] = latent_dim
settings["Noise Factor"] = noise_factor
settings["Dataset"] = dataset_title
settings["Time Dimension"] = time_dimension
settings["Seed Val"] = seed
settings["Reconstruction Threshold"] = reconstruction_threshold


#%% - Functions
### Random Noise Generator Function
def add_noise(input, noise_factor=0.3, debug_noise_function=False):
    noise = torch.randn_like(input) * noise_factor
    noised_img = input + noise
    noised_img = torch.clip(noised_img, 0., 1.)
    if debug_noise_function:
        plt.imshow(noise)
        plt.show()   
    return noised_img

###RNG Seeding for Determinism Function
def Determinism_Seeding(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

### Training Function
def train_epoch_den(encoder, decoder, device, dataloader, loss_fn, optimizer, noise_factor=0.3, print_partial_training_losses=print_partial_training_losses):
    # Set train mode for both the encoder and the decoder
    # train mode makes the autoencoder know the parameters can change
    encoder.train()
    decoder.train()
    train_loss = []
    
    if print_partial_training_losses:  # Prints partial train losses per batch
        image_loop  = (dataloader)
    else:                              # No print partial train losses per batch, instead create progress bar
        image_loop  = tqdm(dataloader, desc='Batches', leave=False)

    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _ in image_loop: # with "_" we just ignore the labels (the second element of the dataloader tuple
        # Move tensor to the proper device
        image_noisy = image_batch###!!!add_noise(image_batch, noise_factor, debug_noise_function)
        image_batch = image_batch.to(device)
        image_noisy = image_noisy.to(device)    
        # Encode data
        encoded_data = encoder(image_noisy)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        if print_partial_training_losses:
            # Print batch loss
            print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())
    return np.mean(train_loss)

### Testing Function
# exact same as train, but doesnt alter the encoder, and defines the loss over the entire batch, not individually.
def test_epoch_den(encoder, decoder, device, dataloader, loss_fn,noise_factor=0.3):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            image_noisy = image_batch#####!!!add_noise(image_batch, noise_factor, debug_noise_function)
            image_noisy = image_noisy.to(device)
            # Encode data
            encoded_data = encoder(image_noisy)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label) 
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return float(val_loss.data)

###Plotting Function
def plot_ae_outputs_den(encoder, decoder, epoch, model_save_name, time_dimension, reconstruction_threshold, n=10, noise_factor=0.3):       #Defines a function for plotting the output of the autoencoder. And also the input + clean training data? Function takes inputs, 'encoder' and 'decoder' which are expected to be classes (defining the encode and decode nets), 'n' which is the number of ?????Images in the batch????, and 'noise_factor' which is a multiplier for the magnitude of the added noise allowing it to be scaled in intensity.  
    """
    n is the number of images to plot i think?
    """

    #%% - 2D Input/Output Comparison Plots 
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
      image_noisy = img#!!!!add_noise(img, noise_factor, debug_noise_function)                   #Runs the function 'add_noise' (in this code) the function adds noise to a set of data, the function takes two arguments, img is the data to add noise to, noise factor is a multiplier for the noise values added, i.e if multiplier is 0 no noise is added, if it is 1 default amount is added, if it is 10 then the values are raised 10x 
      #image_noisy_list.append(image_noisy)                        #Adds the just generated noise image to the list of all the noisy images
      #image_noisy = image_noisy_list[i].to(device)                    #moves the list (i think of tensors?) to the device that will process it i.e either cpu or gpu, we have a check elsewhere in the code that detects if gpu is availible and sets the value of 'device' to gpu or cpu depending on availibility (look for the line that says "device = 'cuda' if torch.cuda.is_available() else 'cpu'"). NOTE: this moves the noised images to device, i think that the original images are already moved to device in previous code
    
      #Following section sets the autoencoder to evaluation mode rather than training (up till line 'with torch.no_grad()')
      encoder.eval()                                   #.eval() is a kind of switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time. For example, Dropouts Layers, BatchNorm Layers etc. You need to turn off them during model evaluation, and .eval() will do it for you. In addition, the common practice for evaluating/validation is using torch.no_grad() in pair with model.eval() to turn off gradients computation
      decoder.eval()                                   #Simarlary as above

      with torch.no_grad():                                               #As mentioned in .eval() comment, the common practice for evaluating/validation is using torch.no_grad() which turns off gradients computation whilst evaluating the model (the opposite of training the model)     
      #Following line runs the autoencoder on the noised data
         rec_img = decoder(encoder(image_noisy))                         #Creates a recovered image (denoised image), by running a noisy image through the encoder and then the output of that through the decoder.

      #Determine the number of signal points on the recovered image 
      int_rec_sig_points = (rec_img >= reconstruction_threshold).sum()      
      number_of_recovered_signal_points.append(int(int_rec_sig_points.numpy()))
      
      test_image = img.squeeze() #.cpu().squeeze().numpy()
      noised_test_image = image_noisy.squeeze() #.cpu().squeeze().numpy()
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
    if (epoch) % print_every_other == 0:
        Out_Label = graphics_dir + f'{model_save_name} - Epoch {epoch}.png'
        plot_save_choice(plot_or_save, Out_Label)
        plt.close()
    else:
        plt.close()

    #%% - 3D Reconstruction Plots 
    in_im = test_image
    noise_im = noised_test_image
    rec_im = recovered_test_image

    ###################WHY IS NORM HERE IN THE CODE??? shouldent it be directly after output and speed time is saved? (i think this is actually for the plots generated during the testig ratehr than afetr so okay maybe not?)
    # RENORMALISATIONN
    if simple_norm_instead_of_custom or all_norm_off:
        rec_im = rec_im * time_dimension
    else: 

        #REMOVE - used for debugging
        if simple_renorm:
            in_im  = in_im * time_dimension
            noise_im  = noise_im * time_dimension
            rec_im  = rec_im * time_dimension

        else:
            in_im = custom_renormalisation(in_im, reconstruction_threshold, time_dimension)
            noise_im = custom_renormalisation(noise_im, reconstruction_threshold, time_dimension)
            rec_im = custom_renormalisation(rec_im, reconstruction_threshold, time_dimension)







    # 3D Reconstruction
    in_im = reconstruct_3D(in_im, reconstruction_threshold)
    noise_im = reconstruct_3D(noise_im, reconstruction_threshold)
    rec_im = reconstruct_3D(rec_im, reconstruction_threshold)
    
    #3D Plottting
    if rec_im.ndim != 1:                       # Checks if there are actually values in the reconstructed image, if not no image is aseved/plotted
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, subplot_kw={'projection': '3d'})
        #ax1 = plt.axes(projection='3d')
        ax1.scatter(in_im[:,0], in_im[:,1], in_im[:,2])
        ax1.set_zlim(0, time_dimension)
        ax2.scatter(noise_im[:,0], noise_im[:,1], noise_im[:,2])
        ax2.set_zlim(0, time_dimension)
        ax3.scatter(rec_im[:,0], rec_im[:,1], rec_im[:,2])
        ax3.set_zlim(0, time_dimension)
        fig.suptitle(f"3D Reconstruction - Epoch {epoch}")
        
        if (epoch) % print_every_other == 0:
            Out_Label = graphics_dir + f'{model_save_name} 3D Reconstruction - Epoch {epoch}.png'
            plot_save_choice(plot_or_save, Out_Label)
        else:
            plt.close()

    if (epoch) % print_every_other == 0:        
        #Telemetry plots
        if plot_cutoff_telemetry == 1:       #needs ttitles and labels etc added
            above_threshold, below_threshold = belief_telemetry(rec_im, reconstruction_threshold, epoch+1, settings, plot_or_save)   
            telemetry.append([epoch, above_threshold, below_threshold])


    return(number_of_true_signal_points, number_of_recovered_signal_points, in_im, noise_im, rec_im)    
    
#%% - Program begins
print("\n \nProgram Initalised - Welcome to DC3D Trainer\n")
#%% - Dataset Pre-tests
# Dataset Integrity Check    #???????? aslso perform on train data dir if ther is one?????? 
scantype = "quick"
if full_dataset_integrity_check:
    scantype = "full"

print(f"Testing training dataset integrity, with {scantype} scan")
dataset_integrity_check(train_dir, full_test=full_dataset_integrity_check, print_output=True)
print("Test completed\n")

if train_dir != test_dir:
    print("Testing test dataset signal distribution")
    dataset_integrity_check(test_dir, full_test=full_dataset_integrity_check, print_output=True)
    print("Test completed\n")

# Dataset Distribution Check
if full_dataset_distribution_check:
    print("\nTesting training dataset signal distribution")
    dataset_distribution_tester(train_dir, time_dimension, ignore_zero_vals_on_plot=True, output_image_dir=graphics_dir)
    print("Test completed\n")

    if train_dir != test_dir:
        print("Testing test dataset signal distribution")
        dataset_distribution_tester(test_dir, time_dimension, ignore_zero_vals_on_plot=True)
        print("Test completed\n")
    

#%% - Data Loader
"""
The DatasetFolder is a generic DATALOADER. It takes arguments:
root - Root directory path
loader - a function to load a sample given its path
others that arent so relevant....
"""

def train_loader2d(path):
    sample = (np.load(path))
    return (sample) #[0]

def test_loader2d(path):
    sample = (np.load(path))            
    return (sample) #[0]

def val_loader2d(path):
    sample = (np.load(path))            
    return (sample)


####check for file count in folder####
files_in_path = os.listdir(data_path + dataset_title + '/Data/') 
num_of_files_in_path = len(files_in_path)

# Report type of gradient descent
learning = batch_learning(num_of_files_in_path, batch_size)
print("%s files in path." %num_of_files_in_path ,"// Batch size =",batch_size, "\nLearning via: " + learning,"\n")

# - Path images, greater than batch choice? CHECK
if num_of_files_in_path < batch_size:
    print("Error, the path selected has", num_of_files_in_path, "image files, which is", (batch_size - num_of_files_in_path) , "less than the chosen batch size. Please select a batch size less than the total number of images in the directory")
    batch_err_message = "Choose new batch size, must be less than total amount of images in directory", (num_of_files_in_path)
    batch_size = int(input(batch_err_message))  #!!! not sure why input message is printing with wierd brakets and speech marks in the terminal? Investigate


#train_dir = r'C:\Users\maxsc\OneDrive - University of Bristol\3rd Year Physics\Project\Autoencoder\2D 3D simple version\Circular and Spherical Dummy Datasets\New big simp\Rectangle\\'
train_dataset = torchvision.datasets.DatasetFolder(train_dir, train_loader2d, extensions='.npy')

#test_dir = r'C:\Users\maxsc\OneDrive - University of Bristol\3rd Year Physics\Project\Autoencoder\2D 3D simple version\Circular and Spherical Dummy Datasets\New big simp test\Rectangle\\'
test_dataset = torchvision.datasets.DatasetFolder(test_dir, train_loader2d, extensions='.npy')


#%% - Data Preparation

if simple_norm_instead_of_custom:
    custom_normalisation_with_args = lambda x: x/time_dimension
else:
    custom_normalisation_with_args = partial(custom_normalisation, reconstruction_threshold=reconstruction_threshold, time_dimension=time_dimension)   #using functools partial to bundle the args into custom norm to use in custom torch transform using lambda function

if all_norm_off:
    custom_normalisation_with_args = lambda x: x


train_transform = transforms.Compose([                                         #train_transform variable holds the tensor tranformations to be performed on the training data.  transforms.Compose([ ,  , ]) allows multiple transforms to be chained together (in serial?) (#!!! does it do more than this??)
                                       #transforms.RandomRotation(30),         #transforms.RandomRotation(angle (degrees?) ) rotates the tensor randomly up to max value of angle argument
                                       #transforms.RandomResizedCrop(224),     #transforms.RandomResizedCrop(pixels) crops the data to 'pixels' in height and width (#!!! and (maybe) chooses a random centre point????)
                                       #transforms.RandomHorizontalFlip(),     #transforms.RandomHorizontalFlip() flips the image data horizontally 
                                       #transforms.Normalize((0.5), (0.5)),    #transforms.Normalize can be used to normalise the values in the array
                                       transforms.Lambda(custom_normalisation_with_args),
                                       transforms.ToTensor(),
                                       #transforms.RandomRotation(180)
                                       ])                 #other transforms can be dissabled but to tensor must be left enabled ! it creates a tensor from a numpy array #!!! ?

test_transform = transforms.Compose([                                          #test_transform variable holds the tensor tranformations to be performed on the evaluation data.  transforms.Compose([ ,  , ]) allows multiple transforms to be chained together (in serial?) (#!!! does it do more than this??)
                                      #transforms.Resize(255),                 #transforms.Resize(pixels? #!!!) ??
                                      #transforms.CenterCrop(224),             #transforms.CenterCrop(pixels? #!!!) ?? Crops the given image at the center. If the image is torch Tensor, it is expected to have […, H, W] shape, where … means an arbitrary number of leading dimensions. If image size is smaller than output size along any edge, image is padded with 0 and then center cropp
                                      #transforms.Normalize((0.5), (0.5)),     #transforms.Normalize can be used to normalise the values in the array
                                      transforms.Lambda(custom_normalisation_with_args),
                                      transforms.ToTensor(),
                                      #transforms.RandomRotation(180)
                                      ])                  #other transforms can be dissabled but to tensor must be left enabled ! it creates a tensor from a numpy array #!!! ?

# this applies above transforms to dataset (dataset transform = transform above)
train_dataset.transform = train_transform       #!!! train_dataset is the class? object 'dataset' it has a subclass called transforms which is the list of transofrms to perform on the dataset when loading it. train_tranforms is the set of chained transofrms we created, this is set to the dataset transforms subclass 
test_dataset.transform = test_transform         #!!! similar to the above but for the test(eval) dataset, check into this for the exact reason for using it, have seen it deone in other ways i.e as in the dataloader.py it is performed differntly. this way seems to be easier to follow
#####For info on all transforms check out: https://pytorch.org/vision/0.9/transforms.html

train_test_split_ratio = 0.8
###Following section splits the training dataset into two, train_data (to be noised) and valid data (to use in eval)
m=len(train_dataset) #Just calculates length of train dataset, m is only used in the next line to decide the values of the split, (4/5 m) and (1/5 m)
train_split=int(m*train_test_split_ratio)
train_data, test2_data = random_split(train_dataset, [train_split, m-train_split])    #random_split(data_to_split, [size of output1, size of output2]) just splits the train_dataset into two parts, 4/5 goes to train_data and 1/5 goes to val_data , validation?
half_slice = int(len(test2_data)/2)
test_data, val_data =  random_split(test2_data, [half_slice, len(test2_data) - half_slice])
###Following section for Dataloaders, they just pull a random sample of images from each of the datasets we now have, train_data, valid_data, and test_data. the batch size defines how many are taken from each set, shuffle argument shuffles them each time?? #!!!
                                                                        #User controll to set batch size for the dataloaders (Hyperparameter)?? #!!!

# required to load the data into the endoder/decoder. Combines a dataset and a sampler, and provides an iterable over the given dataset.
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)                 #Training data loader, can be run to pull training data as configured
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)   #Testing data loader, can be run to pull training data as configured. Also is shuffled using parameter shuffle #!!! why is it shuffled?
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)                   #Validation data loader, can be run to pull training data as configured


#%% - Setup model, loss criteria and optimiser    
### Define a learning rate for the optimiser. 
# Its how much to change the model in response to the estimated error each time the model weights are updated.
lr = learning_rate                                     #Just sets the learing rate value from the user inputs pannel at the top           

### Initialize the two networks
# use encoder and decoder classes, providing dimensions for your dataset. FC2_INPUT_DIM IS NOT USED!! This would be extremely useful.
encoder = Encoder(encoded_space_dim=latent_dim,fc2_input_dim=128, encoder_debug=print_encoder_debug, record_activity=record_activity)
decoder = Decoder(encoded_space_dim=latent_dim,fc2_input_dim=128, decoder_debug=print_decoder_debug, record_activity=record_activity)
encoder.double()   
decoder.double()
params_to_optimize = [{'params': encoder.parameters()} ,{'params': decoder.parameters()}] #Selects what to optimise, 


### Define an optimizer (both for the encoder and the decoder!)
optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=optim_w_decay)


#%% - Load in pretrained netwrok if needed 

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



#%% - Compute
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
                               encoder=encoder, 
                               decoder=decoder, 
                               device=device, 
                               dataloader=train_loader, 
                               loss_fn=loss_fn, 
                               optimizer=optim,
                               noise_factor=noise_factor)
    
    ### Validation (use the testing function)
    # does all validation batches. single average loss produced.
    val_loss = test_epoch_den(
                              encoder=encoder, 
                              decoder=decoder, 
                              device=device, 
                              dataloader=valid_loader, 
                              loss_fn=loss_fn,
                              noise_factor=noise_factor)
    
    # Print Validation_loss and plots at end of each epoch
    history_da['train_loss'].append(train_loss)
    history_da['val_loss'].append(val_loss)
    
    #Updates the epoch reached counter  
    max_epoch_reached = epoch    

    if print_partial_training_losses:
        print('\n End of EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))     #epoch +1 is to make up for the fact the range spans 0 to epoch-1 but we want to numerate things from 1 upwards for sanity
    
    if epoch % print_every_other == 0 and epoch != 0:
        
        print("\n## EPOCH {} PLOTS DRAWN ## \n  \n".format(epoch))
        
        # finally plot the figure with all images on it.
        encoder.eval()
        decoder.eval()
        number_of_true_signal_points, number_of_recovered_signal_points, in_data, noisy_data, rec_data = plot_ae_outputs_den(encoder, decoder, epoch, model_save_name, time_dimension, reconstruction_threshold, noise_factor=noise_factor)
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
        full_data_output["true_signal_points"] = number_of_true_signal_points
        full_data_output["recovered_signal_points"] = number_of_recovered_signal_points
    except:
        pass
    
    # Save .txt Encoder/Decoder Network Summary
    with open(full_netsum_filepath, 'w', encoding='utf-8') as output_file:    #utf_8 encoding needed as default (cp1252) unable to write special charecters present in the summary
        # Write the local date and time to the file
        TD_now = datetime.datetime.now()         # Get the current local date and time
        output_file.write(f"Date data taken: {TD_now.strftime('%Y-%m-%d %H:%M:%S')}\n")    

        output_file.write(("Model ID: " + model_save_name + f"\nTrained on device: {device}\n"))
        output_file.write((f"\nMax Epoch Reached: {max_epoch_reached}\n"))
        
        timer_warning = "(Not accurate - not recorded in speed_test mode)\n"
        if speed_test:
            timer_warning = "\n"
        output_file.write((f"Training Time: {training_time:.2f} seconds\n{timer_warning}\n"))
        
        output_file.write("Input Settings:\n")
        for key, value in settings.items():
            output_file.write(f"{key}: {value}\n")
        
        output_file.write("\nNormalisation:\n")
        output_file.write(f"simple_norm_instead_of_custom: {simple_norm_instead_of_custom}\n")
        output_file.write(f"all_norm_off: {all_norm_off}\n")

        output_file.write("\Pre Training:\n")
        if start_from_pretrained_model:
            output_file.write(f"full_statedict_path: {full_statedict_path}\n")    
        output_file.write(f"start_from_pretrained_model: {start_from_pretrained_model}\n")
        output_file.write(f"load_pretrained_optimser: {load_pretrained_optimser}\n")  
        
        output_file.write("\nAutoencoder Network:\n")
        output_file.write((f"AE File ID: {AE_file_name}\n"))
        output_file.write("\n" + summary_str)
        
        output_file.write("\n \nFull Data Readouts:\n")
        for key, value in full_data_output.items():
            output_file.write(f"{key}: {value}\n")

        output_file.write("\nPython Lib Versions:\n")
        output_file.write((f"PyTorch: {torch.__version__}\n"))
        output_file.write((f"Torchvision: {torchvision.__version__}\n"))
        output_file.write((f"Numpy: {np.__version__}\n"))
        output_file.write((f"Matplotlib: {plt.__version__}\n"))

        system_information = get_system_information()
        output_file.write("\n" + system_information)
    print("- Completed -")
#%% - End of Program - Printing message to notify user!
print("\nProgram Complete - Shutting down...\n")    
    
    