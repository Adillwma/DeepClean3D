# -*- coding: utf-8 -*-
"""
Created on Sat Feb 1 2022
DeepClean v0.3.1

@authors: Adill Al-Ashgar & Max Carter
# University of Bristol

Possible improvements:
### ~~~~~ Possibly set all to double?
dtype (torch.dtype, optional) – the desired data type of returned tensor. 
Default: if None, uses a global default (see torch.set_default_tensor_type()).!!! 

### ~~~~~ Update the model and data paths to folders inside the root dir so that they do not need be defined, and so that people can doanload the git repo and just press run without setting new paths etc 
### ~~~~~ fix epoch numbering printouts? they seem to report 1 epoch greater than they should

### ~~~~~ clear up visulisations

### ~~~~~ Functionalise things

### ~~~~~ Move things to other files (AE, Helper funcs, Visulisations etc)

### ~~~~~ Fix reconstruction threshold, use recon threshold to set bottom limit in custom normalisation

### ~~~~~ Turn plot or save into a function 

### ~~~~~ change telemetry variable name to output_pixel_telemetry

### ~~~~~ Fix this " if plot_higher_dim: AE_visulisation(en...)" break out all seperate plotting functions
    
### ~~~~~ adapt new version for masking - DeepMask3D 

### ~~~~~ sort out val, test and train properly

### ~~~~~ FC2_INPUT_DIM IS NOT USED!! This would be extremely useful. ?? is this for dynamic layer sizing?
"""
#%% - Dependencies
import numpy as np 
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import random 
import time
from torchinfo import summary
#import torch.nn.functional as F
#import torch.optim as optim
#import os
#from Calc import conv_calculator
from tqdm import tqdm
from torchviz import make_dot
import pandas as pd
import plotly.express as px
import json


#%% - User Inputs
mode = 0 ### 0=Data_Gathering, 1=Testing, 2=Speed_Test, 3=Debugging

num_epochs = 51                                              #User controll to set number of epochs (Hyperparameter)
batch_size = 1                                  #User controll to set batch size (Hyperparameter) - #Data Loader, number of Images to pull per batch 
latent_dim = 10                      #User controll to set number of nodes in the latent space, the bottleneck layer (Hyperparameter)

learning_rate = 0.001  #User controll to set optimiser learning rate(Hyperparameter)
optim_w_decay = 1e-05  #User controll to set optimiser weight decay (Hyperparameter)
loss_fn = torch.nn.MSELoss()          #(mean square error) User controll to set loss function (Hyperparameter)

time_dimension = 100
noise_factor = 0                                          #User controll to set the noise factor, a multiplier for the magnitude of noise added. 0 means no noise added, 1 is defualt level of noise added, 10 is 10x default level added (Hyperparameter)
reconstruction_threshold = 0.5      #MUST BE BETWEEN 0-1        #Threshold for 3d reconstruction, values below this confidence level are discounted
###FIX RECON!!!

#%% - Advanced Debugging Settings
print_encoder_debug = False                     #[default = False]
print_decoder_debug = False                     #[default = False]
debug_noise_function = False                    #[default = False]
debug_loader_batch = False     #(Default = False) //INPUT 0 or 1//   #Setting debug loader batch will print to user the images taken in by the dataoader in this current batch and print the corresponding labels

print_network_summary = False     #deault = False
print_partial_training_losses = False            #[default = True]
telemetry_on = False                            #[default = False]
seed = 0              #0 is default which gives no seeeding to RNG, if the value is not zero then this is used for the RNG seeding for numpy, random, and torch libraries

#%% - Plotting Control Settings
print_every_other = 10
plot_or_save = 0                            #[default = 0] 0 is normal behavior, If set to 1 then saves all end of epoch printouts to disk, if set to 2 then saves outputs whilst also printing for user

dataset_title = "Dataset 12_X10K"
outputfig_title = "X10K V1"                    #Must be string, value is used in the titling of the output plots if plot_or_save is selected above
model_save_name = "X10K V1"

#%% - Advanced Visulisation Settings

plot_pixel_difference = True #False
plot_im_stats = False

plot_latent_information = False
plot_higher_dim = False
plot_TSNE_dim = False

plot_train_loss = False
plot_validation_loss = False


#%% - Program Settings
speed_test = False      # [speed_test=False]Defualt    true sets number of epocs to print to larger than number of epochs to run so no plotting time wasted etc
data_gathering = True

#%% - Data Path Settings
data_path = "C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/Yr 3 Project/Circular and Spherical Dummy Datasets/"
#ADILL - "C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/Yr 3 Project/Circular and Spherical Dummy Datasets/"
#MAX - 

model_save_path = "C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/Git Hub Repos/DeepClean Repo/DeepClean-Noise-Suppression-for-LHC-B-Torch-Detector/Models/"
#ADILL - "C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/Git Hub Repos/DeepClean Repo/DeepClean-Noise-Suppression-for-LHC-B-Torch-Detector/Models/"
#MAX - 

#%% - Helper functions
def custom_normalisation(input, time_dimension=100):
    input = (input / (2 * time_dimension)) + reconstruction_threshold
    for row in input:
        for i, ipt in enumerate(row):
            if ipt == reconstruction_threshold:
                row[i] = 0
    return input

###Ploting confidence of each pixel as histogram per epoch with line showing the detection threshold
def belief_telemetry(data, reconstruction_threshold, epoch, settings, plot_or_save=0):
    data2 = data.flatten()

    #Plots histogram showing the confidence level of each pixel being a signal point
    values, bins, bars = plt.hist(data2, 10, histtype='bar')
    plt.axvline(x= reconstruction_threshold, color='red', marker='|', linestyle='dashed', linewidth=2, markersize=12)
    plt.title("Epoch %s" %epoch)
    plt.bar_label(bars, fontsize=10, color='navy') 
    if plot_or_save == 0:
        plt.show()
    else:
        Out_Label = 'Output_Graphics/{}, Confidence Histogram, Epoch {}, {} .png'.format(outputfig_title, epoch, settings) #!!!
        plt.savefig(Out_Label, format='png')        
        plt.close()

    above_threshold = (data2 >= reconstruction_threshold).sum()
    below_threshold = (data2 < reconstruction_threshold).sum()
    return (above_threshold, below_threshold)

def plot_telemetry(telemetry):
    tele = np.array(telemetry)
    #!!! Add labels to lines
    plt.plot(tele[:,0],tele[:,1], color='r', label="Points above threshold") #red = num of points above threshold
    plt.plot(tele[:,0],tele[:,2], color='b', label="Points below threshold") #blue = num of points below threshold
    plt.title("Telemetry over epochs")
    plt.xlabel("Epoch number")
    plt.ylabel("Number of Signal Points")
    plt.legend()
    plt.show()    

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

#%% - Parameters Initialisation
# Joins up the parts of the model save path
modal_save = model_save_path + model_save_name + ".pth"

# Sets program into speed test mode
if speed_test:
    print_every_other = num_epochs + 5

# Initialises pixel belief telemetry
telemetry = [[0,0.5,0.5]]  #Initalises the telemetry memory, starting values are 0, 0.5, 0.5 which corrspond to epoch(0), above_threshold(0.5), below_threshold(0.5)

# Initialises list for noised image storage
image_noisy_list = []

# Initialises seeding values to RNGs
if seed != 0: 
    Determinism_Seeding(seed)

# Create record of all user input settings, to add to output data for testing and keeping track of settings
settings = {} #"Settings = [ep {}][bs {}][lr {}][od {}][ls {}][nf {}][ds {}][sd {}]".format(num_epochs, batch_size, learning_rate, optim_w_decay, latent_dim, noise_factor, dataset_title, seed)
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

#%% - Convoloution + Linear Autoencoder
###Encoder
class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim, encoder_debug):
        super().__init__()
        
        self.encoder_debug=encoder_debug

        ###Convolutional Encoder Layers
        
        # Arguments: (input channels, output channels, kernel size/receiptive field (), stride and padding. # here for more info on arguments https://uob-my.sharepoint.com/personal/ex18871_bristol_ac_uk/_layouts/15/Doc.aspx?sourcedoc={109841d1-79cb-45c2-ac39-0fd24300c883}&action=edit&wd=target%28Code%20Companion.one%7C%2FUntitled%20Page%7C55b8dfad-d53c-4d8e-a2ef-de1376232896%2F%29&wdorigin=NavigationUrl
        
        # Input channels is 1 as the image is black and white so only has luminace values no rgb channels, 
        # Output channels which is the amount of output tensors. Defines number of seperate kernels that will be run across image, all which produce thier own output arrays
        # The kernal size, is the size of the convolving layer. Ours is 3 means 3x3 kernel matrix.
        # Stride is how far the kernal moves across each time, the default is across by one pixel at a time.
        # Padding adds padding (zeros) to the edges of the data array before convoloution filtering. This is to not neglect edge pixels.
        # Dilation spreads out kernel
        
        self.encoder_cnn = nn.Sequential(
            # N.B. input channel dimensions are not the same as output channel dimensions:
            # the images will get smaller into the encoded layer
            #Convolutional encoder layer 1                 
            nn.Conv2d(1, 8, 3, stride=2, padding=1),       #Input_channels, Output_channels, Kernal_size, Stride, Padding
            nn.ReLU(True),                                 #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
            #Convolutional encoder layer 2
            nn.Conv2d(8, 16, 3, stride=2, padding=1),      #Input_channels, Output_channels, Kernal_size, Stride, Padding
            nn.BatchNorm2d(16),                            #BatchNorm normalises the outputs as a batch? #!!!. argument is 'num_features' (expected input of size)
            nn.ReLU(True),                                 #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
            #Convolutional encoder layer 3
            nn.Conv2d(16, 32, 3, stride=2, padding=0),     #Input_channels, Output_channels, Kernal_size, Stride, Padding
            nn.ReLU(True)                                  #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
        )
 
        ###Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
    
        ###Linear Encoder Layers
        self.encoder_lin = nn.Sequential(
            #Linear encoder layer 1  
            nn.Linear(4800, 128),                   #!!! linear network layer. arguuments are input dimensions/size, output dimensions/size. Takes in data of dimensions 3* 3 *32 and outputs it in 1 dimension of size 128
            # nn.Linear(L3[0] * L3[1] * 32, 128),
            nn.ReLU(True),                                #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
            #Linear encoder layer 2
            nn.Linear(128, encoded_space_dim)             #Takes in data of 1 dimension with size 128 and outputs it in one dimension of size defined by encoded_space_dim (this is the latent space? the smalle rit is the more comression but thte worse the final fidelity)
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
            self.encoder_debug = 0         
        return x                                          #Return final result

###Decoder
class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim, decoder_debug):
        super().__init__()
        
        self.decoder_debug = decoder_debug

        ###Linear Decoder Layers
        self.decoder_lin = nn.Sequential(
            #Linear decoder layer 1            
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),                                 #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
            #Linear decoder layer 2
            nn.Linear(128, 4800),
            nn.ReLU(True)                                  #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
        )
        ###Unflatten layer
        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 15, 10))
        
        self.decoder_conv = nn.Sequential(
            #Convolutional decoder layer 1
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=1),             #Input_channels, Output_channels, Kernal_size, Stride, padding(unused), Output_padding
            nn.BatchNorm2d(16),                                                    #BatchNorm normalises the outputs as a batch? #!!!. argument is 'num_features' (expected input of size)
            nn.ReLU(True),                                                         #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
            #Convolutional decoder layer 2
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),   #Input_channels, Output_channels, Kernal_size, Stride, padding, Output_padding
            nn.BatchNorm2d(8),                                                     #BatchNorm normalises the outputs as a batch? #!!!. argument is 'num_features' (expected input of size)
            nn.ReLU(True),                                                         #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
            #Convolutional decoder layer 3
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)     #Input_channels, Output_channels, Kernal_size, Stride, padding, Output_padding
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
            self.decoder_debug = 0         
        x = torch.sigmoid(x)          #THIS IS IMPORTANT PART OF FINAL OUTPUT!: Runs sigmoid function which turns the output data values to range (0-1)#!!! ????    Also can use tanh fucntion if wanting outputs from -1 to 1
        return x                      #Retuns the final output
    
    
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
        image_noisy = add_noise(image_batch, noise_factor, debug_noise_function)
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
            image_noisy = add_noise(image_batch, noise_factor, debug_noise_function)
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
def plot_ae_outputs_den(encoder, decoder, epoch, outputfig_title, time_dimension, reconstruction_threshold, n=10, noise_factor=0.3):       #Defines a function for plotting the output of the autoencoder. And also the input + clean training data? Function takes inputs, 'encoder' and 'decoder' which are expected to be classes (defining the encode and decode nets), 'n' which is the number of ?????Images in the batch????, and 'noise_factor' which is a multiplier for the magnitude of the added noise allowing it to be scaled in intensity.  
    """
    n is the number of images to plot i think?
    """

    #Initialise lists for true and recovered signal point values 
    number_of_true_signal_points = []
    number_of_recovered_signal_points = []


    plt.figure(figsize=(16,4.5))                                      #Sets the figure size
    # numpy array of correct unnoised data
    # targets = test_dataset.targets.numpy()                            #Creates a numpy array (from the .numpy part) the array is created from the values in the specified tensor, which in this case is test_dataset.targets (test_dataset is the dataloader, .targets is a subclass of the dataloader that holds the labels, i.e the correct answer data (in this case the unnoised images).)                          
    # defines dictionary keys 0-(n-1), values are indices in the targets array where those integers can be found 
    # t_idx = {i:np.where(targets==i)[0][0] for i in range(n)}          #!!! ????
    
    for i in range(n):                                                #Runs for loop where 'i' itterates over 'n' total values which range from 0 to n-1
        
      #Following section creates the noised image data drom the original clean labels (images)   
      ax = plt.subplot(3,n,i+1)                                       #Creates a number of subplots for the 'Original images??????' i.e the labels. the position of the subplot is i+1 as it falls in the first row
      img = test_dataset[i][0].unsqueeze(0) # [t_idx[i]][0].unsqueeze(0)                    #!!! ????
      
      
      #Determine the number of signal points on the input image (have to change this to take it directly from the embeded val in the datsaset as when addig noise this method will break)   
      int_sig_points = (img >= reconstruction_threshold).sum()
      number_of_true_signal_points.append(int(int_sig_points.numpy()))
      
      
      
      if epoch <= print_every_other:                                                  #CHECKS TO SEE IF THE EPOCH IS LESS THAN ZERO , I ADDED THIS TO GET THE SAME NOISED IMAGES EACH EPOCH THOUGH THIS COULD BE WRONG TO DO?
          global image_noisy                                          #'global' means the variable (image_noisy) set inside a function is globally defined, i.e defined also outside the function
          image_noisy = add_noise(img, noise_factor, debug_noise_function)                   #Runs the function 'add_noise' (in this code) the function adds noise to a set of data, the function takes two arguments, img is the data to add noise to, noise factor is a multiplier for the noise values added, i.e if multiplier is 0 no noise is added, if it is 1 default amount is added, if it is 10 then the values are raised 10x 
          image_noisy_list.append(image_noisy)                        #Adds the just generated noise image to the list of all the noisy images
      image_noisy = image_noisy_list[i].to(device)                    #moves the list (i think of tensors?) to the device that will process it i.e either cpu or gpu, we have a check elsewhere in the code that detects if gpu is availible and sets the value of 'device' to gpu or cpu depending on availibility (look for the line that says "device = 'cuda' if torch.cuda.is_available() else 'cpu'"). NOTE: this moves the noised images to device, i think that the original images are already moved to device in previous code

    
      #Following section sets the autoencoder to evaluation mode rather than training (up till line 'with torch.no_grad()')
      encoder.eval()                                   #.eval() is a kind of switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time. For example, Dropouts Layers, BatchNorm Layers etc. You need to turn off them during model evaluation, and .eval() will do it for you. In addition, the common practice for evaluating/validation is using torch.no_grad() in pair with model.eval() to turn off gradients computation
      decoder.eval()                                   #Simarlary as above

      with torch.no_grad():                                               #As mentioned in .eval() comment, the common practice for evaluating/validation is using torch.no_grad() which turns off gradients computation whilst evaluating the model (the opposite of training the model)     
      #Following line runs the autoencoder on the noised data
         rec_img  = decoder(encoder(image_noisy))                         #Creates a recovered image (denoised image), by running a noisy image through the encoder and then the output of that through the decoder.

      #Determine the number of signal points on the recovered image 
      int_rec_sig_points = (rec_img >= reconstruction_threshold).sum()      
      number_of_recovered_signal_points.append(int(int_rec_sig_points.numpy()))


      #Following section generates the img plots for the original(labels), noised, and denoised data)
      plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')           #plt.imshow plots an image. The arguments for imshow are, 'image data array' and cmap= which is the colour map. #.squeeze() acts on a tensor and returns a tensor, it removes all dimensions of the tensor that are of length 1, (A×1×B) becomes (AxB) where A and B are greater than 1 #.numpy() creates a numpy array from a tensor #!!! is the .cpu part becuase the code was not made to accept the gpu/cpu check i made????
      ax.get_xaxis().set_visible(False)                                   #Hides the x axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
      ax.get_yaxis().set_visible(False)                                   #Hides the y axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
      if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
        ax.set_title('Original images')                                   #When above condition is reached, the plots title is set

      ax = plt.subplot(3, n, i + 1 + n)                                   #Creates a number of subplots for the 'Corrupted images??????' i.e the labels. the position of the subplot is i+1+n as it falls in the second row
      plt.imshow(image_noisy.cpu().squeeze().numpy(), cmap='gist_gray')   #plt.imshow plots an image. The arguments for imshow are, 'image data array' and cmap= which is the colour map. #.squeeze() acts on a tensor and returns a tensor, it removes all dimensions of the tensor that are of length 1, (A×1×B) becomes (AxB) where A and B are greater than 1 #.numpy() creates a numpy array from a tensor #!!! is the .cpu part becuase the code was not made to accept the gpu/cpu check i made????
      ax.get_xaxis().set_visible(False)                                   #Hides the x axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
      ax.get_yaxis().set_visible(False)                                   #Hides the y axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
      if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
        ax.set_title('Corrupted images')                                  #When above condition is reached, the plots title is set

      ax = plt.subplot(3, n, i + 1 + n + n)                               #Creates a number of subplots for the 'Reconstructed images??????' i.e the labels. the position of the subplot is i+1+n+n as it falls in the third row
      plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')       #plt.imshow plots an image. The arguments for imshow are, 'image data array' and cmap= which is the colour map. #.squeeze() acts on a tensor and returns a tensor, it removes all dimensions of the tensor that are of length 1, (A×1×B) becomes (AxB) where A and B are greater than 1 #.numpy() creates a numpy array from a tensor #!!! is the .cpu part becuase the code was not made to accept the gpu/cpu check i made????
      ax.get_xaxis().set_visible(False)                                   #Hides the x axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
      ax.get_yaxis().set_visible(False)                                   #Hides the y axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
      if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
         ax.set_title('Reconstructed images')                             #When above condition is reached, the plots title is set 
    
    plt.subplots_adjust(left=0.1,              #Adjusts the exact layout of the plots including whwite space round edges
                    bottom=0.1, 
                    right=0.7, 
                    top=0.9, 
                    wspace=0.3, 
                    hspace=0.3)     
    plt.show()                                 #After entire loop is finished, the generated plot is printed to screen
    
    # 3D Reconstruction
    data = rec_img.cpu().squeeze().numpy()
    #print(np.shape(data))
    def rev_norm(data, time_dimension):
        data_output = []
        for cdx, row in enumerate(data):
            for idx, num in enumerate(row):
                if num > 0.5:
                    num -= 0.5
                    num = num * (time_dimension-1*2)
                    data_output.append([cdx,idx,num])
        return np.array(data_output)
    
    rec_data = rev_norm(data, time_dimension)
    if rec_data.ndim != 1:
        # print(np.shape(rec_data))
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(rec_data[:,0], rec_data[:,1], rec_data[:,2])
        ax.set_zlim(0,28)
        plt.show()

    
    ### - PLOT SAVING, CLEAN UP!!! TURN INTO A FUCNTION FOR USE ON ALL PLOTS?????
    if plot_or_save == 0:
        if (epoch+1) % print_every_other == 0:
            plt.show()                                 #After entire loop is finished, the generated plot is printed to screen
        else:
            plt.close()


    elif plot_or_save == 1:
        Out_Label = 'Output_Graphics/{}, Epoch {}, {} .png'.format(outputfig_title, epoch+1, settings) #!!!
        plt.savefig(Out_Label, format='png')
        plt.close()
        print("\n# SAVED OUTPUT TEST IMAGE TO DISK #\n")    

    if (epoch+1) % print_every_other == 0:        
        ###3D Reconstruction
        in_data = img.cpu().squeeze().numpy()
        noisy_data = image_noisy.cpu().squeeze().numpy()
        rec_data = rec_img.cpu().squeeze().numpy()
        
        #Telemetry plots
        if telemetry_on == 1:       #needs ttitles and labels etc added
            above_threshold, below_threshold = belief_telemetry(rec_data, reconstruction_threshold, epoch+1, settings, plot_or_save)   
            telemetry.append([epoch, above_threshold, below_threshold])


    return(number_of_true_signal_points, number_of_recovered_signal_points)    
    



    

#%% - Data Importer
data_dir = 'dataset'

# # mnist data is 28x28, and black and white (so 1x28x28)
# train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
# # train= true
# # train argument selects folder. download argument decides whether to download from internet.
# test_dataset  = torchvision.datasets.MNIST(data_dir, train=False, download=True)
# print(type(test_dataset)) # a dataset
# print(type(test_dataset[0][0])) # PIL image

# in this, were now going to try to work the data generator for a super simple 28x28 cross. This will be
# generated in the 'supersimp' then added here through the data_directory function:

"""
The DatasetFolder is a generic DATALOADER. It takes arguments:
root - Root directory path
loader - a function to load a sample given its path
others that arent so relevant....
"""
# root to files
# data_directory = r'C:\Users\maxsc\OneDrive - University of Bristol\3rd Year Physics\Project\Autoencoder\2D 3D simple version\Circular and Spherical Dummy Datasets\Simple Cross\\'

def train_loader2d(path):
    sample = (np.load(path))
    return (sample) #[0]

def test_loader2d(path):
    sample = (np.load(path))            
    return (sample) #[0]

def val_loader2d(path):
    sample = (np.load(path))            
    return (sample)

# the train_epoch_den and test both add noise themselves?? so i will have to call all of the clean versions:
train_dir = data_path + dataset_title

#train_dir = r'C:\Users\maxsc\OneDrive - University of Bristol\3rd Year Physics\Project\Autoencoder\2D 3D simple version\Circular and Spherical Dummy Datasets\New big simp\Rectangle\\'
train_dataset = torchvision.datasets.DatasetFolder(train_dir, train_loader2d, extensions='.npy')

# N.B. We will use the train loader for this as it takes the clean data, and thats what we want as theres a built in nois adder here already:
test_dir = data_path + dataset_title

#test_dir = r'C:\Users\maxsc\OneDrive - University of Bristol\3rd Year Physics\Project\Autoencoder\2D 3D simple version\Circular and Spherical Dummy Datasets\New big simp test\Rectangle\\'
test_dataset = torchvision.datasets.DatasetFolder(test_dir, train_loader2d, extensions='.npy')


#%% - Data Preparation

train_transform = transforms.Compose([                                         #train_transform variable holds the tensor tranformations to be performed on the training data.  transforms.Compose([ ,  , ]) allows multiple transforms to be chained together (in serial?) (#!!! does it do more than this??)
                                       #transforms.RandomRotation(30),         #transforms.RandomRotation(angle (degrees?) ) rotates the tensor randomly up to max value of angle argument
                                       #transforms.RandomResizedCrop(224),     #transforms.RandomResizedCrop(pixels) crops the data to 'pixels' in height and width (#!!! and (maybe) chooses a random centre point????)
                                       #transforms.RandomHorizontalFlip(),     #transforms.RandomHorizontalFlip() flips the image data horizontally 
                                       #transforms.Normalize((0.5), (0.5)),    #transforms.Normalize can be used to normalise the values in the array
                                       transforms.Lambda(custom_normalisation),
                                       transforms.ToTensor()])                 #other transforms can be dissabled but to tensor must be left enabled ! it creates a tensor from a numpy array #!!! ?

test_transform = transforms.Compose([                                          #test_transform variable holds the tensor tranformations to be performed on the evaluation data.  transforms.Compose([ ,  , ]) allows multiple transforms to be chained together (in serial?) (#!!! does it do more than this??)
                                      #transforms.Resize(255),                 #transforms.Resize(pixels? #!!!) ??
                                      #transforms.CenterCrop(224),             #transforms.CenterCrop(pixels? #!!!) ?? Crops the given image at the center. If the image is torch Tensor, it is expected to have […, H, W] shape, where … means an arbitrary number of leading dimensions. If image size is smaller than output size along any edge, image is padded with 0 and then center cropp
                                      #transforms.Normalize((0.5), (0.5)),     #transforms.Normalize can be used to normalise the values in the array
                                      transforms.Lambda(custom_normalisation),
                                      transforms.ToTensor()])                  #other transforms can be dissabled but to tensor must be left enabled ! it creates a tensor from a numpy array #!!! ?

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
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)                 #Training data loader, can be run to pull training data as configured
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)                   #Validation data loader, can be run to pull training data as configured
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)   #Testing data loader, can be run to pull training data as configured. Also is shuffled using parameter shuffle #!!! why is it shuffled?


#%% - Setup model, loss criteria and optimiser    
### Define a learning rate for the optimiser. 
# Its how much to change the model in response to the estimated error each time the model weights are updated.
lr = learning_rate                                     #Just sets the learing rate value from the user inputs pannel at the top           

### Initialize the two networks

# use encoder and decoder classes, providing dimensions for your dataset. FC2_INPUT_DIM IS NOT USED!! This would be extremely useful.
encoder = Encoder(encoded_space_dim=latent_dim,fc2_input_dim=128, encoder_debug=print_encoder_debug)
decoder = Decoder(encoded_space_dim=latent_dim,fc2_input_dim=128, decoder_debug=print_decoder_debug)
encoder.double()   
decoder.double()
params_to_optimize = [{'params': encoder.parameters()} ,{'params': decoder.parameters()}] #Selects what to optimise, 


### Define an optimizer (both for the encoder and the decoder!)
optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=optim_w_decay)

#%% - Initalise Model on compute device
# Following section checks if a CUDA enabled GPU is available. If found it is selected as the 'device' to perform the tensor opperations. If no CUDA GPU is found the 'device' is set to CPU (much slower) 
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}\n')  #Informs user if running on CPU or GPU

# Following section moves both the encoder and the decoder to the selected device i.e detected CUDA enabled GPU or to CPU
encoder.to(device)   #Moves encoder to selected device, CPU/GPU
decoder.to(device)   #Moves decoder to selected device, CPU/GPU


#%% - Prepare Network Summary
# Create dummy input tensor
enc_input_size = (batch_size, 1, 128, 88)
enc_input_tensor = torch.randn(enc_input_size).double()  # Cast input tensor to double precision

# Join the encoder and decoder models
full_network_model = torch.nn.Sequential(encoder, decoder)

# Generate network summary and then convert to string
model_stats = summary(full_network_model, input_data=enc_input_tensor, device=device, verbose=0)
summary_str = str(model_stats)             

# Print Encoder/Decoder Network Summary
if print_network_summary:
    print(summary_str)

#%% - Compute
# this is a dictionary ledger of train val loss history
history_da={'train_loss':[],'val_loss':[]}                   #Just creates a variable called history_da which contains two lists, 'train_loss' and 'val_loss' which are both empty to start with. value are latter appeneded to the two lists by way of history_da['val_loss'].append(x)

# Begin the training timer
start_time = time.time()

#for j in tqdm(range(5), desc='Inner loop', leave=False
if print_partial_training_losses:  # Prints partial train losses per batch
    loop_range = range(num_epochs)
else:                              # No print partial train losses per batch, instead create progress bar
    loop_range = tqdm(range(num_epochs), desc='Epochs')


# bringing everything together to train model
print("\nTraining Initiated")
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
    if print_partial_training_losses:
        print('\n End of EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))     #epoch +1 is to make up for the fact the range spans 0 to epoch-1 but we want to numerate things from 1 upwards for sanity
    
    if epoch % print_every_other == 0 and epoch != 0:
        
        print("\n \n## EPOCH {} PLOTS DRAWN ##".format(epoch))
        
        # finally plot the figure with all images on it.
        number_of_true_signal_points, number_of_recovered_signal_points = plot_ae_outputs_den(encoder, decoder, epoch, outputfig_title, time_dimension, reconstruction_threshold, noise_factor=noise_factor)
        
        # Allow user to exit training loop    
        max_epoch_reached = epoch    #Updates the epoch reached counter
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

#%% - Old Visulisations
###Loss function plots
epochs_range = range(1,max_epoch_reached+2)
plt.plot(epochs_range, history_da['train_loss']) 
plt.title("Training loss")   
plt.xlabel("Epoch number")
plt.ylabel("Train loss (MSE)")
plt.show()

plt.plot(epochs_range, history_da['train_loss'])   #ERROR SHOULD BE VAL LOSS!
plt.title("Validation loss") 
plt.xlabel("Epoch number")
plt.ylabel("Val loss (MSE)")
plt.show()

if telemetry_on:
    plot_telemetry(telemetry)

#Comparison of true signal points to recovered signal points
print("True signal points",number_of_true_signal_points)
print("Recovered signal points: ",number_of_recovered_signal_points, "\n")
full_data_output["true_signal_points"] = number_of_true_signal_points
full_data_output["recovered_signal_points"] = number_of_recovered_signal_points

#%% - New Visulisations
image = test_dataset[0][0].unsqueeze(0)                      
noised_image = add_noise(image, noise_factor, debug_noise_function) 
cleaned_image = image

from AE_Visulisations import AE_visulisation
from AE_Visulisations import AE_visual_difference

if plot_pixel_difference:
    num_diff_noised, num_same_noised, num_diff_cleaned, num_same_cleaned = AE_visual_difference(image, noised_image, cleaned_image)
    full_data_output["num_diff_noised"] = num_diff_noised
    full_data_output["num_same_noised"] = num_same_noised
    full_data_output["num_diff_cleaned"] = num_diff_cleaned
    full_data_output["num_same_cleaned"] = num_same_cleaned

if plot_higher_dim:
    AE_visulisation(encoder, decoder, latent_dim, device, test_loader, test_dataset, batch_size)
    
if data_gathering:
    # Save and export trained model to user  
    torch.save((encoder, decoder), modal_save)

    # Save .txt Encoder/Decoder Network Summary
    with open(model_save_path + model_save_name + ' - Network Summary.txt', 'w', encoding='utf-8') as output_file:    #utf_8 encoding needed as default (cp1252) unable to write special charecters present in the summary
        output_file.write(("Model ID: " + model_save_name + f"\nTrained on device: {device}"))
        output_file.write((f"\nMax Epoch Reached: {max_epoch_reached}\n"))
        timer_warning = "(Not accurate - not recorded in speed_test mode)\n"
        if speed_test:
            timer_warning = "\n"
        output_file.write((f"Training Time: {training_time:.2f} seconds\n{timer_warning}\n"))
        output_file.write("Input Settings:\n")
        for key, value in settings.items():
            output_file.write(f"{key}: {value}\n")
        output_file.write("\n" + summary_str)
        output_file.write("\n \nFull Data Readouts:\n")
        #output_file.write(str(full_data_output))
        for key, value in full_data_output.items():
            output_file.write(f"{key}: {value}\n")
            
print("\nProgram Complete - Shutting down...")    
    
    