# -*- coding: utf-8 -*-
"""
Created on Sat Oct 8 2022
DeepClean 3D v0.0.2
@author: Adill Al-Ashgar
"""
#%% - Dependencies
import numpy as np 
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader,random_split
from torch import nn
import random 
#import pandas as pd 
#import torch.nn.functional as F
#import torch.optim as optim
import os

from DataLoader_Functions_V2 import initialise_data_loader
from autoencoders.autoencoder_3D_V2 import Encoder, Decoder

#%% - User Inputs (Hyperparameters)
learning_rate = 0.001  #User controll to set optimiser learning rate(Hyperparameter)
optim_w_decay = 1e-05  #User controll to set optimiser weight decay (Hyperparameter)
latent_space_nodes = 4
noise_factor = 0                                           #User controll to set the noise factor, a multiplier for the magnitude of noise added. 0 means no noise added, 1 is defualt level of noise added, 10 is 10x default level added (Hyperparameter)
num_epochs = 40   #40                                            #User controll to set number of epochs (Hyperparameter)

#%% - Program Settings
seed = 0              #0 is default which gives no seeeding to RNG, if the value is not zero then this is used for the RNG seeding for numpy, random, and torch libraries
encoder_debug = 0
decoder_debug = 0
reconstruction_threshold = 0.5
telemetry_on = 1
simple_belief_reports = 1


#%% Dataloading
# - Data Loader User Inputs
batch_size = 1            #Data Loader # of Images to pull per batch (add a check to make sure the batch size is smaller than the total number of images in the path selected)
dataset_title = "S_Dataset 8"
data_path = "C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/Yr 3 Project/Circular and Spherical Dummy Datasets/" #"C:/Users/Student/Desktop/fake im data/"  #"/local/path/to/the/images/"

# - Advanced Data Loader Settings
debug_loader_batch = 0    #(Default = 0 = [OFF]) //INPUT 0 or 1//   #Setting debug loader batch will print to user the images taken in by the dataoader in this current batch and print the corresponding labels
plot_every_other = 1       #(Default = 1) //MUST BE INTEGER INPUT//  #If debug loader batch is enabled this sets the interval for printing for user, 1 is every single img in the batch, 2 is every other img, 5 is every 5th image etc 
batch_size_protection = 1  #(Default = 1 = [ON]) //INPUT 0 or 1//    #WARNING if turned off, debugging print will cause and exeption due to the index growing too large in the printing loop (img = train_features[i])

# - Data Loader Preparation Transforms 
#####For info on all transforms check out: https://pytorch.org/vision/0.9/transforms.html
train_transforms = transforms.Compose([#transforms.RandomRotation(30),         #Compose is required to chain together multiple transforms in serial 
                                       #transforms.RandomResizedCrop(224),
                                       #transforms.RandomHorizontalFlip(),
                                       #transforms.ToTensor()               #other transforms can be dissabled but to tensor must be left enabled
                                       ]) 
test_transforms = transforms.Compose([#transforms.Resize(255),
                                      #transforms.CenterCrop(224),
                                      #transforms.ToTensor()
                                      ])

# - Initialise Data Loader
train_loader, test_loader, val_loader, train_dataset, test_dataset, val_dataset = initialise_data_loader(dataset_title, data_path, batch_size, train_transforms, test_transforms, debug_loader_batch, plot_every_other, batch_size_protection)


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

#%% - Functions

###Ploting beleif of each pixel as histogram per epoch with line showing the detection threshold
def belief_telemetry(data, reconstruction_threshold, epoch):
    data2 = data.flatten()

    #Plots histogram showing the confidence level of each pixel being a signal point
    values, bins, bars = plt.hist(data2, 10, histtype='bar')
    plt.axvline(x= reconstruction_threshold, color='red', marker='|', linestyle='dashed', linewidth=2, markersize=12)
    plt.title("Epoch %s" %epoch)
    plt.bar_label(bars, fontsize=10, color='navy')
    plt.show()    
    above_threshold = (data2 >= reconstruction_threshold).sum()
    below_threshold = (data2 < reconstruction_threshold).sum()
 
    
    
    return (above_threshold, below_threshold)

def plot_telemetry(telemetry):
    tele = np.array(telemetry)
    #!!! Add labels to lines
    plt.plot(tele[:,0],tele[:,1], color='r') #red = points above threshold
    plt.plot(tele[:,0],tele[:,2], color='b') #blue = points below threshold
    plt.show()    

###3D plotter
def plot_3D(pixel_block_3d):
    #pixel_block_3d_est = np.around(pixel_block_3d)
    #print(pixel_block_3d_est)
    
    if simple_belief_reports == 1:
        print("Max belief value:", np.amax(pixel_block_3d))
        print("Min belief value:", np.amin(pixel_block_3d))         
    
    hits_3d = np.argwhere(pixel_block_3d.squeeze() >= reconstruction_threshold)    
    print("NUMBER OF HITS", np.shape(hits_3d))
    x3d = hits_3d[:,2]
    y3d = hits_3d[:,1]
    z3d = hits_3d[:,0]
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #plt.axes(projection='3d')
    ax.scatter(x3d, y3d, z3d)  
    plt.show()
    
    
    
    
### Random Noise Generator Function
def add_noise(inputs,noise_factor=0.3):
     noisy = inputs + torch.randn_like(inputs) * noise_factor
     noisy = torch.clip(noisy,0.,1.)
     return noisy

###RNG Seeding for Determinism Function
def Determinism_Seeding(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

### Training Function
def train_epoch_den(encoder, decoder, device, dataloader, loss_fn, optimizer,noise_factor=0):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _ in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        image_noisy = add_noise(image_batch,noise_factor)
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
        # Print batch loss
        print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())
    return np.mean(train_loss)

### Testing Function
def test_epoch_den(encoder, decoder, device, dataloader, loss_fn,noise_factor=0):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            image_noisy = add_noise(image_batch,noise_factor)
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
    return val_loss.data

###Plotting Function
def plot_ae_outputs_den(encoder, decoder, epoch, n=10, noise_factor=0):       #Defines a function for plotting the output of the autoencoder. And also the input + clean training data? Function takes inputs, 'encoder' and 'decoder' which are expected to be classes (defining the encode and decode nets), 'n' which is the number of ?????Images in the batch????, and 'noise_factor' which is a multiplier for the magnitude of the added noise allowing it to be scaled in intensity.  
    plt.figure(figsize=(16,4.5))                                      #Sets the figure size

    for i in range(n):                                                #Runs for loop where 'i' itterates over 'n' total values which range from 0 to n-1
      
        #Following section creates the noised image data from the original clean labels (images)   
        ax = plt.subplot(3,n,i+1)                                       #Creates a number of subplots for the 'Original images??????' i.e the labels. the position of the subplot is i+1 as it falls in the first row
        img = test_dataset[i][0].unsqueeze(0)
        #print("img!!!TEST:",np.shape(img))      
        image_noisy = img  #add_noise(img,noise_factor)     #use this noise to simulate gaussian thermal noise in detectro? wheras generated noise is from other tracks
        image_noisy = image_noisy.to(device)
        #print("image_noisy!!!TEST:",np.shape(image_noisy))
        
        #Following section sets the autoencoder to evaluation mode rather than training (up till line 'with torch.no_grad()')
        encoder.eval()                                   #.eval() is a kind of switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time. For example, Dropouts Layers, BatchNorm Layers etc. You need to turn off them during model evaluation, and .eval() will do it for you. In addition, the common practice for evaluating/validation is using torch.no_grad() in pair with model.eval() to turn off gradients computation
        decoder.eval()                                   #Simarlary as above
    
    
        with torch.no_grad():                                               #As mentioned in .eval() comment, the common practice for evaluating/validation is using torch.no_grad() which turns off gradients computation whilst evaluating the model (the opposite of training the model)     
        #Following line runs the autoencoder on the noised data
           rec_img  = decoder(encoder(image_noisy))                        #Creates a recovered image (denoised image), by running a noisy image through the encoder and then the output of that through the decoder.
    
        #Following section generates the img plots for the original(labels), noised, and denoised data)
        raw_data = img.cpu().squeeze().numpy()
        #print("raw_data!!!TEST:",np.shape(raw_data))
        ax.scatter(raw_data[1], raw_data[0], raw_data[2], cmap='gist_gray')
        #plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')           #plt.imshow plots an image. The arguments for imshow are, 'image data array' and cmap= which is the colour map. #.squeeze() acts on a tensor and returns a tensor, it removes all dimensions of the tensor that are of length 1, (A×1×B) becomes (AxB) where A and B are greater than 1 #.numpy() creates a numpy array from a tensor #!!! is the .cpu part becuase the code was not made to accept the gpu/cpu check i made????
    
        ax.get_xaxis().set_visible(False)                                   #Hides the x axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
        ax.get_yaxis().set_visible(False)                                   #Hides the y axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
        if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
          ax.set_title('Original images')                                   #When above condition is reached, the plots title is set
    
        ax = plt.subplot(3, n, i + 1 + n)                                   #Creates a number of subplots for the 'Corrupted images??????' i.e the labels. the position of the subplot is i+1+n as it falls in the second row
        imn_data = image_noisy.cpu().squeeze().numpy()
        #print("imn_data!!!TEST:",np.shape(imn_data))
        ax.scatter(imn_data[1], imn_data[0], imn_data[2])   #plt.imshow plots an image. The arguments for imshow are, 'image data array' and cmap= which is the colour map. #.squeeze() acts on a tensor and returns a tensor, it removes all dimensions of the tensor that are of length 1, (A×1×B) becomes (AxB) where A and B are greater than 1 #.numpy() creates a numpy array from a tensor #!!! is the .cpu part becuase the code was not made to accept the gpu/cpu check i made????
        ax.get_xaxis().set_visible(False)                                   #Hides the x axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
        ax.get_yaxis().set_visible(False)                                   #Hides the y axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
        if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
          ax.set_title('Corrupted images')                                  #When above condition is reached, the plots title is set
    
        ax = plt.subplot(3, n, i + 1 + n + n)#, projection='3d')                               #Creates a number of subplots for the 'Reconstructed images??????' i.e the labels. the position of the subplot is i+1+n+n as it falls in the third row
        rec_data = rec_img.cpu().squeeze().numpy()
        #print("rec_data!!!TEST:",np.shape(rec_data))      
        #print("rec=", rec_data)
        ax.scatter(rec_data[1], rec_data[0], rec_data[2])
        #plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')       #plt.imshow plots an image. The arguments for imshow are, 'image data array' and cmap= which is the colour map. #.squeeze() acts on a tensor and returns a tensor, it removes all dimensions of the tensor that are of length 1, (A×1×B) becomes (AxB) where A and B are greater than 1 #.numpy() creates a numpy array from a tensor #!!! is the .cpu part becuase the code was not made to accept the gpu/cpu check i made????
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
    #plt.axes(projection='3d')
    plt.show()                                 #After entire loop is finished, the generated plot is printed to screen
    
    print("\nOriginal")
    plot_3D(raw_data)
    #plot_3D(imn_data)
    print("\nRecovered")
    plot_3D(rec_data)
  
    if telemetry_on == 1:
        above_threshold, below_threshold = belief_telemetry(rec_data, reconstruction_threshold, epoch)   
        telemetry.append([epoch, above_threshold, below_threshold])
 
    
    print("End of Epoch %s \n \n" %epoch)


   
    
#%% - Program Internal Setup
#image_noisy_list = []
telemetry = [[0,0.5,0.5]]


if seed != 0: 
    Determinism_Seeding(seed)

#%% - Setup model, loss criteria and optimiser    
    
### Define the loss function
loss_fn = torch.nn.MSELoss()

### Define a learning rate for the optimiser
lr = learning_rate                                     #Just sets the learing rate value from the user inputs pannel at the top

### Set the random seed for reproducible results
torch.manual_seed(seed)              

### Initialize the two networks
d = latent_space_nodes #!!!d is passed to the encoder & decoder in the lines below and represents the encoded space dimension. This is the number of layers the linear stages will shrink to? #!!!

#model = Autoencoder(encoded_space_dim=encoded_space_dim)
encoder = Encoder(encoded_space_dim=d,fc2_input_dim=128, encoder_debug=encoder_debug)
decoder = Decoder(encoded_space_dim=d,fc2_input_dim=128, decoder_debug=decoder_debug)
params_to_optimize = [{'params': encoder.parameters()} ,{'params': decoder.parameters()}] #Selects what to optimise, 

### Define an optimizer (both for the encoder and the decoder!)
wd = optim_w_decay                                                           #Just sets the weight decay value from the user inputs pannel at the top
optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=wd)

#%% - Compute device check

#Following section checks if a CUDA enabled GPU is available. If found it is selected as the 'device' to perform the tensor opperations. If no CUDA GPU is found the 'device' is set to CPU (much slower) 
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')  #Informs user if running on CPU or GPU

#Following section moves both the encoder and the decoder to the selected device i.e detected CUDA enabled GPU or to CPU
encoder.to(device)   #Moves encoder to selected device, CPU/GPU
decoder.to(device)   #Moves decoder to selected device, CPU/GPU


#%% - Compute

history_da={'train_loss':[],'val_loss':[]}                   #Just creates a variable called history_da which contains two lists, 'train_loss' and 'val_loss' which are both empty to start with. value are latter appeneded to the two lists by way of history_da['val_loss'].append(x)

for epoch in range(num_epochs):                              #For loop that iterates over the number of epochs where 'epoch' takes the values (0) to (num_epochs - 1)
    print('EPOCH %d/%d' % (epoch + 1, num_epochs))
    ### Training (use the training function)
    train_loss=train_epoch_den(
                               encoder=encoder, 
                               decoder=decoder, 
                               device=device, 
                               dataloader=train_loader, 
                               loss_fn=loss_fn, 
                               optimizer=optim,
                               noise_factor=noise_factor)
    
    ### Validation (use the testing function)
    val_loss = test_epoch_den(
                              encoder=encoder, 
                              decoder=decoder, 
                              device=device, 
                              dataloader=val_loader, 
                              loss_fn=loss_fn,
                              noise_factor=noise_factor)
    
    # Print Validation_loss and plots at end of each epoch
    history_da['train_loss'].append(train_loss)
    history_da['val_loss'].append(val_loss)
    print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))     #epoch +1 is to make up for the fact the range spans 0 to epoch-1 but we want to numerate things from 1 upwards for sanity
    plot_ae_outputs_den(encoder,decoder,epoch + 1, noise_factor=noise_factor)

if telemetry_on == 1:    
    plot_telemetry(telemetry)

###Loss function plots
epochs_range = range(1,num_epochs+1)
plt.plot(epochs_range, history_da['train_loss']) 
plt.title("Training loss")   
plt.show()

plt.plot(epochs_range, history_da['train_loss']) 
plt.title("Validation loss") 
plt.show()