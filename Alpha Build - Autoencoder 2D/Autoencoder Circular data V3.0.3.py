# -*- coding: utf-8 -*-
"""
Created on Sat Oct 8 2022
Circular & Spherical data test Autoencoder v3.0.3
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


def train_loader_func(path):   #fix need for two seperate loads, one on each loader
    sample = (np.load(path))
    sample = sample[0]               
    return sample

def test_loader_func(path):
    load = 1 # Set manually, 0 = Blank, no data, 1 = just signal, 2 = just noise, 3 = both, but with differing values (1,2)    #!!! OPION 3 NOT WORKING
    sample = (np.load(path)) 
    sample2 = np.ma.masked_where(sample[1] == load, sample[1])                   
    return sample2

#%% - User Inputs
learning_rate = 0.001  #User controll to set optimiser learning rate(Hyperparameter)
optim_w_decay = 1e-05  #User controll to set optimiser weight decay (Hyperparameter)

#%% - Program Settings
seed = 10              #0 is default which gives no seeeding to RNG, if the value is not zero then this is used for the RNG seeding for numpy, random, and torch libraries
#path = "C:/Users/Student/Desktop/fake im data/"  #"/path/to/your/images/"

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

###Convoloution + Linear Autoencoder
###Encoder
class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        
        ###Convolutional Encoder Layers
        
        #Conv2d function takes arguments: (input channels, output channels, kernel size/receiptive field (), stride and padding. # here for more info on arguments https://uob-my.sharepoint.com/personal/ex18871_bristol_ac_uk/_layouts/15/Doc.aspx?sourcedoc={109841d1-79cb-45c2-ac39-0fd24300c883}&action=edit&wd=target%28Code%20Companion.one%7C%2FUntitled%20Page%7C55b8dfad-d53c-4d8e-a2ef-de1376232896%2F%29&wdorigin=NavigationUrl
        
        #Input channels is 1 as the image is black and white so only has luminace values no rgb channels, 
        #Output channels which is the amount of seperate output tensors?????????, this defines the number of seperate kernals that will be run across image, all which produce thier own output arrays
        #The kernal size, is the size of the convolving layer. best visualised at the link above. Ours is set at is 3 so i guess a line of 3 pixels? it can be square ie give a tuple (x,x) but out data has been linearlised for some reason? NOTE: Kernal size can be even but is almost always an odd number to there is symetry round a central pixel
        #Stride is how far the kernal moves across each time, the default is across by one pixel a time untill end of line then back to start of line and down by a pixel, followed by the cycle of across etc again. 
        #    setting the stride to larger values makes it move across jumping pixels, i.e (1,3) the filter moves across by 3 pixels, and then back to begining of line and down by 1 pixel, this is used to downsample data, so that is why our stride is larger than default. 
        #Padding adds padding to the edges of the data array  before the convoloution filtering, the padding value are noramlly zero but can be made other things. this is normally used to mamke the input array same size as the output
        #Dilation (not used param) sets the spread of the kernal, by defualt it is one contigous block, dilation spreads it out. best visulaised in the link above
        
        self.encoder_cnn = nn.Sequential(
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
        
        #NEW Encoder nodes: 
        #input data format = [batchsize, 1, 128 88] # [batchsize, channels, pixels_in_x, pixels_in_y]
        #conv layers: input---Conv_L1---> [batchsize, 8, 64, 44]
        #conv layers: input---Conv_L2---> [batchsize, 16, 32, 22]
        #conv layers: input---Conv_L3---> [batchsize, 32, 15, 10]
        #Flatten layer: input --flat_L1-> [10 * 15 * 32, 512]
        #Linear layer: input--Lin_L1----> [512, encoded_space_dim]        
                                        # [batchsize, 32, 15, 10]
                                                
                                                
        #conv layers: input---T Conv_L1---> [batchsize, 16, 31, 21]
        #conv layers: input---T Conv_L2---> [batchsize, 8, 62, 42]        
        #conv layers: input---T Conv_L3---> [batchsize, 1, 124, 84]  

        
        #OLD Encoder nodes: 
        #input data format = [batchsize, 1, 28, 28] # [batchsize, channels, pixels_in_x, pixels_in_y]
        #conv layers: input---Conv_L1---> [batchsize, 8, 14, 14]
        #conv layers: input---Conv_L2---> [batchsize, 16, 7, 7]
        #conv layers: input---Conv_L3---> [batchsize, 32, 3, 3]
        #Flatten layer: input --flat_L1-> [3 * 3 * 32, 128]
        #Linear layer: input--Lin_L1----> [128, encoded_space_dim]
        
        #Linear layer: input--Lin_L1----> [encoded_space_dim, 128]        
        #Flatten layer: input --flat_L1-> [128, 3 * 3 * 32]        
        #conv layers: input---Conv_L1---> [batchsize, 32, 43, 63]
        #conv layers: input---Conv_L2---> [batchsize, 16, 21, 31]
        #conv layers: input---Conv_L3---> [batchsize, 8, 9,  14]        
        #output data format = [batchsize, 1, 28, 28] # [batchsize, channels, pixels_in_x, pixels_in_y]
        
        
        ###Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        ###Linear Encoder Layers
        
        #nn.Linear arguments are: in_features – size of each input sample, out_features – size of each output sample, bias – If set to False, the layer will not learn an additive bias. Default: True
        self.encoder_lin = nn.Sequential(
            #Linear encoder layer 1  
            nn.Linear(10 * 15 * 32, 512),                   #!!! linear network layer. arguuments are input dimensions/size, output dimensions/size. Takes in data of dimensions 3* 3 *32 and outputs it in 1 dimension of size 128
            nn.ReLU(True),                                #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
            #Linear encoder layer 2
            nn.Linear(512, encoded_space_dim)             #Takes in data of 1 dimension with size 128 and outputs it in one dimension of size defined by encoded_space_dim (this is the latent space? the smalle rit is the more comression but thte worse the final fidelity)
        )
        
    def forward(self, x):
        print("x in", x.size())
        x = self.encoder_cnn(x)                           #Runs convoloutional encoder on x #!!! input data???
        print("x CNN out", x.size())
        x = self.flatten(x)                               #Runs flatten  on output of conv encoder #!!! what is flatten?
        print("x Flat out", x.size())
        x = self.encoder_lin(x)                           #Runs linear encoder on flattened output 
        print("x Lin out", x.size())
        return x                                          #Return final result

###Decoder
class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        
        ###Linear Decoder Layers
        self.decoder_lin = nn.Sequential(
            #Linear decoder layer 1            
            nn.Linear(encoded_space_dim, 512),
            nn.ReLU(True),                                 #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
            #Linear decoder layer 2
            nn.Linear(512, 10 * 15 * 32),
            nn.ReLU(True)                                  #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
        )
        ###Unflatten layer
        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 15, 10))

        ###Convolutional Decoder Layers
        #NOTE - as this is the decoder and it must perform the reverse operations to the encoder, instead of using conv2d here ConvTranspose2d is used which is the inverse opperation
        #ConvTransopose2d function takes arguments: (input channels, output channels, kernel size/receiptive field (), stride and padding. # here for more info on arguments https://uob-my.sharepoint.com/personal/ex18871_bristol_ac_uk/_layouts/15/Doc.aspx?sourcedoc={109841d1-79cb-45c2-ac39-0fd24300c883}&action=edit&wd=target%28Code%20Companion.one%7C%2FUntitled%20Page%7C55b8dfad-d53c-4d8e-a2ef-de1376232896%2F%29&wdorigin=NavigationUrl
        
        #Input channels 
        #Output channels which is the amount of seperate output tensors?????????, 
        #The kernal size, 
        #Stride 
        #Padding ??? ##!!!!
        #Output_Padding adds padding to the edges of the data array
        #Dilation (not used param) 
        
        self.decoder_conv = nn.Sequential(
            #Convolutional decoder layer 1
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=0, output_padding=0),             #Input_channels, Output_channels, Kernal_size, Stride, padding(unused), Output_padding
            nn.BatchNorm2d(16),                                                    #BatchNorm normalises the outputs as a batch? #!!!. argument is 'num_features' (expected input of size)
            nn.ReLU(True),                                                         #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
            #Convolutional decoder layer 2
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=0, output_padding=1),   #Input_channels, Output_channels, Kernal_size, Stride, padding, Output_padding
            nn.BatchNorm2d(8),                                                     #BatchNorm normalises the outputs as a batch? #!!!. argument is 'num_features' (expected input of size)
            nn.ReLU(True),                                                         #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
            #Convolutional decoder layer 3
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)     #Input_channels, Output_channels, Kernal_size, Stride, padding, Output_padding
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)       #Runs linear decoder on x #!!! is x input data? where does it come from??
        x = self.unflatten(x)         #Runs unflatten on output of linear decoder #!!! what is unflatten?
        x = self.decoder_conv(x)      #Runs convoloutional decoder on output of unflatten
        x = torch.sigmoid(x)          #THIS IS IMPORTANT PART OF FINAL OUTPUT!: Runs sigmoid function which turns the output data values to range (0-1)#!!! ????    Also can use tanh fucntion if wanting outputs from -1 to 1
        return x                      #Retuns the final output
    
    
#%% - Functions
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
def train_epoch_den(encoder, decoder, device, dataloader, loss_fn, optimizer,noise_factor=0.3):
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

def plot_ae_outputs_den(encoder,decoder,n=10,noise_factor=0):       #Defines a function for plotting the output of the autoencoder. And also the input + clean training data? Function takes inputs, 'encoder' and 'decoder' which are expected to be classes (defining the encode and decode nets), 'n' which is the number of ?????Images in the batch????, and 'noise_factor' which is a multiplier for the magnitude of the added noise allowing it to be scaled in intensity.  
    plt.figure(figsize=(16,4.5))                                      #Sets the figure size
    targets = np.array(test_dataset.targets, dtype = np.float32)                            #Creates a numpy array (from the .numpy part) the array is created from the values in the specified tensor, which in this case is test_dataset.targets (test_dataset is the dataloader, .targets is a subclass of the dataloader that holds the labels, i.e the correct answer data (in this case the unnoised images).)                          
    print("TARget",targets)
    print("tsize", np.shape(targets))
    print("ttype", type(targets))
    t_idx = {i:np.where(targets==i)[0] for i in range(n)}          #!!! ????
    for i in range(n):                                                #Runs for loop where 'i' itterates over 'n' total values which range from 0 to n-1
      
      #Following section creates the noised image data drom the original clean labels (images)   
      ax = plt.subplot(3,n,i+1)                                       #Creates a number of subplots for the 'Original images??????' i.e the labels. the position of the subplot is i+1 as it falls in the first row
      img = test_dataset[i][0].unsqueeze(0)
      image_noisy = img  #add_noise(img,noise_factor)     
      image_noisy = image_noisy.to(device)
      
      
      #Following section sets the autoencoder to evaluation mode rather than training (up till line 'with torch.no_grad()')
      encoder.eval()                                   #.eval() is a kind of switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time. For example, Dropouts Layers, BatchNorm Layers etc. You need to turn off them during model evaluation, and .eval() will do it for you. In addition, the common practice for evaluating/validation is using torch.no_grad() in pair with model.eval() to turn off gradients computation
      decoder.eval()                                   #Simarlary as above


      with torch.no_grad():                                               #As mentioned in .eval() comment, the common practice for evaluating/validation is using torch.no_grad() which turns off gradients computation whilst evaluating the model (the opposite of training the model)     
      #Following line runs the autoencoder on the noised data
         rec_img  = decoder(encoder(image_noisy))                        #Creates a recovered image (denoised image), by running a noisy image through the encoder and then the output of that through the decoder.

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
    
    
#%% NEW DATALOADER
#%% - User Inputs

dataset_title = "S_Dataset 3"
data_path = "C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/Yr 3 Project/Circular and Spherical Dummy Datasets/" #"C:/Users/Student/Desktop/fake im data/"  #"/local/path/to/the/images/"
batch_size = 5         #Data Loader # of Images to pull per batch (add a check to make sure the batch size is smaller than the total number of images in the path selected)


#%% - Advanced Settings
debug_loader_batch = 1     #(Default = 0 = [OFF]) //INPUT 0 or 1//   #Setting debug loader batch will print to user the images taken in by the dataoader in this current batch and print the corresponding labels
plot_every_other = 1       #(Default = 1) //MUST BE INTEGER INPUT//  #If debug loader batch is enabled this sets the interval for printing for user, 1 is every single img in the batch, 2 is every other img, 5 is every 5th image etc 
batch_size_protection = 1  #(Default = 1 = [ON]) //INPUT 0 or 1//    #WARNING if turned off, debugging print will cause and exeption due to the index growing too large in the printing loop (img = train_features[i])


#%% - Path images, greater than batch choice? CHECK
####check for file count in folder####
if batch_size_protection == 1:
    files_in_path = os.listdir(data_path + dataset_title + '/Data/') 
    num_of_files_in_path = len(files_in_path)
    print("#files in path=",num_of_files_in_path ,"batch size=",batch_size)
    if num_of_files_in_path < batch_size:
        print("Error, the path selected has", num_of_files_in_path, "image files, which is", (batch_size - num_of_files_in_path) , "less than the chosen batch size. Please select a batch size less than the total number of images in the directory")
        
        #!!!Need code to make this event cancel the running of program and re ask for user input on batch size or just reask for the batch size
        batch_err_message = "Choose new batch size, must be less than total amount of images in directory", (num_of_files_in_path)
        batch_size = int(input(batch_err_message))  #!!! not sure why input message is printing with wierd brakets and speech marks in the terminal? Investigate


#%% - Data Preparation  #!!!Perhaps these should be passed ino the loader as user inputs, that allows for ease of changing between differnt tranforms in testing without having to flip to the data loader code
train_transforms = transforms.Compose([#transforms.RandomRotation(30),         #Compose is required to chain together multiple transforms in serial 
                                       #transforms.RandomResizedCrop(224),
                                       #transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor()])                #other transforms can be dissabled but to tensor must be left enabled

test_transforms = transforms.Compose([#transforms.Resize(255),
                                      #transforms.CenterCrop(224),
                                      transforms.ToTensor()])

#%% - Data Loading
train_dataset = datasets.DatasetFolder(data_path + dataset_title, loader=train_loader_func, extensions='.npy', transform=train_transforms)
test_dataset = datasets.DatasetFolder(data_path + dataset_title, loader=test_loader_func, extensions='.npy', transform=test_transforms)

print("TRAINING DATA SHAPE:", train_dataset[0][0].shape)
print("TESTING DATA SHAPE:", test_dataset[0][0].shape)


#m=len(train_dataset)
#train_data, val_data = random_split(train_dataset, [int(m-m*0.2), int(m*0.2)])

#print("TRAIN DATA SHAPE:", train_data[0][0].shape)
#print("VALID DATA SHAPE:", val_data[0][0].shape)
#print("TEST DATA SHAPE:", test_dataset[0][0].shape)

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size)
#valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)  #!!! train datset

#%% - Debugging Outputs
if debug_loader_batch == 1:
    train_features, train_labels = next(iter(train_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    
    for i in range (0, batch_size, plot_every_other):   # Display image and label.
        print ("\nImage #",i+1)
        img = train_features[i].squeeze()
        label = train_labels[i]
        plt.imshow(img, cmap="gray")  #.T, cmap="gray")   #!!!fix the need for img.T which is the transpose, as it flips the image, 
        plt.show()
        print(f"Label: {label}")




#%% END OF NEW DATA LOADER #    

#%% - Program Internal Setup
image_noisy_list = []

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
d = 4 #!!!d is passed to the encoder & decoder in the lines below and represents the encoded space dimension. This is the number of layers the linear stages will shrink to? #!!!

#model = Autoencoder(encoded_space_dim=encoded_space_dim)
encoder = Encoder(encoded_space_dim=d,fc2_input_dim=128)
decoder = Decoder(encoded_space_dim=d,fc2_input_dim=128)
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
noise_factor = 0                                           #User controll to set the noise factor, a multiplier for the magnitude of noise added. 0 means no noise added, 1 is defualt level of noise added, 10 is 10x default level added (Hyperparameter)
num_epochs = 8                                               #User controll to set number of epochs (Hyperparameter)
batch_size= 10

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
                              dataloader=test_loader, 
                              loss_fn=loss_fn,
                              noise_factor=noise_factor)
    
    # Print Validation_loss and plots at end of each epoch
    history_da['train_loss'].append(train_loss)
    history_da['val_loss'].append(val_loss)
    print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))     #epoch +1 is to make up for the fact the range spans 0 to epoch-1 but we want to numerate things from 1 upwards for sanity
    plot_ae_outputs_den(encoder,decoder,noise_factor=noise_factor)
    
    
    
    