# -*- coding: utf-8 -*-
"""
Created on Sat Oct 8 2022

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
#import pandas as pd 
#import torch.nn.functional as F
#import torch.optim as optim
#import os

#%% - User Inputs
learning_rate = 0.001  #User controll to set optimiser learning rate(Hyperparameter)
optim_w_decay = 1e-05  #User controll to set optimiser weight decay (Hyperparameter)

#%% - Program Settings
seed = 10              #0 is default which gives no seeeding to RNG, if the value is not zero then this is used for the RNG seeding for numpy, random, and torch libraries
path = "C:/Users/Student/Desktop/fake im data/"  #"/path/to/your/images/"

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
        #Encoder nodes: 
        #input data format = [batchsize, 1, 28, 28] # [batchsize, channels, pixels_in_x, pixels_in_y]
        #conv layers: input---Conv_L1---> []
        
        
        
        ###Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        ###Linear Encoder Layers
        
        #nn.Linear arguments are: in_features – size of each input sample, out_features – size of each output sample, bias – If set to False, the layer will not learn an additive bias. Default: True
        self.encoder_lin = nn.Sequential(
            #Linear encoder layer 1  
            nn.Linear(3 * 3 * 32, 128),                   #!!! linear network layer. arguuments are input dimensions/size, output dimensions/size. Takes in data of dimensions 3* 3 *32 and outputs it in 1 dimension of size 128
            nn.ReLU(True),                                #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
            #Linear encoder layer 2
            nn.Linear(128, encoded_space_dim)             #Takes in data of 1 dimension with size 128 and outputs it in one dimension of size defined by encoded_space_dim (this is the latent space? the smalle rit is the more comression but thte worse the final fidelity)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)                           #Runs convoloutional encoder on x #!!! input data???
        x = self.flatten(x)                               #Runs flatten  on output of conv encoder #!!! what is flatten?
        x = self.encoder_lin(x)                           #Runs linear encoder on flattened output 
        return x                                          #Return final result

###Decoder
class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        
        ###Linear Decoder Layers
        self.decoder_lin = nn.Sequential(
            #Linear decoder layer 1            
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),                                 #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
            #Linear decoder layer 2
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)                                  #ReLU activation function - Activation function determinines if a neuron fires, i.e is the output of the node considered usefull. also allows for backprop. the arg 'True' makes the opperation carry out in-place(changes the values in the input array to the output values rather than making a new one), default would be false
        )
        ###Unflatten layer
        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 3, 3))

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
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),             #Input_channels, Output_channels, Kernal_size, Stride, padding(unused), Output_padding
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




##########################################################################################################################################################################

#Second half 



###Plotting Function
def plot_ae_outputs_den(encoder,decoder,n=10,noise_factor=0.3):       #Defines a function for plotting the output of the autoencoder. And also the input + clean training data? Function takes inputs, 'encoder' and 'decoder' which are expected to be classes (defining the encode and decode nets), 'n' which is the number of ?????Images in the batch????, and 'noise_factor' which is a multiplier for the magnitude of the added noise allowing it to be scaled in intensity.  
    plt.figure(figsize=(16,4.5))                                      #Sets the figure size
    targets = test_dataset.targets.numpy()                            #Creates a numpy array (from the .numpy part) the array is created from the values in the specified tensor, which in this case is test_dataset.targets (test_dataset is the dataloader, .targets is a subclass of the dataloader that holds the labels, i.e the correct answer data (in this case the unnoised images).)                          
    t_idx = {i:np.where(targets==i)[0][0] for i in range(n)}          #!!! ????
    for i in range(n):                                                #Runs for loop where 'i' itterates over 'n' total values which range from 0 to n-1
        
      #Following section creates the noised image data drom the original clean labels (images)   
      ax = plt.subplot(3,n,i+1)                                       #Creates a number of subplots for the 'Original images??????' i.e the labels. the position of the subplot is i+1 as it falls in the first row
      img = test_dataset[t_idx[i]][0].unsqueeze(0)                    #!!! ????
      if epoch <= 0:                                                  #CHECKS TO SEE IF THE EPOCH IS LESS THAN ZERO , I ADDED THIS TO GET THE SAME NOISED IMAGES EACH EPOCH THOUGH THIS COULD BE WRONG TO DO?
          global image_noisy                                          #'global' means the variable (image_noisy) set inside a function is globally defined, i.e defined also outside the function
          image_noisy = add_noise(img,noise_factor)                   #Runs the function 'add_noise' (in this code) the function adds noise to a set of data, the function takes two arguments, img is the data to add noise to, noise factor is a multiplier for the noise values added, i.e if multiplier is 0 no noise is added, if it is 1 default amount is added, if it is 10 then the values are raised 10x 
          image_noisy_list.append(image_noisy)                        #Adds the just generated noise image to the list of all the noisy images
      image_noisy = image_noisy_list[i].to(device)                    #moves the list (i think of tensors?) to the device that will process it i.e either cpu or gpu, we have a check elsewhere in the code that detects if gpu is availible and sets the value of 'device' to gpu or cpu depending on availibility (look for the line that says "device = 'cuda' if torch.cuda.is_available() else 'cpu'"). NOTE: this moves the noised images to device, i think that the original images are already moved to device in previous code

    
      #Following section sets the autoencoder to evaluation mode rather than training (up till line 'with torch.no_grad()')
      encoder.eval()                                   #.eval() is a kind of switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time. For example, Dropouts Layers, BatchNorm Layers etc. You need to turn off them during model evaluation, and .eval() will do it for you. In addition, the common practice for evaluating/validation is using torch.no_grad() in pair with model.eval() to turn off gradients computation
      decoder.eval()                                   #Simarlary as above


      with torch.no_grad():                                               #As mentioned in .eval() comment, the common practice for evaluating/validation is using torch.no_grad() which turns off gradients computation whilst evaluating the model (the opposite of training the model)     
      #Following line runs the autoencoder on the noised data
         rec_img  = decoder(encoder(image_noisy))                         #Creates a recovered image (denoised image), by running a noisy image through the encoder and then the output of that through the decoder.

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
    
    
    

#%% - Program Internal Setup
image_noisy_list = []

if seed != 0: 
    Determinism_Seeding(seed)

#%% - Data Importer
# data_dir = 'dataset'
# train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
# test_dataset  = torchvision.datasets.MNIST(data_dir, train=False, download=True)

data_directory = 'dataset'
train_dataset = torchvision.datasets.DatasetFolder(data_dir, train=True, download=True)
# as MNIST is a dataset on web, the download argument authorises download from there.
# Train argument decides whether to take from train or test folder.
test_dataset  = torchvision.datasets.DatasetFolder(data_dir, train=False, download=True)

# in this, were now going to try to work the data generator for a super simple 28x28 cross. This will be
# generated in the 'supersimp' then added here through the data_directory function:

#%% - Data Preparation  #!!!Perhaps these should be passed ino the loader as user inputs, that allows for ease of changing between differnt tranforms in testing without having to flip to the data loader code

####SECTION MODIFIED#####
train_transform = transforms.Compose([                                         #train_transform variable holds the tensor tranformations to be performed on the training data.  transforms.Compose([ ,  , ]) allows multiple transforms to be chained together (in serial?) (#!!! does it do more than this??)
                                       #transforms.RandomRotation(30),         #transforms.RandomRotation(angle (degrees?) ) rotates the tensor randomly up to max value of angle argument
                                       #transforms.RandomResizedCrop(224),     #transforms.RandomResizedCrop(pixels) crops the data to 'pixels' in height and width (#!!! and (maybe) chooses a random centre point????)
                                       #transforms.RandomHorizontalFlip(),     #transforms.RandomHorizontalFlip() flips the image data horizontally 
                                       #transforms.Normalize((0.5), (0.5)),    #transforms.Normalize can be used to normalise the values in the array
                                       transforms.ToTensor()])                 #other transforms can be dissabled but to tensor must be left enabled ! it creates a tensor from a numpy array #!!! ?

test_transform = transforms.Compose([                                          #test_transform variable holds the tensor tranformations to be performed on the evaluation data.  transforms.Compose([ ,  , ]) allows multiple transforms to be chained together (in serial?) (#!!! does it do more than this??)
                                      #transforms.Resize(255),                 #transforms.Resize(pixels? #!!!) ??
                                      #transforms.CenterCrop(224),             #transforms.CenterCrop(pixels? #!!!) ?? Crops the given image at the center. If the image is torch Tensor, it is expected to have […, H, W] shape, where … means an arbitrary number of leading dimensions. If image size is smaller than output size along any edge, image is padded with 0 and then center cropp
                                      #transforms.Normalize((0.5), (0.5)),     #transforms.Normalize can be used to normalise the values in the array
                                      transforms.ToTensor()])                  #other transforms can be dissabled but to tensor must be left enabled ! it creates a tensor from a numpy array #!!! ?

# this applies above transforms to dataset (dataset transform = transform above)
train_dataset.transform = train_transform       #!!! train_dataset is the class? object 'dataset' it has a subclass called transforms which is the list of transofrms to perform on the dataset when loading it. train_tranforms is the set of chained transofrms we created, this is set to the dataset transforms subclass 
test_dataset.transform = test_transform         #!!! similar to the above but for the test(eval) dataset, check into this for the exact reason for using it, have seen it deone in other ways i.e as in the dataloader.py it is performed differntly. this way seems to be easier to follow
#####For info on all transforms check out: https://pytorch.org/vision/0.9/transforms.html
####SECTION MODIFIED END#####



###Following section splits the training dataset into two, train_data (to be noised) and valid data (to use in eval)
m=len(train_dataset) #Just calculates length of train dataset, m is only used in the next line to decide the values of the split, (4/5 m) and (1/5 m)
# 80% 20% is the absolute classic split
train_data, val_data = random_split(train_dataset, [int(m-m*0.2), int(m*0.2)])    #random_split(data_to_split, [size of output1, size of output2]) just splits the train_dataset into two parts, 4/5 goes to train_data and 1/5 goes to val_data , validation?


###Following section for Dataloaders, they just pull a random sample of images from each of the datasets we now have, train_data, valid_data, and test_data. the batch size defines how many are taken from each set, shuffle argument shuffles them each time?? #!!!
batch_size=256                                                                                #User controll to set batch size for the dataloaders (Hyperparameter)?? #!!!

# required to load the data into the endoder/decoder. Combines a dataset and a sampler, and provides an iterable over the given dataset.
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)                 #Training data loader, can be run to pull training data as configured
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)                   #Validation data loader, can be run to pull training data as configured
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True)   #Testing data loader, can be run to pull training data as configured. Also is shuffled using parameter shuffle #!!! why is it shuffled?



#%% - Setup model, loss criteria and optimiser    
    
### Define the loss function (mean square error)
loss_fn = torch.nn.MSELoss()

### Define a learning rate for the optimiser. 
# Its how much to change the model in response to the estimated error each time the model weights are updated.
lr = learning_rate                                     #Just sets the learing rate value from the user inputs pannel at the top

### Set the random seed for reproducible results
torch.manual_seed(seed)              

### Initialize the two networks
d = 4 #!!!d is passed to the encoder & decoder in the lines below and represents the encoded space dimension. This is the number of layers the linear stages will shrink to? #!!!

# model = Autoencoder(encoded_space_dim=encoded_space_dim)
# use encoder and decoder classes, providing dimensions for your dataset. FC2_INPUT_DIM IS NOT USED!! This would be extremely useful.
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
noise_factor = 0.4                                           #User controll to set the noise factor, a multiplier for the magnitude of noise added. 0 means no noise added, 1 is defualt level of noise added, 10 is 10x default level added (Hyperparameter)
num_epochs = 5                                               #User controll to set number of epochs (Hyperparameter)


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
                              dataloader=valid_loader, 
                              loss_fn=loss_fn,
                              noise_factor=noise_factor)
    
    # Print Validation_loss and plots at end of each epoch
    history_da['train_loss'].append(train_loss)
    history_da['val_loss'].append(val_loss)
    print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))     #epoch +1 is to make up for the fact the range spans 0 to epoch-1 but we want to numerate things from 1 upwards for sanity
    plot_ae_outputs_den(encoder,decoder,noise_factor=noise_factor)
    
    
    
    