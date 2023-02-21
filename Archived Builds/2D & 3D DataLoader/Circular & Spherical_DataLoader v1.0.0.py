# -*- coding: utf-8 -*-
"""
DeepClean_DataLoader Driver v1.0.0
@author: Adill Al-Ashgar
Created on Mon Oct 10 04:20:57 2022

USER NOTICE!
x Takes in a batch size, a data path, debugging choice, and debugging step size.
x Returns two dataloaders (trainloader and testloader) prepared with a randomised?(not yet) batch of images from the path. 
x Also prints the images in the batch to user, if debug mode is on (ie = 1). debugging step size sets for how 
  many images one will be printed, in form (step_size:1) ie 1:1 2:1 5:1 etc 
"""

#%% - START OF DATA LOADING
#%% - Dependencies
from torchvision import transforms
from DataLoader_Functions_V1 import initialise_data_loader
       
#%% - Data Loader User Inputs
batch_size = 10            #Data Loader # of Images to pull per batch (add a check to make sure the batch size is smaller than the total number of images in the path selected)
dataset_title = "S_Dataset 4"
data_path = "C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/Yr 3 Project/Circular and Spherical Dummy Datasets/" #"C:/Users/Student/Desktop/fake im data/"  #"/local/path/to/the/images/"

#%% - Advanced Data Loader Settings
debug_loader_batch = 1     #(Default = 0 = [OFF]) //INPUT 0 or 1//   #Setting debug loader batch will print to user the images taken in by the dataoader in this current batch and print the corresponding labels
plot_every_other = 1       #(Default = 1) //MUST BE INTEGER INPUT//  #If debug loader batch is enabled this sets the interval for printing for user, 1 is every single img in the batch, 2 is every other img, 5 is every 5th image etc 
batch_size_protection = 1  #(Default = 1 = [ON]) //INPUT 0 or 1//    #WARNING if turned off, debugging print will cause and exeption due to the index growing too large in the printing loop (img = train_features[i])

#%% - Data Loader Preparation Transforms 
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

#%% - Initialise Data Loader
initialise_data_loader(dataset_title, data_path, batch_size, train_transforms, test_transforms, debug_loader_batch, plot_every_other, batch_size_protection)

# dataset_title, data_path, batch_size, train_transforms, test_transforms, debug_loader_batch = 0, plot_every_other = 1, batch_size_protection = 1
#%% - END OF DATA LOADING
