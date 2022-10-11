# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 04:20:57 2022
DeepClean_DataLoader v0.0.2
@authors: Adill Al-Ashgar & Max Carter

#############
#Takes in a batch size, a data path, debugging choice, and debugging step size

#Returns two dataloaders prepared with a randomised?(not yet) batch of images from the path. 
#trainloader and testloader
#Also prints the images in the batch to user, if debug mode is on (ie = 1). debugging step size sets for how many images 1 is printed, in form (step_size:1) ie 1:1 2:1 5:1 etc 
#############

"""

#%% - Dependencies
import torch
import matplotlib.pyplot as plt
from torchvision import transforms, datasets


#%% - User Inputs
data_path = "C:/Users/Student/Desktop/fake im data/"  #"/local/path/to/the/images/"

batch_size = 4             #Data Loader # of Images to pull per batch (add a check to make sure the batch size is smaller than the total number of images in the path selected)
debug_loader_batch = 0     #(Default = 0 = [OFF]) //INPUT 0 or 1// Setting debug loader batch will print to user the images taken in by the dataoader in this current batch and print the corresponding labels
plot_every_other = 1       #(Default = 1) //MUST BE INTEGER INPUT// If debug loader batch is enabled this sets the interval for printing for user, 1 is every single img in the batch, 2 is every other img, 5 is every 5th image etc 


#%% - Data Preparation  #!!!Perhaps these should be passed ino the loader as user inputs, that allows for ease of changing between differnt tranforms in testing without having to flip to the data loader code
train_transforms = transforms.Compose([#transforms.RandomRotation(30),         #Compose is required to chain together multiple transforms in serial 
                                       #transforms.RandomResizedCrop(224),
                                       #transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor()])                #other transforms can be dissabled but to tensor must be left enabled

test_transforms = transforms.Compose([#transforms.Resize(255),
                                      #transforms.CenterCrop(224),
                                      transforms.ToTensor()])



#%% - Data Loading
train_data = datasets.ImageFolder(data_path,transform=train_transforms)   #Training data and test data can pull from the same source, but then have differnt transforms applied to each ie there might be noise added to one ?  or maybe we will be given clean data? or perhaps we just have to work with the bad data and work out how to make a good loss function                                    
test_data = datasets.ImageFolder(data_path,transform=test_transforms)
#specific_folder_data = datasets.ImageFolder(data_path + '/specific folder name', transform=test_transforms)   #If we need to call data from a specific subfolder

trainloader = torch.utils.data.DataLoader(train_data,batch_size=batch_size)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

#%% - Debugging Outputs
if debug_loader_batch == 1:
    train_features, train_labels = next(iter(trainloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    
    for i in range (0, batch_size, plot_every_other):   # Display image and label.
        print ("Image #",i+1)
        img = train_features[i].squeeze()
        label = train_labels[i]
        plt.imshow(img.T, cmap="gray")   #!!!fix the need for img.T which is the transpose, as it flips the image, 
        plt.show()
        print(f"Label: {label}")

