# -*- coding: utf-8 -*-
"""
DataLoader Functions V1
@author: Adill Al-Ashgar
Created on Tue Nov 15 19:02:32 2022

USER NOTICE!
x Must be inside the root dir of the DataLoader or the Neural Net that calls it.
"""
#%% - Dependencies
import torch
import matplotlib.pyplot as plt
from torchvision import datasets
import os
import numpy as np
from torch.utils.data import random_split

#%%
def train_loader2d(path):   #fix need for two seperate loads, one on each loader
    sample = (np.load(path))
    sample = sample[0]               
    return (sample)

#%%
def test_loader2d(path):
    load = 1 # Set manually, 0 = Blank, no data, 1 = just signal, 2 = just noise, 3 = both, but with differing values (1,2)    #!!! OPION 3 NOT WORKING
    sample = (np.load(path))
    sample2 = np.ma.masked_where(sample[1] == load, sample[1])                   
    return (sample2)

#%%
def train_loader3d(path):   #fix need for two seperate loads, one on each loader
    sample = (np.load(path))
    sample = sample               
    return torch.tensor(sample)

#%%
def test_loader3d(path):
    #load = 1 # Set manually, 0 = Blank, no data, 1 = just signal, 2 = just noise, 3 = both, but with differing values (1,2)    #!!! OPION 3 NOT WORKING
    sample = (np.load(path))
    #sample2 = np.ma.masked_where(sample                   
    return torch.tensor(sample)

#%%
def batch_learning(training_dataset_size, batch_size):
    if batch_size == 1: 
        output = "Stochastic Gradient Descent"
    elif batch_size == training_dataset_size:
        output = "Batch Gradient Descent"        
    else:
        output = "Mini-Batch Gradient Descent"
    return(output) 

#%%
def initialise_data_loader (dataset_title, data_path, batch_size, train_transforms, test_transforms, debug_loader_batch = 0, plot_every_other = 1, batch_size_protection = 1):
    # Input type check, 2D or 3D. Based on dataset foldr name. 3D if folder starts with S_
    if dataset_title.startswith('S_'):
        print("Detected 3D Input")
        circular_or_spherical = 1 #2d or 3d loader. 0 = 2d, 1 = 3d
    else:
        print("Detected 2D Input")
        circular_or_spherical = 0 #2d or 3d loader. 0 = 2d, 1 = 3d  
    
    # - Path images, greater than batch choice? CHECK
    ####check for file count in folder####
    if batch_size_protection == 1:
        files_in_path = os.listdir(data_path + dataset_title + '/Data/') 
        num_of_files_in_path = len(files_in_path)
        learning = batch_learning(num_of_files_in_path, batch_size)
        print("%s files in path." %num_of_files_in_path ,"// Batch size =",batch_size, "\nLearning via: " + learning,"\n")
        if num_of_files_in_path < batch_size:
            print("Error, the path selected has", num_of_files_in_path, "image files, which is", (batch_size - num_of_files_in_path) , "less than the chosen batch size. Please select a batch size less than the total number of images in the directory")
            
            #!!!Need code to make this event cancel the running of program and re ask for user input on batch size or just reask for the batch size
            batch_err_message = "Choose new batch size, must be less than total amount of images in directory", (num_of_files_in_path)
            batch_size = int(input(batch_err_message))  #!!! not sure why input message is printing with wierd brakets and speech marks in the terminal? Investigate

    # - Data Loading
    if circular_or_spherical == 0:
        train_data = datasets.DatasetFolder(data_path + dataset_title, loader=train_loader2d, extensions='.npy', transform=train_transforms)
        test_data = datasets.DatasetFolder(data_path + dataset_title, loader=test_loader2d, extensions='.npy', transform=test_transforms)
    
    else:
        train_data = datasets.DatasetFolder(data_path + dataset_title, loader=train_loader3d, extensions='.npy', transform=train_transforms)
        test_data = datasets.DatasetFolder(data_path + dataset_title, loader=test_loader3d, extensions='.npy', transform=test_transforms)
            
    ###Following section splits the training dataset into two, train_data (to be noised) and valid data (to use in eval)
    m=len(train_data) #Just calculates length of train dataset, m is only used in the next line to decide the values of the split, (4/5 m) and (1/5 m)
    train_data, val_data = random_split(train_data, [int(round((m-m*0.2))), int(round((m*0.2)))])    #random_split(data_to_split, [size of output1, size of output2]) just splits the train_dataset into two parts, 4/5 goes to train_data and 1/5 goes to val_data , validation?
    
    trainloader = torch.utils.data.DataLoader(train_data,batch_size=batch_size)
    validloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size) 
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    # - Debugging Outputs
    if debug_loader_batch == 1:
        train_features, train_labels = next(iter(trainloader))
        print(f"Feature batch shape: {train_features.size()}")
        print(f"Labels batch shape: {train_labels.size()}")

        for i in range (0, batch_size, plot_every_other):   # Display image and label.
            label = train_labels[i]
            #print ("\nImage #",i+1)
            #print(f"Label: {label}")   
    
            ##Checks if data is 2d if so plots 2d image
            if circular_or_spherical == 0:
                img = train_features[i].squeeze()
                plt.imshow(img, cmap="gray")  #.T, cmap="gray")   #!!!fix the need for img.T which is the transpose, as it flips the image, 
                plt.show()
            
            ##Checks if data is 3d if so plots 3d image
            else:           
                hits_3d = np.nonzero(train_features[i].squeeze())
                #print(hits_3d)
                x3d = hits_3d.T[2]
                y3d = hits_3d.T[1]
                z3d = hits_3d.T[0]
                
                fig = plt.figure()               #Plots spherical data
                ax = plt.axes(projection='3d')
                ax.scatter(x3d, y3d, z3d)#, s = signal_hit_size, c = "b") #Plots spherical data in blue
                #ax.scatter(x_sph_noise_data,y_sph_noise_data,z_sph_noise_data, s = noise_hit_size, c = noise_colour) #Plots spherical noise in blue or red depending on the user selection of seperate_noise_colour
                ax.set_xlim(0, 100) #Time resoloution of detector
                ax.set_ylim(0, 88)  #width (px) of detector
                ax.set_zlim(0, 128) #hight (px) of detector
                plt.show()
    

    return(trainloader, testloader, validloader, train_data, test_data, val_data)





"""
#%% - Data Importer
data_dir = 'dataset'
train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
test_dataset  = torchvision.datasets.MNIST(data_dir, train=False, download=True)


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


train_dataset.transform = train_transform       #!!! train_dataset is the class? object 'dataset' it has a subclass called transforms which is the list of transofrms to perform on the dataset when loading it. train_tranforms is the set of chained transofrms we created, this is set to the dataset transforms subclass 
test_dataset.transform = test_transform         #!!! similar to the above but for the test(eval) dataset, check into this for the exact reason for using it, have seen it deone in other ways i.e as in the dataloader.py it is performed differntly. this way seems to be easier to follow

####SECTION MODIFIED END#####



###Following section splits the training dataset into two, train_data (to be noised) and valid data (to use in eval)
m=len(train_dataset) #Just calculates length of train dataset, m is only used in the next line to decide the values of the split, (4/5 m) and (1/5 m)
train_data, val_data = random_split(train_dataset, [int(round((m-m*0.2)), int(round((m*0.2))])    #random_split(data_to_split, [size of output1, size of output2]) just splits the train_dataset into two parts, 4/5 goes to train_data and 1/5 goes to val_data , validation?


###Following section for Dataloaders, they just pull a random sample of images from each of the datasets we now have, train_data, valid_data, and test_data. the batch size defines how many are taken from each set, shuffle argument shuffles them each time?? #!!!
batch_size=256                                                                                #User controll to set batch size for the dataloaders (Hyperparameter)?? #!!!

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)                 #Training data loader, can be run to pull training data as configured
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)                   #Validation data loader, can be run to pull training data as configured
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True)   #Testing data loader, can be run to pull training data as configured. Also is shuffled using parameter shuffle #!!! why is it shuffled?
"""