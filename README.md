# DeepClean - [PyTorch Convolutional Neural Network Architecture to Perform Noise Suppression for LHCb Torch Detector]
# Authors: Adill Al-Ashgar & Max Carter
# Physics Department - University of Bristol 


# Motivation / Background:
One of the four largest experiments on the LHC, the LHCb's (Large Hadron Collider Beauty) main focus is in investigating
baryon anti-baryon asymmetry, responsible for the matter dominated universe we inhabit today.

As part of the planned upgrade for the LHCb in 2022, a new detector is proposed to be added called TORCH (Time Of internally Reflected Cherenkov light) 
which is closely related  to the PANDA DIRC and Belle TOP detectors, combining timing information with DIRC-style reconstruction, but aiming 
for higher TOF resolution, 10–15 ps (per track). The detecttor is sensetive to incoming photons, it is arranged in a grid format, its output is a list of 
grid coocrdinates which detected photons

A current challenge is in reconstructing detected photons path data, it is a computationally costly process and the data flow is incredible (Xinclude data rates here from the 40Mhz streamX). 
All detections are currently reconstructed. However, once reconstructed the events are filtered down to those that correlated to a particular event, a large 
proportion of the detected photons (Xquote average percentage of noise? or sig to noise ratioX) are noise from other collisions or the detector and electronic subsystem.

Our desire is to reduce the number of signals that require path reconstruction by using a neural network to detect the signal, only passing these 
points on to the reconstruction algorithm saving the computation time spent on reconstructing the noise signals only to throw them away.

This is critical in the efficiency of the processing pipeline as the LHC moves into its high luminosity upgrade where data rates will be further increased.

# Quick Start for Denoising
To denoise images they must first be numpy arrays with shape 88 x 128.
Use the DC3D_V3 Denoiser file located in the folder DC3D_V3.
To use the file all you need to do is input the file dir to the foldr that contains the images to be denoised, and
spcify the output dir where the resulting denoised images should be saved. 
Then run the file.



# Quick start for Training
Using the settins guide below set the trainer settings to the desired values.
Then run the file DC3D_V3_Trainer 3.py



# DC3D Settings
Program Settings

This section contains the inputs that the user should provide to the program. These inputs are as follows:

    dataset_title: a string representing the title of the dataset that the program will use.
    model_save_name: a string representing the name of the model that the program will save.
    time_dimension: an integer representing the number of time steps in the data.
    reconstruction_threshold: a float representing the threshold for 3D reconstruction; values below this confidence level are discounted.

Hyperparameter Settings

This section contains the hyperparameters that the user can adjust to train the model. These hyperparameters are as follows:

    num_epochs: an integer representing the number of epochs for training.
    batch_size: an integer representing the number of images to pull per batch.
    latent_dim: an integer representing the number of nodes in the latent space, which is the bottleneck layer.
    learning_rate: a float representing the optimizer learning rate.
    optim_w_decay: a float representing the optimizer weight decay for regularization.
    dropout_prob: a float representing the dropout probability.

Dataset Split Settings

This section contains the hyperparameters that control how the dataset is split. These hyperparameters are as follows:

    train_test_split_ratio: a float representing the ratio of the dataset to be used for training.
    val_test_split_ratio: a float representing the ratio of the non-training data to be used for validation as opposed to testing.

Loss Function Settings

This section contains the hyperparameters that control the loss function used in training. These hyperparameters are as follows:

    loss_function_selection: an integer representing the selected loss function; see the program code for the list of options.
    zero_weighting: a float representing the zero weighting for ada_weighted_mse_loss.
    nonzero_weighting: a float representing the nonzero weighting for ada_weighted_mse_loss.
    zeros_loss_choice: an integer representing the selected loss function for zero values in ada_weighted_custom_split_loss.
    nonzero_loss_choice: an integer representing the selected loss function for nonzero values in ada_weighted_custom_split_loss.

Image Preprocessing Settings

This section contains the hyperparameters that control image preprocessing. These hyperparameters are as follows:

    signal_points: an integer representing the number of signal points to add.
    noise_points: an integer representing the number of noise points to add.
    x_std_dev: a float representing the standard deviation of the detector's error in the x-axis.
    y_std_dev: a float representing the standard deviation of the detector's error in the y-axis.
    tof_std_dev: a float representing the standard deviation of the detector's error in the time of flight.

Pretraining Settings

This section contains the hyperparameters that control pretraining. These hyperparameters are as follows:

    start_from_pretrained_model: a boolean representing whether to start from a pretrained model.
    load_pretrained_optimser: a boolean representing whether to load the pretrained optimizer.
    pretrained_model_path: a string representing the path to the saved full state dictionary for pretraining.

Normalisation Settings

This section contains the hyperparameters that control normalization. These hyperparameters are as follows:

    simple_norm_instead_of_custom: a boolean representing whether to use simple normalization instead of custom normalization.
    all_norm_off: a boolean representing whether to use any input normalization.
    simple_renorm: a boolean representing whether to use simple output renormalization instead of custom output renormal





# Code Contents:
01 - Simple Circular & Spherical Dataset Generator
Genrates a simple dataset of circular and spherical data, used to test the autoencoder architecture.

02 - Dataset Validator
Used to validate the dataset generated in 01
  

   
03 - 2D & 3D DataLoader
   - DataLoader_Functions_V1
   
04 - Convolution Layer Output Size Calculator
 Calcaultes the output dimensions of a convolution layer, works for 2D and 3D convolutions and thier corresponding transpose layers.


05 - 2D Conv Autoencoder - Simple Data
   - DataLoader_Functions_V1
   
06 - 3D Conv Autoencoder - Simple Data
   - DataLoader_Functions_V1