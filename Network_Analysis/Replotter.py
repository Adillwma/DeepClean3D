# -*- coding: utf-8 -*-
# Build created on Wednesday May 6th 2023
# Author: Adill Al-Ashgar
# University of Bristol
# @Adill: adillwmaa@gmail.co.uk - ex18871@bristol.ac.uk


import os 
import re
import torch
import numpy as np  
from tqdm import tqdm  # Progress bar
import matplotlib.pyplot as plt     
import pickle
import pandas as pd

# User Inputs
raw_data_path = r"N:\Audio Autoencoder Results\kernal experiments93!! - Training Results\Raw_Data_Output\\"    # Set the directory path where the raw plot data files are located
epoch_selection = 5
sample_rate= 48000

# Helper functions
def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
    #plt.show(block=False)
    plt.show()

def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    #plt.show(block=False)
    plt.show()

def compare_waveforms(waveform1, waveform2, sample_rate):
    waveform1 = waveform1.numpy()
    waveform2 = waveform2.numpy()

    num_channels1, num_frames1 = waveform1.shape
    num_channels2, num_frames2 = waveform2.shape

    time_axis1 = torch.arange(0, num_frames1) / sample_rate
    time_axis2 = torch.arange(0, num_frames2) / sample_rate

    figure, axes = plt.subplots(max(num_channels1, num_channels2), 1)

    if num_channels1 == 1:
        axes = [axes]

    for c in range(num_channels1):
        axes[c].plot(time_axis1, waveform1[c], linewidth=1)
        axes[c].grid(True)
        axes[c].set_ylabel(f"Channel 1")

    if num_channels2 > 0:
        for c in range(num_channels2):
            axes[c].plot(time_axis2, waveform2[c], linewidth=1)
            axes[c].grid(True)
            axes[c].set_ylabel(f"Channel 2")

    figure.suptitle("Waveform Comparison")
    plt.show()

### Plotting Functions
### Waveform Plotting Function
def plot_ae_outputs_den(clean_list, label_list, noised_list, recovered_list, epoch, num_to_plot):       #Defines a function for plotting the output of the autoencoder. And also the input + clean training data? Function takes inputs, 'encoder' and 'decoder' which are expected to be classes (defining the encode and decode nets), 'n' which is the number of ?????Images in the batch????, and 'noise_factor' which is a multiplier for the magnitude of the added noise allowing it to be scaled in intensity.  
    """
    n is the number of images to plot
    """

    loop  = tqdm(range(num_to_plot), desc='Plotting Waveform Comparisons', leave=False, colour="green") # Creates a progress bar for the batches

    ### Input/Output Comparison Plots 
    plt.figure(figsize=(16,9))                                      #Sets the figure size
        
    for i in loop:                                                #Runs for loop where 'i' itterates over 'n' total values which range from 0 to n-1
        audio_clean = clean_list[i]
        audio_labels = label_list[i]
        audio_noised = noised_list[i]
        audio_recovered = recovered_list[i]

        ### Following section generates the img plots for the original(labels), noised, and denoised data)
        waveform = audio_clean.squeeze(0).numpy()
        _, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / sample_rate

        #Following section creates the noised image data drom the original clean labels (images)   
        ax = plt.subplot(4,num_to_plot,i+1)                                       #Creates a number of subplots for the 'Original images??????' i.e the labels. the position of the subplot is i+1 as it falls in the first row
        plt.plot(time_axis, waveform.squeeze(0), linewidth=1)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (unit?)')
        plt.ylim(-1.0, 1.0)
        plt.grid(True)     
        if i == num_to_plot//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
            ax.set_title('EPOCH %s \nOriginal Audio' %(epoch))               #When above condition is reached, the plots title is set                                   #When above condition is reached, the plots title is set

        ax = plt.subplot(4, num_to_plot, i + 1 + num_to_plot)                                   
        waveform = audio_labels.squeeze(0).numpy()
        plt.plot(time_axis, waveform.squeeze(0), linewidth=1)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (unit?)')
        plt.grid(True)    
        plt.ylim(-1.0, 1.0) 
        if i == num_to_plot//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
            ax.set_title('Audio Labels')               #When above condition is reached, the plots title is set                                   #When above condition is reached, the plots title is set

        ax = plt.subplot(4, num_to_plot, i + 1 + num_to_plot + num_to_plot)                                   #Creates a number of subplots for the 'Corrupted images??????' i.e the labels. the position of the subplot is i+1+n as it falls in the second row
        waveform = audio_noised.squeeze(0).numpy()
        plt.plot(time_axis, waveform.squeeze(0), linewidth=1)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (unit?)')
        plt.grid(True)     
        plt.ylim(-1.0, 1.0) 
        if i == num_to_plot//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
            ax.set_title('Corrupted Audio')                                  #When above condition is reached, the plots title is set

        ax = plt.subplot(4, num_to_plot, i + 1 + num_to_plot + num_to_plot + num_to_plot)                               #Creates a number of subplots for the 'Reconstructed images??????' i.e the labels. the position of the subplot is i+1+n+n as it falls in the third row
        waveform = audio_recovered.squeeze(0).numpy()
        plt.plot(time_axis, waveform.squeeze(0), linewidth=1)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (unit?)')
        plt.grid(True)
        plt.ylim(-1.0, 1.0)
        if i == num_to_plot//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
            ax.set_title('Reconstructed Audio')                             #When above condition is reached, the plots title is set 

    plt.subplots_adjust(left=0.1,              #Adjusts the exact layout of the plots including whwite space round edges
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.1, 
                    hspace=0.3)     
    plt.show()

### Spectral Plotting Function
def plot_ae_outputs_den_spectral(clean_list, label_list, noised_list, recovered_list, epoch, num_to_plot):       #Defines a function for plotting the output of the autoencoder. And also the input + clean training data? Function takes inputs, 'encoder' and 'decoder' which are expected to be classes (defining the encode and decode nets), 'n' which is the number of ?????Images in the batch????, and 'noise_factor' which is a multiplier for the magnitude of the added noise allowing it to be scaled in intensity.  
    """
    n is the number of images to plot
    """


    loop  = tqdm(range(num_to_plot), desc='Plotting Spectral Comparisons', leave=False, colour="green") # Creates a progress bar for the batches

    n = num_to_plot

    ### Input/Output Comparison Plots 
    plt.figure(figsize=(16,9))                                      #Sets the figure size
    for i in loop:                                                #Runs for loop where 'i' itterates over 'n' total values which range from 0 to n-1
        audio_clean = clean_list[i]
        audio_labels = label_list[i]
        audio_noised = noised_list[i]
        audio_recovered = recovered_list[i]

        #Following section creates the noised image data drom the original clean labels (images)   
        ax = plt.subplot(4,n,i+1)                                       #Creates a number of subplots for the 'Original images??????' i.e the labels. the position of the subplot is i+1 as it falls in the first row
        waveform = audio_clean.squeeze(0).numpy()
        plt.specgram(waveform.squeeze(0), Fs=sample_rate)
        plt.xlabel('Time [sec]')
        plt.ylabel('Frequency [Hz]')
        plt.grid(alpha=0.2)
        if i == n-1:
            plt.colorbar()
        if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
            ax.set_title('EPOCH %s \nOriginal Audio' %(epoch))               #When above condition is reached, the plots title is set                                   #When above condition is reached, the plots title is set

        ax = plt.subplot(4, n, i + 1 + n)                                   
        waveform = audio_labels.squeeze(0).numpy()
        plt.specgram(waveform.squeeze(0), Fs=sample_rate)
        plt.xlabel('Time [sec]')
        plt.ylabel('Frequency [Hz]')
        plt.grid(alpha=0.2)
        if i == n-1:
            plt.colorbar()
        if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
            ax.set_title('Audio Labels')               #When above condition is reached, the plots title is set                                   #When above condition is reached, the plots title is set

        ax = plt.subplot(4, n, i + 1 + n + n)                                   #Creates a number of subplots for the 'Corrupted images??????' i.e the labels. the position of the subplot is i+1+n as it falls in the second row
        waveform = audio_noised.squeeze(0).numpy()
        plt.specgram(waveform.squeeze(0), Fs=sample_rate)
        plt.xlabel('Time [sec]')
        plt.ylabel('Frequency [Hz]')
        plt.grid(alpha=0.2)
        if i == n-1:
            plt.colorbar()
        if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
            ax.set_title('Corrupted Audio')                                  #When above condition is reached, the plots title is set

        ax = plt.subplot(4, n, i + 1 + n + n + n)                               #Creates a number of subplots for the 'Reconstructed images??????' i.e the labels. the position of the subplot is i+1+n+n as it falls in the third row
        waveform = audio_recovered.squeeze(0).numpy()
        plt.specgram(waveform.squeeze(0), Fs=sample_rate)
        plt.xlabel('Time [sec]')
        plt.ylabel('Frequency [Hz]')
        plt.grid(alpha=0.2)
        if i == n-1:
            plt.colorbar()
        if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
            ax.set_title('Reconstructed Audio')                             #When above condition is reached, the plots title is set 

    plt.subplots_adjust(left=0.1,              #Adjusts the exact layout of the plots including whwite space round edges
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.1, 
                    hspace=0.3)     
    
    plt.show()

### Histogram Plotting Function
def plot_ae_outputs_den_histogram(clean_list, label_list, noised_list, recovered_list, epoch, num_to_plot):       #Defines a function for plotting the output of the autoencoder. And also the input + clean training data? Function takes inputs, 'encoder' and 'decoder' which are expected to be classes (defining the encode and decode nets), 'n' which is the number of ?????Images in the batch????, and 'noise_factor' which is a multiplier for the magnitude of the added noise allowing it to be scaled in intensity.  
    """
    n is the number of images to plot
    """

    loop  = tqdm(range(num_to_plot), desc='Plotting Spectral Comparisons', leave=False, colour="green") # Creates a progress bar for the batches

    n = num_to_plot

    ### Input/Output Comparison Plots 
    plt.figure(figsize=(16,9))                                      #Sets the figure size
    for i in loop:                                                #Runs for loop where 'i' itterates over 'n' total values which range from 0 to n-1
        audio_clean = clean_list[i]
        audio_labels = label_list[i]
        audio_noised = noised_list[i]
        audio_recovered = recovered_list[i]

        ### Following section generates the img plots for the original(labels), noised, and denoised data)
        #Following section creates the noised image data drom the original clean labels (images)   
        axW = plt.subplot(4,n,i+1)                                       #Creates a number of subplots for the 'Original images??????' i.e the labels. the position of the subplot is i+1 as it falls in the first row
        waveform = audio_clean.squeeze(0).squeeze(0).T
        axW.hist(waveform, bins=100, density=True, histtype='step', color='black')  
        axW.grid(alpha=0.2)  
        if i == 0:
            axW.set_ylabel('Density')
        if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
            axW.set_title('EPOCH %s \nOriginal Audio' %(epoch))               #When above condition is reached, the plots title is set                                   #When above condition is reached, the plots title is set

        axE = plt.subplot(4, n, i + 1 + n)                                   
        waveform = audio_labels.squeeze(0).squeeze(0).T
        axE.hist(waveform, bins=100, density=True, histtype='step', color='black')
        axE.grid(alpha=0.2)
        if i == 0:
            axE.set_ylabel('Density')
        if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
            axE.set_title('Audio Labels')               #When above condition is reached, the plots title is set                                   #When above condition is reached, the plots title is set

        axR = plt.subplot(4, n, i + 1 + n + n)                                   #Creates a number of subplots for the 'Corrupted images??????' i.e the labels. the position of the subplot is i+1+n as it falls in the second row
        waveform = audio_noised.squeeze(0).squeeze(0).T
        axR.hist(waveform, bins=100, density=True, histtype='step', color='black')
        axR.grid(alpha=0.2)
        if i == 0:
            axR.set_ylabel('Density')

        if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
            axR.set_title('Corrupted Audio')                                  #When above condition is reached, the plots title is set

        axT = plt.subplot(4, n, i + 1 + n + n + n)                               #Creates a number of subplots for the 'Reconstructed images??????' i.e the labels. the position of the subplot is i+1+n+n as it falls in the third row
        waveform = audio_recovered.squeeze(0).squeeze(0).T
        axT.hist(waveform, bins=100, density=True, histtype='step', color='black')
        axT.set_xlabel('Amplitude')
        axT.set_xlim(-1, 1)
        axT.grid(alpha=0.2)
        if i == 0:
            axT.set_ylabel('Density')     
        if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
            axT.set_title('Reconstructed Audio')                             #When above condition is reached, the plots title is set 
   
    plt.tight_layout()                          #Adjusts the exact layout of the plots including whwite space round edges

    plt.show()

### Plotting the difference between the denoised and original data
def plot_ae_outputs_den_difference(clean_list, label_list, noised_list, recovered_list, epoch, num_to_plot):       #Defines a function for plotting the output of the autoencoder. And also the input + clean training data? Function takes inputs, 'encoder' and 'decoder' which are expected to be classes (defining the encode and decode nets), 'n' which is the number of ?????Images in the batch????, and 'noise_factor' which is a multiplier for the magnitude of the added noise allowing it to be scaled in intensity.  
    """
    n is the number of images to plot
    """

    loop  = tqdm(range(num_to_plot), desc='Plotting Spectral Comparisons', leave=False, colour="green") # Creates a progress bar for the batches

    n = num_to_plot

    ### Input/Output Comparison Plots 
    plt.figure(figsize=(16,9))                                      #Sets the figure size
    for i in loop:                                                #Runs for loop where 'i' itterates over 'n' total values which range from 0 to n-1

        audio_clean = clean_list[i]
        audio_labels = label_list[i]
        audio_noised = noised_list[i]
        audio_recovered = recovered_list[i]

        difference = audio_clean - audio_recovered
        difference = difference.squeeze(0).squeeze(0).T
        waveform = difference
        num_frames = int(len(waveform))
        time_axis = torch.arange(0, num_frames) / sample_rate

        axW = plt.subplot(4,n,i+1)                                       #Creates a number of subplots for the 'Original images??????' i.e the labels. the position of the subplot is i+1 as it falls in the first row
        axW.plot(time_axis, audio_clean.squeeze(0).squeeze(0).T, linewidth=1)
        axW.plot(time_axis, audio_recovered.squeeze(0).squeeze(0).T, linewidth=1)
        axW.grid(alpha=0.4)  
        axW.set_ylim(-1.0, 1.0)
        axW.set_xlabel('Time (s)')
        if i == 0:
            axW.set_ylabel('Amplitude (unit?)')
        if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
            axW.set_title('EPOCH %s \nWaveform Comparison' %(epoch))               #When above condition is reached, the plots title is set                                   #When above condition is reached, the plots title is set

        axE = plt.subplot(4, n, i + 1 + n)                                   
        axE.plot(time_axis, waveform, linewidth=1)
        axE.set_xlabel('Time (s)')
        axE.set_ylim(-1.0, 1.0)
        axE.grid(alpha=0.4)  
        if i == 0:
            axE.set_ylabel('Amplitude (unit?)')
        if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
            axE.set_title('Waveform Difference')               #When above condition is reached, the plots title is set                                   #When above condition is reached, the plots title is set

        axR = plt.subplot(4, n, i + 1 + n + n)                                   #Creates a number of subplots for the 'Corrupted images??????' i.e the labels. the position of the subplot is i+1+n as it falls in the second row
        axR.specgram(waveform, Fs=sample_rate)
        if i == 0:
            axR.set_ylabel('Time (s)')
        axR.set_xlabel('Frequency [Hz]')
        if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
            axR.set_title('Spectrogram Difference')                                  #When above condition is reached, the plots title is set

        axT = plt.subplot(4, n, i + 1 + n + n + n)                               #Creates a number of subplots for the 'Reconstructed images??????' i.e the labels. the position of the subplot is i+1+n+n as it falls in the third row
        axT.hist(waveform, bins=100, density=True, histtype='step', color='black')
        axT.set_xlabel('Amplitude')
        axT.grid(alpha=0.4)  
        axT.set_xlim(-1.0, 1.0)
        if i == 0:
            axT.set_ylabel('Density')     
        if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
            axT.set_title('Histogram Difference')                             #When above condition is reached, the plots title is set 
        
    plt.tight_layout()                          #Adjusts the exact layout of the plots including whwite space round edges
    plt.show()




# Load the raw plot data from the dictionary file saved earlier
def load_plot_data(model_save_name, epoch):
    plot_data = torch.load(f'{model_save_name} - Plot Data - Epoch {epoch}.pt')
    return plot_data

def load_variable(path):

    extension = path.split('.')[-1]
    
    if extension == 'pkl':
        with open(path, 'rb') as file:
            return pickle.load(file)
        
    elif extension == 'npy':
        return np.load(path)
    
    elif extension == 'pt':
        return torch.load(path)
    
    elif extension == 'csv':
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            return df.iloc[:, 0].values.tolist()
        else:
            return df.values.tolist()
    else:
        raise ValueError("Unsupported file format.")




# Example usage:
settings = load_variable(raw_data_path + f'settings dict_dict.pkl')
print("Loaded settings dictionary")

history_da = load_variable(raw_data_path + f'history_da_dict.pkl')
print("Loaded history dictionary")

epochs_range = load_variable(raw_data_path + f'epochs_range_array.npy')
print("Loaded epochs range array")



clean_list = load_variable(raw_data_path + f'{epoch_selection}_clean_list_forcepkl.pkl')
print("Loaded settings dictionary")

label_list = load_variable(raw_data_path + f'{epoch_selection}_label_list_forcepkl.pkl')
print("Loaded settings dictionary")

noised_list = load_variable(raw_data_path + f'{epoch_selection}_noised_list_forcepkl.pkl')
print("Loaded settings dictionary")

recovered_list = load_variable(raw_data_path + f'{epoch_selection}_recovered_list_forcepkl.pkl')
print("Loaded settings dictionary")


try:
    hyperparam_to_optimise = history_da['Hyperparameter_to_optimise']
except:
    print("No hyperparameter to optimise found in history dictionary. Setting to None.")

max_epoch_reached = epochs_range[-1]


# GET THESE BOOLEAN VALUES FORM THE PLOT DATA BY CHECKING WHAT PLOTS IT HAS AVAILIBLE TO LOAD
plot_waveform_comparison = True   #BROKEN!!! IF SET TO OFF THEN THE AUDIO SAVE IS NOT WOKRING AND CODE BREAKS!!! FIX   #[default = True]               Generate plot of waveform comparison between input and output
plot_spectral_comparison = True      #[default = False]               Generate plot of spectral comparison between input and output
plot_histogram_comparison = True     #[default = True]               Generate plot of histogram comparison between input and output
plot_audio_difference = True #rename        #[default = True]           Generate plot of audio difference between input and output

plot_train_loss = True               #[default = True]        Generate plot of training loss across epochs
plot_test_loss = True                #[default = True]      Generate plot of test loss across epochs
plot_validation_loss = True          #[default = True]                Generate plot of validation loss across epochs
plot_training_time = True            #[default = True]              Generate plot of training time across epochs

plot_losses_per_epoch_group = True        #[default = True]  #Plots the losses per epoch in a group plot
loss_group_size = 20                      #[default = 50]  #Number of epochs to group together for the extra loss plots if above is set to true

plot_latent_generations = True       #[default = True]               Generate plot of latent space generations
plot_higher_dim = False              #[default = True]         Generate plot of PCA?? and TSNE





print("\nInput Settings:\n")  # Write the input settings to the file
for key, value in settings.items():
    print(f"{key}: {value}\n")



if plot_waveform_comparison: 
    plot_ae_outputs_den(clean_list, label_list, noised_list, recovered_list, epoch_selection, num_to_plot=5)

if plot_spectral_comparison:
    plot_ae_outputs_den_spectral(clean_list, label_list, noised_list, recovered_list, epoch_selection, num_to_plot=5)

if plot_histogram_comparison:
    plot_ae_outputs_den_histogram(clean_list, label_list, noised_list, recovered_list, epoch_selection, num_to_plot=5)

if plot_audio_difference:
    plot_ae_outputs_den_difference(clean_list, label_list, noised_list, recovered_list, epoch_selection, num_to_plot=5)


if plot_train_loss:
    plt.plot(epochs_range, history_da['train_loss']) 
    plt.title("Training loss")   
    plt.xlabel("Epoch number")
    plt.ylabel("Train loss (MSE)")
    plt.grid(alpha=0.2)
    plt.show()

if plot_test_loss:
    plt.plot(epochs_range, history_da['test_loss']) 
    plt.title("Test loss")   
    plt.xlabel("Epoch number")
    plt.ylabel("Test loss (MSE)")
    plt.grid(alpha=0.2)
    plt.show()

if plot_validation_loss:
    plt.plot(history_da['HTO_val'], history_da['val_loss'])   
    plt.title("Validation loss") 
    plt.xlabel(hyperparam_to_optimise)
    plt.ylabel("Val loss (MSE)")
    plt.grid(alpha=0.2)
    plt.show()

if plot_training_time:
    plt.plot(history_da['HTO_val'], history_da['training_time'])   
    plt.title("Training time") 
    plt.xlabel(hyperparam_to_optimise)
    plt.ylabel("Training time (s)")
    plt.grid(alpha=0.2)
    plt.show()

if plot_losses_per_epoch_group: 

    for i in range(0, max_epoch_reached, loss_group_size):

        if plot_train_loss:
            plt.plot(epochs_range[i:i+loss_group_size], history_da['train_loss'][i:i+loss_group_size]) 
            plt.title("Training loss")   
            plt.xlabel("Epoch number")
            plt.ylabel("Train loss (MSE)")
            plt.grid(alpha=0.2)
            plt.xlim(i, i + loss_group_size)
            plt.show()

        if plot_test_loss:
            plt.plot(epochs_range[i:i+loss_group_size], history_da['test_loss'][i:i+loss_group_size]) 
            plt.title("Test loss")   
            plt.xlabel("Epoch number")
            plt.ylabel("Test loss (MSE)")
            plt.grid(alpha=0.2)
            plt.xlim(i, i + loss_group_size)
            plt.show()

        if plot_validation_loss:
            plt.plot(history_da['HTO_val'][i:i+loss_group_size], history_da['val_loss'][i:i+loss_group_size])  
            plt.title("Validation loss")
            plt.xlabel(hyperparam_to_optimise)
            plt.ylabel("Val loss (MSE)")
            plt.grid(alpha=0.2)
            plt.show()

        if plot_training_time:
            plt.plot(history_da['HTO_val'][i:i+loss_group_size], history_da['training_time'][i:i+loss_group_size])   
            plt.title("Training time") 
            plt.xlabel(hyperparam_to_optimise)
            plt.ylabel("Training time (s)")
            plt.grid(alpha=0.2)
            plt.show()




