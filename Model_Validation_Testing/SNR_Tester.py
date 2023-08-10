# -*- coding: utf-8 -*-
"""
Image Quality Tester
@author: Adill Al-Ashgar
Created on Fri Feb 10 15:32:02 2023
"""
#%% - Dependencies
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
import os

from DC3D_V3_Denoiser import DeepClean3D
from DC3D_Autoencoder_V1 import Encoder, Decoder

def load_numpy_files(folder_path):

    for file in os.listdir(folder_path):
        if file.endswith(".npy"):
            numpy_array = np.load(os.path.join(folder_path, file))
            if numpy_array.ndim == 3:
                torch_array = torch.from_numpy(numpy_array[0]) 
                print("The input image array has 3 dimensions!")
            elif numpy_array.ndim == 2:
                torch_array = torch.from_numpy(numpy_array) 
            else:
                print("The array has a incorrect number of dimensions")
                break
        break
    return torch_array

#%% Helper functions for evaluating performance of the denoiser:
#Signal to Noise Ratio (SNR)
def SNR(clean_input, noised_target):
    """
    Calculates the Signal to Noise Ratio (SNR) of a given signal and noise.
    SNR is defined as the ratio of the magnitude of the signal and the magnitude of the noise.
    
    Args:
    clean_input (torch.Tensor): The original signal.
    noised_target (torch.Tensor): The signal with added noise.
    
    Returns:
    The calculated SNR value.    
    """
    signal_power = torch.mean(torch.pow(clean_input, 2))

    noise = noised_target - clean_input
    noise_power = torch.mean(torch.pow(noise, 2))

    snr = 10 * torch.log10(signal_power / noise_power)
    return (snr.numpy())

#Peak Signal-to-Noise Ratio (PSNR):
def PSNR(clean_input, noised_target, time_dimension):
    """
    Calculates the Peak Signal to Noise Ratio (PSNR) of a given image and its recovered version. PSNR is defined as the ratio of 
    the maximum possible power of a signal and the power of corrupting noise. The measure focuses on how well high-intensity 
    regions of the image come through the noise, and pays much less attention to low intensity regions.

    Args:
    clean_input (torch.Tensor): The original image.
    noised_target (torch.Tensor): The recovered image.
    
    Returns:
    The calculated PSNR value.
    """
    mse = torch.mean(torch.pow(clean_input - noised_target, 2))   #Finds the mean square error
    max_value = time_dimension
    psnr = 10 * torch.log10((max_value**2) / mse)
    return (psnr.numpy())

#Mean Squared Error (MSE):
def MSE(clean_input, noised_target):
    """
    Mean Squared Error (MSE)

    Args:
    clean_input (torch.Tensor): The original image.
    noised_target (torch.Tensor): The recovered image.
    
    Returns:
    The calculated Mean Squared Error value.
    """
    mse = torch.mean(torch.pow(clean_input - noised_target, 2))
    return (mse.numpy())

#Mean Absolute Error (MAE):
def MAE(clean_input, noised_target):
    """
    Mean Absolute Error (MAE)

    Args:
    clean_input (torch.Tensor): The original image.
    noised_target (torch.Tensor): The recovered image.
    
    Returns:
    The calculated Mean Absolute Error value.
    """
    return (torch.mean(torch.abs(clean_input - noised_target))).numpy()

#Structural Similarity Index (SSIM):
def SSIM(clean_input, noised_target):
    """
    Structural Similarity Index Measure (SSIM), is a perceptual quality index that measures the structural similarity between 
    two images. SSIM takes into account the structural information of an image, such as luminance, contrast, and structure, 
    and compares the two images based on these factors. SSIM is based on a three-part similarity metric that considers the 
    structural information in the image, the dynamic range of the image, and the luminance information of the image. SSIM is 
    designed to provide a more perceptually relevant measure of image similarity than traditional metrics such as Mean Squared 
    Error or Peak Signal-to-Noise Ratio.

    Args:
    clean_input (torch.Tensor): The original image.
    noised_target (torch.Tensor): The recovered image.
    
    Returns:
    The calculated Structural Similarity Index Measure value.
    """
    mu1 = torch.mean(clean_input)
    mu2 = torch.mean(noised_target)
    sigma1_sq = torch.mean(clean_input ** 2) - mu1 ** 2
    sigma2_sq = torch.mean(noised_target ** 2) - mu2 ** 2
    sigma12 = torch.mean(clean_input * noised_target) - mu1 * mu2
    return ((2 * mu1 * mu2 + 0.0001) * (2 * sigma12 + 0.0001) / (mu1 ** 2 + mu2 ** 2 + 0.0001) / (sigma1_sq + sigma2_sq + 0.0001)).numpy()

#Correlation Coefficent
def correlation_coeff(clean_input, noised_target):
    """
    Correlation coefficient is a scalar value that measures the linear relationship between two signals. The correlation 
    coefficient ranges from -1 to 1, where a value of 1 indicates a perfect positive linear relationship, a value of -1 indicates 
    a perfect negative linear relationship, and a value of 0 indicates no linear relationship between the two signals. Correlation 
    coefficient only measures the linear relationship between two signals, and does not take into account the structure of the signals.

    ρ = cov(x,y) / (stddev(x) * stddev(y))

    The function first computes the mean and standard deviation of each tensor, and then subtracts the mean from each element 
    to get the centered tensors x_center and y_center. The numerator is the sum of the element-wise product of x_center 
    and y_center, and the denominator is the product of the standard deviations of the two centered tensors multiplied by the 
    number of elements in the tensor. The function returns the value of the correlation coefficient ρ as the ratio of the numerator 
    and denominator.

    Args:
    clean_input (torch.Tensor): The original image.
    noised_target (torch.Tensor): The recovered image.
    
    Returns:
    The calculated correlation coefficient value.
    """
    clean_mean = clean_input.mean()
    noised_mean = noised_target.mean()
    clean_std = clean_input.std()
    noised_std = noised_target.std()
    clean_center = clean_input - clean_mean
    noised_center = noised_target - noised_mean
    numerator = (clean_center * noised_center).sum()
    denominator = clean_std * noised_std * clean_input.numel()
    return (numerator / denominator).numpy()

#Mutual Information:
def MutualInformation(clean_input, noised_target):
    H_x = -torch.sum(clean_input * torch.log2(clean_input + 1e-8))
    H_y = -torch.sum(noised_target * torch.log2(noised_target + 1e-8))
    H_xy = -torch.sum(torch.mul(clean_input, torch.log2(torch.mul(clean_input, noised_target) + 1e-8)))
    return (H_x + H_y - H_xy).numpy()

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def accuracy(output, target):
    """
    Calculates accuracy of a given output and target tensor.
    :param output: (torch.Tensor) Output tensor of shape (batch_size, num_classes)
    :param target: (torch.Tensor) Target tensor of shape (batch_size,)
    :return: (float) Accuracy score.
    """
    with torch.no_grad():
        # Convert the predicted class from one-hot encoded form to integer form
        pred = output.argmax(dim=1)
        # Calculate the number of correctly predicted examples
        correct = (pred == target).sum().item()
        # Calculate the total number of examples
        total = target.shape[0]
        # Return the accuracy score as the ratio of correctly predicted examples to total examples
        return correct / total
    
def precision(output, target):
    """
    Calculates precision of a given output and target tensor.
    :param output: (torch.Tensor) Output tensor of shape (batch_size, num_classes)
    :param target: (torch.Tensor) Target tensor of shape (batch_size,)
    :return: (float) Precision score.
    """
    with torch.no_grad():
        # Convert the predicted class from one-hot encoded form to integer form
        pred = output.argmax(dim=1)
        # Calculate the number of correctly predicted positive examples
        true_positive = ((pred == 1) & (target == 1)).sum().item()
        # Calculate the number of predicted positive examples
        predicted_positive = (pred == 1).sum().item()
        # Return the precision score as the ratio of correctly predicted positive examples to predicted positive examples
        return true_positive / predicted_positive if predicted_positive != 0 else 0
    
def recall(output, target):
    """
    Calculates recall of a given output and target tensor.
    :param output: (torch.Tensor) Output tensor of shape (batch_size, num_classes)
    :param target: (torch.Tensor) Target tensor of shape (batch_size,)
    :return: (float) Recall score.
    """
    with torch.no_grad():
        # Convert the predicted class from one-hot encoded form to integer form
        pred = output.argmax(dim=1)
        # Calculate the number of correctly predicted positive examples
        true_positive = ((pred == 1) & (target == 1)).sum().item()
        # Calculate the number of actual positive examples
        actual_positive = (target == 1).sum().item()
        # Return the recall score as the ratio of correctly predicted positive examples to actual positive examples
        return true_positive / actual_positive if actual_positive != 0 else 0

def f1_score(output, target):
    """
    Compute the F1 score for the output and target tensors.
    
    Parameters:
    - output (torch.Tensor): Tensor of predicted values
    - target (torch.Tensor): Tensor of ground truth values
    
    Returns:
    - f1 (float): The computed F1 score
    """
    with torch.no_grad():
        # Compute the precision of the model's predictions
        precision_val = precision(output, target)
        
        # Compute the recall of the model's predictions
        recall_val = recall(output, target)
        
        # Compute the F1 score as 2 * (precision * recall) / (precision + recall + 1e-15)
        # The 1e-15 term is added to avoid division by zero
        f1 = 2 * (precision_val * recall_val) / (precision_val + recall_val + 1e-15)
        
        # Return the computed F1 score
        return f1


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


#Combine all performance metrics into simple test script
def quantify_performance(clean_input, noised_target, cleaned_image, label):
    performance = {}
    performance['SNR'] = SNR(clean_input, noised_target), SNR(clean_input, cleaned_image)
    performance['PSNR'] = PSNR(clean_input, noised_target, time_dimension), PSNR(clean_input, cleaned_image, time_dimension)
    performance['MSE'] = MSE(clean_input, noised_target), MSE(clean_input, cleaned_image)
    performance['MAE'] = MAE(clean_input, noised_target), MAE(clean_input, cleaned_image)
    performance['SSIM'] = SSIM(clean_input, noised_target), SSIM(clean_input, cleaned_image)
    performance['Correlation Coefficient'] = correlation_coeff(clean_input, noised_target), correlation_coeff(clean_input, cleaned_image)
    performance['Mutual Information'] = MutualInformation(clean_input, noised_target), MutualInformation(clean_input, cleaned_image)    #BROKEN
    
    #Iterativly plotting results
    print("\n",label)
    for item in performance:
        print(item, performance[item])
    
    #Return dictionary with results for image
    return performance





#%% - Noise functions
#White Additive Noise :
def add_white_noise(clean_input, time_dimension):
    noise = (torch.normal(time_dimension/2,time_dimension/2, clean_input.shape)).clip(0,time_dimension)
    noisy_images = clean_input + noise
    #print("add_white_noise", torch.max(noisy_images))
    return (noisy_images)

#Masking Noise :
def add_masking_noise(clean_input, time_dimension):         ###Simulates pixels failing to fire when struck by photon
    a = 0.7*torch.ones(clean_input.shape)
    noisy_images = clean_input*torch.bernoulli(a)
    #print("add_masking_noise", torch.max(noisy_images))
    return (noisy_images)

#Poisson Noise :
def add_poisson_noise(clean_input, time_dimension):
    a = time_dimension*torch.ones(clean_input.shape)
    p = torch.poisson(a)
    p_norm = p/p.max()
    noisy_images = (clean_input + p_norm).clip(0,time_dimension)
    #print("add_poisson_noise", torch.max(noisy_images))
    return (noisy_images)

#Gaussian Noise:
def add_gaussian_noise(clean_input, time_dimension):
    noise = torch.normal(time_dimension/2,time_dimension/2, clean_input.shape)
    noisy_images = clean_input + noise
    #print("add_gaussian_noise", torch.max(noisy_images))
    return noisy_images

#Impulse Noise:
def add_impulse_noise(clean_input, time_dimension):
    noise = torch.rand(*clean_input.shape)
    noise[noise <= 0.9] = 0
    noise[noise > 0.1] = np.random.random()
    noisy_images = clean_input + (noise*time_dimension)
    #print("add_impulse_noise", torch.max(noisy_images))
    return noisy_images

#%% Main body
def results_tester(clean_image, model_det):
    image_white_noise = add_white_noise(clean_image, time_dimension)
    image_masking_noise = add_masking_noise(clean_image, time_dimension)
    image_poisson_noise = add_poisson_noise(clean_image, time_dimension)
    image_impulse_noise = add_impulse_noise(clean_image, time_dimension)
    image_gaussian_noise = add_gaussian_noise(clean_image, time_dimension)
    
    images = [clean_image, image_white_noise, image_masking_noise, image_poisson_noise, image_impulse_noise, image_gaussian_noise]
    titles = ["Clean Image", "White Noise", "Masking Noise", "Poisson Noise", "Impulse Noise", "Gaussian Noise"]
    colours = ["k","y","m","c","r","g","b"]  # Define unique plotting colours for each noised image type 

    fig, ax = plt.subplots(2, len(images), figsize=(15, 8))
    performance_comparison={}
    for i, img in enumerate(images):
        col = i % len(images)
        ax[0][col].imshow(img)
        ax[0][col].set_title(titles[i])
        cleaned_img = DeepClean3D(img, model_det)
        ax[1][col].imshow(cleaned_img)
        performance_comparison[titles[i]] = quantify_performance(clean_image, img, cleaned_img, titles[i])  #between clean input and noised img
        #quantify_performance(clean_image, cleaned_img, titles[i])  #between clean input and cleaned output
    plt.tight_layout()
    plt.show()

    fig2, ax = plt.subplots(2,4)
    for i, item in enumerate(performance_comparison):
        for j, test in enumerate(performance_comparison[item]):
            row = j // 4
            col = j % 4
 
            ax[row][col].set_title(test)
            ax[row][col].set_xticks(range(len(images))) # set x-ticks to be integers  
            ax[row][col].scatter(i, performance_comparison[item][test][0], marker="x", c=colours[i], label=item)
            ax[row][col].scatter(i, performance_comparison[item][test][1], marker="o", c=colours[i])
            ax[row][col].set_xticklabels(["C", "WN", "MN", "PN", "IN", "GN"]) # X tick labels

            ax[row][col].grid(True, linestyle='--', linewidth=0.5, color='gray')   
      
    handles, labels = ax[0][0].get_legend_handles_labels()   #Gets labels from plot
    ax[1][3].legend(handles, labels)                         #Uses the empty space for ax8 for the shared legend
    ax[1][3].set_axis_off()                                  #Removes the axis for ax8 as it is just a legend not a plot
    
    plt.tight_layout()                                   
    plt.show()                                   

        


### - Settings
model_name = "16_X_15K_Blanks - Model"
model_path = "C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/Git Hub Repos/DeepClean Repo/DeepClean-Noise-Suppression-for-LHC-B-Torch-Detector/Models/16_X_15K_Blanks - Training Results/"
model_det = model_path + model_name +".pth"


EPS = 1e-8  
time_dimension = 100
#image_raw = np.zeros((128,88))  
#image_raw = np.load("C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/Git Hub Repos/DeepClean Repo/DeepClean-Noise-Suppression-for-LHC-B-Torch-Detector/Datasets/Simple Cross/Data/Simple Cross (flat pixel block data) 5.npy") #np.array([[0,0,9,0,0,0,0],[0,0,5,0,4,0,0],[0,1,4,0,0,6,0],[0,0,0,7,0,0,1],[1,0,1,1,0,7,0]])
#image_raw[10][10] = time_dimension
#image_test = torch.tensor(image_raw)
#image_test = torch.tensor(image_raw[0])

dataset_title = "Dataset 10_X"
data_path = "C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/Yr 3 Project/Circular and Spherical Dummy Datasets/"
filepath = data_path + dataset_title + "/Data/"

input_image = load_numpy_files(filepath)
results_tester(input_image, model_det)



