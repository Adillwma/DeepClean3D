# -*- coding: utf-8 -*-
"""
Image Quality Tester
@author: Adill Al-Ashgar
Created on Fri Feb 10 15:32:02 2023
"""

from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import os

def load_numpy_files(folder_path, N):
    input_images_list = []
    count = 0
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
                      #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! [0] only needed for arrrys that come in with the singal [0] noise[1] situation, fix this
            input_images_list.append(torch_array)
            count += 1
            if count >= N:
                break
    return input_images_list


from DC3D_V3.DC3D_V3_Denoiser import DeepClean3D
#def DeepClean3D(img, model_weights):     #PLACEHOLDER/// REPLACE WITH ABOVE IMPORT LINE INSTEAD
#return(img)

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
    return (float(snr.numpy()))

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
    return (float(psnr.numpy()))

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
    return (float(mse.numpy()))

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
    return float((torch.mean(torch.abs(clean_input - noised_target))).numpy())

#Structural Similarity Index (SSIM):
import torch.nn.functional as F
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
    return float(((2 * mu1 * mu2 + 0.0001) * (2 * sigma12 + 0.0001) / (mu1 ** 2 + mu2 ** 2 + 0.0001) / (sigma1_sq + sigma2_sq + 0.0001)).numpy())

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
    return float((numerator / denominator).numpy())

#Mutual Information:
def MutualInformation(clean_input, noised_target):
    H_x = -torch.sum(clean_input * torch.log2(clean_input + 1e-8))
    H_y = -torch.sum(noised_target * torch.log2(noised_target + 1e-8))
    H_xy = -torch.sum(torch.mul(clean_input, torch.log2(torch.mul(clean_input, noised_target) + 1e-8)))
    return float((H_x + H_y - H_xy).numpy())

#Combine all performance metrics into simple test script
def quantify_performance(clean_input, noised_target, label):
    performance = {}
    performance['SNR'] = SNR(clean_input, noised_target)
    performance['MSE'] = MSE(clean_input, noised_target)
    performance['SSIM'] = SSIM(clean_input, noised_target)
    performance['Mutual Information'] = MutualInformation(clean_input, noised_target)    #BROKEN
    performance['PSNR'] = PSNR(clean_input, noised_target, time_dimension)
    performance['MAE'] = MAE(clean_input, noised_target)
    performance['Correlation Coefficient'] = correlation_coeff(clean_input, noised_target)

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
def results_box_plots(axes, images, titles):
    fig2, ax = plt.subplots(2, 4)
    data = []
    for i, row in enumerate(axes):
        for j, ax_col in enumerate(row):
            data.append([c.get_offsets()[:, 1] for c in ax_col.collections])
            ax[i, j].boxplot(data[-1], showfliers=False, notch=True)
            ax[i, j].set_xticks(range(len(images)))
            ax[i, j].set_xticklabels([t for t in titles], rotation=30, ha="right")
            ax[i, j].grid(True, linestyle='--', linewidth=0.5, color='gray')
            ax[i, j].set_title(ax_col.get_title())
    
    plt.tight_layout()
    plt.show()

def results_tester(clean_images, model_path, set_target):
    """
    set target must be : 1, or 2 : 1 compares the original to the denoised, 2 compares the original to the noised, 
    """
    fig2, ax = plt.subplots(2,4)

    for image_number, clean_image in enumerate(clean_images):                           # Selects current loaded image from the image batch
        print(image_number)
        # Creates the noised images form the currently loaded image
        image_white_noise = add_white_noise(clean_image, time_dimension)                
        image_masking_noise = add_masking_noise(clean_image, time_dimension)
        image_poisson_noise = add_poisson_noise(clean_image, time_dimension)
        image_impulse_noise = add_impulse_noise(clean_image, time_dimension)
        image_gaussian_noise = add_gaussian_noise(clean_image, time_dimension)
        
        # Sets the labels and images for iteration
        images = [clean_image, image_white_noise, image_masking_noise, image_poisson_noise, image_impulse_noise, image_gaussian_noise]
        titles = ["Clean Image", "White Noise", "Masking Noise", "Poisson Noise", "Impulse Noise", "Gaussian Noise"]
        colours = ["k","y","m","c","r","g","b"]  # Define unique plotting colours for each noised image type 

        # Calculates performance results for each type of noise for the current loaded image
        performance_comparison={}
        for i, img in enumerate(images):
            cleaned_img = DeepClean3D(img, model_path)
            if set_target == 1:
                targ = cleaned_img
            else:
                targ = img
            performance_comparison[titles[i]] = quantify_performance(clean_image, targ, titles[i])  #between clean input and denoised img, change cleaned_img to just img to test differnce between clean input and noised files

        # Plots performance results for each type of noise for the current loaded image
        for i, item in enumerate(performance_comparison):
            for j, test in enumerate(performance_comparison[item]):
                row = j // 4
                col = j % 4

                if image_number == 0:
                    ax[row][col].set_title(test)
                    ax[row][col].set_xticks(range(len(images))) # Set x-ticks to be integers  
                    ax[row][col].grid(True, linestyle='--', linewidth=0.5, color='gray')   
                    ax[row][col].set_xticklabels(["C", "WN", "MN", "PN", "IN", "GN"]) # X tick labels
                ax[row][col].scatter(i, performance_comparison[item][test], marker="x", c=colours[i]  ,label=item)

        if image_number == 0:
            print("set")
            handles, labels = ax[0][0].get_legend_handles_labels()   # Gets labels from plot (above condition makes it only the first time round so as not to relist the same handles again)
    ax[1][3].legend(handles, labels)                                 # Uses the empty space for ax8 for the shared legend
    ax[1][3].set_axis_off()                                          # Removes the axis for ax8 as it is just a legend not a plot
    
    plt.tight_layout()                                   
    plt.show()                                   

### - Settings
model_name = "TEST"
model_path = "C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/Git Hub Repos/DeepClean Repo/DeepClean-Noise-Suppression-for-LHC-B-Torch-Detector/Models/"
model_det = model_path + model_name +".pth"
N = 30 #number og images to load

EPS = 1e-8  
time_dimension = 100
#image_raw = np.zeros((128,88))  
#image_raw = np.load("C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/Git Hub Repos/DeepClean Repo/DeepClean-Noise-Suppression-for-LHC-B-Torch-Detector/Datasets/Simple Cross/Data/Simple Cross (flat pixel block data) 5.npy") #np.array([[0,0,9,0,0,0,0],[0,0,5,0,4,0,0],[0,1,4,0,0,6,0],[0,0,0,7,0,0,1],[1,0,1,1,0,7,0]])
#image_raw[10][10] = time_dimension
#image_test = torch.tensor(image_raw)
#image_test = torch.tensor(image_raw[0])
# - Data Loader User Inputs
dataset_title = "Dataset 1_Realistic"
data_path = "C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/Yr 3 Project/Circular and Spherical Dummy Datasets/"
filepath = data_path + dataset_title + "/Data/"

#filepath = (r"C:\Users\Student\Documents\UNI\Onedrive - University of Bristol\Git Hub Repos\DeepClean Repo\DeepClean-Noise-Suppression-for-LHC-B-Torch-Detector\Datasets\Dataset 3\Data")
#input_image_list = [image_test, image_test*2, image_test*5, image_test/2, image_test*1.3, image_test*4]
input_image_list = load_numpy_files(filepath, N)
plt.imshow(input_image_list[0])
plt.show()

results_tester(input_image_list, model_det, set_target=1)
results_tester(input_image_list, model_det, set_target=2)



