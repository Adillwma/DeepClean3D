#%% NEW!! IMAGE METRICS - NEEDS CLEANING UP!!!
import torch
import numpy as np
from skimage.metrics import structural_similarity, normalized_mutual_information, normalized_root_mse

# Nomalised Root Mean Squared Error (NMSRE)
def NMSRE(clean_input, noised_target):
    inp = clean_input.detach().numpy()
    tar = noised_target.detach().numpy()
    norm_rootmnsq = normalized_root_mse(inp, tar)
    return float(norm_rootmnsq)

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

    noise = clean_input - noised_target 
    noise_power = torch.mean(torch.pow(noise, 2))

    snr = 10 * torch.log10(signal_power / noise_power)
       
    return (float(snr.cpu().numpy()))

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
    return (float(psnr.cpu().numpy()))

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
    mse = torch.mean((torch.pow(clean_input - noised_target, 2)))
    return (float(mse.cpu().numpy()))

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
    return float((torch.mean(torch.abs(clean_input - noised_target))).cpu().numpy())

#Structural Similarity Index (SSIM):
def SSIM(clean_input, noised_target, time_dimension):
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
    clean_image = clean_input.detach().cpu().numpy()
    recovered_image = noised_target.detach().cpu().numpy()
    return structural_similarity(clean_image, recovered_image, data_range=float(time_dimension))

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
    return float((numerator / denominator).cpu().numpy())

#Mutual Information:
def NomalisedMutualInformation(clean_input, noised_target):
    clean_image = clean_input.detach().cpu().numpy()
    recovered_image = noised_target.detach().cpu().numpy()
    return normalized_mutual_information(clean_image, recovered_image)-1

def compare_images_pixels(clean_img, denoised_img, terminal_print=False):   ###!!!INVESTIGATE USING PRINT = TRUE !!!!
    clean_img = clean_img.detach().cpu().numpy()
    denoised_img = denoised_img.detach().cpu().numpy()
    ###TRUE HITS STATS###
    if terminal_print:
        print("###TRUE HITS STATS###")
    
    ##X,Y##
    true_hits_indexs = np.nonzero(clean_img)     # Find the indexs of the non zero pixels in clean_img
    numof_true_hits = len(true_hits_indexs[0])   # Find the number of lit pixels in clean_img
    if terminal_print:
        print("numof_true_hits:", numof_true_hits)
    
    # Check the values in corresponding indexs in denoised_img, retunr the index's and number of them that are also non zero
    true_positive_xy_indexs = np.nonzero(denoised_img[true_hits_indexs]) 
    numof_true_positive_xy = len(true_positive_xy_indexs[0])                     # Calculate the number of pixels in clean_img that are also in denoised_img ###NUMBER OF SUCSESSFUL X,Y RECON PIXELS
    if terminal_print:
        print("numof_true_positive_xy:", numof_true_positive_xy)

    # Calculate the number of true hit pixels in clean_img that are not lit at all in denoised_img  ###NUMBER OF LOST TRUE PIXELS
    false_negative_xy = numof_true_hits - numof_true_positive_xy
    if terminal_print:
        print("false_negative_xy:", false_negative_xy)
    
    # Calculate the percentage of non zero pixels in clean_img that are also non zero in denoised_img   ###PERCENTAGE OF SUCSESSFUL X,Y RECON PIXELS
    if numof_true_hits == 0:
        percentage_of_true_positive_xy = 0
    else:
        percentage_of_true_positive_xy = (numof_true_positive_xy / numof_true_hits) * 100
    
    if terminal_print:
        print(f"percentage_of_true_positive_xy: {percentage_of_true_positive_xy}%")
    

    ##TOF##
    # Calculate the number of pixels in clean_img that are also in denoised_img and have the same TOF value  ###NUMBER OF SUCSESSFUL X,Y,TOF RECON PIXELS
    num_of_true_positive_tof = np.count_nonzero(np.isclose(clean_img[true_hits_indexs], denoised_img[true_hits_indexs], rtol=1e-6))
    if terminal_print:
        print("num_of_true_positive_tof:", num_of_true_positive_tof)
    
    # Calculate the percentage of pixels in clean_img that are also in denoised_img and have the same value   ###PERCENTAGE OF SUCSESSFUL X,Y,TOF RECON PIXELS
    if numof_true_hits == 0:
        percentage_of_true_positive_tof = 0
    else:
        percentage_of_true_positive_tof = (num_of_true_positive_tof / numof_true_hits) * 100
    if terminal_print:
        print(f"percentage_of_true_positive_tof: {percentage_of_true_positive_tof}%")    
    

    ###FALSE HIT STATS###
    if terminal_print:
        print("\n###FALSE HIT STATS###")        
    clean_img_zero_indexs = np.where(clean_img == 0.0)   # find the index of the 0 valued pixels in clean image 
    number_of_zero_pixels = np.sum(clean_img_zero_indexs[0])   # Find the number of pixels in clean image that are zero
    if terminal_print:
        print("number_of_true_zero_pixels:",number_of_zero_pixels)

    #check the values in corresponding indexs in denoised_img, return the number of them that are non zero
    denoised_img_false_lit_pixels = np.nonzero(denoised_img[clean_img_zero_indexs])
    numof_false_positives_xy = len(denoised_img_false_lit_pixels[0])
    if terminal_print:
        print("numof_false_positives_xy:",numof_false_positives_xy)

    # Calculate the percentage of pixels in clean_img that are zero and are also non zero in denoised_img   ###PERCENTAGE OF FALSE LIT PIXELS

    if number_of_zero_pixels == 0:
        percentage_of_false_lit_pixels = 0
    else:
        percentage_of_false_lit_pixels = ((number_of_zero_pixels - denoised_img_false_lit_pixels)/number_of_zero_pixels) * 100
        percentage_of_false_lit_pixels = percentage_of_false_lit_pixels[0][0]
        #percentage_of_false_lit_pixels = (numof_false_positives_xy / number_of_zero_pixels) * 100
    
    
    if terminal_print:
        print(f"percentage_of_false_positives_xy: {percentage_of_false_lit_pixels}%")
    
    return percentage_of_true_positive_xy, percentage_of_true_positive_tof, numof_false_positives_xy
