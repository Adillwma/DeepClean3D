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

def compare_images_pixels(target_image, reconstructed_image, debug_mode=False): 
    """
    Takes in the clean image and the denoised image and compares them to find the percentages of signal spatial retention, signal temporal retention, and the raw count of false positives and false negatives.

    Args:
        target_image (torch tensor): The clean image
        reconstructed_image (torch tensor): The denoised image
        debug_mode (bool): If True, the function will print out the total number of pixels, the true signal points, and the true zero points in the target image. Default = False
        
    Returns:
        signal_spatial_retention_percentage (float): The percentage of signal spatial retention
        signal_temporal_retention_percentage (float): The percentage of signal temporal retention
        false_positive_count_raw (int): The raw count of false positives
        false_negative_count_raw (int): The raw count of false negatives
    """
    target_image = target_image.detach().cpu()
    reconstructed_image = reconstructed_image.detach().cpu()
    
    # determine the total number of pixels in the data
    total_pixels = target_image.numel()
    if debug_mode:
        print("Total Pixels: ", total_pixels)

    # Genrates a index mask for the zero values in the target image
    zero_mask = (target_image == 0)

    # Inverts the zero mask to get the non-zero mask from the target image indicating the signal points
    nonzero_mask = ~zero_mask

    # True Number of Signal Points in the Target Image
    true_signal_points = len(target_image[nonzero_mask])
    if debug_mode:
        print("True Signal Points: ", true_signal_points)

    # True Zero Points in the Target Image
    true_zero_points = len(target_image[zero_mask])
    if debug_mode:
        print("True Zero Points: ", true_zero_points)


    # detemine how many of the values in the reconstructed image that fall in the nonzero mask are non zero
    signal_spatial_retention_raw = len(reconstructed_image[nonzero_mask].nonzero())
    if debug_mode:
        print("Signal Spatial Retention Raw: ", signal_spatial_retention_raw)
    signal_spatial_retention_percentage = (signal_spatial_retention_raw / len(reconstructed_image[nonzero_mask])) * 100
    if debug_mode:
        print("Signal Spatial Retention Percentage: ", signal_spatial_retention_percentage, "%")

    # determine how many of those indexs have matching ToF values
    signal_temporal_retention_raw = len(reconstructed_image[nonzero_mask] == target_image[nonzero_mask])
    if debug_mode:
        print("Signal Temporal Retention Raw: ", signal_temporal_retention_raw)
    signal_temporal_retention_percentage = (signal_temporal_retention_raw / len(reconstructed_image[nonzero_mask])) * 100
    if debug_mode:
        print("Signal Temporal Retention Percentage: ", signal_temporal_retention_percentage, "%")


    # determine how many of the values in the reconstructed image that fall under the zero mask are not zero 
    false_positive_count_raw = len(reconstructed_image[zero_mask].nonzero())
    if debug_mode:
        print("False Positive Count: ", false_positive_count_raw)


    # Determine how many of the values in the reconstructed image that fall under the non zero mask are zero
    data = reconstructed_image[nonzero_mask]
    false_negative_count_raw = len(data[data == 0])
    if debug_mode:
        print("False negative count: ", false_negative_count_raw)

    return signal_spatial_retention_percentage, signal_temporal_retention_percentage, false_positive_count_raw, false_negative_count_raw



















#### ADD IN THESE NEW METRICS TOO !!!!


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
    
def precisionMETRIC(output, target):
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


