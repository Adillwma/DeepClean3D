import numpy as np
import torch


#%% DC3D Special Functions
# Special normalisation for pure masking
def mask_optimised_normalisation(data, ):
    """
    Normalisation function for pure masking output, trhe function takes any non zero valu eto 1 and any zero value is left as is. [EXPLAIN WHY!!!!!]

    Args:
        data (torch tensor): The input data to be normalised.

    Returns:
        data (torch tensor): The normalised data.
    
    """
    data = torch.where(data > 0, 1.0, 0.0)
    return data

# Custom normalisation function
def gaped_normalisation(data, reconstruction_threshold, time_dimension=100):
    """
    Normalisation function that normalised values in range [0 < value <= 'time_dimension'] to new range of [reconstruction_threshold' < norm_value <= 1], with a gap at the bottom of the range to allow for the reconstruction threshold to be applied later. All zero values are left untouched. This is used for the direct network output.
    
    Args:
        data (torch tensor): The input data to be normalised. [WARNING: the data must be (integer values? and) in the range [0 < value <= 'time_dimension'] or the normalisation will not work correctly]
        reconstruction_threshold (float): The threshold used in the custom normalisation, used to set the lower limit of the normalised values.
        time_dimension (int): The number of time steps in the data set, used to set the upper limit of the normalised values. Default = 100  

    Returns:
        data (torch tensor): The normalised data.
    """
    data = torch.where(data > 0, (((data / time_dimension) / (1 / (1 - reconstruction_threshold))) + reconstruction_threshold), 0 ) 
    
    """
    print(data.shape)
    print(type(data))
    print(data)
    if data is torch.Tensor:    
        data = torch.where(data > 0, (((data / time_dimension) / (1 / (1 - reconstruction_threshold))) + reconstruction_threshold), 0 )
    elif data is np.array:
        data = np.where(data > 0, (((data / time_dimension) / (1 / (1 - reconstruction_threshold))) + reconstruction_threshold), 0 )
    else:
        raise ValueError("ERROR: The data type is not recognised by the gaped_normalisation function, please use a torch tensor or a numpy array")
    """
    return data

# Custom renormalisation function
def gaped_renormalisation(data, reconstruction_threshold, time_dimension=100):
    """
    torch version of our Renormalisation function that renormalises values in range [reconstruction_threshold < value <= 1] to new range of [0 < renorm_value <= 'time_dimension'] removing any values that fell below the 'reconstruction_threshold' by setting thwm to zero. All zero values are left untouched. This is used for the direct network output.

    Args:
        data (torch tensor): The input data to be renormalised. [WARNING: the data must be (float values? and) in the range [reconstruction_threshold < value <= 1] or the renormalisation will not work correctly]
        reconstruction_threshold (float): The threshold used in the custom normalisation, used to set the lower limit of the normalised values.
        time_dimension (int): The number of time steps in the data set, used to set the upper limit of the normalised values. Default = 100

    Returns:
        data (torch tensor): The renormalised data.
    """
    data = torch.where(data > reconstruction_threshold, ((data - reconstruction_threshold)*(1 / (1 - reconstruction_threshold))) * (time_dimension), 0)
    """
    if data is torch.Tensor:
        data = torch.where(data > reconstruction_threshold, ((data - reconstruction_threshold)*(1 / (1 - reconstruction_threshold))) * (time_dimension), 0)
    elif data is np.array:
        data = np.where(data > reconstruction_threshold, ((data - reconstruction_threshold)*(1 / (1 - reconstruction_threshold))) * (time_dimension), 0)
    else:
        raise ValueError("ERROR: The data type is not recognised by the gaped_renormalisation function, please use a torch tensor or a numpy array")
    """
    return data

# 3D Reconstruction function
def reconstruct_3D(*args):
    """
    3D reconstruction function that takes any number of 2D arrays created through our 3D to 2D with embedded ToF method and reconstructs them into a 3D arrays. 
    
    Args:
        *args (np array): Any number of 2D arrays to be reconstructed.

    Returns:
        results (np array): The reconstructed 3D arrays. Same number of arrays as input.
    """
    results = []
    for data in args:
        res = np.nonzero(data > 0)
        data_output = np.column_stack((res[0], res[1], data[res]))
        results.append(data_output)
    return results

# Masking technique
def masking_recovery(input_image, recovered_image, time_dimension, debug=False):
    """
    Applies my masking technique to the recovered image, this is essential and and condational for both the input and the recovered image as to weatehr a pixel is allowed to stay in the final output or not, which relies on the differing distortion profiles of the two images.

    Args:
        input_image (np array): The input image to be masked.
        recovered_image (np array): The recovered image to be masked.
        time_dimension (int): The number of time steps in the data set, used to set the upper limit of the normalised values.
        print_result (bool, optional): If set to true then the function will print a text string reporting the masking usefullness evaluation metrics. Defaults to False.

    Returns:
        result (np array): The masked recovered image.
    """
    raw_input_image = input_image.clone()
    net_recovered_image = recovered_image.clone()
    #Evaluate usefullness 
    # count the number of non-zero values
    masking_pixels = torch.count_nonzero(net_recovered_image)
    image_shape = net_recovered_image.shape
    total_pixels = image_shape[0] * image_shape[1] * time_dimension
    # print the count
    if debug:
        print(f"Total number of pixels in the timescan: {format(total_pixels, ',')}\nNumber of pixels returned by the masking: {format(masking_pixels, ',')}\nNumber of pixels removed from reconstruction by masking: {format(total_pixels - masking_pixels, ',')}")

    # use np.where and boolean indexing to update values in a
    mask_indexs = np.where(net_recovered_image != 0)
    net_recovered_image[mask_indexs] = raw_input_image[mask_indexs]
    result = net_recovered_image

    #assert result.shape == raw_input_image.shape, "ERROR: Masking has failed, the recovered image is not the same shape as the input image"
    if result.shape != raw_input_image.shape:
        print("ERROR: Masking has failed, the recovered image is not the same shape as the input image")
    
    return result