import torch


def weighted_perfect_recovery_lossOLD(reconstructed_image, target_image, zero_weighting=1, nonzero_weighting=1):

    # Get the indices of 0 and non 0 values in target_image as a mask for speed
    zero_mask = (target_image == 0)
    nonzero_mask = ~zero_mask         # Invert mask
    
    # Get the values in target_image
    values_zero = target_image[zero_mask]
    values_nonzero = target_image[nonzero_mask]

    #Calualte the number of value sin each of values_zero and values_nonzero for use in the class balancing
    zero_n = len(values_zero)
    nonzero_n = len(values_nonzero)
    
    # Get the corresponding values in reconstructed_image
    corresponding_values_zero = reconstructed_image[zero_mask]
    corresponding_values_nonzero = reconstructed_image[nonzero_mask]

    if zero_n == 0:
        zero_loss = 0
    else:
        # Calculate the loss for zero values
        loss_value_zero = (values_zero != corresponding_values_zero).float().sum() 
        zero_loss = zero_weighting*( (1/zero_n) * loss_value_zero)

    if nonzero_n == 0:
        nonzero_loss = 0
    else:
        # Calculate the loss for non-zero values
        loss_value_nonzero = (values_nonzero != corresponding_values_nonzero).float().sum() 
        nonzero_loss = nonzero_weighting*( (1/nonzero_n) * loss_value_nonzero) 

    # Calculate the total loss with automatic class balancing and user class weighting
    loss_value = zero_loss + nonzero_loss

    return loss_value
