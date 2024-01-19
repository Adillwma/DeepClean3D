import torch

class WeightedPerfectRecoveryLoss(torch.nn.Module):
    def __init__(self, zero_weighting=1, nonzero_weighting=1):
        super(WeightedPerfectRecoveryLoss, self).__init__()
        self.zero_weighting = zero_weighting
        self.nonzero_weighting = nonzero_weighting

    def backward(self, grad_output):
        # Retrieve the tensors saved in the forward method
        reconstructed_image, target_image = self.saved_tensors  # <----- Remove this line

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

        # Calculate the gradients
        grad_reconstructed_image = torch.zeros_like(reconstructed_image)
        grad_reconstructed_image[zero_mask] += self.zero_weighting*(1/zero_n)*(corresponding_values_zero != values_zero).float()
        grad_reconstructed_image[nonzero_mask] += self.nonzero_weighting*(1/nonzero_n)*(corresponding_values_nonzero != values_nonzero).float()

        return grad_reconstructed_image * grad_output.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    
    def forward(self, reconstructed_image_in, target_image_in):
        reconstructed_image = reconstructed_image_in.clone()
        target_image = target_image_in.clone()

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
            zero_loss = self.zero_weighting*( (1/zero_n) * loss_value_zero)

        if nonzero_n == 0:
            nonzero_loss = 0
        else:
            # Calculate the loss for non-zero values
            loss_value_nonzero = (values_nonzero != corresponding_values_nonzero).float().sum() 
            nonzero_loss = self.nonzero_weighting*( (1/nonzero_n) * loss_value_nonzero) 

        # Calculate the total loss with automatic class balancing and user class weighting
        loss_value = zero_loss + nonzero_loss


        return loss_value

