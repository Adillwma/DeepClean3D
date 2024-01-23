import torch



##### NEW NEW NEW !!!
class NEWESTACB3dloss2024(torch.nn.Module):
    def __init__(self, b, m, n):

        """
        # Number of dp is related to timestep size. 1,000 = 3dp, 10,000 = 4dp, 100,000 = 5dp, 1,000,000 = 6dp etc
        
        """
        super().__init__()   
        self.mse_loss = torch.nn.MSELoss(reduction='mean')
        self.zero_weighting = 1
        self.nonzero_weighting = 1


        # create a batches tensor
        batches = torch.arange(b).repeat_interleave(m*n).view(-1,1)

        # Create indices tensor
        indices = torch.stack( torch.meshgrid(torch.arange(m), torch.arange(n), indexing='ij'), dim=-1).view(-1, 2) 
        indices = indices.repeat(b, 1)

        self.batches = batches  
        self.indices = indices

    def transform_to_3d_coordinates(self, input_tensor):
        # Reshape the input tensor and concatenate with indices
        output_tensor = torch.cat((self.batches, self.indices, input_tensor.reshape(-1, 1)), dim=1)

        return output_tensor


    def forward(self, reconstructed_image, target_image):
        reconstructed_image = self.transform_to_3d_coordinates(reconstructed_image)
        target_image = self.transform_to_3d_coordinates(target_image)


        # target_image is a tensor of shape (m, 4)
        # Example:
        # target_image = np.array([[1, 2, 3, 4],
        #                         [5, 6, 7, 0],
        #                         [8, 9, 10, 11]])

        # Identify 0 values in the 4th column of the second dimension

        zero_mask = (target_image[:, 3] == 0)
        nonzero_mask = ~zero_mask

        values_zero = target_image[zero_mask]
        values_nonzero = target_image[nonzero_mask]

        corresponding_values_zero = reconstructed_image[zero_mask]
        corresponding_values_nonzero = reconstructed_image[nonzero_mask]

        zero_loss = self.mse_loss(corresponding_values_zero, values_zero)
        nonzero_loss = self.mse_loss(corresponding_values_nonzero, values_nonzero)

        if torch.isnan(zero_loss):
            zero_loss = 0
        if torch.isnan(nonzero_loss):
            nonzero_loss = 0

        weighted_mse_loss = (self.zero_weighting * zero_loss) + (self.nonzero_weighting * nonzero_loss)

        return weighted_mse_loss



###### OLD AFTER THIS!!"!"


# Weighted Custom Split Loss Function

class ada_weighted_custom_split_loss(torch.nn.Module): 
    """
    Calculates the weighted error loss between target_image and reconstructed_image.
    The loss for zero pixels in the target_image is weighted by zero_weighting, and the loss for non-zero
    pixels is weighted by nonzero_weighting and both have loss functions as passed in by user.

    Args:
    - target_image: a tensor of shape (B, C, H, W) containing the target image
    - reconstructed_image: a tensor of shape (B, C, H, W) containing the reconstructed image
    - zero_weighting: a scalar weighting coefficient for the MSE loss of zero pixels
    - nonzero_weighting: a scalar weighting coefficient for the MSE loss of non-zero pixels

    Returns:
    - weighted_mse_loss: a scalar tensor containing the weighted MSE loss
    """


    def __init__(self, split_loss_functions, zero_weighting=1, nonzero_weighting=1):
        super().__init__()   
        self.zero_weighting = zero_weighting
        self.nonzero_weighting = nonzero_weighting
        self.loss_func_zeros = split_loss_functions[0]
        self.loss_func_nonzeros = split_loss_functions[1]


    def forward(self, reconstructed_image, target_image):
        # Get the indices of 0 and non 0 values in target_image as a mask for speed
        zero_mask = (target_image == 0)
        nonzero_mask = ~zero_mask         # Invert mask
        
        # Get the values in target_image
        values_zero = target_image[zero_mask]
        values_nonzero = target_image[nonzero_mask]
        
        # Get the corresponding values in reconstructed_image
        corresponding_values_zero = reconstructed_image[zero_mask]
        corresponding_values_nonzero = reconstructed_image[nonzero_mask]
        
        zero_loss = self.loss_func_zeros(corresponding_values_zero, values_zero)
        nonzero_loss = self.loss_func_nonzeros(corresponding_values_nonzero, values_nonzero)

        # Protection from there being no 0 vals or no non zero vals, which then retunrs nan for MSE and creates a nan overall MSE return (which is error)
        if torch.isnan(zero_loss):
            zero_loss = 0
        if torch.isnan(nonzero_loss):
            nonzero_loss = 0
        
        # Sum losses with weighting coefficiants 
        weighted_split_loss = (self.zero_weighting * zero_loss) + (self.nonzero_weighting * nonzero_loss) 
        
        return weighted_split_loss


class ACBLoss(torch.nn.Module):
    def __init__(self, zero_weighting=1, nonzero_weighting=1):
        """
        Initializes the ACB-MSE Loss Function class with weighting coefficients.

        Args:
        - zero_weighting: a scalar weighting coefficient for the MSE loss of zero pixels
        - nonzero_weighting: a scalar weighting coefficient for the MSE loss of non-zero pixels
        """
        super().__init__()   
        self.zero_weighting = zero_weighting
        self.nonzero_weighting = nonzero_weighting
        self.mse_loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, reconstructed_image, target_image):
        """
        Calculates the weighted mean squared error (MSE) loss between target_image and reconstructed_image.
        The loss for zero pixels in the target_image is weighted by zero_weighting, and the loss for non-zero
        pixels is weighted by nonzero_weighting.

        Args:
        - target_image: a tensor of shape (B, C, H, W) containing the target image
        - reconstructed_image: a tensor of shape (B, C, H, W) containing the reconstructed image

        Returns:
        - weighted_mse_loss: a scalar tensor containing the weighted MSE loss
        """
        zero_mask = (target_image == 0)
        nonzero_mask = ~zero_mask

        values_zero = target_image[zero_mask]
        values_nonzero = target_image[nonzero_mask]

        corresponding_values_zero = reconstructed_image[zero_mask]
        corresponding_values_nonzero = reconstructed_image[nonzero_mask]

        zero_loss = self.mse_loss(corresponding_values_zero, values_zero)
        nonzero_loss = self.mse_loss(corresponding_values_nonzero, values_nonzero)

        if torch.isnan(zero_loss):
            zero_loss = 0
        if torch.isnan(nonzero_loss):
            nonzero_loss = 0

        weighted_mse_loss = (self.zero_weighting * zero_loss) + (self.nonzero_weighting * nonzero_loss)

        return weighted_mse_loss




class ACBLoss3D(torch.nn.Module):
    def __init__(self, zero_weighting=1, nonzero_weighting=1, virtual_t_weighting=1, virtual_x_weighting=1, virtual_y_weighting=1, timesteps=1000):
        """
        Initializes the ACB-MSE-3D Holographic Loss Function class with weighting coefficients.

        Args:
        - zero_weighting: a scalar weighting coefficient for the MSE loss of zero pixels
        - nonzero_weighting: a scalar weighting coefficient for the MSE loss of non-zero pixels
        - virtual_t_weighting: a scalar weighting coefficient for the MSE loss of virtual t pixels
        - virtual_x_weighting: a scalar weighting coefficient for the MSE loss of virtual x pixels
        - virtual_y_weighting: a scalar weighting coefficient for the MSE loss of virtual y pixels
        """
        super().__init__()   
        self.zero_weighting = zero_weighting
        self.nonzero_weighting = nonzero_weighting
        self.virtual_t_weighting = virtual_t_weighting
        self.virtual_x_weighting = virtual_x_weighting
        self.virtual_y_weighting = virtual_y_weighting
        self.mse_loss = torch.nn.MSELoss(reduction='mean')
        self.timesteps = timesteps

        if self.timesteps <= 0:
            raise ValueError("Timesteps to Holographic loss should be a positive integer")
        
    def holographic_transform(self, input_image, virtual_dim='y', binary_values=True, debug=False):

        input_batch = input_image.clone()
        output_batch_1 = torch.zeros((input_batch.shape[0], input_batch.shape[1], input_batch.shape[2], self.timesteps), dtype=input_batch.dtype, requires_grad=True)
        output_batch = output_batch_1.clone()

        for b in range(input_batch.shape[0]):

            input_tensor = input_batch[b, 0]
            
            if virtual_dim == 'x':
                input_tensor = input_tensor.T

            non_zero_indices_x, non_zero_indices_y = torch.nonzero(input_tensor, as_tuple=True)
            values = input_tensor[non_zero_indices_x, non_zero_indices_y]
            quantized_values = (values * self.timesteps).to(torch.int64) - 1                         # Values lie in range 0.0 - 1.0 

            for quantized_value, non_zero_indice_x, non_zero_indice_y in zip(quantized_values, non_zero_indices_x, non_zero_indices_y):
                output_batch[b, 0, non_zero_indice_x, quantized_value] = non_zero_indice_y

        return output_batch


    def ACB_MSE_Loss (self, reconstructed, target):
        reconstructed_image = reconstructed.clone()
        target_image = target.clone()
        """
        Calculates the weighted mean squared error (MSE) loss between target_image and reconstructed_image.
        The loss for zero pixels in the target_image is weighted by zero_weighting, and the loss for non-zero
        pixels is weighted by nonzero_weighting.

        Args:
        - target_image: a tensor of shape (B, C, H, W) containing the target image
        - reconstructed_image: a tensor of shape (B, C, H, W) containing the reconstructed image

        Returns:
        - weighted_mse_loss: a scalar tensor containing the weighted MSE loss
        """
        zero_mask = (target_image == 0)
        nonzero_mask = ~zero_mask

        values_zero = target_image[zero_mask]
        values_nonzero = target_image[nonzero_mask]

        corresponding_values_zero = reconstructed_image[zero_mask]
        corresponding_values_nonzero = reconstructed_image[nonzero_mask]

        zero_loss = self.mse_loss(corresponding_values_zero, values_zero)
        nonzero_loss = self.mse_loss(corresponding_values_nonzero, values_nonzero)

        if torch.isnan(zero_loss):
            zero_loss = 0
        if torch.isnan(nonzero_loss):
            nonzero_loss = 0

        weighted_mse_loss = (self.zero_weighting * zero_loss) + (self.nonzero_weighting * nonzero_loss)

        return weighted_mse_loss

    def forward(self, reconstructed_image, target_image):
        if self.virtual_t_weighting:
            vt_loss = (self.ACB_MSE_Loss(reconstructed_image, target_image)) * self.virtual_t_weighting
        else:
            vt_loss = torch.tensor(0, dtype=reconstructed_image.dtype, requires_grad=True)
        
        if self.virtual_x_weighting:
            reconstructed_hologram_vx = self.holographic_transform(reconstructed_image, virtual_dim='x')
            target_hologram_vx = self.holographic_transform(target_image, virtual_dim='x')
            vx_loss = (self.ACB_MSE_Loss(reconstructed_hologram_vx, target_hologram_vx)) * self.virtual_x_weighting
        else:
            vx_loss = torch.tensor(0, dtype=reconstructed_image.dtype, requires_grad=True)
        
        if self.virtual_y_weighting:
            reconstructed_hologram_vy = self.holographic_transform(reconstructed_image)
            target_hologram_vy = self.holographic_transform(target_image)
            vy_loss = self.ACB_MSE_Loss(reconstructed_hologram_vy, target_hologram_vy) * self.virtual_y_weighting
        else:
            vy_loss = torch.tensor(0, dtype=reconstructed_image.dtype, requires_grad=True)

        return vt_loss + vx_loss + vy_loss
    

#  Adaptive Sum of Squared Errors loss function
def ada_SSE_loss(reconstructed_image, target_image):
    """
    Sum Squared Error Loss
    """
    sse_loss = ((reconstructed_image-target_image)**2).sum()
    
    return sse_loss




class ffACBLoss(torch.nn.Module):
    def __init__(self, zero_weighting=1, nonzero_weighting=1, fullframe_weighting=1):
        """
        Initializes the ACB-MSE Loss Function class with weighting coefficients.

        Args:
        - zero_weighting: a scalar weighting coefficient for the MSE loss of zero pixels
        - nonzero_weighting: a scalar weighting coefficient for the MSE loss of non-zero pixels
        """
        super().__init__()   
        self.zero_weighting = zero_weighting
        self.nonzero_weighting = nonzero_weighting
        self.fullframe_weighting = fullframe_weighting
        self.mse_loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, reconstructed_image, target_image):
        """
        Calculates the weighted mean squared error (MSE) loss between target_image and reconstructed_image.
        The loss for zero pixels in the target_image is weighted by zero_weighting, and the loss for non-zero
        pixels is weighted by nonzero_weighting.

        Args:
        - target_image: a tensor of shape (B, C, H, W) containing the target image
        - reconstructed_image: a tensor of shape (B, C, H, W) containing the reconstructed image

        Returns:
        - weighted_mse_loss: a scalar tensor containing the weighted MSE loss
        """
        zero_mask = (target_image == 0)
        nonzero_mask = ~zero_mask

        values_zero = target_image[zero_mask]
        values_nonzero = target_image[nonzero_mask]

        corresponding_values_zero = reconstructed_image[zero_mask]
        corresponding_values_nonzero = reconstructed_image[nonzero_mask]

        zero_loss = self.mse_loss(corresponding_values_zero, values_zero)
        nonzero_loss = self.mse_loss(corresponding_values_nonzero, values_nonzero)
        full_frame_loss = self.mse_loss(reconstructed_image, target_image)

        if torch.isnan(zero_loss):
            zero_loss = 0
        if torch.isnan(nonzero_loss):
            nonzero_loss = 0

        weighted_mse_loss = (self.zero_weighting * zero_loss) + (self.nonzero_weighting * nonzero_loss) + (self.fullframe_weighting * full_frame_loss)

        return weighted_mse_loss


class HistogramLoss(torch.nn.Module):
    def __init__(self, num_bins=256):
        super(HistogramLoss, self).__init__()
        self.num_bins = num_bins

    def histogram_intersection(hist1, hist2):
        min_hist = torch.min(hist1, hist2)
        return torch.sum(min_hist)

    def forward(self, input_image, target_image):
        hist_input = torch.histc(input_image.view(-1), bins=self.num_bins, min=0, max=255)
        hist_target = torch.histc(target_image.view(-1), bins=self.num_bins, min=0, max=255)

        hist_input = hist_input / hist_input.sum()
        hist_target = hist_target / hist_target.sum()

        loss = 1 - self.histogram_intersection(hist_input, hist_target)

        return loss




class simple3Dloss(torch.nn.Module):
    def __init__(self, zero_weighting=1, nonzero_weighting=1, virtual_t_weighting=1, virtual_x_weighting=1, virtual_y_weighting=1, timesteps=1000):
        super().__init__()   
        self.mse_loss = torch.nn.MSELoss(reduction='mean')
        self.timesteps = timesteps

    def holographic_transform(self, input_image, virtual_dim='y', binary_values=True, debug=False):
        input_batch = input_image.clone()

        # Create a tensor of zeros with the desired last dimension size
        new_size = list(input_batch.size())
        new_size[-1] = self.timesteps
        output_batch = torch.zeros(new_size, dtype=input_batch.dtype, requires_grad=True)

        for b in range(input_batch.shape[0]):
            input_tensor = input_batch[b, 0]
            
            if virtual_dim == 'x':
                input_tensor = input_tensor.T

            non_zero_indices_x, non_zero_indices_y = torch.nonzero(input_tensor, as_tuple=True)
            values = input_tensor[non_zero_indices_x, non_zero_indices_y]
            quantized_values = (values * self.timesteps).to(torch.int64) - 1


            for quantized_value, non_zero_indice_x, non_zero_indice_y in zip(quantized_values, non_zero_indices_x, non_zero_indices_y):
                #output_batch[b, 0, non_zero_indice_x, quantized_value] = non_zero_indice_y

                new_values = output_batch.clone()  # Create a new tensor with the same values as output_batch
                new_values[b, 0, non_zero_indice_x, quantized_value] = non_zero_indice_y
                output_batch = new_values  # Update output_batch with the modified values
        return output_batch

    def forward(self, reconstructed_image, target_image):
        reconstructed_hologram_vy = self.holographic_transform(reconstructed_image)
        target_hologram_vy = self.holographic_transform(target_image)
        vy_loss = self.mse_loss(reconstructed_hologram_vy, target_hologram_vy)
        return vy_loss
    

class simple3DlossOLD(torch.nn.Module):
    def __init__(self, zero_weighting=1, nonzero_weighting=1, virtual_t_weighting=1, virtual_x_weighting=1, virtual_y_weighting=1, timesteps=1000):
        super().__init__()   
        self.mse_loss = torch.nn.MSELoss(reduction='mean')
        self.timesteps = timesteps

    def holographic_transform(self, input_image, virtual_dim='y', binary_values=True, debug=False):

        input_batch = input_image.clone()
        output_batch_1 = torch.zeros((input_batch.shape[0], input_batch.shape[1], input_batch.shape[2], self.timesteps), dtype=input_batch.dtype, requires_grad=True)
        output_batch = output_batch_1.clone()

        for b in range(input_batch.shape[0]):

            input_tensor = input_batch[b, 0]
            
            if virtual_dim == 'x':
                input_tensor = input_tensor.T

            non_zero_indices_x, non_zero_indices_y = torch.nonzero(input_tensor, as_tuple=True)
            values = input_tensor[non_zero_indices_x, non_zero_indices_y]
            quantized_values = (values * self.timesteps).to(torch.int64) - 1                         # Values lie in range 0.0 - 1.0 

            for quantized_value, non_zero_indice_x, non_zero_indice_y in zip(quantized_values, non_zero_indices_x, non_zero_indices_y):
                output_batch[b, 0, non_zero_indice_x, quantized_value] = non_zero_indice_y

        return output_batch

    def forward(self, reconstructed_image, target_image):
        reconstructed_hologram_vy = self.holographic_transform(reconstructed_image)
        target_hologram_vy = self.holographic_transform(target_image)
        vy_loss = self.mse_loss(reconstructed_hologram_vy, target_hologram_vy)
        return vy_loss
    

class True3DLoss(torch.nn.Module):
    def __init__(self, zero_weighting=1, nonzero_weighting=1, timesteps=1000):
        """
        # Number of dp is related to timestep size. 1,000 = 3dp, 10,000 = 4dp, 100,000 = 5dp, 1,000,000 = 6dp etc
        
        """
        super().__init__()   
        self.mse_loss = torch.nn.MSELoss(reduction='mean')
        self.timesteps = timesteps
        self.zero_weighting = zero_weighting
        self.nonzero_weighting = nonzero_weighting

    def ACB_MSE_Loss (self, reconstructed, target):
        reconstructed_image = reconstructed.clone()
        target_image = target.clone()
        """
        Calculates the weighted mean squared error (MSE) loss between target_image and reconstructed_image.
        The loss for zero pixels in the target_image is weighted by zero_weighting, and the loss for non-zero
        pixels is weighted by nonzero_weighting.

        Args:
        - target_image: a tensor of shape (B, C, H, W) containing the target image
        - reconstructed_image: a tensor of shape (B, C, H, W) containing the reconstructed image

        Returns:
        - weighted_mse_loss: a scalar tensor containing the weighted MSE loss
        """
        zero_mask = (target_image == 0)
        nonzero_mask = ~zero_mask

        values_zero = target_image[zero_mask]
        values_nonzero = target_image[nonzero_mask]

        corresponding_values_zero = reconstructed_image[zero_mask]
        corresponding_values_nonzero = reconstructed_image[nonzero_mask]

        zero_loss = self.mse_loss(corresponding_values_zero, values_zero)
        nonzero_loss = self.mse_loss(corresponding_values_nonzero, values_nonzero)

        if torch.isnan(zero_loss):
            zero_loss = 0
        if torch.isnan(nonzero_loss):
            nonzero_loss = 0

        weighted_mse_loss = (self.zero_weighting * zero_loss) + (self.nonzero_weighting * nonzero_loss)

        return weighted_mse_loss

    def expand_data_to_new_dimPREFERED(self, input_tensor):
        """
        # Number of dp is related to timestep size. 1,000 = 3dp, 10,000 = 4dp, 100,000 = 5dp, 1,000,000 = 6dp etc
        
        """

        # QUANITSE VALUES IN THE TENSOR TO steps of (1/timesteps) then multiply by timesteps to arrive at integers (this simplifies to juyt * timestep then round)
        #quantised_tensor = torch.round(input_tensor * self.timesteps) # / timesteps

        # Determine the number of classes (third dimension size)
        num_classes = self.timesteps #int(input_tensor.max()) + 1

        # Convert the input tensor to indices
        indices = input_tensor.long()

        # Create a mask for non-zero values
        #non_zero_mask = input_tensor != 0

        # Create one-hot encoded tensor
        one_hot_encoded = torch.nn.functional.one_hot(indices, num_classes=num_classes).float()

        # Apply the mask to exclude zero values
        #one_hot_encoded = one_hot_encoded * non_zero_mask.unsqueeze(-1)

        # Permute dimensions to move the new dimension to the front
        #one_hot_encoded = one_hot_encoded.permute(0, 1, 4, 2, 3)
        
        return one_hot_encoded

    def expand_data_to_new_dim(self, input_tensor):
        input_tensor = (input_tensor * self.timesteps) - 1

        input_tensor = torch.where(input_tensor < 0, torch.tensor(0.0), input_tensor)   # could try input_tensor + 1.0 instead of  torch.tensor(0.0)

        # Assuming you have 'indices' and 'num_classes' defined
        num_classes =  self.timesteps
        indices = input_tensor.long()

        # Reshape the indices tensor for compatibility with scatter_
        reshaped_indices = indices.view(indices.size(0), indices.size(1), -1)

        # Create a tensor of zeros with the same shape as 'reshaped_indices'
        one_hot_encoded = torch.zeros(reshaped_indices.size(0), reshaped_indices.size(1), num_classes, reshaped_indices.size(2), requires_grad=True)
        
        # Use scatter to fill the one-hot encoded tensor (without in-place operation)
        one_hot_encoded = one_hot_encoded.scatter(2, reshaped_indices.unsqueeze(2), 1)

        # Convert the tensor to float
        one_hot_encoded = one_hot_encoded.float()

        return one_hot_encoded

    def forward(self, reconstructed_image, target_image):
        reconstructed_3D_view = self.expand_data_to_new_dim(reconstructed_image)
        target_3D_view = self.expand_data_to_new_dim(target_image)
        true3d_loss = self.ACB_MSE_Loss(reconstructed_3D_view, target_3D_view)
        return true3d_loss




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
