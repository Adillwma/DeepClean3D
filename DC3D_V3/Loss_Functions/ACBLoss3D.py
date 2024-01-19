import torch

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
    