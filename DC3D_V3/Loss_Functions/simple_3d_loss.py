import torch

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
    
