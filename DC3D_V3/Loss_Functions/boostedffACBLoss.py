import torch




class boostedffACBLoss(torch.nn.Module):
    def __init__(self, zero_weighting=1, nonzero_weighting=1, fullframe_weighting=1, ff_loss='mse', boost=1):
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
        self.boost = boost

        if ff_loss == 'mse':
            self.ff_loss = torch.nn.MSELoss(reduction='mean')
        elif ff_loss == 'mae':
            self.ff_loss = torch.nn.L1Loss(reduction='mean')
        elif ff_loss == 'bce':
            self.ff_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

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

        zero_loss = self.mse_loss(corresponding_values_zero * self.boost, values_zero * self.boost)
        nonzero_loss = self.mse_loss(corresponding_values_nonzero * self.boost, values_nonzero * self.boost)
        full_frame_loss = self.ff_loss(reconstructed_image * self.boost, target_image * self.boost)

        if torch.isnan(zero_loss):
            zero_loss = 0
        if torch.isnan(nonzero_loss):
            nonzero_loss = 0

        weighted_mse_loss = (self.zero_weighting * zero_loss) + (self.nonzero_weighting * nonzero_loss) + (self.fullframe_weighting * full_frame_loss)

        return weighted_mse_loss
