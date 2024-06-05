import torch 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm
from DC3D_V3.Helper_files.DC3D_Core_Functions import *
from Helper_files.Helper_Functions import plot_save_choice
from DC3D_V3_Trainer_3 import quantify_loss_performance


### Training Function
def train_epoch(epoch, encoder, decoder, device, dataloader, loss_fn, optimizer, time_dimension=100, reconstruction_threshold=0.5, print_partial_training_losses=False, masking_optimised_binary_norm=False, loss_vs_sparse_img=False, renorm_for_loss_calc=False, use_tensorboard=False, writer=None):
    """
    Training loop for a single epoch

    Args:
        encoder (torch model): The encoder model
        decoder (torch model): The decoder model
        device (torch device): The device to run the training on
        dataloader (torch dataloader): The dataloader to iterate over
        loss_fn (torch loss function): The loss function to use
        optimizer (torch optimizer): The optimizer to use
        signal_points (int) or (tuple): The number of signal points to retain in the signal sparsification preprocess. If given as int then number will be constant over training, if given as tuple then a random value will be selected from the range given for each batch
        noise_points (int) or (tuple): The number of noise points to add in the preprocessing. If given as int then number will be constant over training, if given as tuple then a random value will be selected from the range given for each batch. Default = 0
        x_std_dev (float) or (tuple): The standard deviation of the gaussian distribution to draw the x shift from. If given as float then number will be constant over training, if given as tuple then a random value will be selected from the range given for each batch. Default = 0
        y_std_dev (float) or (tuple): The standard deviation of the gaussian distribution to draw the y shift from. If given as float then number will be constant over training, if given as tuple then a random value will be selected from the range given for each batch. Default = 0
        tof_std_dev (float) or (tuple): The standard deviation of the gaussian distribution to draw the ToF shift from. If given as float then number will be constant over training, if given as tuple then a random value will be selected from the range given for each batch. Default = 0
        time_dimension (int): The number of time steps in the data set, used to set the upper limit of the noise point values amonst other things. Default = 100
        reconstruction_threshold (float): The threshold used in the custom normalisation, also used to set the lower limit of the noise point values. Default = 0.5
        print_partial_training_losses (bool): A flag to set if the partial training losses are printed to terminal. If set to true partial loss is printed to termial whilst training. If set to false the a TQDM progress bar is used to show processing progress. Default = False
        masking_optimised_binary_norm (bool): A flag to set if the masking optimised binary normalisation is used. If set to true the masking optimised binary normalisation is used, if set to false the gaped normalisation is used. Default = False  [EXPLAIN IN MORE DETAIL!!!!!]
        loss_vs_sparse_img (bool): A flag to set if the loss is calculated against the sparse image or the clean image. If set to true the loss is calculated against the sparse image, if set to false the loss is calculated against the clean image. Default = False

    Returns:
        loss_total/batches (float): The average loss over the epoch calulated by dividing the total sum of loss by the number of batches !! EXPLAIN THIS SIMPLER !!! 

    """
    encoder.train()   
    decoder.train()   

    loss_total = 0.0
    batches = 0

    iterator = (dataloader) if print_partial_training_losses else tqdm(dataloader, desc='Batches', leave=False)                  # If print_partial_training_losses is true then we just iterate the dataloder without genrating a progress bar as partial losses will be printed instead. If set to 'False' use the tqdm progress bar wrapper for the dataset for user feedback on progress
    for image_batch, sparse_output_batch, sparse_and_resolution_limited_batch, noised_sparse_reslimited_batch in iterator: 

        # DATA PREPROCESSING
        with torch.no_grad(): # No need to track the gradients
            if masking_optimised_binary_norm:
                normalised_batch = mask_optimised_normalisation(noised_sparse_reslimited_batch)
                norm_sparse_output_batch = mask_optimised_normalisation(sparse_output_batch)
                normalised_inputs = mask_optimised_normalisation(image_batch)
            else:
                normalised_batch = gaped_normalisation(noised_sparse_reslimited_batch, reconstruction_threshold, time_dimension)
                norm_sparse_output_batch = gaped_normalisation(sparse_output_batch, reconstruction_threshold, time_dimension)
                normalised_inputs = gaped_normalisation(image_batch, reconstruction_threshold, time_dimension)
            
        # Move tensor to the proper device
        image_clean = normalised_inputs.to(device) # Move the clean image batch to the device
        image_sparse = norm_sparse_output_batch.to(device) # Move the sparse image batch to the device
        image_noisy = normalised_batch.to(device) # Move the noised image batch to the device
        
        # Encode data
        encoded_data = encoder(image_noisy) # Encode the noised image batch
        # Decode data
        decoded_data = decoder(encoded_data) # Decode the encoded image batch
        
        if loss_vs_sparse_img:
            loss_comparator = image_sparse
        else:
            loss_comparator = image_clean

        # Evaluate loss
        if renorm_for_loss_calc:
            decoded_data = gaped_renormalisation_torch(decoded_data, reconstruction_threshold, time_dimension)
            loss_comparator = gaped_renormalisation_torch(loss_comparator, reconstruction_threshold, time_dimension)

        loss = loss_fn(decoded_data, loss_comparator)  # Compute the loss between the decoded image batch and the clean image batch
        
        # Backward pass
        optimizer.zero_grad() # Reset the gradients
        loss.backward() # Compute the gradients
        optimizer.step() # Update the parameters
        batches += 1
        loss_total += loss.item()
        avg_epoch_loss = loss_total/batches

        if print_partial_training_losses:         # Prints partial train losses per batch
            print('\t partial train loss (single batch): %f' % (loss.data))  # Print batch loss value
    
        if use_tensorboard:
            # Add the gradient values to Tensorboard
            for name, param in encoder.named_parameters():
                writer.add_histogram(name + '/grad', param.grad, global_step=epoch)

            for name, param in decoder.named_parameters():
                writer.add_histogram(name + '/grad', param.grad, global_step=epoch)

            writer.add_scalar('Loss/train', avg_epoch_loss, epoch)

    return avg_epoch_loss

### Testing Function
def test_epoch(encoder, decoder, device, dataloader, loss_fn, time_dimension=100, reconstruction_threshold=0.5, print_partial_training_losses=False, masking_optimised_binary_norm=False, loss_vs_sparse_img=False, renorm_for_loss_calc=False):
    """
    Testing (Evaluation) loop for a single epoch. This function is identical to the training loop except that it does not perform the backward pass and parameter update steps and the model is run in eval mode. Additionaly the dataset used is the test dataset rather than the training dataset so that the data is unseen by the model.

    Args:

        encoder (torch model): The encoder model
        decoder (torch model): The decoder model
        device (torch device): The device to run the training on
        dataloader (torch dataloader): The dataloader to iterate over
        loss_fn (torch loss function): The loss function to use
        signal_points (int) or (tuple): The number of signal points to retain in the signal sparsification preprocess. If given as int then number will be constant over training, if given as tuple then a random value will be selected from the range given for each batch
        noise_points (int) or (tuple): The number of noise points to add in the preprocessing. If given as int then number will be constant over training, if given as tuple then a random value will be selected from the range given for each batch. Default = 0
        x_std_dev (float) or (tuple): The standard deviation of the gaussian distribution to draw the x shift from. If given as float then number will be constant over training, if given as tuple then a random value will be selected from the range given for each batch. Default = 0
        y_std_dev (float) or (tuple): The standard deviation of the gaussian distribution to draw the y shift from. If given as float then number will be constant over training, if given as tuple then a random value will be selected from the range given for each batch. Default = 0
        tof_std_dev (float) or (tuple): The standard deviation of the gaussian distribution to draw the ToF shift from. If given as float then number will be constant over training, if given as tuple then a random value will be selected from the range given for each batch. Default = 0
        time_dimension (int): The number of time steps in the data set, used to set the upper limit of the noise point values amonst other things. Default = 100
        reconstruction_threshold (float): The threshold used in the custom normalisation, also used to set the lower limit of the noise point values. Default = 0.5
        print_partial_training_losses (bool): A flag to set if the partial training losses are printed to terminal. If set to true partial loss is printed to termial whilst training. If set to false the a TQDM progress bar is used to show processing progress. Default = False
        masking_optimised_binary_norm (bool): A flag to set if the masking optimised binary normalisation is used. If set to true the masking optimised binary normalisation is used, if set to false the gaped normalisation is used. Default = False  [EXPLAIN IN MORE DETAIL!!!!!]
        loss_vs_sparse_img (bool): A flag to set if the loss is calculated against the sparse image or the clean image. If set to true the loss is calculated against the sparse image, if set to false the loss is calculated against the clean image. Default = False

    Returns:
        loss_total/batches (float): The average loss over the epoch calulated by dividing the total sum of loss by the number of batches !! EXPLAIN THIS SIMPLER !!!

    """
    # Set evaluation mode for encoder and decoder
    encoder.eval() # Evaluation mode for the encoder
    decoder.eval() # Evaluation mode for the decoder
    with torch.no_grad(): # No need to track the gradients

        loss_total = 0.0
        batches = 0
        
        iterator = (dataloader) if print_partial_training_losses else tqdm(dataloader, desc='Testing', leave=False, colour="yellow")                  # If print_partial_training_losses is true then we just iterate the dataloder without genrating a progress bar as partial losses will be printed instead. If set to 'False' use the tqdm progress bar wrapper for the dataset for user feedback on progress
        for image_batch, sparse_output_batch, sparse_and_resolution_limited_batch, noised_sparse_reslimited_batch in iterator: 

            if masking_optimised_binary_norm:
                normalised_batch = mask_optimised_normalisation(noised_sparse_reslimited_batch)
                image_batch_norm = mask_optimised_normalisation(image_batch)
            else:
                normalised_batch = gaped_normalisation(noised_sparse_reslimited_batch, reconstruction_threshold, time_dimension)
                image_batch_norm = gaped_normalisation(image_batch, reconstruction_threshold, time_dimension)
            
            image_clean = image_batch_norm.to(device) # Move the clean image batch to the device
            image_noisy = normalised_batch.to(device) # Move the noised image batch to the device

            # Encode data
            encoded_data = encoder(image_noisy) # Encode the noised image batch
            # Decode data
            decoded_data = decoder(encoded_data) # Decode the encoded image batch

            if loss_vs_sparse_img:
                loss_comparator = sparse_output_batch
            else:
                loss_comparator = image_clean

            # Evaluate loss
            if renorm_for_loss_calc:
                decoded_data = gaped_renormalisation_torch(decoded_data, reconstruction_threshold, time_dimension)
                loss_comparator = gaped_renormalisation_torch(loss_comparator, reconstruction_threshold, time_dimension)

            # Evaluate loss
            loss = loss_fn(decoded_data, loss_comparator)  # Compute the loss between the decoded image batch and the clean image batch
            batches += 1
            loss_total += loss.item()

            #Run additional perfomrnace metric loss functions for final plots, this needs cleaning up!!!!!
            quantify_loss_performance(loss_comparator, decoded_data, time_dimension)

    return loss_total/batches

### Validation Function
def validation_routine(encoder, decoder, device, dataloader, loss_fn, time_dimension=100, reconstruction_threshold=0.5, print_partial_training_losses=False, masking_optimised_binary_norm=False, loss_vs_sparse_img=False, renorm_for_loss_calc=False):
    """
    Validation loop for a single epoch. This function is identical to the test/evaluation loop except that it is used for hyperparamter evaluation to evaluate between differnt models. This function is not used during training, only for hyperparameter evaluation. Again it uses a previosuly unseen dataset howevr this one is fixed and not randomly selected from the dataset so as to provide a fixed point of reference for direct model comparison.
    
    Args:
        encoder (torch model): The encoder model
        decoder (torch model): The decoder model
        device (torch device): The device to run the training on
        dataloader (torch dataloader): The dataloader to iterate over
        loss_fn (torch loss function): The loss function to use
        signal_points (int) or (tuple): The number of signal points to retain in the signal sparsification preprocess. If given as int then number will be constant over training, if given as tuple then a random value will be selected from the range given for each batch
        noise_points (int) or (tuple): The number of noise points to add in the preprocessing. If given as int then number will be constant over training, if given as tuple then a random value will be selected from the range given for each batch. Default = 0
        x_std_dev (float) or (tuple): The standard deviation of the gaussian distribution to draw the x shift from. If given as float then number will be constant over training, if given as tuple then a random value will be selected from the range given for each batch. Default = 0
        y_std_dev (float) or (tuple): The standard deviation of the gaussian distribution to draw the y shift from. If given as float then number will be constant over training, if given as tuple then a random value will be selected from the range given for each batch. Default = 0
        tof_std_dev (float) or (tuple): The standard deviation of the gaussian distribution to draw the ToF shift from. If given as float then number will be constant over training, if given as tuple then a random value will be selected from the range given for each batch. Default = 0
        time_dimension (int): The number of time steps in the data set, used to set the upper limit of the noise point values amonst other things. Default = 100
        reconstruction_threshold (float): The threshold used in the custom normalisation, also used to set the lower limit of the noise point values. Default = 0.5
        print_partial_training_losses (bool): A flag to set if the partial training losses are printed to terminal. If set to true partial loss is printed to termial whilst training. If set to false the a TQDM progress bar is used to show processing progress. Default = False
        masking_optimised_binary_norm (bool): A flag to set if the masking optimised binary normalisation is used. If set to true the masking optimised binary normalisation is used, if set to false the gaped normalisation is used. Default = False  [EXPLAIN IN MORE DETAIL!!!!!]
        loss_vs_sparse_img (bool): A flag to set if the loss is calculated against the sparse image or the clean image. If set to true the loss is calculated against the sparse image, if set to false the loss is calculated against the clean image. Default = False

    Returns:
        loss_total/batches (float): The average loss over the epoch calulated by dividing the total sum of loss by the number of batches !! EXPLAIN THIS SIMPLER !!!

    """
    # Set evaluation mode for encoder and decoder
    encoder.eval() # Evaluation mode for the encoder
    decoder.eval() # Evaluation mode for the decoder
    with torch.no_grad(): # No need to track the gradients

        loss_total = 0.0
        batches = 0

        iterator = (dataloader) if print_partial_training_losses else tqdm(dataloader, desc='Validation', leave=False, colour="green")                  # If print_partial_training_losses is true then we just iterate the dataloder without genrating a progress bar as partial losses will be printed instead. If set to 'False' use the tqdm progress bar wrapper for the dataset for user feedback on progress
        for image_batch, sparse_output_batch, sparse_and_resolution_limited_batch, noised_sparse_reslimited_batch in iterator: 

            if masking_optimised_binary_norm:
                normalised_batch = mask_optimised_normalisation(noised_sparse_reslimited_batch)
                image_batch_norm = mask_optimised_normalisation(image_batch)
            else:
                normalised_batch = gaped_normalisation(noised_sparse_reslimited_batch, reconstruction_threshold, time_dimension)
                image_batch_norm = gaped_normalisation(image_batch, reconstruction_threshold, time_dimension)
            
            image_clean = image_batch_norm.to(device) # Move the clean image batch to the device
            image_noisy = normalised_batch.to(device) # Move the noised image batch to the device

            # Encode data
            encoded_data = encoder(image_noisy) # Encode the noised image batch
            # Decode data
            decoded_data = decoder(encoded_data) # Decode the encoded image batch

            if loss_vs_sparse_img:
                loss_comparator = sparse_output_batch
            else:
                loss_comparator = image_clean

            # Evaluate loss
            if renorm_for_loss_calc:
                decoded_data = gaped_renormalisation_torch(decoded_data, reconstruction_threshold, time_dimension)
                loss_comparator = gaped_renormalisation_torch(loss_comparator, reconstruction_threshold, time_dimension)

            # Evaluate loss
            loss = loss_fn(decoded_data, loss_comparator)  # Compute the loss between the decoded image batch and the clean image batch
            batches += 1
            loss_total += loss.item()

            #Run additional perfomrnace metric loss functions for final plots, this needs cleaning up!!!!!
            #quantify_loss_performance(loss_comparator, decoded_data, time_dimension)
    
    return loss_total/batches



