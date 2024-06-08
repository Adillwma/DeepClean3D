import torch
import os
from functools import partial


#%% - Network Hook Functions
def activation_hook_fn(module, input, output, layer_index, activations):
    """
    This function will be called whenever a layer is called during the forward pass.
    It records the activations and saves them in the specified dictionary.
    """
    activations[layer_index] = output.detach()

def weights_hook_fn(module, input, output, layer_index, weights_data):
    """
    This function will be called whenever a layer is called during the forward pass.
    It records the weights and saves them in the specified dictionary.
    """
    weights_data[layer_index] = module.weight.data.clone().detach()

def biases_hook_fn(module, input, output, layer_index, biases_data):
    """
    This function will be called whenever a layer is called during the forward pass.
    It records the biases and saves them in the specified dictionary.
    """
    biases_data[layer_index] = module.bias.data.clone().detach()

def register_network_hooks(encoder, decoder, record_activity, record_weights, record_biases, activations, weights_data, biases_data, debug=False):
    # Loop through all the modules (layers) in the encoder and register the hooks
    for idx, module in enumerate(encoder.encoder_lin.modules()):
        if isinstance(module, torch.nn.Linear):
            if debug:   
                print("Registering hooks for encoder layer: ", idx)
            if record_activity:
                # Register the hook with the layer_index as the key to identify activations for this layer
                module.register_forward_hook(partial(activation_hook_fn, layer_index=idx, activations=activations))
            if record_weights:
                # Register the hook with the layer_index as the key to identify activations for this layer
                module.register_forward_hook(partial(weights_hook_fn, layer_index=idx, weights_data=weights_data))
            if record_biases:
                # Register the hook with the layer_index as the key to identify activations for this layer
                module.register_forward_hook(partial(biases_hook_fn, layer_index=idx, biases_data=biases_data))

    enc_max_idx = idx
    # Loop through all the modules (layers) in the decoder and register the hooks
    for idx, module in enumerate(decoder.decoder_lin.modules()):
        if isinstance(module, torch.nn.Linear):
            if debug:
                print("Registering hooks for decoder layer: ", enc_max_idx + idx)
            if record_activity:
                # Register the hook with the layer_index as the key to identify activations for this layer
                module.register_forward_hook(partial(activation_hook_fn, layer_index = enc_max_idx + idx, activations=activations))
            if record_weights:
                # Register the hook with the layer_index as the key to identify activations for this layer
                module.register_forward_hook(partial(weights_hook_fn, layer_index = enc_max_idx + idx, weights_data=weights_data))
            if record_biases:
                # Register the hook with the layer_index as the key to identify activations for this layer
                module.register_forward_hook(partial(biases_hook_fn, layer_index = enc_max_idx + idx, biases_data=biases_data))

    print("All hooks registered\n")


### FIX !!!! 
def write_hook_data_to_disk_and_clear(activations, weights_data, biases_data, epoch, output_dir):
    """
    This function saves the activation, weights and biases tracking data to disk and then clears the tracking dictionaries to avoid unnecaesary memory usage and potential overflows during longer training runs.

    Args:
        activations (dict): A dictionary containing the activations of each layer in the network
        weights_data (dict): A dictionary containing the weights of each layer in the network
        biases_data (dict): A dictionary containing the biases of each layer in the network
        epoch (int): The current epoch number
        output_dir (str): The path to the directory to save the data to
    """

    if len(activations) != 0:
        activ_path = output_dir + "Activation Data/"
        os.makedirs(activ_path, exist_ok=True)
        # Save the activations to a file named 'activations_epoch_{epoch}.pt'
        torch.save(activations, activ_path + f'activations_epoch_{epoch}.pt')

        # Clear the activations dictionary to free up memory
        activations.clear()

    if len(weights_data) != 0:
        weight_path = output_dir + "Weights Data/"
        os.makedirs(weight_path, exist_ok=True)
        # Save the weights to a file named 'weights_epoch_{epoch}.pt'
        torch.save(weights_data, weight_path + f'weights_epoch_{epoch}.pt')

        # Clear the weights dictionary to free up memory
        weights_data.clear()

    if len(biases_data) != 0:
        bias_path = output_dir + "Biases Data/"
        os.makedirs(bias_path, exist_ok=True)
        # Save the weights to a file named 'weights_epoch_{epoch}.pt'
        torch.save(biases_data, bias_path + f'biases_epoch_{epoch}.pt')

        # Clear the weights dictionary to free up memory
        biases_data.clear()