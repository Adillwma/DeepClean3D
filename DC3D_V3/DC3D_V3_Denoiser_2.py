# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 2023
DeepClean Model Runner
@author: Adill Al-Ashgar
"""
"""
This is simply code that takes in a noised image and runs one of the saved, trained models on it to denoise it.
"""
import torch
#from Autoencoders.DC3D_Autoencoder_V1 import Encoder, Decoder
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import numpy as np  
from tqdm import tqdm  # Progress bar
import importlib.util
import re
from tqdm import tqdm

#User Inputs
input_images_path = "N:\Yr 3 Project Datasets\RDT 50KM\Data\\"      #Dataset 24_X10ks\Data"
output_images_path = "N:\Year 3 + Project Continued\Results\\"
time_dimension = 1000

#AE Settings
reconstruction_threshold = 0.5
latent_dim = 10
model_name =  "RDT 10kM tdim1000 AE2PROTECT 30 sig"

#Path settings
pretrained_model_path = f"N:\\Yr 3 Project Results\\{model_name} - Training Results\\{model_name} - Model + Optimiser State Dicts.pth"
AE_file_folder_path = f"N:\\Yr 3 Project Results\\{model_name} - Training Results\\"

#%% - Functions

# Import and prepare Autoencoder model
def setup_encoder_decoder(latent_dim,pretrained_model_path, AE_file_folder_path):
    #from DC3D_Autoencoder_V1 import Encoder, Decoder   # make this programatic from the model folder as it also contains the backup AE file

    def import_encoder_decoder(folder_path):
        module_name_pattern = r"DC3D_Autoencoder_V\w+\.py"
        module_path_pattern = os.path.join(folder_path, module_name_pattern)

        matching_files = [file for file in os.listdir(folder_path) if re.match(module_name_pattern, file)]
        if not matching_files:
            raise ImportError(f"No DC3D_Autoencoder module found in {folder_path}\n")

        module_name = matching_files[0][:-3]
        module_path = os.path.join(folder_path, f"{module_name}.py")
        print(f"Loaded {module_name}")
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module.Encoder, module.Decoder
    
    Encoder, Decoder = import_encoder_decoder(AE_file_folder_path)
    encoder = Encoder(encoded_space_dim=latent_dim, fc2_input_dim=128, encoder_debug=False, record_activity=False)
    decoder = Decoder(encoded_space_dim=latent_dim, fc2_input_dim=128, decoder_debug=False, record_activity=False)
    encoder.double()   
    decoder.double()

    # load the full state dictionary into memory
    full_state_dict = torch.load(pretrained_model_path)

    # load the state dictionaries into the models
    encoder.load_state_dict(full_state_dict['encoder_state_dict'])
    decoder.load_state_dict(full_state_dict['decoder_state_dict'])

    encoder.eval()                                   #.eval() is a kind of switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time. For example, Dropouts Layers, BatchNorm Layers etc. You need to turn off them during model evaluation, and .eval() will do it for you. In addition, the common practice for evaluating/validation is using torch.no_grad() in pair with model.eval() to turn off gradients computation
    decoder.eval()    
    return encoder, decoder





def custom_normalisation(data, reconstruction_threshold, time_dimension=100):
    data = ((data / time_dimension) / (1/(1-reconstruction_threshold))) + reconstruction_threshold
    for row in data:   ###REPLACE USING NP.WHERE
        for i, ipt in enumerate(row):
            if ipt == reconstruction_threshold:
                row[i] = 0
    return data

def custom_renormalisation(data, reconstruction_threshold, time_dimension=100):
    data = np.where(data > reconstruction_threshold, ((data - reconstruction_threshold)*(1/(1-reconstruction_threshold)))*(time_dimension), 0)
    return data



#Following function runs the autoencoder on the input data
def deepclean3(input_image_tensor, reconstruction_threshold, encoder, decoder, time_dimension=100):
    
    with torch.no_grad():
        norm_image = custom_normalisation(input_image_tensor, reconstruction_threshold, time_dimension)
        image_prepared = norm_image.unsqueeze(0).unsqueeze(0)   #Adds two extra dimesnions to start of array so shape goes from (x,y) to (1,1,x,y) to represent batch and channel dims
        rec_image = decoder(encoder(image_prepared))                         #Creates a recovered image (denoised image), by running a noisy image through the encoder and then the output of that through the decoder.
        rec = rec_image.squeeze().numpy()
        rec_image_renorm = custom_renormalisation(rec, reconstruction_threshold, time_dimension)
    return rec_image_renorm

# Masking technique
def masking_recovery(input_image, recovered_image, print_result=True):
    raw_input_image = input_image.clone()
    net_recovered_image = recovered_image.copy()
    #Evaluate usefullness 
    # count the number of non-zero values
    masking_pixels = np.count_nonzero(net_recovered_image)
    image_shape = net_recovered_image.shape
    total_pixels = image_shape[0] * image_shape[1] * time_dimension
    # print the count
    if print_result:
        print(f"Total number of pixels in the timescan: {format(total_pixels, ',')}\nNumber of pixels returned by the masking: {format(masking_pixels, ',')}\nNumber of pixels removed from reconstruction by masking: {format(total_pixels - masking_pixels, ',')}")

    # use np.where and boolean indexing to update values in a
    mask_indexs = np.where(net_recovered_image != 0)
    net_recovered_image[mask_indexs] = raw_input_image[mask_indexs]
    result = net_recovered_image
    return result

### tests our model
def deepclean_images(input_images_path, output_images_path, time_dimension, reconstruction_threshold, latent_dim, pretrained_model_path, AE_file_folder_path):


    # Setup encoder and decoder and load models
    encoder, decoder = setup_encoder_decoder(latent_dim, pretrained_model_path, AE_file_folder_path)

    ### Load input images dataset
    # Get a list of all the .npy files in the folder
    file_list = [f for f in os.listdir(input_images_path) if f.endswith('.npy')]
    #Sort list by date created
    file_list.sort(key=lambda x: os.path.getctime(os.path.join(input_images_path, x)))

    ### Loop through the files 
    for i, file_name in tqdm(enumerate(file_list), desc='Img Files'):
        input_image_path = os.path.join(input_images_path, file_name)
        
        # Load the input image
        input_image = np.load(input_image_path)
        input_image_tensor = torch.from_numpy(input_image)

        # Run the autoencoder on the input image
        recovered_image = deepclean3(input_image_tensor, reconstruction_threshold, encoder, decoder, time_dimension)
        masking_rec_image = masking_recovery(input_image_tensor, recovered_image, print_result=False)

        #Create the output folders if they do not exist
        if not os.path.exists(output_images_path + "\Direct_Output\\"):
            os.makedirs(output_images_path + "\Direct_Output\\")
        if not os.path.exists(output_images_path + "\Masked_Output\\"):
            os.makedirs(output_images_path + "\Masked_Output\\")

        # Save the recovered image
        recovered_image_path = os.path.join(output_images_path + "\Direct_Output\\", f'{file_name[:-4]}_recovered.npy')
        np.save(recovered_image_path, recovered_image)

        # Save the recovered image
        masking_rec_image_path = os.path.join(output_images_path + "\Masked_Output\\", f'{file_name[:-4]}_masking_rec.npy')
        np.save(masking_rec_image_path, masking_rec_image)
    
    print("Program Completed")


#%% - Run the DeepClean3D function on the input data

#Func Driver
deepclean_images(input_images_path, output_images_path, time_dimension, reconstruction_threshold, latent_dim, pretrained_model_path, AE_file_folder_path)

