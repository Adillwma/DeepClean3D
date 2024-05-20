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



class DC3D_Inference:
    def __init__(self, time_dimension, reconstruction_threshold, latent_dim, pretrained_model_path, AE_file_folder_path):
        self.time_dimension = time_dimension
        self.reconstruction_threshold = reconstruction_threshold
        self.encoder, self.decoder = self.setup_encoder_decoder(latent_dim, pretrained_model_path, AE_file_folder_path)

    # Import and prepare Autoencoder model
    def setup_encoder_decoder(self, latent_dim, pretrained_model_path, AE_file_folder_path):
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
        encoder.double()   # allow setting precision by variable from model folder
        decoder.double()

        # load the full state dictionary into memory
        full_state_dict = torch.load(pretrained_model_path)

        # load the state dictionaries into the models
        encoder.load_state_dict(full_state_dict['encoder_state_dict'])
        decoder.load_state_dict(full_state_dict['decoder_state_dict'])

        encoder.eval()  
        decoder.eval()

        return encoder, decoder                                

    def custom_normalisation(self, data):
        data = ((data / self.time_dimension) / (1/(1-self.reconstruction_threshold))) + self.reconstruction_threshold
        for row in data:    
            for i, ipt in enumerate(row):
                if ipt == self.reconstruction_threshold:
                    row[i] = 0
        return data

    def custom_renormalisation(self, data):
        data = np.where(data > self.reconstruction_threshold, ((data - self.reconstruction_threshold)*(1/(1-self.reconstruction_threshold)))*(self.time_dimension), 0)
        return data

    def deepclean3(self, input_image_tensor):
            with torch.no_grad():
                norm_image = self.custom_normalisation(input_image_tensor)
                image_prepared = norm_image.unsqueeze(0).unsqueeze(0)
                rec_image = self.decoder(self.encoder(image_prepared))
                rec = rec_image.squeeze().numpy()
                rec_image_renorm = self.custom_renormalisation(rec)
            return rec_image_renorm

    def masking_recovery(self, input_image, recovered_image, print_result):
        raw_input_image = input_image.clone()
        net_recovered_image = recovered_image.copy()
        masking_pixels = np.count_nonzero(net_recovered_image)
        image_shape = net_recovered_image.shape
        total_pixels = image_shape[0] * image_shape[1] * time_dimension
        if print_result:
            print(f"Total number of pixels in the timescan: {format(total_pixels, ',')}\nNumber of pixels returned by the masking: {format(masking_pixels, ',')}\nNumber of pixels removed from reconstruction by masking: {format(total_pixels - masking_pixels, ',')}")

        mask_indexs = np.where(net_recovered_image != 0)
        net_recovered_image[mask_indexs] = raw_input_image[mask_indexs]
        result = net_recovered_image
        return result

    def run(self, input_image_tensor):
        recovered_image = self.deepclean3(input_image_tensor)
        masking_rec_image = self.masking_recovery(input_image_tensor, recovered_image, print_result=False)
        return recovered_image, masking_rec_image


if __name__ == "__main__":

    #%% - Run the DeepClean3D function on the input data
    #User Inputs
    input_images_path = "N:\Yr 3 Project Datasets\RDT 50KM Fix\Data\\"      #Dataset 24_X10ks\Data"
    output_images_path = "N:\Year 3 + Project Continued\Results\class\\"
    reconstruction_threshold = 0.5

    #AE Settings
    time_dimension = 1000
    latent_dim = 10
    model_name =  "RDT 10kM tdim1000 AE2PROTECT 30 sig"

    # Paths
    pretrained_model_path = f"N:\\Yr 3 Project Results\\Archived\\{model_name} - Training Results\\{model_name} - Model + Optimiser State Dicts.pth"
    AE_file_folder_path = f"N:\\Yr 3 Project Results\\Archived\\{model_name} - Training Results\\"

    # Inference Driver
    dc3d_inference = DC3D_Inference(time_dimension, reconstruction_threshold, latent_dim, pretrained_model_path, AE_file_folder_path)
    
    # Demo Run using dummy data on disk (Replace with actual data input pipeline)
    file_list = [f for f in os.listdir(input_images_path) if f.endswith('.npy')]
    file_list.sort(key=lambda x: os.path.getctime(os.path.join(input_images_path, x)))

    # Run Inference on all files in the input folder
    for i, file_name in tqdm(enumerate(file_list), desc='Img Files'):
        input_image_path = os.path.join(input_images_path, file_name)
        input_image_tensor = torch.from_numpy(np.load(input_image_path))

        recovered_image, masking_rec_image = dc3d_inference.run(input_image_tensor)

        if not os.path.exists(output_images_path + "\Direct_Output\\"):
            os.makedirs(output_images_path + "\Direct_Output\\")
        if not os.path.exists(output_images_path + "\Masked_Output\\"):
            os.makedirs(output_images_path + "\Masked_Output\\")

        recovered_image_path = os.path.join(output_images_path + "\Direct_Output\\", f'{file_name[:-4]}_recovered.npy')
        np.save(recovered_image_path, recovered_image)

        masking_rec_image_path = os.path.join(output_images_path + "\Masked_Output\\", f'{file_name[:-4]}_masking_rec.npy')
        np.save(masking_rec_image_path, masking_rec_image)

    print("Program Completed")