# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 2023
DeepClean Model Runner
@author: Adill Al-Ashgar
"""
import torch
from torchinfo import summary
from Autoencoders.DC3D_Autoencoder_V1 import Encoder, Decoder
import numpy as np
import matplotlib.pyplot as plt

###Plotting Function
def DeepClean3D(image_noisy, model_file_path):       #Defines a function for plotting the output of the autoencoder. And also the input + clean training data? Function takes inputs, 'encoder' and 'decoder' which are expected to be classes (defining the encode and decode nets), 'n' which is the number of ?????Images in the batch????, and 'noise_factor' which is a multiplier for the magnitude of the added noise allowing it to be scaled in intensity.  
    if image_noisy.ndim == 2:
        x_px_len = len(image_noisy[0])
        y_px_len = len(image_noisy[:,0])
    else:
        print("The array has a different number of dimensions")
        
    encoder, decoder = torch.load(model_file_path)
    print("MODEL LOADED SUCCESSFULLY")
    
    encoder.double()   
    decoder.double()

    #Following section checks if a CUDA enabled GPU is available. If found it is selected as the 'device' to perform the tensor opperations. If no CUDA GPU is found the 'device' is set to CPU (much slower) 
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')  #Informs user if running on CPU or GPU - (NVIDIA CUDA)
        
    #Following section moves both the encoder and the decoder to the selected device i.e detected CUDA enabled GPU or to CPU
    encoder.to(device)   #Moves encoder to selected device, CPU/GPU
    decoder.to(device)   #Moves decoder to selected device, CPU/GPU

    #Following section sets the autoencoder to evaluation mode rather than training (up till line 'with torch.no_grad()')
    encoder.eval()                                   #.eval() is a kind of switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time. For example, Dropouts Layers, BatchNorm Layers etc. You need to turn off them during model evaluation, and .eval() will do it for you. In addition, the common practice for evaluating/validation is using torch.no_grad() in pair with model.eval() to turn off gradients computation
    decoder.eval()                                   #Simarlary as above

    #print(summary(encoder, input_size=((1, 1, y_px_len, x_px_len))))   #!!!!UPDATE TO AUTOMATICALLY DETERMINE X AND Y SIZES????
    image_noisy = image_noisy[np.newaxis, np.newaxis, :, :]
    
    with torch.no_grad():                                               #As mentioned in .eval() comment, the common practice for evaluating/validation is using torch.no_grad() which turns off gradients computation whilst evaluating the model (the opposite of training the model)     
    #Following line runs the autoencoder on the noised data
        rec_img  = decoder(encoder(image_noisy.double())) # Sets input img to double prescision (fp64) then runs through encoder then decoder in series
    rec_img = np.squeeze(rec_img) #Removes batch and channel dims
    return (rec_img)  #return the recovered image

"""
model_name = "10X_Activation_V1"
model_path = "C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/Git Hub Repos/DeepClean Repo/DeepClean-Noise-Suppression-for-LHC-B-Torch-Detector/Models/"
model_full_file_path = model_path + model_name +".pth"
AE_file_path = model_path + model_name +".pth"

im_in = torch.tensor(np.zeros((128,88), dtype=np.float32))
plt.imshow(im_in)
plt.show()
ret_im = DeepClean3D(im_in, model_full_file_path)
plt.imshow(ret_im)
plt.show()
"""