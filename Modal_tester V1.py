# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 2022
Modal_tester V1.0.0
Author: Adill Al-Ashgar
University of Bristol

# Modal_tester V1
If the saved model is successfully loaded, the program will print out the contents of the model. 
If the "AttributeError" occurs, you can try to investigate the saved model file to check whether it 
contains the expected "Encoder" attribute. You can also check the code used to save the model to ensure 
that the "Encoder" attribute is included.

### Posible improvements
# 
"""
#%% - Dependancies
import torch

#%% - Function
def modal_tester(model_filename, model_filepath):
    # Construct full file path
    model_full_path = model_filepath + model_filename + ".pth"

    # Return print to user so they can verify correct path 
    print(f"Loading Model: {model_filename} \nFrom path: {model_full_path}\n")

    # Test the model load
    try:
        encoder, decoder = torch.load(model_full_path)
        print(encoder, decoder)
        testpass = True

    # Error handling in case model loading has issues
    except: 
        print("\nError in loading modal! Please check that you selected the correct directory and that model filetype is .pth.\n")
        testpass = False

    return(testpass)       # Return boolean variable incase external code caling this function may need to be aware of the test status to enact its own conditional response

#%% - Test Driver
# User Paths for Saved Model
model_filename = "10X_Activation_V1"                # NOTE Do not add file extension, it is automaticalaly added. 
model_filepath = "C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/Git Hub Repos/DeepClean Repo/DeepClean-Noise-Suppression-for-LHC-B-Torch-Detector/Models/"                          # NOTE If in root dir then leave this as "" with nothing contained in the speech marks, otherise input dir here

if modal_tester(model_filename, model_filepath):
    print("continuing")
