# -*- coding: utf-8 -*-
"""
Robust Pytorch Model Exporter V1.0.0
@author: Adill Al-Ashgar
Created on Tue Feb 21 16:43:10 2023

#Implemented due to:
If you are saving/loading the entire model, then according to the documentation:

> The disadvantage of this (standard torch.save) approach is that the serialized data is bound to the specific classes 
and the exact directory structure used when the model is saved. The reason for this is because pickle 
does not save the model class itself. Rather, it saves a path to the file containing the class, which 
is used during load time. Because of this, your code can break in various ways when used in other 
projects or after refactors.

"""

import torch
import os

def save_pytorch_model(model, input_shape, output_shape, filename):
    """
    Saves a PyTorch model along with its input and output shapes to a file.
    Args:
        model (nn.Module): PyTorch model to save
        input_shape (tuple): shape of the input tensor(s) for the model
        output_shape (tuple): shape of the output tensor(s) for the model
        filename (str): name of the file to save the model to
    """
    # Save the state dictionary of the model
    state_dict_path = f"{filename}.pth"
    torch.save(model.state_dict(), state_dict_path)
    
    # Define a script to save the model architecture and other information
    script = f"""
        import torch
        import torch.nn as nn
        
        class {model.__class__.__name__}(nn.Module):
            def __init__(self):
                super({model.__class__.__name__}, self).__init__()
                {model.__repr__().replace('    ','').replace('\n','')}     # Cant fix it 
            
            def forward(self, input_tensor):
                return self.__call__(input_tensor)
        
        def get_model():
            model = {model.__class__.__name__}()
            model.load_state_dict(torch.load('{state_dict_path}'))
            model.eval()
            return model
        
        INPUT_SHAPE = {input_shape}
        OUTPUT_SHAPE = {output_shape}
    """
    
    # Save the script to a file
    script_filename = f"{filename}.py"
    with open(script_filename, "w") as f:
        f.write(script)
    
    # Print message to user
    print(f"PyTorch model saved to {script_filename} and {state_dict_path}")
