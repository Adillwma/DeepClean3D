# -*- coding: utf-8 -*-
"""
Robust Pytorch Model Exporter V1.0.0
@author: Adill Al-Ashgar
Created on Tue Feb 24 2023

#Implemented due to:
If you are saving/loading the entire model, then according to the documentation:

> The disadvantage of this (standard torch.save) approach is that the serialized data is bound to the specific classes 
and the exact directory structure used when the model is saved. The reason for this is because pickle 
does not save the model class itself. Rather, it saves a path to the file containing the class, which 
is used during load time. Because of this, your code can break in various ways when used in other 
projects or after refactors.



# Imporvement to be added 
# make sure it ifnds the file, if not foun in dir its in then just scan the whole working dir to make sure not missing it 

"""

import os
import shutil


def Robust_model_export(function_name, search_dir, output_dir, debug = False):

    def parse_module_name(string):
        # Protection from module being in subfolder so _get_module retuns subfolder.module_name we want just module name, so following lines check for any dots in the module name and if so find the right most dot and returns the string following it, otheriwse just retunrs it's input 
        dot_index = string.rfind('.')
        if dot_index != -1:
            left_string = string[:dot_index].replace('.', '\\')     # Text to left of right most dot (this is the subfolder(s) names if there is) the replace replaces any dots with / for proper folder naming
            right_string = string[dot_index+1:]  # Text to right of right most dot (this is module name)
            return right_string, left_string # Module name, subfolder
        else:
            return string, None # Module name, None (there is no subfolder in this case)

    # Find autoencoder module name
    module_name = function_name.__module__
    module_name, subfolder = parse_module_name(module_name)

    # Define the filename to search for ()
    filename = module_name + '.py'

    #cahnged from working dir to current file dir as the file is not in top level wdir
    #search_dir = os.getcwd()  # current working directory
    
    if subfolder != None:
        search_dir = search_dir + "\\" + subfolder
        if debug:
            print("subfolder")
    if debug:
        print("search loc", search_dir)
    # Traverse the directory tree to find the file
    file_path = None
    try:
        for root, dirs, files in os.walk(search_dir):
            if filename in files:
                file_path = os.path.join(root, filename)
                break
    except:
        print("Error! Autoencoder File Not Found for Copy")   #just to stop this not working from holding up the rest of the main program 
        pass

    # Copy the file to a new location with modified filename
    if file_path is None:
        print(f"File '{filename}' not found in '{search_dir}' or its subdirectories")
        AE_file_name = "AE run from main code body, not external file!"
    else:
        new_filename = os.path.splitext(filename)[0] + os.path.splitext(filename)[1]
        os.makedirs(output_dir, exist_ok=True)
        new_file_path = os.path.join(output_dir, new_filename)
        shutil.copyfile(file_path, new_file_path)
        if debug:
            print(f"File '{filename}' copied to '{new_file_path}'")
        AE_file_name = filename

    return (AE_file_name)




def Robust_model_export2(function_name, search_dir, debug = False):

    def parse_module_name(string):
        # Protection from module being in subfolder so _get_module retuns subfolder.module_name we want just module name, so following lines check for any dots in the module name and if so find the right most dot and returns the string following it, otheriwse just retunrs it's input 
        dot_index = string.rfind('.')
        if dot_index != -1:
            left_string = string[:dot_index].replace('.', '\\')     # Text to left of right most dot (this is the subfolder(s) names if there is) the replace replaces any dots with / for proper folder naming
            right_string = string[dot_index+1:]  # Text to right of right most dot (this is module name)
            return right_string, left_string # Module name, subfolder
        else:
            return string, None # Module name, None (there is no subfolder in this case)

    # Find autoencoder module name
    module_name = function_name.__module__
    module_name, subfolder = parse_module_name(module_name)

    # Define the filename to search for ()
    filename = module_name + '.py'

    #cahnged from working dir to current file dir as the file is not in top level wdir
    #search_dir = os.getcwd()  # current working directory
    
    if subfolder != None:
        search_dir = search_dir + "\\" + subfolder
        if debug:
            print("subfolder")
    if debug:
        print("search loc", search_dir)
    # Traverse the directory tree to find the file
    file_path = None
    try:
        for root, dirs, files in os.walk(search_dir):
            if filename in files:
                file_path = os.path.join(root, filename)
                break
    except:
        print("Error! Autoencoder File Not Found for Copy")   #just to stop this not working from holding up the rest of the main program 
        pass
    


    # read in the data from the file
    if file_path is None:
        print(f"File '{filename}' not found in '{search_dir}' or its subdirectories")
        AE_file_name = "AE run from main code body, not external file!"
    else:

        # Read the content of the Python file
        with open(file_path, 'r') as file:
            script_content = file.read()

        AE_file_name = filename

    return script_content, AE_file_name


