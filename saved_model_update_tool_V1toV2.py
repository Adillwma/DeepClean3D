import os
import pickle
import torch

def find_data(path):

    filenames = os.listdir(path)
    # find the filepath for a file that ends with .pth and begins with "Statedicts"
    for file in filenames:
        # check if already processed the folder (for multiple future runs on same main results folder)
        if file == "Combined_V2_checkpoint.pth": ### imporve this method of skipping because atm it still loads most data from the folders as it loops looking for this, just search for this filename directly before the loop???
            return False

        if file.endswith(".py"):
            with open(os.path.join(path, file), "r") as f:
                script = f.read()


        elif file.endswith(".pkl") and file.startswith("deployment_variables_fc_input_dim_int"):
            with open(os.path.join(path, file), 'rb') as f:
                fc_input_dim = pickle.load(f)

        # if filename includes word precision at all  
        elif file.endswith(".pkl") and "precision" in file:
            with open(os.path.join(path, file), 'rb') as f:
                precision = pickle.load(f)

        elif file.endswith(".pkl") and file.startswith("deployment_variables_ld_int"):
            with open(os.path.join(path, file), 'rb') as f:
                latent_dim = pickle.load(f)

    
        elif file.endswith(".pth") and file.startswith("Statedicts"):
            statedict = os.path.join(path, file)
            statedicts_all = torch.load(statedict, map_location=torch.device('cpu'))
            encoder_state_dict = statedicts_all["encoder_state_dict"]
            decoder_state_dict = statedicts_all["decoder_state_dict"]
            optimiser_state_dict = statedicts_all["optimizer_state_dict"]

        elif file.endswith(".pth") and "State" in file:
            combined = os.path.join(path, file)
            combined_all = torch.load(combined, map_location=torch.device('cpu'))
            encoder_state_dict = combined_all["encoder_state_dict"]
            decoder_state_dict = combined_all["decoder_state_dict"]
            optimiser_state_dict = combined_all["optimizer_state_dict"]

    # make sure all the variables are loaded without causing any errors if they are not found
    try:
        script
    except NameError:
        script = None
    
    try:
        encoder_state_dict
    except NameError:
        encoder_state_dict = None   

    try:
        decoder_state_dict
    except NameError:
        decoder_state_dict = None

    try:
        optimiser_state_dict
    except NameError:
        optimiser_state_dict = None

    try:
        fc_input_dim
    except NameError:
        fc_input_dim = None

    try:
        precision
    except NameError:
        precision = None

    try:
        latent_dim
    except NameError:
        latent_dim = None

    if script != None and encoder_state_dict != None and decoder_state_dict != None and optimiser_state_dict != None and fc_input_dim != None and precision != None and latent_dim != None:

        # Construct the V2 model structure #  dict_keys(['encoder_state_dict', 'decoder_state_dict', 'optimizer_state_dict', 'latent_dim', 'fc_input_dim', 'precision', 'autoencoder_py_file_script']

        model_checkpoint_data = {}
        model_checkpoint_data["encoder_state_dict"] = encoder_state_dict
        model_checkpoint_data["decoder_state_dict"] = decoder_state_dict
        model_checkpoint_data["optimizer_state_dict"] = optimiser_state_dict
        model_checkpoint_data["latent_dim"] = latent_dim
        model_checkpoint_data["fc_input_dim"] = fc_input_dim
        model_checkpoint_data["precision"] = precision
        model_checkpoint_data["autoencoder_py_file_script"] = script

        # Save the model checkpoint data back to same folder with the name Combined_V2_checkpoint
        torch.save(model_checkpoint_data, path + "\\Combined_V2_checkpoint.pth")

        return True
    
    else:
        return False



main_path = r"N:\Yr 3 Project Results"  #r"N:\Yr 3 Project Results\\"

model_checpoint_folder_paths = []
model_deployment_folder_paths = []
# scan every folder and sub folder of the main path to find every single folder path that includes the terms "Model_Checkpoints" or "Model_Deployment" and add each to a seperate list
for root, dirs, files in os.walk(main_path):
    for dir in dirs:
        if "Model_Checkpoints" in dir:
            # walk through each subfolder in the Model_Checkpoints folder to find the data
            for root2, dirs2, files2 in os.walk(os.path.join(root, dir)):
                for dir2 in dirs2:
                    data = find_data(os.path.join(root2, dir2))
                    if data:
                        model_checpoint_folder_paths.append(os.path.join(root2, dir2))

    
        if "Model_Deployment" in dir:
            data = find_data(os.path.join(root, dir))
            if data:
                model_deployment_folder_paths.append(os.path.join(root, dir))

print(model_checpoint_folder_paths)
print(model_deployment_folder_paths)

