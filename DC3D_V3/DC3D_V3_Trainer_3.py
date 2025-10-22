# DeepClean Trainer v1.5.0
# Build created on Tuesday October 22nd 2025
# Author: Adill Al-Ashgar
# University of Bristol
# adill@neuralworkx.com

#from pickle import FALSE  #??? 
#from sympy import use   ##???
import torch

"""
Possible improvements:

#### ~~~~

#### ~~~~

#### ~~~~

#### ~~~~

#### ~~~~

#### ~~~~

#### ~~~~

#### ~~~~

#### ~~~~ Fix the belif telemety output to be integer as described, make sure all downstream is okay with it not being a tensor anymore!

### ~~~~ Auto name and number each model (use a local stoareg file to track model numbber) add model number to file info and use to auto genrate a model family tree when pretraining from others 

### ~~~~~ FIX: 'Epoch Times' list not being appended to correctly, only final? epoch time recorded?? (Obbserved taling the mean of the array always rssults in 1/2 of the total processing time)? 

### ~~~~~ [DONE!] FIX: Datalaoder calls for every individual sample in the ifnal bacth instead of loading them all at once in a singl eslice with the rang eof index's ... this will require reevaluation of the metho dused to iterate the large datloader as woul dno longe rbe checking index by index ?

### ~~~~~ [DONE!] FIX: INcorrct average epoch time calculation in final summary

### ~~~~~ FIX: async load the large datlaoder

### ~~~~~ FIX: degradation fucntions, costly and need optimisation

### ~~~~~ FIX: Torch profiler is not working, need to investigate, possibly due to not using the when loop 

### ~~~~~ FIX: 3d reconstrucxtion costly and requires reoptimisations

### ~~~~~ FIX: Handle execution timer misiing start/stop data gracefully, do not allow program crash 

### ~~~~~ [DONE!] FIX plotting fucntion iterates through test_loader batches choosing first image in each batch, this leads to error if there are less batches than the number to plot selected

### ~~~~~ FIX hook_flush_interval, setting this to greater than 1 results in just th eepoch in question being saved, i.,e setting to 4 saves every fourth epoch rather than stroing each epoch between then and then dumping them all!!

### ~~~~~ FIX TOF std_dev which if applied ends up with values outside of the range 0-time_dimension, either clamp the values to this range, or if values exceed it then remove those points is in practice the std_dev would carry them into previous or post frames rather than current frame. 

### ~~~~~  [DONE!] FIX the way the precision value is saved for model reruns, currently saves the double_precision boolean although have now moved to numerical 'precision' variable that accepts 16. 32 and 64 corresponding to half, single and double 

### ~~~~~ FIX, [UPDATE: THIS IS AN ERROR WITH ACB_MSE CREATING NANS WHICH THEN ESULT IN 0 VALUES BEING ASSINGED, THE PROGRAM DOES WORK USING STANDARD MSE LOASS AT FP16] when running with precision = 16 an error ocurs in backwards pass saying cannot perfrom backward pass of int, i think its saying the result of the loss is an int?? Investigate


### REFACTOR: func: quantify_loss_performance~ Appends the average performance metrics for the batch to the corresponding list of average metrics results for each epoch which is tracked eleshwehre in the enhanced performance tracking system and needs simplifying and condensing!!!

### REFACTOR: func: belief_telemetry ~ This function creates histogram plots of the pixel values recorded by the belief telemtry system [which needs renaming and reworking to simplify. (should be reconstruction thresholding telemtetry?)]  whihc records the values directly out of the netwrok before our reconstruction thresholding in the custom renomalisation is applied. This is important to keep ttrack of what is occusring before our iytput processing as it may be hiding errors.

### REFACTOR: func: create_settings_dict ~ Terrible, nneds compleate clean up and overhaul

### REFACTOR: func: plot_detailed_performance_loss etc  ~ PART OF THE LOSS PERFORMANCE SYSTEM NEEDEING OVERHAUL SIMPLIFICATION MODUIOARISATION AND THEN MOVING TO SEPERATE FILE 

### REFACTOR: func:  ~ 

### REFACTOR: func:  ~ 

### REFACTOR: func:  ~ 

### REFACTOR: func:  ~ 

### REFACTOR: func:  ~ 

### REFACTOR: func:  ~ 


### ~~~~~ [DONE!] Connect the inject_seed sytem up and add an interval

### ~~~~~ FIX BUG : xlsxwriter.exceptions.InvalidWorksheetName: Excel worksheet name 'T_fullframe_weighting_0.5 direct' must be <= 31 chars.

### ~~~~~ [DONE!] USe seeding value to controll data spareness and noise addition so that thay can be set to fixed

### ~~~~~ [DONE!] Create new unseen dataset for performance analysis on hyperparameter optimisation, could have a switch to make it use the val/test data?

### ~~~~~ [DONE!] Record the time taken per epoch, and then alongside the loss vs epoch plot (which dosent show time, so 1 epoch that takes 2 hrs is same as one that takes 1 min) plot loss vs time as a seperate plot

### ~~~~~ [DONE!] Add user controll to overide double precision processing

### ~~~~~ [DONE!] Improve the pixel telemtry per epoch by adding a dotted green line indicating the true number of signal points\

### ~~~~~ Attach the new masking optimised normalisation check if it need a corresponding renorm

### ~~~~~ [DONE!] Make sure that autoecoder Encoder and Decoder are saved along with model in the models folder 

### ~~~~~ Add the new performance metrics per epoch to the history da dictionary to clean up??

### ~~~~~ [DONE!] Fix memory leak in testing function loss calulation

### ~~~~~ [DONE!] Investigate and fix memory leak in plotting function

### ~~~~~ [DONE!] Reduce memory usage in loss calulation by removing lists

### ~~~~~ clean up the perforance loss plotting metircs calulation section, move to external script?

### ~~~~~ [DONE!] Ground plots and std devs in physical units

### ~~~~~ Clean up the performance loss plottsing code, it is too long and unwieldy, move it to an external file and load as a function

### ~~~~~~ [DONE!] Allow normalisation/renorm to be bypassed, to check how it affects results 

### ~~~~~~ [DONE!] Find out what is going on with recon threshold scaling issue

### ~~~~~~ [DONE!] fix noise adding to the data, it is not working as intended, need to retain clean images for label data 

### ~~~~~ Update the completed prints to terminal after saving things to only say that if the task is actually completeted, atm it will do it regardless, as in the error on .py save 

### ~~~~~ [DONE!] MUST SEPERATE 3D recon and flattening from the normalisation and renormalisation

### ~~~~~ #Noticed we need to update and cleanup all the terminal printing during the visulisations, clean up the weird line spaces and make sure to print what plot is currntly generating as some take a while and the progress bar doesn’t let user know what plot we are currently on

### ~~~~~ Update the model and data paths to folders inside the root dir so that they do not need be defined, and so that people can doanload the git repo and just press run without setting new paths etc 

### ~~~~~ fix epoch numbering printouts? they seem to report 1 epoch greater than they should

### ~~~~~ Clean up custom autoencoder.py file saver terminal prints left over from debugging

### ~~~~~ [DONE!] Fix Val loss save bug

### ~~~~~ [DONE!] Custom MSE loss fucntion with weighting on zero vals to solve class imbalence

### ~~~~~ [DONE!] Move things to other files (AE, Helper funcs, Visulisations etc)

### ~~~~~ [DONE!] Fix reconstruction threshold, use recon threshold to set bottom limit in custom normalisation

### ~~~~~ [DONE!] Turn plot or save into a function 

### ~~~~~ [DONE!] Add in a way to save the model after each epoch, so that if the program crashes we can still use the last saved model

### ~~~~~ [DONE!] Find way to allow user to exit which is non blocking 

### ~~~~~ [DONE!] Train on labeld data which has the fill line paths in labels and then just points on line in the raw data?

### ~~~~~ [DONE!] change telemetry variable name to output_pixel_telemetry

### ~~~~~ [DONE!] Fix this " if plot_higher_dim: AE_visulisation(en...)" break out all seperate plotting functions
    
### ~~~~~ [DONE!] sort out val, test and train properly

### ~~~~~ [DONE!] Update all layer activation tracking from lists and numpy to torch tensors throughout pipeline for speed

### ~~~~~ Add in automatic Enc/Dec layer size calulations

### ~~~~~ [DONE!] Search for and fix errors in custom norm an renorm

### ~~~~~ [DONE!] Seperate and moularise renorm and 3D reconstruction

### ~~~~~ Create flatten module in main body so noise can be added to the 3D cube rather than slicewise?

### ~~~~~ [DONE!] Add all advanced program settings to end of net summary txt file i.e what typ eof normalisation used etc, also add th enam eof the autoencoder file i.e AE_V1 etc from the module name 

### ~~~~~ [DONE!] update custom mse loss fucntion so that user arguments are set in settings page rather than at function def by defualts i.e (zero_weighting=1, nonzero_weighting=5)

### ~~~~~ [DONE!] could investigate programatically setting the non_zero weighting based on the ratio of zero points to non zero points in the data set which would balance out the two classes in the loss functions eyes

### ~~~~~ [DONE!] Add way for program to save the raw data for 3d plots so that they can be replotted after training and reviewed in rotatable 3d 

### ~~~~~ [DONE!] Check if running model in dp (fp64) is causing large slow down???

### ~~~~~ Allow seperate loss fucntion for testing/validation phase?

### ~~~~~ Properly track the choices for split loss funcs in txt output file 

### ~~~~~ Explicitlly pass in split_loss_functions to the split custom weigted func atm is not done to simplify the code but is not ideal

### ~~~~~ [DONE!] Update noise points to take a range as input and randomly select number for each image from the range

### ~~~~~ [DONE!] Add fcuntion next to noise adder that drops out pixels, then can have the labeld image with high signal points and then dropout the points in the input image to network so as to train it to find dense line from sparse points!

### ~~~~~ [DONE!] Add plots of each individual degradation step rathert than just all shown on one (this could be done instead of the current end of epoch 10 plots or alongside)

### ~~~~~ colour true signal points red in the input distroted image so that viewer can see the true signal points and the noise added

### ~~~~~ [DONE!] add masking directly to the trainer so we can see masked output too 

### ~~~~~  Label loss plots y axis programatically based on user loss function selection
"""

#%% - First time setup
results_output_path = "N:/DeepClean3D Project Folder/Yr 3 Project Results/"             # Path to the results output folder, this is the folder that will contains all your results folders, not the results folder for a particular run.

dataset_paths = [#"N:/DeepClean3D Project Folder/Yr 3 Project Datasets/V3_PDT_10K/",
                 #"N:/DeepClean3D Project Folder/Yr 3 Project Datasets/[V3]_RDT_50K/",    # Path to the dataset folder, this is the folder that contains your dataset folders, not a dataset folder itself. This allows fast switching between datasets by changing just the dataset title below without having to change the full path each time
                "N:/DeepClean3D Project Folder/Yr 3 Project Datasets/V3_RDT_150K/",
               ]                             

counters_local_filepath = 'DC3D_V3/Local_Program_Data/counters.json'

#%% - Data Path Settings
model_save_name = "Long150K_AdamW"      # Name of the model to be saved, this will be the name of the folder that the model is saved in
model_checkpoint_interval = 20               # Number of epochs between each model checkpoint save, set to False for no checkpointing. If set to a value then the model will save a checkpoint of the model and optimiser state dicts at the end of each 'model_checkpointing_interval' epochs
timeout_time = False                         # Time in minuits to wait before stopping training, set to False to disable time based stopping. {Default = False}. NOTE! Program will allow the epoch currently in progress to conclude before stopping.

#%% - Training Hyperparameter Settings
num_epochs = 500                              # Set max number of epochs to run for. {Default = 100}
batch_size = 250                             # Set batch size - number of samples to pull per batch. {Default = 64}
learning_rate = 0.004                       # Set optimiser learning rate. {Default = 1e-3}
optim_w_decay = 1e-07  #1e-05                       # Set optimiser weight decay for regularisation. {Default = 1e-5}
precision = 32                               # Set the precision of the data and model (16, 32 or 64) NOTE 16 only availble if runing on supported GPU hardware. {Default = 32}
 
# Architecture Settings
latent_dim = 24                              # Set number of nodes in the latent space, the bottleneck layer. {Default = 12}
fc_input_dim = 128                           # Set number of nodes in the intermediate fully connected layers. {Default = 128}
dropout_prob = 0.2                           # [NOTE Not connected yet] Set dropout probability. {Default = 0.2}
activation_function = torch.nn.ReLU(True)    # Set activation function for the network. {Default = nn.ReLU()}
use_adamw_optimiser = True                   # If set to true then AdamW optimiser is used, if false then Adam optimiser is used.

# Reconstruction Settings
reconstruction_threshold = 0.25               # MUST BE BETWEEN 0-1  #Threshold for 3d reconstruction, values below this confidence level are discounted. {Default = 0.5}
quantise_reconstructed_output = True          #[SHOULD BE TRUE ALWAYS!] # If set to true then the reconstructed output will be quantised in the 2D and 3D plots and for the detailed performance metrics calulations

#%% - Image Preprocessing Settings
signal_points = 300#(25, 35)                    # Set the number of signal points to add
noise_points = (0, 100)                      # Set the number of noise points to add
x_std_dev = 0                                # (mm) Set the standard deviation of the detectors error in the x axis
y_std_dev = 0                                # (mm) Set the standard deviation of the detectors error in the y axis
tof_std_dev = 0                              # (ns) Set the standard deviation of the detectors error in the time of flight 

#%% - Dataset Settings
train_test_split_ratio = 0.9                 # Set the ratio of the dataset to be used for training (Hyperparameter)

# Data Loader Settings
large_data_bundles_size = 10000              # Bundle size of dataset, 1 for V1 ~0.98MB, 1000 for V2 ~98MB, 10K for V3 = ~980MB
number_of_bundles_to_memory = 1              # Essentially the 'Large batch Size' i.e how many bundles to load into memory at once [Must be less than or equal to the number of bundles in the dataset on disk]

preprocess_on_gpu = False                    # Only woirks if cuda gpu is found, else will defulat back to cpu preprocess
shuffle_train_data = False #FIX CURRENLT BREAKS CODE CAUSING SAME ERROR AS FP16 i.e NANS IN ACBMSE RETURNING INT AS 0 FOR LOSS             # If set to true then will shuffle the training data each epoch
data_loader_workers = 0# FIX!                # Number of workers to use for the data loader, 0 means all data loading will be done on the main thread, 1 means one worker will be used to load data in the background, 2 means two workers will be used etc. 
pin_memory = False                           # [default = False] If set to true then the data loader will pin the memory for faster data transfer to the GPU, if set to false then the data loader will not pin the memory

inject_seed_interval = 10                    # Number of epochs between each seed injection, set to False for no seed injection. Reinjects the original seeding value to deterministically recreate the same noise and signal points each epoch, this is useful for testing the effect of different hyperparameters on the same data or for allowing the training to see same images multiple times to improve performance
seeding_value = 10 #None                     # None gives no seeeding to RNG, if the value is set this is used for the RNG seeding for numpy, and torch libraries. {Default = None}

# Input Data Settings
TORCHSIM_data = False                        #!!!!!!!!!!!!!!!!!!!!!!
time_dimension = 1000                        # Set the number of time steps in the data
xdim = 88                                    # Set the x dimension of the data in pixels
ydim = 128                                   # Set the y dimension of the data in pixels
channels = 1      #CONNECT                   # Set the number of channels in the data

use_physical_values_for_plot_axis = True     # If false then axis are label by pixel indicies, if true then axis are labelled by physical units
x_length = 800                               # mm
y_length = 1500                              # mm
time_length = 1000                           # ns

#%% - Loss Function Settings
loss_vs_sparse_img = False                               # Set if the loss is calculated against the sparse image or the full image (Hyperparameter)
loss_function_selection = "TrippleLossNI_CS"                  # Select loss function (string):# "MAE", "MSE", "SSE", "BCE", "ACB_MSE", "ffACB_MSE", "Split_Loss", "True3D", "Simple3D", "ACB3D", "Fast3D", "WPR_Loss"
renorm_for_loss_calc = False                             # Set if the loss is calculated against the renormalised image or the network output data (Hyperparameter)

# ACB
zero_weighting = 1                                      # Set zero weighting for ACBMSE or ffACBMSE (Hyperparameter)
nonzero_weighting = 1#0.4 #0.4                          # Set non zero weighting for ACBMSE or ffACBMSE  (Hyperparameter)

# Weights used for ACBMSE, Triple loss and varients
time_weighting = 1                                      # Set time weighting for ffACBMSE (Hyperparameter) [# Only used for ffACBMSE loss function]
tp_weighting = 8                                        # Set time point weighting for ffACBMSE (Hyperparameter) [# Only used for ffACBMSE loss function]
tn_weighting = 1                                        # Set frame point weighting for ffACBMSE (Hyperparameter) [# Only used for ffACBMSE loss function]
fp_weighting = 1                                        # Set frame point weighting for ffACBMSE (Hyperparameter) [# Only used for ffACBMSE loss function]
fn_weighting = 10                                        # Set frame point weighting for ffACBMSE (Hyperparameter) [# Only used for ffACBMSE loss function]
loss_penalty = 0.1
loss_reduction = 'mean'
fullframe_weighting = 0.00001 #1.5                         # Set full frame weighting for ffACBMSE (Hyperparameter) [# Only used for ffACBMSE loss function]
ff_loss = 'mse'                                         # Set loss function for full frame loss (Hyperparameter) [# Only used for ffACBMSE loss function] # "MAE", "MSE", "SSE", "BCE"

# Selections for split loss function
zeros_loss_choice = 1                                   # Select loss function for zero values (Hyperparameter): 0 = Maxs_Loss_Func, 1 = torch.nn.MSELoss(), 2 = torch.nn.BCELoss(), 3 = torch.nn.L1Loss(), 4 = ada_SSE_loss
nonzero_loss_choice = 1                                 # Select loss function for non zero values (Hyperparameter): 0 = Maxs_Loss_Func, 1 = torch.nn.MSELoss(), 2 = torch.nn.BCELoss(), 3 = torch.nn.L1Loss(), 4 = ada_SSE_loss

#%% - Pretraining settings
start_from_pretrained_model = False                      # If set to true then the model will load the pretrained model and optimiser state dicts from the path below
load_pretrained_optimiser = True                         # Only availible if above is set to true - (pretrain seems to perform better if this is set to true)

pretrained_model_path = r'N:\Yr 3 Project Results\Online_RUN_020_PTfrom online19 PTfrm47 - Training Results\Model_Checkpoints\Epoch_170\Combined_V2_checkpoint.pth'      # Specify the path to the saved full state dictionary for pretraining

#%% - Normalisation Settings 
masking_optimised_binary_norm = False        # If set to true then the model will use the binary normalisation method optimised for masking output. Otherwise will use the gaped custom normalisation optimised for the direct network output

#%% - New beta feature settings 
compile_model = False    # NOT AVAILIBE ON WINDOWS?!                    # [default = True] If set to true then the model will be compiled using torch.compile for faster execution, if set to false then the model will not be compiled further information is availible at: 
compile_mode = 'default' # mode for the torch.compile, can be set to default, reduce-overhead or max-autotune, see here: https://pytorch.org/docs/stable/generated/torch.compile.html for more information

# Neural Net Telemetry
record_weights = False                       # If set to true then the model will record the weights of the network at the end of each epoch
record_biases = False                        # If set to true then the model will record the biases of the network at the end of each epoch
record_activity = False                      # If set to true then the model will record the activations of the network at the end of each epoch
hook_flush_interval = 1          #FIX!            # Number of epochs between each flush of the network hooks to disk or set to False for no flush. Larger values will consume more memory but will reduce the overhead of flushing the hooks to disk    # If set to true then the model will flush the network hooks to disk periodically, this is useful for reducing memory usage but will slow down training due to the time taken to flush the hooks
hook_max_memory_usage = 0.5         #CONNECT!         # Maximum memory usage in GB before the hooks are automatically flushed to disk, this is useful for reducing memory usage but will slow down training due to the time taken to flush the hooks

#%% - Visulisation Settings
plot_or_save = 'save'                            # [default = 1] 0 prints plots to terminal (blocking till closed), If set to 1 then saves all end of epoch printouts to disk (non-blocking), if set to 2 then saves outputs whilst also printing for user (blocking till closed)
save_all_raw_plot_data = False                # [default = False] If set to true then all raw data for plots is saved to disk for replotting and analysis later

## During training plots
print_every_other = 1                        # [default = 2] 1 is to save/print all training plots every epoch, 2 is every other epoch, 3 is every 3rd epoch etc
num_to_plot = 10 # 2d plots                   # [default = 10] Number of images to plot during training

plot_live_time_loss = True                   # [default = True] Generate plot of live training loss vs time during trainig which is overwritten each epoch, this is useful for seeing how the training is progressing
plot_live_training_loss = True               # [default = True] Generate plot of live training loss vs epoch during trainig which is overwritten each epoch, this is useful for seeing how the training is progressing

plot_comparative_loss = True
comparative_live_loss = True                 # [default = True] Adds comparative lines to the live plots, the models for comparison are selected below
slide_live_plot_size = 0                     # [default = 0] Number of epochs to show on the live plot (if set to 0 then will show all epochs)
comparative_loss_titles = ["S1 10K", "S1 100K", "S1 500K", "S2 10K"]
comparative_loss_paths = [r'N:\DeepClean3D Project Folder\Yr 3 Project Results\RDT 10K 1000ToF timed - Training Results\\',   # Settings V1
                          r'N:\DeepClean3D Project Folder\Yr 3 Project Results\RDT 100K 1000ToF timed - Training Results\\',  # Settings V1
                          r'N:\DeepClean3D Project Folder\Yr 3 Project Results\RDT 500K 1000ToF timed - Training Results\\',  # Settings V1
                          r'N:\DeepClean3D Project Folder\Yr 3 Project Results\RDT 10K S2 - Training Results\\'               # Settings V2
                          ] 


## Post training plots
plot_train_loss = True                       # [default = True]       
plot_test_loss = True                        # [default = True]
plot_validation_loss = True                  # [default = True]               

plot_detailed_performance_loss = True        # Plots ssim nmi etc for each epoch 
plot_normalised_radar = True                # [default = True]
         
plot_pixel_threshold_telemetry = True        # [default = False] # Update name to pixel_cuttoff_telemetry    #Very slow, reduces net performance by XXXXXX%
plot_pixel_difference = False #BROKEN        # [default = True]          
plot_latent_generations = True              # [default = True]              
plot_higher_dim = False                      # [default = True]  
plot_Graphwiz = True                         # [default = True]       

use_tensorboard = False                       # [default = False]

#%% - Advanced Debugging Settings
print_encoder_debug = False                     # [default = False]  
print_decoder_debug = False                     # [default = False] 
print_network_summary = False                   # [Default = False] Prints the network summary to terminal

debug_noise_function = False                    # [default = False]  
debug_loader_batch = False                      # SAFELY REMOVE THIS PARAM!!!  #(Default = False) //INPUT 0 or 1//   #Setting debug loader batch will print to user the images taken in by the dataoader in this current batch and print the corresponding labels
debug_model_exporter  = False                   # [default = False]

full_dataset_integrity_check = False            # [Default = False] V slow  #Checks the integrity of the dataset by checking shape of each item as opposed to when set to false which only checks one single random file in the dataset
full_dataset_distribution_check = False         # [Default = False] V slow  #Checks the distribution of the dataset , false maesn no distributionn check is done

use_execution_timer = True    #FIX!!     ###CONNECT!! # [Default = True] Uses the execution timer to time each module in the process
run_profiler = False                            # [Default = False] Runs the cProfiler on the training loop to check for bottlenecks and slow functions
run_pytorch_profiler = False

#%% Hyperparameter Optimisation Settings  
optimise_hyperparameter = False                               # Set if hyperparameter optimisation is used
hyperparam_to_optimise = 'zero_weighting'                     # Set which hyperparameter to optimise  - options are: 'batch_size', 'learning_rate', 'optim_w_decay', 'dropout_prob', 'loss_function_selection', 'conv_layers', 'conv_filter_multiplier', 'latent_dim'
set_optimisiation_list_manually = [0.3, 0.5, 0.7, 1.0, 1.3]   # Set this param = to your list i.e [[12, 120], 35, 87]
val_set_on = False                                            # Set if a validation set is used
val_test_split_ratio = 0.9                                    # This needs to be better explained its actually test_val ration ratehr than oterh way round     # [NOTE LEAVE AT 0.5, is for future update, not working currently] Set the ratio of the non-training data to be used for validation as opposed to testing (Hyperparameter)

## Simple Performance Measure
print_validation_results = True                               # Set if the validation results are printed to terminal 
plot_training_time = True                                     # Set if the training time is plotted 

## Full Performance Analysis - Performed in addition to and seperatly from the validation stage for automatic data analysis
perf_analysis_num_files = 5000                                                                     # Number of files to test
perf_analysis_plot = 100                                                                           # The number of results to print imshow plots for for each model tested, set to False for none
perf_analysis_dataset_dir = (r"N:\\DeepClean3D Project Folder\\Yr 3 Project Datasets\\PERF VALIDATION SETS\\40K 100N 30S\\")   # Directory of dataset to test - (Best to use totally unseen data files that are not contianed within the train, test or validation sets)
debug_hpo_perf_analysis = False


#%% ################################## END OF USER SETTINGS ###################################

def override_globals_with_cli():
    """
    Detects top-level user-defined variables (excluding functions, modules,
    and built-ins), exposes them as CLI arguments, and overrides them if provided.
    Supports int, float, str, bool, list, dict with automatic type conversion.
    """
    import argparse
    import types

    def convert_value(val):
        """Convert string to int or float if possible, else leave as string."""
        try:
            if '.' in val:
                return float(val)
            else:
                return int(val)
        except ValueError:
            return val
    
    # Filter globals: only variables that are not callable, not modules, and do not start with "_"
    user_vars = {
        k: v for k, v in globals().items()
        if not k.startswith("_") and not isinstance(v, (types.FunctionType, types.ModuleType))
    }

    parser = argparse.ArgumentParser(description="Override user settings from command line")

    for var, val in user_vars.items():
        if isinstance(val, bool):
            parser.add_argument(f"--{var}", dest=var, action='store_true', help=f"Set {var} True")
            parser.add_argument(f"--no-{var}", dest=var, action='store_false', help=f"Set {var} False")
            parser.set_defaults(**{var: val})
        elif isinstance(val, list):
            parser.add_argument(f"--{var}", type=str, help=f"Override {var} as comma-separated list", default=",".join(map(str, val)))
        elif isinstance(val, dict):
            parser.add_argument(f"--{var}", type=str, help=f"Override {var} as key=value pairs, comma-separated", default=",".join(f"{k}={v}" for k,v in val.items()))
        else:
            parser.add_argument(f"--{var}", type=type(val), help=f"Override {var}", default=val)

    args = parser.parse_args()

    # Override globals dynamically
    for var, val in user_vars.items():
        arg_value = getattr(args, var)
        if isinstance(val, bool):
            globals()[var] = arg_value
        elif isinstance(val, list):
            globals()[var] = [convert_value(v.strip()) for v in arg_value.split(",")]
        elif isinstance(val, dict):
            d = {}
            if arg_value:
                for item in arg_value.split(","):
                    k, v = item.split("=")
                    d[k.strip()] = convert_value(v.strip())
            globals()[var] = d
        else:
            globals()[var] = arg_value

override_globals_with_cli()  # This function will overide any of the above global settings with command line inputs if they are provided

#%% - Load in Dependencies
# External Libraries
import os
import time     # Used to time the training loop
import json
import torch
from datetime import datetime 
import torchvision 
import numpy as np  
import pandas as pd
from tqdm.auto import tqdm  # Progress bar, auto automtically selects between normal TQDM or notebook version for running in Jupiter/Google collab style notebooks
from torchinfo import summary # function to get the summary of the model layers structure, trainable parameters and memory usage
import matplotlib.pyplot as plt     
from matplotlib.ticker import FuncFormatter
from torch.utils.tensorboard.writer import SummaryWriter, FileWriter
import cProfile
from torch.utils.data import Subset
import torch.profiler

# - General Functions
from Helper_files.Robust_model_exporter_V1 import Robust_model_export, Robust_model_export2                   # Function to export the raw .py file that contains the autoencoder class
from Helper_files.System_Information_check import get_system_information                # Function to get the host system performance specs of the training machine
from Helper_files.Dataset_Integrity_Check_V1 import dataset_integrity_check             # Function to check the integrity of the datasets values
from Helper_files.Dataset_distribution_tester_V1 import dataset_distribution_tester     # Function to check the distribution of the datasets values
from Helper_files.ExcelExtractor import extract_data_to_excel                           # This is a custom function to extract data from excel files
from Helper_files.model_perf_analysis2 import run_full_perf_tests                       # Function to run the full performance analysis on the model
from Helper_files.ExecutionTimer import Execution_Timer

#from Helper_files.Export_User_Settings import create_settings_dict
from Dataloader_V4 import *
from Helper_files.DC3D_Core_Functions import *
#from train_test_val import train_epoch, test_epoch, validation_routine
from Helper_files.Network_Hooks import *
from Helper_files.Plotting_Helpers import *
from Autoencoders.DC3D_AE_V2_2L import Encoder, Decoder    # - Autoencoder
from Helper_files.Helper_Functions import *   # - Helper functions
from Helper_files.Image_Metrics import *    # - Image metrics
from Helper_files.AE_Visulisations import *   # - Visulisations 
from Loss_Functions.Loss_Fn_Classes import *   # - Loss Functions
from Helper_files.Data_Degradation_Functions import *   # Data Degredation Functions



#%% - HACKS NEED FIXING!!!

# Set the Model ID for this run and dynamically add to save path title
with open(counters_local_filepath, 'r') as f:
    counters = json.load(f)

counters["Model_ID"] += 1
model_id = counters["Model_ID"]

# Save updated counter
with open(counters_local_filepath, 'w') as f:
    json.dump(counters, f)

model_save_name = f"MID{model_id}__{model_save_name}" # Model name with incremented ID




# Set the parent ID if this is a child run from a pretrained model
if start_from_pretrained_model:
    parent_model_id = extract_model_id_from_path(pretrained_model_path)
else:
    parent_model_id = None
    



# Set print every other to not exceed num_epochs
if print_every_other > num_epochs:                                                                   
    print_every_other = num_epochs

results_output_path_1 = results_output_path  # HACK FOR HYPERPARAM OPTIMISATION FIX IT!!!

history_da = {'train_loss':[], 'test_loss':[], 'val_loss':[], 'HTO_val':[], 'training_time':[]}      # Needs better placement???
max_epoch_reached = 0                                                                                # In case user exits before end of first epoch 

# Physical scaling parameters for plotting
time_scale = time_dimension / time_length                                                            # ns per t pixel   # MOVE THESE LINES ELSEWHERE TO CLEAN UP
x_scale = xdim / x_length                                                                            # mm per x pixel
y_scale = ydim / y_length                                                                            # mm per y pixel
physical_scale_parameters = [x_scale, y_scale, time_scale]

input_signal_settings = [signal_points, x_std_dev, y_std_dev, tof_std_dev, noise_points] #move!

# Create a dictionary to store the activations
activations = {}
weights_data = {}
biases_data = {}




#%% - Performance Profiling and Logging Setup
if use_execution_timer:
    execution_timer = Execution_Timer()

if compile_model:
    gaped_renormalisation = torch.compile(gaped_renormalisation)
    #gaped_normalisation = torch.compile(gaped_normalisation)
    reconstruct_3D = torch.compile(reconstruct_3D)
    signal_degredation = torch.compile(signal_degredation)

if run_profiler:
    profiler = cProfile.Profile()      # Create a cProfile object to store the profiler data

if run_pytorch_profiler:
    torch_profiler = torch.profiler.profile(profile_memory=True)
    #torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    #                                        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    #                                        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    #                                        record_shapes=True,
    #                                        profile_memory=True,
    #                                        with_stack=True
    #                                        )

    # Start the profiler
    torch_profiler.start()


#%% - Load in Comparative Live Loss Data
if comparative_live_loss:
    comparative_history_da, comparative_epoch_times = load_comparative_data(comparative_loss_paths, plot_live_training_loss, plot_live_time_loss)

#%% - Data Gathering Functions
### CLEAN UP THIS FUCNTION !!!!!!!!!!!!
def full_model_export(checkpoint_save, checkpoint, model_output_dir, model_checkpoints_dir, epoch, encoder, decoder, optim, latent_dim, fc_input_dim, precision, Encoder, debug_model_exporter=False):

    def save_model_data(output_path, output_filename, encoder, decoder, optim, latent_dim, fc_input_dim, precision, model_output_dir, model_save_name, checkpoint, model_checkpoints_dir, epoch, debug_model_exporter=False):
        # Get the directory name of the current Python file for the autoencoder export
        search_dir = os.path.abspath(__file__)
        search_dir = os.path.dirname(search_dir)


        if start_from_pretrained_model:
            autoencoder_py_file_script = checkpoint["autoencoder_py_file_script"]
            AE_file_name = "DC3D_Autoencoder_V1_Protected2_V2"  # REPLACE WITH USING THE THIRSD LIN EOF THE AUTOENCODER FILE TEXT ONCE FILES HAVE BEEN UPDATED
        else:
            autoencoder_py_file_script, AE_file_name = Robust_model_export2(Encoder, search_dir, debug=debug_model_exporter) #Only need to run on encoder as encoder and decoder are both in the same file so both get saved this way

        torch.save({'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'latent_dim': latent_dim,
                    'fc_input_dim': fc_input_dim,
                    'precision': precision,
                    'autoencoder_py_file_script': autoencoder_py_file_script,
                    },
                output_filename)
        
        return AE_file_name

    if checkpoint_save:
        model_checkpoints_path = model_checkpoints_dir + f"Epoch_{epoch}\\"
        os.makedirs(model_checkpoints_path, exist_ok=True)
        new_combined_filename = model_checkpoints_path + f"Combined_checkpoint_epoch_{epoch}.pth"
        AE_file_name = save_model_data(model_checkpoints_path, new_combined_filename, encoder, decoder, optim, latent_dim, fc_input_dim, precision, model_output_dir, model_save_name, checkpoint, model_checkpoints_dir, epoch, debug_model_exporter=False)

    else:
        # Joins up the parts of the different output files save paths
        new_combined_filename = model_output_dir + model_save_name + f"Combined_checkpoint_epoch_{epoch}.pth"
        print("\nSaving model to .pth file...")
        AE_file_name = save_model_data(model_output_dir, new_combined_filename, encoder, decoder, optim, latent_dim, fc_input_dim, precision, model_output_dir, model_save_name, checkpoint, model_checkpoints_dir, epoch, debug_model_exporter=False)
        print("- Completed -") 
        
        return AE_file_name

    
# Tracks network output pixel value distribution as histogram pre-reconstruction threshold and renomalisation to understand the effect of the reconstruction thresholding
def belief_telemetry(data:torch.Tensor, reconstruction_threshold: float, epoch: int, plot_or_save="save") -> tuple[int, int]:

    """
    This function creates histogram plots of the pixel values recorded by the belief telemtry system [which needs renaming and reworking to simplify. (should be reconstruction thresholding telemtetry?)]  whihc records the values directly out of the netwrok before our reconstruction thresholding in the custom renomalisation is applied. This is important to keep ttrack of what is occusring before our iytput processing as it may be hiding errors.
    The histogram also shows the reconstruction threshold to show the distribution of values above and below the threshold.

    Args:
        data (torch tensor): The image batch. Shape [B, C, H, W]?????????????????????????
        reconstruction_threshold (float): The threshold used in the custom normalisation, used to set the lower limit of the noise point values. Default = 0.5
        epoch (int): The current epoch number
        settings (dict): A dictionary containing the settings for the current training run
        plot_or_save (int): A flag to set if the plot is printed to terminal or saved to disk. 0 prints plots to terminal (blocking till closed), If set to 1 then saves all end of epoch printouts to disk (non-blocking), if set to 2 then saves outputs whilst also printing for user (blocking till closed)

    Generates: 
        Histogram plot, either printed to terminal or saved to disk based on the plot_or_save flag

    Returns:
        above_threshold (int): The number of pixels in the image above the reconstruction threshold
        below_threshold (int): The number of pixels in the image below the reconstruction threshold
    """
    

    data2 = data.flatten()

    #Plots histogram showing the confidence level of each pixel being a signal point
    _, _, bars = plt.hist(data2, 10, histtype='bar')
    plt.axvline(x= reconstruction_threshold, color='red', marker='|', linestyle='dashed', linewidth=2, markersize=12)
    plt.title("Epoch %s" %epoch)
    plt.bar_label(bars, fontsize=10, color='navy') 
    plt.xlabel("Output Values")
    plt.ylabel("Number of Pixels")
    plt.grid(alpha=0.2)
    Out_Label = graphics_dir + f'{model_save_name} - Reconstruction Telemetry Histogram - Epoch {epoch}.png'
    plot_save_choice(plot_or_save, Out_Label)

    above_threshold = (data2 >= reconstruction_threshold).sum()
    below_threshold = (data2 < reconstruction_threshold).sum()
    return (above_threshold, below_threshold)   # REPLACE WITH return (int(above_threshold.item()), int(below_threshold.item())) once everything downstream has been checked for compatibility!
                

def quantify_loss_performance(clean_input_batch, noised_target_batch, time_dimension):
    """
    This function compares each image in the batch to its corresponding target image and calculates a range of enhanced performance metrics for each. The results for each metric are then avaeraged over the batch and this average is appended to the corresponding list of average metrics results for each epoch which is tracked eleshwehre in the enhanced performance tracking system

        Args:
        clean_input_batch (torch tensor): The input image batch. Shape [B, C, H, W]
        noised_target_batch (torch tensor): The target image batch. Shape [B, C, H, W]
        time_dimension (int): The number of time steps in the data set, used to set the upper limit of the noise point values. Default = 100  
    
    Generates:
        Appends the average performance metrics for the batch to the corresponding list of average metrics results for each epoch which is tracked eleshwehre in the enhanced performance tracking system and needs simplifying and condensing!!!

    """
    batch_size = len(clean_input_batch)
    
    # Calculates the mean of the performance metrics for the batch and appends this to the corresponding list of average metrics results for each epoch
    #avg_loss_nrmse.append(np.mean(loss_nrmse))
    avg_loss_mse.append(MSE2(clean_input_batch.squeeze(1), noised_target_batch.squeeze(1)))
    avg_loss_mae.append(MAE2(clean_input_batch.squeeze(1), noised_target_batch.squeeze(1)))
    avg_loss_snr.append(SNR2(clean_input_batch.squeeze(1), noised_target_batch.squeeze(1)))
    avg_loss_psnr.append(PSNR2(clean_input_batch.squeeze(1), noised_target_batch.squeeze(1), time_dimension))
    avg_loss_ssim.append(SSIM2(clean_input_batch, noised_target_batch, time_dimension))
    avg_loss_nmi.append(NomalisedMutualInformation2(clean_input_batch.squeeze(1), noised_target_batch.squeeze(1)))
    avg_loss_cc.append(correlation_coeff2(clean_input_batch.squeeze(1), noised_target_batch.squeeze(1)))

    percentage_of_true_positive_xy, percentage_of_true_positive_tof, numof_false_positives_xy, num_false_negative_xy = compare_images_pixels2(clean_input_batch.squeeze(1), noised_target_batch.squeeze(1))    
    avg_loss_true_positive_xy.append(percentage_of_true_positive_xy)
    avg_loss_true_positive_tof.append(percentage_of_true_positive_tof)
    avg_loss_false_positive_xy.append(numof_false_positives_xy / batch_size)
    #avg_loss_false_negative_xy.append(loss_false_negative_xy / batch_size)



def create_comparison_plot_data(slide_live_plot_size, epoch, max_epoch_reached, comparative_live_loss, comparative_loss_titles, comparative_epoch_times, comparative_history_da, data=history_da['train_loss']):

    """
    This function creates the comparative live loss plot data that updates during training. It takes the data from the current training model and the comparative models and creates the data lists for the live loss plots then calls the comparative_loss_plot fucntion to create the plots.

    Args:
        slide_live_plot_size (int): The number of epochs to show on the live loss plots
        epoch (int): The current epoch number
        max_epoch_reached (int): The maximum epoch number reached by the current training model
        comparative_live_loss (bool): A flag to set if the comparative live loss plots are enabled
        comparative_loss_titles (list): A list of the titles of the comparative models
        comparative_epoch_times (list): A list of the epoch times of the comparative models
        comparative_history_da (list): A list of the history_da dictionaries of the comparative models
        data (list): The data to be plotted on the live loss plots. Default = history_da['train_loss']
    
    Generates:
        A plot of the live loss data vs epoch number, saved to disk
        A plot of the live loss data vs train time elapsed, saved to disk
    """

    ## Slide View
    low_lim = 0
    if slide_live_plot_size > 0:
        low_lim = max(0, epoch - slide_live_plot_size)

    ## Set data for current training model
    x_list_epochs = [range(low_lim, epoch)]
    x_list_time = [epoch_times_list]

    y_list = [data]
    legend_label_list = ["Training loss"]

    ## Add data for comparative models if user desired
    if comparative_live_loss:
        legend_label_list.extend(comparative_loss_titles)

        for loss_dictionary, epoch_t_list in zip(comparative_history_da, comparative_epoch_times):
            epoch_t_list = epoch_t_list[1:]   # NOTE: FIX FOR EPOCH LISTS BEING SAVeD INCLUDING 0 time which breaks the plot by having one more value than the hsitory da train loss. INVESTIGATE THIS ERROR ROOT CAUSE

            #if epoch > len(loss_dictionary['train_loss']):
                #print("WARNING: Epoch is greater than the number of epochs in the comparative model, this will cause an error in the live loss plot")
                #epoch_t_list.append(np.nan) # protection from surpassing the comaprison data during training.
                #loss_dictionary['train_loss'].append(np.nan) # protection from surpassing the comaprison data during training. 
            
            x_list_epochs.append(range(low_lim,len(loss_dictionary['train_loss'])))    
            x_list_time.append(epoch_t_list)
            y_list.append(loss_dictionary['train_loss'])
    
    ## Create plots
    # can have if staments here for protin time and epochs etc
    Out_Label1 = dir + f'{model_save_name} - Live Train loss.png'
    comparitive_loss_plot(x_list_epochs, y_list, legend_label_list, "Epoch number", "Train loss (ACB-MSE)", "Live Training loss", Out_Label1, plot_or_save)
    Out_Label2 = dir + f'{model_save_name} - Live time loss.png'
    comparitive_loss_plot(x_list_time, y_list, legend_label_list, "Time (s)", "Train loss (ACB-MSE)", "Live Time loss", Out_Label2, plot_or_save)

#%% - Train, Test, Val and Plot Functions

### Plotting function
def plot_epoch_data(end_of_epoch_plotting_data, epoch, model_save_name, time_dimension, reconstruction_threshold, signal_points, graphics_dir, plot_or_save, n=10, quantise_time_domain=False):       #Defines a function for plotting the output of the autoencoder. And also the input + clean training data? Function takes inputs, 'encoder' and 'decoder' which are expected to be classes (defining the encode and decode nets), 'n' which is the number of ?????Images in the batch????, and 'noise_factor' which is a multiplier for the magnitude of the added noise allowing it to be scaled in intensity.  
    
    """
    Plots the output of the autoencoder in a variety of ways to track its perfromance and abilities during the training cycle. 

    Args:
        end_of_epoch_plotting_data (torch tensor): The image batch. Shape [5, B=1, C=1, H, W]
        epoch (int): The current epoch number
        model_save_name (str): The name of the model being trained
        time_dimension (int): The number of time steps in the data set, used to set the upper limit of the noise point values amonst other things. Default = 100
        reconstruction_threshold (float): The threshold used in the custom normalisation, also used to set the lower limit of the noise point values. Default = 0.5
        signal_points (int) or (tuple): The number of signal points to retain in the signal sparsification preprocess. If given as int then number will be constant over training, if given as tuple then a random value will be selected from the range given for each batch
        noise_points (int) or (tuple): The number of noise points to add in the preprocessing. If given as int then number will be constant over training, if given as tuple then a random value will be selected from the range given for each batch. Default = 0
        x_std_dev (float) or (tuple): The standard deviation of the gaussian distribution to draw the x shift from. If given as float then number will be constant over training, if given as tuple then a random value will be selected from the range given for each batch. Default = 0
        y_std_dev (float) or (tuple): The standard deviation of the gaussian distribution to draw the y shift from. If given as float then number will be constant over training, if given as tuple then a random value will be selected from the range given for each batch. Default = 0
        tof_std_dev (float) or (tuple): The standard deviation of the gaussian distribution to draw the ToF shift from. If given as float then number will be constant over training, if given as tuple then a random value will be selected from the range given for each batch. Default = 0
        physical_scale_parameters (list): A list of the physical scale parameters for the data set. Default = [1,1,1]
        n is the number of images to plot in the 2d comparison. Default = 10
        quantise_time_domain (bool): A flag to set if the time domain should be quantised to integer values for plotting. Default = False

    Generates:
        A variety of plots showing the output of the autoencoder
    
    """



    # Initialise lists for true and recovered signal point values 
    number_of_true_signal_points = []
    number_of_recovered_signal_points = []

    # 2D Input/Output Comparison Plots 
    plt.figure(figsize=(16,9))                                      #Sets the figure size
    #loop = tqdm(range(n), desc='Plotting 2D Comparisons', leave=False, colour="green") 
    for i, (img, sparse_output_batch, sparse_and_resolution_limited_batch, noised_sparse_reslimited_batch, recovered_batch, recovered_masking_batch) in tqdm(enumerate(end_of_epoch_plotting_data), desc='Plotting 2D Comparisons', leave=False, colour="green", dynamic_ncols=True) :     # CLEAN UP START 

        # Update tqdm bar
        #loop.update(1)

        # Save the number of true signal points on the input image !!# THIS NEEDS TO BE ABLE TO DEAL WITH CASES WHERE IT IS A RANGE NOT AIFXED VALUE!
        number_of_true_signal_points.append(signal_points)  

        #Determine the number of signal points on the recovered image batch
        int_rec_sig_points = (recovered_batch >= reconstruction_threshold*time_dimension).sum()      # recon threshold * time dim beacus e rec image has been renomalised by this point and the recon threshol applies to the normalised state
        number_of_recovered_signal_points.append(int(int_rec_sig_points.cpu().numpy())) # clean up the .cpu.numpy? its not needed??

        in_im = img.cpu().numpy()
        sparse_im = sparse_output_batch.cpu().numpy()
        reslim_im = sparse_and_resolution_limited_batch.cpu().numpy()
        noise_im = noised_sparse_reslimited_batch.cpu().numpy()
        rec_im = recovered_batch.cpu().numpy()
        masked_im = recovered_masking_batch.cpu().numpy()

        if quantise_time_domain:
            in_im = np.round(in_im)
            sparse_im = np.round(sparse_im)
            reslim_im = np.round(reslim_im)
            noise_im = np.round(noise_im)
            rec_im = np.round(rec_im)
            masked_im = np.round(masked_im)

        #cmap = cm.get_cmap('viridis')
        #cmap.set_under('k') # set the color for 0 to black ('k')

        #Following section generates the img plots for the original(labels), noised, and denoised data)
        ax = plt.subplot(6,n,i+1)                                       #Creates a number of subplots for the 'Original images??????' i.e the labels. the position of the subplot is i+1 as it falls in the first row
        plt.imshow(in_im.T, cmap='gist_gray', vmin=0, vmax=time_dimension)           #plt.imshow plots an image. The arguments for imshow are, 'image data array' and cmap= which is the colour map. #.squeeze() acts on a tensor and returns a tensor, it removes all dimensions of the tensor that are of length 1, (A×1×B) becomes (AxB) where A and B are greater than 1 #.numpy() creates a numpy array from a tensor #!!! is the .cpu part becuase the code was not made to accept the gpu/cpu check i made????
        ax.get_xaxis().set_visible(False)                                   #Hides the x axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
        ax.get_yaxis().set_visible(False)                                   #Hides the y axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
        if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
            ax.set_title('EPOCH %s \nOriginal images' %(epoch))               #When above condition is reached, the plots title is set                                   #When above condition is reached, the plots title is set

        ax = plt.subplot(6, n, i + 1 + n)                                   #Creates a number of subplots for the 'Corrupted images??????' i.e the labels. the position of the subplot is i+1+n as it falls in the second row
        plt.imshow(sparse_im.T, cmap='gist_gray', vmin=0, vmax=time_dimension)   #plt.imshow plots an image. The arguments for imshow are, 'image data array' and cmap= which is the colour map. #.squeeze() acts on a tensor and returns a tensor, it removes all dimensions of the tensor that are of length 1, (A×1×B) becomes (AxB) where A and B are greater than 1 #.numpy() creates a numpy array from a tensor #!!! is the .cpu part becuase the code was not made to accept the gpu/cpu check i made????
        ax.get_xaxis().set_visible(False)                                   #Hides the x axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
        ax.get_yaxis().set_visible(False)                                   #Hides the y axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
        if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
            ax.set_title('Sparsity Applied')                                  #When above condition is reached, the plots title is set

        ax = plt.subplot(6, n, i + 1 + (2*n))                               #Creates a number of subplots for the 'Reconstructed images??????' i.e the labels. the position of the subplot is i+1+n+n as it falls in the third row
        plt.imshow(reslim_im.T, cmap='gist_gray', vmin=0, vmax=time_dimension)       #plt.imshow plots an image. The arguments for imshow are, 'image data array' and cmap= which is the colour map. #.squeeze() acts on a tensor and returns a tensor, it removes all dimensions of the tensor that are of length 1, (A×1×B) becomes (AxB) where A and B are greater than 1 #.numpy() creates a numpy array from a tensor #!!! is the .cpu part becuase the code was not made to accept the gpu/cpu check i made????
        ax.get_xaxis().set_visible(False)                                   #Hides the x axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
        ax.get_yaxis().set_visible(False)                                   #Hides the y axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
        if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
            ax.set_title('Resoloution Limited')                             #When above condition is reached, the plots title is set 

        ax = plt.subplot(6, n, i + 1 + (3*n))                                   #Creates a number of subplots for the 'Corrupted images??????' i.e the labels. the position of the subplot is i+1+n as it falls in the second row
        plt.imshow(noise_im.T, cmap='gist_gray', vmin=0, vmax=time_dimension)   #plt.imshow plots an image. The arguments for imshow are, 'image data array' and cmap= which is the colour map. #.squeeze() acts on a tensor and returns a tensor, it removes all dimensions of the tensor that are of length 1, (A×1×B) becomes (AxB) where A and B are greater than 1 #.numpy() creates a numpy array from a tensor #!!! is the .cpu part becuase the code was not made to accept the gpu/cpu check i made????
        ax.get_xaxis().set_visible(False)                                   #Hides the x axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
        ax.get_yaxis().set_visible(False)                                   #Hides the y axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
        if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
            ax.set_title('Corrupted images')                                  #When above condition is reached, the plots title is set

        ax = plt.subplot(6, n, i + 1 + (4*n))                               #Creates a number of subplots for the 'Reconstructed images??????' i.e the labels. the position of the subplot is i+1+n+n as it falls in the third row
        plt.imshow(rec_im.T, cmap='gist_gray', vmin=0, vmax=time_dimension)       #plt.imshow plots an image. The arguments for imshow are, 'image data array' and cmap= which is the colour map. #.squeeze() acts on a tensor and returns a tensor, it removes all dimensions of the tensor that are of length 1, (A×1×B) becomes (AxB) where A and B are greater than 1 #.numpy() creates a numpy array from a tensor #!!! is the .cpu part becuase the code was not made to accept the gpu/cpu check i made????
        ax.get_xaxis().set_visible(False)                                   #Hides the x axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
        ax.get_yaxis().set_visible(False)                                   #Hides the y axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
        if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
            ax.set_title('Reconstructed images')                             #When above condition is reached, the plots title is set 

        ax = plt.subplot(6, n, i + 1 + (5*n))                               #Creates a number of subplots for the 'Reconstructed images??????' i.e the labels. the position of the subplot is i+1+n+n as it falls in the third row
        plt.imshow(masked_im.T, cmap='gist_gray', vmin=0, vmax=time_dimension)       #plt.imshow plots an image. The arguments for imshow are, 'image data array' and cmap= which is the colour map. #.squeeze() acts on a tensor and returns a tensor, it removes all dimensions of the tensor that are of length 1, (A×1×B) becomes (AxB) where A and B are greater than 1 #.numpy() creates a numpy array from a tensor #!!! is the .cpu part becuase the code was not made to accept the gpu/cpu check i made????
        ax.get_xaxis().set_visible(False)                                   #Hides the x axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
        ax.get_yaxis().set_visible(False)                                   #Hides the y axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
        if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
            ax.set_title('Masked Reconstruction')                             #When above condition is reached, the plots title is set 

    plt.subplots_adjust(left=0.01,              #Adjusts the exact layout of the plots including whwite space round edges
                    bottom=0.02, 
                    right=0.99, 
                    top=0.95, 
                    hspace=0.3
                    )     

    Out_Label = graphics_dir + f'{model_save_name} - Epoch {epoch}.png' #creates the name of the file to be saved
    plot_save_choice(plot_or_save, Out_Label) #saves the plot if plot_or_save is set to 1, if 0 it displays, if 2 it displays and saves
    plt.close()

    # 3D Reconstruction
    in_im_3d, sparse_im, reslim_im, noise_im, rec_im_3d, masked_im = reconstruct_3D(in_im.squeeze(0), sparse_im.squeeze(0), reslim_im.squeeze(0), noise_im.squeeze(0), rec_im.squeeze(0), masked_im.squeeze(0)) #reconstructs the 3D image using the reconstruct_3D function

    # 3D Plottting
    if rec_im.ndim != 1:                       # Checks if there are actually values in the reconstructed image, if not no image is aseved/plotted
        fig, axs = plt.subplots(2, 3, figsize=(32,20), subplot_kw={'projection': '3d'})
        ax1, ax2, ax3, ax4, ax5, ax6 = axs.flatten()
        fig.suptitle(f"3D Reconstruction - Epoch {epoch}") #sets the title of the plot

        ax1.set_title("Input Image") #sets the title of the plot
        ax1.scatter(in_im_3d[:,0], in_im_3d[:,1], in_im_3d[:,2]) #plots the 3D scatter plot for input 

        ax2.set_title("Sparse Image") #sets the title of the plot
        ax2.scatter(sparse_im[:,0], sparse_im[:,1], sparse_im[:,2]) #plots the 3D scatter plot for sparse image
    
        ax3.set_title("Resolution Limited Image") #sets the title of the plot
        ax3.scatter(reslim_im[:,0], reslim_im[:,1], reslim_im[:,2]) #plots the 3D scatter plot for reslim image

        ax4.set_title("Noised Image") #sets the title of the plot
        ax4.scatter(noise_im[:,0], noise_im[:,1], noise_im[:,2]) #plots the 3D scatter plot for noised image

        ax5.set_title("Reconstructed Image") #sets the title of the plot
        ax5.scatter(rec_im_3d[:,0], rec_im_3d[:,1], rec_im_3d[:,2]) #plots the 3D scatter plot for reconstructed image

        try: ## NOTE THIS ERROR NEEDS FIXING. IT IS CAUSED BY A SITUATION WHERE DATA DOES COME BACK THAT IS GREATER THAN THE RECON CUTTOFF SO 3D PLOTS ARE GENERATED HOWVER NO POINTS LIE IN CORRECT PLACE FOR MASKING SO THE MASK CONTAINS NOTHING. THEN THE MASK WILL BE WRONG DIMS AND CASUE THE PLOT ERROR HERE. FIX
            ax6.set_title("Masked Reconstructed Image") #sets the title of the plot
            ax6.scatter(masked_im[:,0], masked_im[:,1], masked_im[:,2]) #plots the 3D scatter plot for masked image
        except:
            print("ERROR OCCURED IN MASKING 3D PLOT! INVESTIGATE!")


        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.set_zlim(0, time_dimension)
            ax.set_xlim(0, 128)
            ax.set_ylim(0, 88)
        
            if use_physical_values_for_plot_axis:
                # Set axis labels
                ax.set_xlabel('x (mm)')
                ax.set_ylabel('y (mm)')
                ax.set_zlabel('time (ns)')
                # Apply tick format conversion for x, y, and z axes from 'pixels' to physical values
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x_scale, tick_number: tick_number * x_scale))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y_scale, tick_number: tick_number * y_scale))
                ax.zaxis.set_major_formatter(FuncFormatter(lambda time_scale, tick_number: tick_number * time_scale))
            
            else:
                # Set axis labels
                ax.set_xlabel('x (pixels)')
                ax.set_ylabel('y (pixels)')
                ax.set_zlabel('time (pixels)')

        fig.tight_layout()

        Out_Label = graphics_dir + f'{model_save_name} 3D Reconstruction - Epoch {epoch}.png' #creates the name of the file to be saved
        plot_save_choice(plot_or_save, Out_Label) #saves the plot if plot_or_save is set to 1, if 0 it displays, if 2 it displays and saves

    return(number_of_true_signal_points, number_of_recovered_signal_points, in_im, noise_im, rec_im)        #returns the number of true signal points, number of recovered signal points, input image, noised image and reconstructed image 

### Training Function
def train_epoch(epoch, encoder, decoder, device, dataloader, loss_fn, optimizer, time_dimension=100, reconstruction_threshold=0.5, masking_optimised_binary_norm=False, loss_vs_sparse_img=False, renorm_for_loss_calc=False, use_tensorboard=False, writer=None):
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
        masking_optimised_binary_norm (bool): A flag to set if the masking optimised binary normalisation is used. If set to true the masking optimised binary normalisation is used, if set to false the gaped normalisation is used. Default = False  [EXPLAIN IN MORE DETAIL!!!!!]
        loss_vs_sparse_img (bool): A flag to set if the loss is calculated against the sparse image or the clean image. If set to true the loss is calculated against the sparse image, if set to false the loss is calculated against the clean image. Default = False

    Returns:
        loss_total/batches (float): The average loss over the epoch calulated by dividing the total sum of loss by the number of batches !! EXPLAIN THIS SIMPLER !!! 

    """
    encoder.train()   
    decoder.train()   

    loss_total = 0.0
    batches = 0


    iterator = tqdm(dataloader, desc='Batches', leave=False, dynamic_ncols=True, postfix="Starting")             
    for data in iterator: 
        image_batch, sparse_output_batch, _, noised_sparse_reslimited_batch = data
        execution_timer.record_time(event_name="Training-preprocess", event_type="start")
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
        execution_timer.record_time(event_name="Training-preprocess", event_type="stop")

        execution_timer.record_time(event_name="Training-forward", event_type="start")
        # Encode data
        encoded_data = encoder(image_noisy) # Encode the noised image batch
        # Decode data
        decoded_data = decoder(encoded_data) # Decode the encoded image batch
        execution_timer.record_time(event_name="Training-forward", event_type="stop")

        execution_timer.record_time(event_name="Training-loss", event_type="start")
        if loss_vs_sparse_img:
            loss_comparator = image_sparse
        else:
            loss_comparator = image_clean

        # Evaluate loss
        if renorm_for_loss_calc:
            decoded_data = gaped_renormalisation(decoded_data, reconstruction_threshold, time_dimension)
            loss_comparator = gaped_renormalisation(loss_comparator, reconstruction_threshold, time_dimension)

        loss = loss_fn(decoded_data, loss_comparator)  # Compute the loss between the decoded image batch and the clean image batch
        execution_timer.record_time(event_name="Training-loss", event_type="stop")

        execution_timer.record_time(event_name="Training-backward", event_type="start")
        # Backward pass
        optimizer.zero_grad() # Reset the gradients - IS THIS CORRECT???
        loss.backward() # Compute the gradients
        optimizer.step() # Update the parameters

        batches += 1
        loss_total += loss.item()
        avg_epoch_loss = loss_total/batches
        
        execution_timer.record_time(event_name="Training-backward", event_type="stop")

        execution_timer.record_time(event_name="Training-postprocess", event_type="start")
        
        iterator.set_postfix({'Partial Batch Loss': loss.data.item()})

        if use_tensorboard:
            # Add the gradient values to Tensorboard
            for name, param in encoder.named_parameters():
                writer.add_histogram(name + '/grad', param.grad, global_step=epoch)

            for name, param in decoder.named_parameters():
                writer.add_histogram(name + '/grad', param.grad, global_step=epoch)

            writer.add_scalar('Loss/train', avg_epoch_loss, epoch)
        execution_timer.record_time(event_name="Training-postprocess", event_type="stop")
    return avg_epoch_loss

### Testing Function
def test_epoch(encoder, decoder, device, dataloader, loss_fn, time_dimension=100, reconstruction_threshold=0.5, masking_optimised_binary_norm=False, loss_vs_sparse_img=False, renorm_for_loss_calc=False, debug_masking_function=False):
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
        masking_optimised_binary_norm (bool): A flag to set if the masking optimised binary normalisation is used. If set to true the masking optimised binary normalisation is used, if set to false the gaped normalisation is used. Default = False  [EXPLAIN IN MORE DETAIL!!!!!]
        loss_vs_sparse_img (bool): A flag to set if the loss is calculated against the sparse image or the clean image. If set to true the loss is calculated against the sparse image, if set to false the loss is calculated against the clean image. Default = False

    Returns:
        loss_total/batches (float): The average loss over the epoch calulated by dividing the total sum of loss by the number of batches !! EXPLAIN THIS SIMPLER !!!

    """
    imgs = []
    sparse_imgs = []
    reslim_imgs = []
    noised_imgs = []
    recovered_imgs = []
    maskrec_imgs = []
    n = 10 

    # Set evaluation mode for encoder and decoder
    encoder.eval() # Evaluation mode for the encoder
    decoder.eval() # Evaluation mode for the decoder
    with torch.no_grad(): # No need to track the gradients

        loss_total = 0.0
        batches = 0
        
        iterator = tqdm(dataloader, desc='Testing', leave=False, colour="yellow", dynamic_ncols=True, postfix="Starting")           
        for data in iterator: 
            image_batch, sparse_output_batch, sparse_and_resolution_limited_batch, noised_sparse_reslimited_batch = data



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
            if renorm_for_loss_calc: # NOTE: IF USED THIS WOULD DUPLICATE THE RENORM DONE FURTHER DOWN AND BE A PROBLEM!!! FIUX !!!!
                data_for_loss = gaped_renormalisation(decoded_data, reconstruction_threshold, time_dimension)
                loss_comparator = gaped_renormalisation(loss_comparator, reconstruction_threshold, time_dimension)
            else:
                data_for_loss = decoded_data   # is this fix for the above working or does it need to copy to not modify the original????

            # Evaluate loss
            loss = loss_fn(data_for_loss, loss_comparator)  # Compute the loss between the decoded image batch and the clean image batch
            batches += 1
            loss_total += loss.item()

            iterator.set_postfix({'Partial Batch Loss': loss.data.item()})

            performance_test_comparator = gaped_renormalisation(loss_comparator.clone(), reconstruction_threshold, time_dimension) #DOES THIS NEED TO BE DETACHED??? # Clone the loss comparator to avoid modifying the original
            performance_test_data = gaped_renormalisation(decoded_data.clone(), reconstruction_threshold, time_dimension, quantise_reconstructed_output) #DOES THIS NEED TO BE DETACHED??? # Clone the decoded data to avoid modifying the original


            #Run additional perfomrnace metrics
            execution_timer.record_time(event_name="Testing-performance_metrics", event_type="start")
            quantify_loss_performance(performance_test_comparator, performance_test_data, time_dimension)
            execution_timer.record_time(event_name="Testing-performance_metrics", event_type="stop")

            if len(imgs) < n:
                num_to_append = min(n - len(imgs), len(image_batch))
                imgs.append(image_batch[:num_to_append])
                sparse_imgs.append(sparse_output_batch[:num_to_append])
                reslim_imgs.append(sparse_and_resolution_limited_batch[:num_to_append])
                noised_imgs.append(noised_sparse_reslimited_batch[:num_to_append])
                
                output_data = gaped_renormalisation(decoded_data[:num_to_append].cpu(), reconstruction_threshold, time_dimension, quantise_reconstructed_output)        # Renormalise output data   avoiding doing it to the entire batch, just the required num of images for the following plotting etc    
                recovered_imgs.append(output_data) 
                maskrec_imgs.append(masking_recovery(noised_sparse_reslimited_batch[:num_to_append].cpu(), output_data, time_dimension, debug_masking_function))

        if plot_pixel_threshold_telemetry and epoch % print_every_other == 0 and epoch !=0:   
            above_threshold, below_threshold = belief_telemetry(decoded_data[0][0].cpu(), reconstruction_threshold, epoch, plot_or_save)    #calls the belief_telemetry function to generate the telemetry plots
            telemetry.append([epoch, above_threshold, below_threshold]) #appends the telemetry data to the telemetry list

        imgs = torch.cat(imgs, dim=0)[:n]
        sparse_imgs = torch.cat(sparse_imgs, dim=0)[:n]
        reslim_imgs = torch.cat(reslim_imgs, dim=0)[:n]
        noised_imgs = torch.cat(noised_imgs, dim=0)[:n]
        recovered_imgs = torch.cat(recovered_imgs, dim=0)[:n]
        maskrec_imgs = torch.cat(maskrec_imgs, dim=0)[:n]

        end_of_epoch_plotting_data = list(zip(imgs, sparse_imgs, reslim_imgs, noised_imgs, recovered_imgs, maskrec_imgs))
        
    return loss_total/batches, end_of_epoch_plotting_data

def test_epoch2(encoder, decoder, device, dataloader, loss_fn, time_dimension=100, reconstruction_threshold=0.5, masking_optimised_binary_norm=False, loss_vs_sparse_img=False, renorm_for_loss_calc=False, debug_masking_function=False):

    imgs = []
    sparse_imgs = []
    reslim_imgs = []
    noised_imgs = []
    recovered_imgs = []
    maskrec_imgs = []
    n = 10 

    # Set evaluation mode for encoder and decoder
    encoder.eval() # Evaluation mode for the encoder
    decoder.eval() # Evaluation mode for the decoder
    with torch.no_grad(): # No need to track the gradients

        loss_total = 0.0
        batches = 0
        
        iterator = tqdm(dataloader, desc='Testing', leave=False, colour="yellow", dynamic_ncols=True, postfix="Starting")           
        for data in iterator: 
            image_batch, sparse_output_batch, sparse_and_resolution_limited_batch, noised_sparse_reslimited_batch = data

            # Move all data to the GPU at once
            image_batch = image_batch.to(device)
            sparse_output_batch = sparse_output_batch.to(device)
            sparse_and_resolution_limited_batch = sparse_and_resolution_limited_batch.to(device)
            noised_sparse_reslimited_batch = noised_sparse_reslimited_batch.to(device)

            if masking_optimised_binary_norm:
                normalised_batch = mask_optimised_normalisation(noised_sparse_reslimited_batch)
                image_batch_norm = mask_optimised_normalisation(image_batch)
            else:
                normalised_batch = gaped_normalisation(noised_sparse_reslimited_batch, reconstruction_threshold, time_dimension)
                image_batch_norm = gaped_normalisation(image_batch, reconstruction_threshold, time_dimension)
            
            image_clean = image_batch_norm # Data already on device
            image_noisy = normalised_batch # Data already on device

            # Encode and decode data
            encoded_data = encoder(image_noisy) # Encode the noised image batch
            decoded_data = decoder(encoded_data) # Decode the encoded image batch

            if loss_vs_sparse_img:
                loss_comparator = sparse_output_batch
            else:
                loss_comparator = image_clean

            # Evaluate loss
            if renorm_for_loss_calc:
                data_for_loss = gaped_renormalisation(decoded_data, reconstruction_threshold, time_dimension)
                loss_comparator = gaped_renormalisation(loss_comparator, reconstruction_threshold, time_dimension)
            else:
                data_for_loss = decoded_data

            # Compute the loss
            loss = loss_fn(data_for_loss, loss_comparator)
            batches += 1
            loss_total += loss.item()

            iterator.set_postfix({'Partial Batch Loss': loss.data.item()})

            # Run additional performance metrics
            performance_test_comparator = gaped_renormalisation(loss_comparator.clone(), reconstruction_threshold, time_dimension)
            performance_test_data = gaped_renormalisation(decoded_data.clone(), reconstruction_threshold, time_dimension)
            quantify_loss_performance(performance_test_comparator, performance_test_data, time_dimension)

            if len(imgs) < n:
                num_to_append = min(n - len(imgs), len(image_batch))
                imgs.append(image_batch[:num_to_append].cpu())
                sparse_imgs.append(sparse_output_batch[:num_to_append].cpu())
                reslim_imgs.append(sparse_and_resolution_limited_batch[:num_to_append].cpu())
                noised_imgs.append(noised_sparse_reslimited_batch[:num_to_append].cpu())
                output_data = gaped_renormalisation(decoded_data[:num_to_append], reconstruction_threshold, time_dimension).cpu()
                recovered_imgs.append(output_data)
                maskrec_imgs.append(masking_recovery(noised_sparse_reslimited_batch[:num_to_append].cpu(), output_data, time_dimension, debug_masking_function))

        if plot_pixel_threshold_telemetry and epoch % print_every_other == 0 and epoch != 0:   
            above_threshold, below_threshold = belief_telemetry(decoded_data[0][0].cpu(), reconstruction_threshold, epoch, plot_or_save)
            telemetry.append([epoch, above_threshold, below_threshold])

        imgs = torch.cat(imgs, dim=0)[:n]
        sparse_imgs = torch.cat(sparse_imgs, dim=0)[:n]
        reslim_imgs = torch.cat(reslim_imgs, dim=0)[:n]
        noised_imgs = torch.cat(noised_imgs, dim=0)[:n]
        recovered_imgs = torch.cat(recovered_imgs, dim=0)[:n]
        maskrec_imgs = torch.cat(maskrec_imgs, dim=0)[:n]

        end_of_epoch_plotting_data = list(zip(imgs, sparse_imgs, reslim_imgs, noised_imgs, recovered_imgs, maskrec_imgs))
        
    return loss_total/batches, end_of_epoch_plotting_data

### Validation Function
def validation_routine(encoder, decoder, device, dataloader, loss_fn, time_dimension=100, reconstruction_threshold=0.5, masking_optimised_binary_norm=False, loss_vs_sparse_img=False, renorm_for_loss_calc=False):
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

        iterator = tqdm(dataloader, desc='Validation', leave=False, colour="green", dynamic_ncols=True, postfix="Starting")                 
        for data in iterator: 
            image_batch, sparse_output_batch, sparse_and_resolution_limited_batch, noised_sparse_reslimited_batch = data
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
                decoded_data = gaped_renormalisation(decoded_data, reconstruction_threshold, time_dimension)
                loss_comparator = gaped_renormalisation(loss_comparator, reconstruction_threshold, time_dimension)

            # Evaluate loss
            loss = loss_fn(decoded_data, loss_comparator)  # Compute the loss between the decoded image batch and the clean image batch
            batches += 1
            loss_total += loss.item()

            iterator.set_postfix({'Partial Batch Loss': loss.data.item()})
    
    return loss_total/batches


#%% - Set compute device
if torch.cuda.is_available():        
    device = torch.device("cuda")  # Nvidia GPU is prefered if availible      

elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():  
    device = torch.device("mps")  # Apple Silicon, Metal Performance Shaders

else:                                                                       
    device = torch.device("cpu")  # Fallback to CPU if no supported GPU is availible    

#%% - Program begins


if __name__ == "__main__":
    print("\n \nProgram Initalised - Welcome to DC3D Trainer")  #prints the welcome message
    print(f'Selected compute device: {device}\n')  #Informs user if running on CPU or GPU

    if use_execution_timer:
        execution_timer.record_time(event_name="Program Initialisation", event_type="start") #records the time the program started

    if run_profiler:
        # Start profiling
        profiler.enable()

    #%% - # Hyperparameter Optimiser
    if optimise_hyperparameter:
        HTO_values = set_optimisiation_list_manually
        print(f"Hyperparameter optimisation is enabled, the following hyperparameters will be optimised: {hyperparam_to_optimise}")
        val_loop_range = HTO_values #val_loop_range is the number of values in the HTO_values list, i.e. the number of times the model will be trained and evaluated
    else: 
        val_loop_range = range(1,2,1)

    for HTO_val in val_loop_range: #val_loop is the number of times the model will be trained and evaluated
        if optimise_hyperparameter:
            globals()[hyperparam_to_optimise] = HTO_val
            print(f"Current Test: {hyperparam_to_optimise} value = {HTO_val}\n")

        #%% - # Input / Output Path Initialisation
        if optimise_hyperparameter:
            results_output_path = results_output_path_1 + model_save_name + "_" + hyperparam_to_optimise + f" Optimisation\\{HTO_val}\\"

        # Create output directory if it doesn't exist
        dir = results_output_path + model_save_name + " - Training Results/"
        os.makedirs(dir, exist_ok=True)

        # Create output directory for images if it doesn't exist
        graphics_dir = dir + "Output_Graphics/"
        os.makedirs(graphics_dir, exist_ok=True)

        raw_plotdata_output_dir = dir + "Raw_Data_Output/"
        os.makedirs(raw_plotdata_output_dir, exist_ok=True)

        model_output_dir = dir + "Model_Deployment/"
        os.makedirs(model_output_dir, exist_ok=True)

        model_checkpoints_dir = dir + "Model_Checkpoints/"
        os.makedirs(model_checkpoints_dir, exist_ok=True)

        full_activity_filepath = dir + model_save_name + " - Activity.npz"
        full_netsum_filepath = dir + model_save_name + " - Network Summary.txt"

        # Joins up the parts of the differnt input dataset load paths
        train_dirs = [dataset_path + 'Data/' for dataset_path in dataset_paths]


        #%% - Bodge clean up 
        ###CLEAN UP THIS METHOD TO SOMTHING BETTER!!!!!!
        epoch_avg_loss_mse = []
        epoch_avg_loss_mae = []
        epoch_avg_loss_snr = []
        epoch_avg_loss_psnr = []
        epoch_avg_loss_ssim = []
        epoch_avg_loss_nmi = []
        epoch_avg_loss_cc = []
        epoch_avg_loss_true_positive_xy = []
        epoch_avg_loss_true_positive_tof = []
        epoch_avg_loss_false_positive_xy = []

        # Initialises pixel belief telemetry
        telemetry = []# [[0, 0, xdim * ydim]]                # Initalises the telemetry memory, starting values are 0, 0.5, 0.5 which corrspond to epoch(0), above_threshold(0.5), below_threshold(0.5)
        
        # Create a summary writer
        if use_tensorboard:
            writer = SummaryWriter()
        else:
            writer = None


        #%% - Initialises seeding values to RNGs
        if seeding_value:
            torch.manual_seed(seeding_value)
            np.random.seed(seeding_value)
            print("Seeding with value:", seeding_value)
        else:
            torch.seed()                                    # Set the seed for the RNGs
            np.random.seed()

        T_seed = torch.initial_seed()           # Seed for PyTorch RNGs
        N_seed = np.random.get_state()[1][0]    # Seed for NumPy RNGs


        #%% - Set loss function choice

        # Set loss function choice for split loss (if loss function choice is set to ada weighted custom split loss)
        availible_split_loss_functions = [torch.nn.MSELoss(), torch.nn.BCELoss(), torch.nn.L1Loss(), ada_SSE_loss]    # List of all availible loss functions is set to ada_weighted_custom_split_loss
        split_loss_functions = [availible_split_loss_functions[zeros_loss_choice], availible_split_loss_functions[nonzero_loss_choice]] # Sets loss functions based on user input


        # List of all availible loss functions - add new loss functions here by adding name and function to dictionary as key and value. Code is her so that falls inside the hyperparam optimser loop so the loss function init params can also be optimised, i.e weights for ACB loss etc.
        availible_loss_functions_dict = {
            "MAE": torch.nn.L1Loss(),                                                                      # Mean Absolute Error Loss from PyTorch Library
            "MSE": torch.nn.MSELoss(),                                                                     # Mean Squared Error Loss from PyTorch Library
            "SSE": ada_SSE_loss,                                                                           # Sum of Squared Error Loss (Custom)
            "BCE": torch.nn.BCELoss(),   

            "LayeredLoss": LayeredLoss(zero_weighting, nonzero_weighting),                                                                  # Custom Layered Loss Function
            "TrippleLoss": TrippleLoss(zero_weighting, nonzero_weighting, fullframe_weighting, time_weighting, tp_weighting, fp_weighting, fn_weighting, tn_weighting, loss_penalty, 'mean'),                                                                  # Custom Tripple Loss Function
            "TrippleLossNI": TrippleLossNI(fullframe_weighting, tp_weighting, fp_weighting, fn_weighting, tn_weighting),                                                 # Tripple Loss Function without time weighting
            "TrippleLossNI_CS": TrippleLossNI_CS(fullframe_weighting, tp_weighting, fp_weighting, fn_weighting, tn_weighting),                                                 
                                                                                                                                            # Binary Cross Entropy Loss from PyTorch Library
            "ACB_MSE": ACBLoss(zero_weighting, nonzero_weighting),     
            "ACBLossNI" : ACBLossNI(zero_weighting, nonzero_weighting),                                                   # My Original Automatically Class Balanced MSE Loss using Class balancing by number of points
    
                                                                                                                     # My Original Automatically Class Balanced MSE Loss using Class balancing by frequency
            "ffACB_MSE": ffACBLoss(zero_weighting, nonzero_weighting, fullframe_weighting),                                       # Update to ACB_MSE, adds new term to the loss function that calulates mse over the full frame with its own weighting
            "Split_Loss": ada_weighted_custom_split_loss(split_loss_functions, zero_weighting, nonzero_weighting),                # Uses my automatic class balancing to split the loss function into two parts, one for zero values and one for non-zero values but instead of using MSE as my ACBMSE implementation this just leaves the loss function open to user setting, and uniquly allows user to select differnt loss fucntion for each class.

            "True3D": True3DLoss(zero_weighting=1, nonzero_weighting=1, timesteps=1000),                                                                          # 3D Loss function that uses the true 3D image as the target, rather than the 2D projection, (Warning: Very Slow)
            "Simple3D": simple3Dloss(zero_weighting=1, nonzero_weighting=1, virtual_t_weighting=1, virtual_x_weighting=1, virtual_y_weighting=1, timesteps=1000), # Simplified 3D loss function (Warning: Very Slow)
            "ACB3D": ACBLoss3D(zero_weighting, nonzero_weighting, virtual_t_weighting=1, virtual_x_weighting=None, virtual_y_weighting=None, timesteps=1000), # Adds my automatic class balancing to the 3D loss function (Warning: Very Slow)
            "Fast3D": NEWESTACB3dloss2024(batch_size, ydim, xdim),                                                                                             # Fast 3D loss method that instea of projecting 2D back to 3D and filling the cube with the values, it unravels the 2D back to an array of 3D coordinates (4D if counting the batch dim) then compares these directly ratehr than taking the next step to fill the cube. This is much faster than the other 3D loss functions and is the current prefered method for 3D loss, although has not demonstared benefits above the 2D ACB loss functions so far.

            "WPR_Loss": WeightedPerfectRecoveryLoss(),           # Custom loss function that contains a weighting term that encorages the network to have points follow the correct gradient ??? 
        }

        # Accessing the selected loss function and label
        loss_fn_label = loss_function_selection                    # Sets loss function label, used on plots and data to clearly identify loss function
        loss_fn = availible_loss_functions_dict[loss_fn_label]       # Sets loss function based on user input of parameter loss_function_selection



        #%% - Create record of all user input settings, to add to output data for testing and keeping track of settings
        settings = {}  # Creates empty dictionary to store settings 
        settings["Epochs"] = num_epochs     # Adds the number of epochs to the settings dictionary
        settings["Batch Size"] = batch_size # Adds the batch size to the settings dictionary
        settings["Learning Rate"] = learning_rate # Adds the learning rate to the settings dictionary
        settings["Optimiser Decay"] = optim_w_decay # Adds the optimiser decay to the settings dictionary
        settings["Dropout Probability"] = dropout_prob # Adds the dropout probability to the settings dictionary
        settings["Latent Dimension"] = latent_dim # Adds the latent dimension to the settings dictionary
        settings["FC Input Dim"] = fc_input_dim # Adds the FC input dimension to the settings dictionary
        settings["Noise Points"] = noise_points # Adds the noise points to the settings dictionary
        settings["Train Split"] = train_test_split_ratio # Adds the train split to the settings dictionary
        settings["Val/Test Split"] = val_test_split_ratio   # Adds the val/test split to the settings dictionary
        settings["Dataset/s used"] = [path.rstrip('/').split('/')[-1] for path in dataset_paths] # Adds the dataset titles to the settings dictionary
        settings["Time Dimension"] = time_dimension # Adds the time dimension to the settings dictionary
        settings["Seed Val"] = seeding_value # Adds the seed value to the settings dictionary
        settings["Reconstruction Threshold"] = reconstruction_threshold # Adds the reconstruction threshold to the settings dictionary
        settings["Precision"] = precision
        settings["shuffle_train_data"] = shuffle_train_data
        settings["record_weights"] = record_weights
        settings["record_biases"] = record_biases
        settings["record_activity"] = record_activity
        settings["timeout_time"] = timeout_time

        #%% - Dataset Pre-tests
        #if full_dataset_integrity_check: #if full_dataset_integrity_check is set to True, then the scan type is set to full (slow)
        #    scantype = "full"
        #else:
        #    scantype = "quick"


        ###### NOTE! REMOVE TESTS FOR DATASET AND INCLUDE THEM IN THE DATSET GENRATOR INSTEAD TO AVOID REPETETIVE CHECKS OF THE SAME SETS OVER AND OVER AGAIN !!!!
        # CHECK: Dataset Integrity Check
        #print(f"Testing training dataset integrity, with {scantype} scan")                                 # Prints the scan type
        #dataset_integrity_check(train_dir, full_test=full_dataset_integrity_check, print_output=True)      # Checks the integrity of the training dataset

        # CHECK: Dataset Distribution Check
        #if full_dataset_distribution_check:                                                                                         # If full_dataset_distribution_check is set to True, then the dataset distribution is checked
        #    print("\nTesting training dataset signal distribution")
        #    dataset_distribution_tester(train_dir, time_dimension, ignore_zero_vals_on_plot=True, output_image_dir=graphics_dir)    # Checks the distribution of the training dataset

        # CHECK: Num of images in path greater than batch size choice? 
        num_of_files_in_path = sum(len(os.listdir(dir_path)) for dir_path in train_dirs) * large_data_bundles_size                                                      # Number of files in path
        if num_of_files_in_path < batch_size:                                                                                              # If the number of files in the path is less than the batch size, user is promted to input a new batch size
            print(f"Error, the path selected has {num_of_files_in_path} image files, which is {(batch_size - num_of_files_in_path)} less than the chosen batch size. Please select a batch size less than the total number of images in the directory")
            batch_err_message = "Choose new batch size, must be less than total amount of images in directory", (num_of_files_in_path)                               # Creates the error message
            batch_size = int(input(batch_err_message))                                                                                                               # NOTE: Not sure why input message is printing with wierd brakets and speech marks in the terminal? Investigate???

        # Report type of gradient descent to user
        print(f"{num_of_files_in_path} files in path. // Batch size = {batch_size}\nLearning via: {batch_learning(num_of_files_in_path, batch_size)}\n")   # Prints the number of files in the path and the batch size and the resultant type of batch learning

        #%% - Data Loader & Preperation



        # Creates the dataset
        train_dataset = MTN_Dataset(train_dirs, large_data_bundles_size, number_of_bundles_to_memory, device, shuffle_train_data, preprocess_on_gpu, precision, timer=execution_timer)

        # Dataset Partitioning
        m = len(train_dataset)  # m is the length of the train_dataset, i.e., the number of images in the dataset
        train_split = int(m * train_test_split_ratio)  # train_split is the ratio of train images to be used in the training set as opposed to non_training set

        # Sequentially split the dataset
        train_indices = list(range(train_split))
        non_training_indices = list(range(train_split, m))

        train_data = Subset(train_dataset, train_indices)
        non_training_data = Subset(train_dataset, non_training_indices)

        m2 = len(non_training_data)  # m2 is the length of the non_training_data, i.e., the number of images in the dataset

        if val_set_on:
            val_split = int(m2 * val_test_split_ratio)  # val_split is the ratio of non-train images to be used in the validation set as opposed to the test set
            test_indices = non_training_indices[:m2 - val_split]
            val_indices = non_training_indices[m2 - val_split:]

            test_data = Subset(train_dataset, test_indices)
            val_data = Subset(train_dataset, val_indices)
        else:
            test_data = non_training_data
            val_data = non_training_data

        # Generating Dataloaders WITH THE WAY I HAVE STORED DATA IN MEMORY IS THIS CAUSING A 3X IN SIZE IN MEMORY????
        train_loader = StackedDatasetLoader(input_signal_settings, physical_scale_parameters, time_dimension, device, preprocess_on_gpu, precision, dataset=train_data, batch_size=batch_size, num_workers=data_loader_workers, timer=execution_timer) # Shuffle is handled by the dataset, set with the variable 'shuffle_data'. It must not be applied here otherwise will induce errors in the logic based on index value
        test_loader = StackedDatasetLoader(input_signal_settings, physical_scale_parameters, time_dimension, device, preprocess_on_gpu, precision, dataset=test_data, batch_size=batch_size, num_workers=data_loader_workers, timer=execution_timer) # Shuffle is handled by the dataset, set with the variable 'shuffle_data'. It must not be applied here otherwise will induce errors in the logic based on index value
        valid_loader = StackedDatasetLoader(input_signal_settings, physical_scale_parameters, time_dimension, device, preprocess_on_gpu, precision, dataset=val_data, batch_size=batch_size, num_workers=data_loader_workers, timer=execution_timer) # Shuffle is handled by the dataset, set with the variable 'shuffle_data'. It must not be applied here otherwise will induce errors in the logic based on index value

        train_batch_sampler = torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(train_data), 
                                                    batch_size=batch_size, 
                                                    drop_last=False)

        test_batch_sampler = torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(test_data), 
                                                    batch_size=batch_size, 
                                                    drop_last=False)

        val_batch_sampler = torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(val_data), 
                                                    batch_size=batch_size, 
                                                    drop_last=False)

        # Generating Dataloaders WITH THE WAY I HAVE STORED DATA IN MEMORY IS THIS CAUSING A 3X IN SIZE IN MEMORY????
        train_loader = StackedDatasetLoader(input_signal_settings, physical_scale_parameters, time_dimension, device, preprocess_on_gpu, precision, dataset=train_data, batch_sampler=train_batch_sampler,  timer=execution_timer) # Shuffle is handled by the dataset, set with the variable 'shuffle_data'. It must not be applied here otherwise will induce errors in the logic based on index value
        test_loader = StackedDatasetLoader(input_signal_settings, physical_scale_parameters, time_dimension, device, preprocess_on_gpu, precision, dataset=test_data, batch_sampler=test_batch_sampler,  timer=execution_timer) # Shuffle is handled by the dataset, set with the variable 'shuffle_data'. It must not be applied here otherwise will induce errors in the logic based on index value
        valid_loader = StackedDatasetLoader(input_signal_settings, physical_scale_parameters, time_dimension, device, preprocess_on_gpu, precision, dataset=val_data, batch_sampler=val_batch_sampler, timer=execution_timer) # Shuffle is handled by the dataset, set with the variable 'shuffle_data'. It must not be applied here otherwise will induce errors in the logic based on index value



        #%% - Setup model, loss criteria and optimiser   
        if start_from_pretrained_model: 

            # Load the saved model data
            checkpoint = torch.load(pretrained_model_path, map_location=device)   # map_location to the currently used device enusures that a model trainied on a different device type is loaded correctly

            # Set correct variables for the model
            latent_dim = checkpoint['latent_dim']
            fc_input_dim = checkpoint["fc_input_dim"]

            dtype = torch.float32   # remove once fixed!!!! BUG
            precision = 32 #checkpoint["precision"]   BUG

            # Update saved settings 
            settings["Latent Dimension"] = latent_dim
            settings["Precision"] = precision
            settings["FC Input Dim"] = fc_input_dim

            # Load the Encoder and Decoder classes by executing the script content directly using exec()
            exec(checkpoint["autoencoder_py_file_script"])

            encoder = Encoder(checkpoint['latent_dim'], False, checkpoint["fc_input_dim"])
            decoder = Decoder(checkpoint['latent_dim'], False, checkpoint["fc_input_dim"])

            # Load the state dicts
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            decoder.load_state_dict(checkpoint['decoder_state_dict'])        

            print("PRECISION:", precision)

            # precision already set so just updating the dtype variable for rest of program, no need for value error check as must have been valid to run in previous model train
            #if precision == "16":
            #    dtype = torch.float16
            #elif precision == "32":
            #    dtype = torch.float32
            #elif precision == "64":
            #    dtype = torch.float64


        else:
            # Initialize the encoder and decoder
            encoder = Encoder(latent_dim, fc_input_dim, activation_function, print_encoder_debug)
            decoder = Decoder(latent_dim, fc_input_dim, activation_function, print_decoder_debug)
            encoder, decoder, dtype = set_model_precision(encoder, decoder, precision) # Sets the precision of the encoder and decoder based on user input and returns the dtype variable used in the rest of the code


            # REMOVE AND FIX THE NEED FOR THIS DUMMY CHECKPOINT DATA IF NOT LOADING FROM PT MODEL
            checkpoint = {}
            #checkpoint['optimizer_state_dict'] = None
            #checkpoint['encoder_state_dict'] = None
            #checkpoint['decoder_state_dict'] = None
            #checkpoint['autoencoder_py_file_script'] = None
            #checkpoint['latent_dim'] = latent_dim
            #checkpoint['fc_input_dim'] = fc_input_dim
            #checkpoint['precision'] = precision

        # Define the optimizer
        params_to_optimize = [{'params': encoder.parameters()} ,{'params': decoder.parameters()}] #Selects what to optimise, 
        if use_adamw_optimiser:
            optim = torch.optim.AdamW(params_to_optimize, lr=learning_rate, weight_decay=optim_w_decay)
        else:
            optim = torch.optim.Adam(params_to_optimize, lr=learning_rate, weight_decay=optim_w_decay)

        if start_from_pretrained_model and load_pretrained_optimiser:
            # load the optimizer state dictionary, if user requested
            optim.load_state_dict(checkpoint['optimizer_state_dict'])

        # Initalise Model on compute device
        encoder.to(device)   #Moves encoder to selected device, CPU/GPU
        decoder.to(device)   #Moves decoder to selected device, CPU/GPU

        if compile_model:
            encoder = torch.compile(encoder, mode=compile_mode)
            decoder = torch.compile(decoder, mode=compile_mode)



        #%% - Prepare Network Summary
        # Set evaluation mode for encoder and decoder

        with torch.no_grad(): # No need to track the gradients
            
            # Create dummy input tensor
            enc_input_tensor = torch.randn(batch_size, channels, ydim, xdim, dtype=dtype) 
                
            # Join the encoder and decoder models
            full_network_model = torch.nn.Sequential(encoder, decoder)   #should this be done with no_grad?

            # Generate network summary and then convert to string
            model_stats = summary(full_network_model, input_data=enc_input_tensor, device=device, verbose=0)
            summary_str = str(model_stats)             

            # Print Encoder/Decoder Network Summary to terminal if user requested (Regardless of user setting, network summary is always added to the final .txt output file along with the final model)
            if print_network_summary:
                print(summary_str)


        #%% - Add Activation data hooks to encoder and decoder layers
        if record_activity or record_weights or record_biases:   # MOVE TO FUCNTION AND THEN TO HELPER FUCNS FILE FOR CLEANER CODE
            register_network_hooks(encoder, decoder, record_activity, record_weights, record_biases, activations, weights_data, biases_data)

        #%% - Running Training Loop

        execution_timer.record_time(event_name="Program Initialisation", event_type="stop") #records the time the program started


        print("Training Initiated\nPress Ctr + c to exit and save the model during training.\n")
        start_time = time.time()                     # Begin the training timer

        epoch_times_list = []                        
        history_da['train_loss']  = []
        history_da['test_loss'] = []                

        try:                                         # Try except clause allows user to exit training gracefully whilst still retaiing a saved model and ouput plots
            loop_range = tqdm(range(1, num_epochs+1), desc='Epochs', colour='red', dynamic_ncols=True, postfix="Starting")
            epoch_start_time = time.time()           # Starts the epoch timer
            for epoch in loop_range:                 # For loop that iterates over the number of epochs where 'epoch' takes the values (0) to (num_epochs - 1)
                execution_timer.record_time(event_name="Epoch", event_type="start") #records the time the program started

                #### IMPLEMNT THIS TO INJECT SEED VAULE AGAIN WEACH EPOCH SO THAT  THE TRAINING DATA IS THE DETERMINISTICALLY THE SAME EACH EPOCH NOT FRESH EACH TIME
                if inject_seed_interval and epoch % inject_seed_interval == 0:
                    torch.manual_seed(T_seed)
                    np.random.seed(N_seed)

                ###CLEAN UP THIS METHOD TO SOMTHING BETTER!!!!!!
                avg_loss_mse = []
                avg_loss_mae = []
                avg_loss_snr = []
                avg_loss_psnr = []
                avg_loss_ssim = []
                avg_loss_nmi = []
                avg_loss_cc = []
                avg_loss_true_positive_xy = []
                avg_loss_true_positive_tof = []
                avg_loss_false_positive_xy = []

                ### Training (use the training function)
                execution_timer.record_time(event_name="Training", event_type="start") #records the time the program started
                train_loss=train_epoch(epoch, 
                                        encoder, 
                                        decoder, 
                                        device, 
                                        train_loader, 
                                        loss_fn, 
                                        optim,
                                        time_dimension,
                                        reconstruction_threshold,
                                        masking_optimised_binary_norm,
                                        loss_vs_sparse_img,
                                        renorm_for_loss_calc,
                                        use_tensorboard,
                                        writer,
                                        )
                execution_timer.record_time(event_name="Training", event_type="stop") #records the time the program started

                ### Testing (use the testing function)
                execution_timer.record_time(event_name="Testing", event_type="start") #records the time the program started
                test_loss, end_of_epoch_plotting_data = test_epoch(encoder, 
                                        decoder, 
                                        device, 
                                        test_loader, 
                                        loss_fn,
                                        time_dimension,
                                        reconstruction_threshold,
                                        masking_optimised_binary_norm,
                                        loss_vs_sparse_img,
                                        renorm_for_loss_calc,
                                        )
                execution_timer.record_time(event_name="Testing", event_type="stop") #records the time the program started


                loop_range.set_postfix({'Train Loss': train_loss, 'Test Loss': test_loss})


                if epoch % print_every_other == 0 and epoch != 0:         
                    execution_timer.record_time(event_name="Plotting Function", event_type="start") #records the time the program started           
                    # Run plotting function for training feedback and telemetry.
                    encoder.eval()
                    decoder.eval()
                    returned_data_from_plotting_function = plot_epoch_data(end_of_epoch_plotting_data,
                                                                            epoch, 
                                                                            model_save_name, 
                                                                            time_dimension, 
                                                                            reconstruction_threshold,
                                                                            signal_points,
                                                                            graphics_dir,
                                                                            plot_or_save,
                                                                            num_to_plot,
                                                                            quantise_reconstructed_output,
                                                                            )
                    
                    number_of_true_signal_points, number_of_recovered_signal_points, in_data, noisy_data, rec_data = returned_data_from_plotting_function
                    if save_all_raw_plot_data:
                        save_variable(number_of_true_signal_points, f'Epoch {epoch}_number_of_true_signal_points', raw_plotdata_output_dir, force_pickle=True)
                        save_variable(number_of_recovered_signal_points, f'Epoch {epoch}_number_of_recovered_signal_points', raw_plotdata_output_dir, force_pickle=True)
                        save_variable(in_data, f'Epoch {epoch}_input_list', raw_plotdata_output_dir, force_pickle=True)
                        save_variable(noisy_data, f'Epoch {epoch}_noised_list', raw_plotdata_output_dir, force_pickle=True)
                        save_variable(rec_data, f'Epoch {epoch}_recovered_list', raw_plotdata_output_dir, force_pickle=True)
            
                    encoder.train()
                    decoder.train()
                    execution_timer.record_time(event_name="Plotting Function", event_type="stop") #records the time the program started

                execution_timer.record_time(event_name="Epoch Results", event_type="start") #records the time the program started      

                epoch_end_time = time.time()
                epoch_time = epoch_end_time - epoch_start_time
                epoch_times_list.append(epoch_time) 

                # Update the epoch reached counter  
                max_epoch_reached = epoch    
                    
                history_da['train_loss'].append(train_loss)
                history_da['test_loss'].append(test_loss)

                ###CLEAN UP THIS METHOD TO SOMTHING BETTER!!!!!!
                epoch_avg_loss_mse.append(np.mean(avg_loss_mse))
                epoch_avg_loss_mae.append(np.mean(avg_loss_mae)) # USelass really so probably depreciate!
                epoch_avg_loss_snr.append(np.mean(avg_loss_snr))  # USelass really so probably depreciate!
                epoch_avg_loss_psnr.append(np.mean(avg_loss_psnr)) 
                epoch_avg_loss_ssim.append(np.mean(avg_loss_ssim))
                epoch_avg_loss_nmi.append(np.mean(avg_loss_nmi))
                epoch_avg_loss_cc.append(np.mean(avg_loss_cc))
                epoch_avg_loss_true_positive_xy.append(np.mean(avg_loss_true_positive_xy))
                epoch_avg_loss_true_positive_tof.append(np.mean(avg_loss_true_positive_tof))
                epoch_avg_loss_false_positive_xy.append(np.mean(avg_loss_false_positive_xy))

                # check if current epoch is a multiple of 'model_checkpoint_interval' and if so save the model
                if model_checkpoint_interval and epoch % model_checkpoint_interval == 0 and epoch != 0:
                    full_model_export(True, checkpoint, model_output_dir, model_checkpoints_dir, epoch, encoder, decoder, optim, latent_dim, fc_input_dim, precision, Encoder, debug_model_exporter=False)

                if epoch % hook_flush_interval == 0 and epoch != 0:
                    write_hook_data_to_disk_and_clear(activations, weights_data, biases_data, epoch, dir)

                ### Live Comparison Plots
                if plot_comparative_loss:
                    create_comparison_plot_data(slide_live_plot_size, epoch, max_epoch_reached, comparative_live_loss, comparative_loss_titles, comparative_epoch_times, comparative_history_da, data=history_da['train_loss'])
                execution_timer.record_time(event_name="Epoch Results", event_type="stop") #records the time the program started
                execution_timer.record_time(event_name="Epoch", event_type="stop") #records the time the program started

                ### Training Timeout Check
                if timeout_time:
                    if time.time() - start_time > timeout_time * 60:  #convert timeout time from minuits to seconds
                        print("Training timed out, exiting training loop")
                        break
                
                pass # end of try clause, if all goes well and user doesen't request an early exit then the training loop will end here

        # If user presses Ctr + c to exit training loop, this handles the exception and allows the code to run its final data and model saving etc before exiting        
        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Exiting training gracefully...")
        
        execution_timer.record_time(event_name="Program Debrief", event_type="start") #records the time the program started

        if run_profiler:
            # Stop profiling
            profiler.disable()

        #%% - After Training
        encoder.eval()
        decoder.eval()    
        
        # Stop timing the training process and calculate the total training time
        end_time = time.time()
        training_time = end_time - start_time

        # Report the training time
        print(f"\nTotal Training Cycle Took {training_time:.2f} seconds")

        # Build Dictionary to collect all output data for .txt file
        full_data_output = {}
        full_data_output["Train Loss"] = round(history_da['train_loss'][-1], 3)
        full_data_output["Test Loss"] = round(history_da['test_loss'][-1], 3)   #Val loss calulaton is broken? check it above

        detailed_performance_loss_dict={}
        detailed_performance_loss_dict["epoch_avg_loss_mse"] = epoch_avg_loss_mse
        detailed_performance_loss_dict["epoch_avg_loss_snr"] = epoch_avg_loss_snr
        detailed_performance_loss_dict["epoch_avg_loss_psnr"] = epoch_avg_loss_psnr
        detailed_performance_loss_dict["epoch_avg_loss_ssim"] = epoch_avg_loss_ssim
        detailed_performance_loss_dict["epoch_avg_loss_nmi"] = epoch_avg_loss_nmi
        detailed_performance_loss_dict["epoch_avg_loss_cc"] = epoch_avg_loss_cc
        detailed_performance_loss_dict["epoch_avg_loss_true_positive_xy"] = epoch_avg_loss_true_positive_xy
        detailed_performance_loss_dict["epoch_avg_loss_true_positive_tof"] = epoch_avg_loss_true_positive_tof
        detailed_performance_loss_dict["epoch_avg_loss_false_positive_xy"] = epoch_avg_loss_false_positive_xy


        #%% - Save any remaining unsaved raw plot data
        epochs_range = range(1,len(history_da["train_loss"])+1) 
        if save_all_raw_plot_data:     ##FIND bETTER PLACE FOR THIS
            save_variable(np.array(epochs_range), "epochs_range", raw_plotdata_output_dir)  #!!!!!! testing raw plot outputter!!!!

            save_variable(history_da, "history_da", raw_plotdata_output_dir) 

            save_variable(epoch_times_list, "epoch_times_list", raw_plotdata_output_dir) 

            save_variable(settings, "settings", raw_plotdata_output_dir)

        # This one we will always save as its neccesary to compare models across different training runs that do not have equivelent loss fucntion scales.
        save_variable(detailed_performance_loss_dict, "detailed_performance_loss", raw_plotdata_output_dir)


        #%% - Output Visulisations
        ###Loss function plots
        Out_Label = graphics_dir + f'{model_save_name} - Train and Val loss v epoch Comb - Epoch {epoch}.png'
        loss_plot_trainval([epochs_range, epochs_range], [history_da['train_loss'], history_da['test_loss']], ["train", "test"], [model_save_name, model_save_name], "Epoch number", f"Loss ({loss_fn_label})", "Train and Val loss v Epoch", Out_Label, plot_or_save)

        Out_Label = graphics_dir + f'{model_save_name} - Train and Val loss v Time Comb - Epoch {epoch}.png'
        loss_plot_trainval([epoch_times_list, epoch_times_list], [history_da['train_loss'], history_da['test_loss']], ["train", "test"], [model_save_name, model_save_name], "Training Time (s)", f"Loss ({loss_fn_label})", "Train and Val loss v Time", Out_Label, plot_or_save)

        if plot_detailed_performance_loss: 
            Out_Label = graphics_dir + f'{model_save_name} - Detailed Performance loss - Epoch {epoch}.png'
            draw_detailed_performance_loss_plots(epochs_range, epoch_avg_loss_mse, epoch_avg_loss_snr, epoch_avg_loss_psnr, epoch_avg_loss_ssim, epoch_avg_loss_nmi, epoch_avg_loss_cc, epoch_avg_loss_true_positive_xy, epoch_avg_loss_true_positive_tof, epoch_avg_loss_false_positive_xy, Out_Label, plot_or_save)

        if plot_pixel_threshold_telemetry:
            Out_Label = graphics_dir + f'{model_save_name} - Reconstruction Threshold Telemetry - Epoch {epoch}.png'
            plot_telemetry(telemetry, signal_points, xdim*ydim, Out_Label, plot_or_save)

        if plot_pixel_difference:
            num_diff_noised, num_same_noised, num_diff_cleaned, num_same_cleaned, im_diff_noised, im_diff_cleaned = AE_visual_difference(in_data, noisy_data, rec_data)
            full_data_output["num_diff_noised"] = num_diff_noised
            full_data_output["num_same_noised"] = num_same_noised
            full_data_output["num_diff_cleaned"] = num_diff_cleaned
            full_data_output["num_same_cleaned"] = num_same_cleaned

            # Create a new figure
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

            # Plot the im_diff_noised subplot
            ax1.imshow(im_diff_noised, cmap='gray')
            ax1.set_title('Original -> Noised Img')

            # Plot the im_diff_cleaned subplot
            ax2.imshow(im_diff_cleaned, cmap='gray')
            ax2.set_title('Original -> Cleaned Img')

            # Add title for the whole figure    
            fig.suptitle('Pixel-wise Difference Comparisons')
            
            Out_Label = graphics_dir + f'{model_save_name} - Pixel Difference Epoch {epoch}.png'
            plot_save_choice(plot_or_save, Out_Label)  

        if plot_latent_generations:
            def show_image(img):
                npimg = img.numpy()
                plt.imshow(np.transpose(npimg, (1, 2, 0)))

            img_recon = Generative_Latent_information_Visulisation(encoder, decoder, latent_dim, device, test_loader, reconstruction_threshold, time_dimension)
            
            fig, ax = plt.subplots(figsize=(20, 8.5))
            show_image(torchvision.utils.make_grid(img_recon[:100],10,10, pad_value=100))
            plt.axis("off")
            Out_Label = graphics_dir + f'{model_save_name} - Latent Generation Epoch {epoch}.png'
            plot_save_choice(plot_or_save, Out_Label)  

        if plot_higher_dim:
            Out_Label_1 = graphics_dir + f'{model_save_name} - Higher Dimensisions Epoch {epoch}.png'
            Out_Label_2 = graphics_dir + f'{model_save_name} - TSNE Epoch {epoch}.png'
            Reduced_Dimension_Data_Representations(encoder, device, train_dataset, Out_Label_1, Out_Label_2, plot_or_save)
            
        if plot_normalised_radar:
            # Data for the radar plot
            categories = list(map(str, detailed_performance_loss_dict.keys())) # Names of each performance measure

            fig = create_radar_plot(categories, [detailed_performance_loss_dict], [model_save_name])
            Out_Label = graphics_dir + f'{model_save_name} - Normalised Radar Epoch {epoch}.png'
            plot_save_choice(plot_or_save, Out_Label)


            fig = create_tri_radar_plot(categories, [detailed_performance_loss_dict], [model_save_name])

            #categories = ['epoch_avg_loss_true_positive_xy', 'epoch_avg_loss_true_positive_tof', 'epoch_avg_loss_false_positive_xy']

            #fig = create_radar_plot(categories, [detailed_performance_loss_dict], [model_save_name])
            Out_Label = graphics_dir + f'{model_save_name} - Normalised Tri-Radar Epoch {epoch}.png'
            plot_save_choice(plot_or_save, Out_Label)

            categories = ['epoch_avg_loss_true_positive_xy', 'epoch_avg_loss_true_positive_tof', 'epoch_avg_loss_false_positive_xy']
            fig = create_3d_radar_plot(categories, detailed_performance_loss_dict, model_save_name)
            Out_Label = graphics_dir + f'{model_save_name} - Normalised 3D Radars per Epoch.png'
            plot_save_choice(plot_or_save, Out_Label)

        #if plot_loss_heatmap:
        #    Out_Label = graphics_dir + f'{model_save_name} - Loss Heatmap Epoch {epoch}.png'
        #    plot_loss_heatmap(loss_fn, encoder, decoder, device, test_loader, Out_Label, plot_or_save)



        if plot_Graphwiz:
            Graphviz_visulisation(encoder, decoder, precision, batch_size, channels, xdim, ydim, graphics_dir)

        #%% - Export model, statedicts and required variables for inference to disk    
        AE_file_name = full_model_export(False, checkpoint, model_output_dir, model_checkpoints_dir, epoch, encoder, decoder, optim, latent_dim, fc_input_dim, precision, Encoder, debug_model_exporter=False)

        #%% - Export all data logged to disk in form of .txt file in the output dir
        print("\nSaving inputs, model stats and results to .txt file in output dir...")
        #Comparison of true signal points to recovered signal points
        try: 
            #print("True signal points",number_of_true_signal_points)
            #print("Recovered signal points: ",number_of_recovered_signal_points, "\n")
            full_data_output["true_signal_points"] = number_of_true_signal_points  # Save the number of true signal points to the full_data_output dictionary
            full_data_output["recovered_signal_points"] = number_of_recovered_signal_points  # Save the number of recovered signal points to the full_data_output dictionary
        except:
            pass
        
        # Save .txt Encoder/Decoder Network Summary
        with open(full_netsum_filepath, 'w', encoding='utf-8') as output_file:    #utf_8 encoding needed as default (cp1252) unable to write special charecters present in the summary
            # Write the local date and time to the file
            TD_now = datetime.now()         # Get the current local date and time
            output_file.write(f"Date data taken: {TD_now.strftime('%Y-%m-%d %H:%M:%S')}\n")     # Write the current local date and time to the file
            output_file.write(("Model Name: " + model_save_name + f"\nTrained on device: {device}\n"))   # Write the model ID and device used to train the model to the file
            output_file.write((f"Model_ID: {model_id}\n"))   # Write the model ID to the file
            output_file.write((f"Max Epoch Reached: {max_epoch_reached}\n"))  # Write the max epoch reached during training to the file
            output_file.write((f"Training Time: {format_time(training_time)}\n")) # Write the training time to the file
            #output_file.write((f"Avg Epoch Time: {format_time(np.mean(epoch_times_list))}\n")) # Write the training time per epoch to the file
            output_file.write((f"Avg Epoch Time: {format_time(training_time/max_epoch_reached)}\n")) # Write the training time per epoch to the file

            output_file.write((f"\nTorch Seed Val to Reproduce Run: {T_seed}\n")) 
            output_file.write((f"Numpy Seed Val to Reproduce Run: {N_seed}\n")) 
            if seeding_value:
                output_file.write((f"User Set Seeding: True\n"))
            else:
                output_file.write((f"User Set Seeding: False\n"))
            
            output_file.write("\nInput Settings:\n")  # Write the input settings to the file
            for key, value in settings.items():
                output_file.write(f"{key}: {value}\n")
            
            output_file.write("\nLoss Function Settings:\n")    # Write the loss function settings to the file
            output_file.write(f"Loss Function Choice: {loss_fn}\n")
            if loss_function_selection == 0:
                output_file.write(f"zero_weighting: {zero_weighting}\n")
                output_file.write(f"nonzero_weighting: {nonzero_weighting}\n")    
            if loss_function_selection == 6:
                output_file.write(f"Split Loss Functions Selected (Hits, Non-Hits): {split_loss_functions}\n")

            output_file.write("\nNormalisation:\n")     # Write the normalisation settings to the file
            output_file.write(f"Mask optimised binary norm: {masking_optimised_binary_norm}\n") 

            output_file.write("\nPre Training:\n")   # Write the pre training settings to the file
            output_file.write(f"start_from_pretrained_model: {start_from_pretrained_model}\n")
            output_file.write(f"parent_model_id: {parent_model_id}\n")
            if start_from_pretrained_model:
                output_file.write(f"pretrained_model_path: {pretrained_model_path}\n")
                output_file.write(f"load_pretrained_optimiser: {load_pretrained_optimiser}\n")  
            
            output_file.write("\n \nFull Data Readouts:\n") 
            for key, value in full_data_output.items(): 
                output_file.write(f"{key}: {value}\n") 

            output_file.write("\nPython Lib Versions:\n") # Write the python library versions to the file
            output_file.write((f"PyTorch: {torch.__version__}\n"))  
            output_file.write((f"Torchvision: {torchvision.__version__}\n"))     
            output_file.write((f"Numpy: {np.__version__}\n")) 
            output_file.write((f"Pandas: {pd.__version__}\n")) 
            #output_file.write((f"MPL: {plt.__version__}\n"))

            output_file.write("\nAutoencoder Network:\n")  # Write the autoencoder network settings to the file
            output_file.write((f"AE File ID: {AE_file_name}\n"))    # Write the autoencoder network file ID to the file
            output_file.write("\n" + summary_str)   # Write the autoencoder network summary to the file

            output_file.write("\nTraining System Information:\n") 
            system_information = get_system_information()  # Get the system information using helper function
            output_file.write("\n" + system_information)  # Write the system information to the file
        print("- Completed -")

        if optimise_hyperparameter:
            val_loss = validation_routine(  encoder, 
                                            decoder, 
                                            device, 
                                            valid_loader, 
                                            loss_fn,
                                            time_dimension,
                                            reconstruction_threshold,
                                            masking_optimised_binary_norm,
                                            loss_vs_sparse_img,
                                            renorm_for_loss_calc)
                    
            history_da['val_loss'].append(val_loss)
            history_da['HTO_val'].append(HTO_val)
            history_da['training_time'].append(training_time)

            if print_validation_results:
                print("\n# VALIDATION RESULTS #")
                print(f"{hyperparam_to_optimise} value: ", HTO_val)
                print("Train loss: ", train_loss)
                print("Test loss: ", test_loss)
                print("Val loss: ", val_loss)
                print(f"Total Training time: {format_time(training_time)}", )

            print("\n\n- Model Training Completed -\n \n \n \n") 

    #%% - Plot the validation loss and training time for each hyperparameter value
    if optimise_hyperparameter:
        history_da['Hyperparameter_to_optimise'] = hyperparam_to_optimise
        if plot_validation_loss:
            Out_Label =  results_output_path_1 + model_save_name + "_" + hyperparam_to_optimise + r" Optimisation\\" + f'{model_save_name} - Val loss - Epoch {epoch}.png'
            loss_plot(history_da['HTO_val'], history_da['val_loss'], hyperparam_to_optimise, "Val loss (MSE)", "Validation loss", Out_Label, plot_or_save)

        if plot_training_time:
            Out_Label =  results_output_path_1 + model_save_name + "_" + hyperparam_to_optimise + r" Optimisation\\" +  f'{model_save_name} - Training time - Epoch {epoch}.png'
            loss_plot(history_da['HTO_val'], history_da['training_time'], hyperparam_to_optimise, "Training time (s)", "Training time", Out_Label, plot_or_save)
        
        print("Hyperparameter optimisation complete.\nExtracting optimisation data to Excel file...")
        netsum_directory = results_output_path_1 + model_save_name + "_" + hyperparam_to_optimise + r" Optimisation\\"                 # set the directory path where the txt files are located
        #print("directory to grab txt data from, seems not to be working, check dir is correct !!!!!!!: ", netsum_directory)
        excel_output_path = netsum_directory + hyperparam_to_optimise + " Optimisation - Settings.xlsx"    # set the output path for the Excel file
        extract_data_to_excel(netsum_directory, excel_output_path)    
        colour_code_excel_file(excel_output_path)        # COMINE THE EXCELL CODE TO ONE¬¬¬!!!!
        print("Extraction complete\n \n")

        #%% Prepare analysis of output models perfromance

        # Run the full performance tests
        output_file_path = netsum_directory + 'Analysis\\'  # directory to save output to
        os.makedirs(output_file_path, exist_ok=True)
        model_names = []
        pretrained_model_folder_paths = []
        for i, HTOsetting in enumerate(history_da['HTO_val']):
            model_names.append(model_save_name + "_" + hyperparam_to_optimise + "_" + str(HTOsetting))
            pretrained_model_folder_paths.append(netsum_directory + str(HTOsetting) + f'\\{model_save_name} - Training Results\Model_Deployment\\')
        
        ### NOTE FIX!!!!!!!!!!!!!!!!!!!!!!!
        run_full_perf_tests(num_files=perf_analysis_num_files, 
                            plot=perf_analysis_plot, 
                            save_recovered_data=True, 
                            dataset_dir=perf_analysis_dataset_dir, 
                            output_file_path=output_file_path, 
                            model_names=model_names, 
                            pretrained_model_folder_paths=pretrained_model_folder_paths,  
                            debug_mode = debug_hpo_perf_analysis)

    execution_timer.record_time(event_name="Program Debrief", event_type="stop") #records the time the program started

    #%% - End of Program - Printing message to notify user!
    print("\nProgram Complete - Shutting down...\n")    

    if run_profiler:
        # save the profiling results to a file
        profiler.dump_stats('profiler_results.prof')

        # use snakeviz to view the profiling results in a browser
        command = "snakeviz -s profiler_results.prof"
        os.system(command)


    if run_pytorch_profiler:
        # Stop the profiler
        torch_profiler.stop()
        print(torch_profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print("Generating execution timer plot...")
    fig = execution_timer.return_plot(dark_mode=True)
    fig.set_size_inches(28.5, 10.8)
    Out_Label = graphics_dir + f'{model_save_name} - Execution Timer.png'
    plot_save_choice(plot_or_save, Out_Label, dpi=100)  

    print("- Completed -")