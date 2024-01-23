import torch
import numpy as np
import pickle
import pandas as pd
import timeit
import matplotlib.pyplot as plt
import os
import sys
import inspect
import datetime
import time


def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h :{int(minutes)}m :{seconds:3f}s"

# Helper function to return the batch learning method string to user
def batch_learning(training_dataset_size, batch_size):
    if batch_size == 1: 
        output = "Stochastic Gradient Descent"
    elif batch_size == training_dataset_size:
        output = "Batch Gradient Descent"        
    else:
        output = "Mini-Batch Gradient Descent"
    return(output)

def load_comparative_data(comparative_loss_paths, plot_live_training_loss=False, plot_live_time_loss=False):
    comparative_history_da = []
    comparative_epoch_times = []

    for loss_path in comparative_loss_paths:
        # load pkl file into dictionary
        if plot_live_training_loss or plot_live_time_loss:
            with open(loss_path + '\\Raw_Data_Output\\history_da_dict.pkl', 'rb') as f:
                comparative_history_da.append(pickle.load(f))
        
        if plot_live_time_loss:
            with open(loss_path + '\\Raw_Data_Output\\epoch_times_list_list.csv', 'rb') as f:
                # load the data from the csv file called f into a list 
                comparative_epoch_times.append(np.loadtxt(f, delimiter=',').tolist())
    
    return comparative_history_da, comparative_epoch_times

def save_variable(variable, variable_name, path, force_pickle=False):

    if force_pickle:
        with open(path + variable_name + "_forcepkl.pkl", 'wb') as file:
            pickle.dump(variable, file)
    else:
        if isinstance(variable, dict):
            with open(path + variable_name + "_dict.pkl", 'wb') as file:
                pickle.dump(variable, file)

        elif isinstance(variable, np.ndarray):
            np.save(path + variable_name + "_array.npy", variable)

        elif isinstance(variable, torch.Tensor):
            torch.save(variable, path + variable_name + "_tensor.pt")

        elif isinstance(variable, list):
            df = pd.DataFrame(variable)
            df.to_csv(path + variable_name + "_list.csv", index=False)

        elif isinstance(variable, int):
            with open(path + variable_name + "_int.pkl", 'wb') as file:
                pickle.dump(variable, file)

        elif isinstance(variable, float):
            with open(path + variable_name + "_float.pkl", 'wb') as file:
                pickle.dump(variable, file)

        elif isinstance(variable, str):
            with open(path + variable_name + "_str.pkl", 'wb') as file:
                pickle.dump(variable, file)

        else:
            raise ValueError("Unsupported variable type.")


# Helper function to allow values to be input as a range from which random values are chosen each time unless the input is a single value in which case it is used as the constant value
def input_range_to_random_value(*args):
    """
    Generate random values based on input ranges or single values.

    This function accepts an arbitrary number of arguments, where each argument
    can be either a single value (int or float) or a range (list or tuple) of
    values. For ranges, it generates a random integer if the range consists of
    integers, or a random float if the range consists of floats.

    Parameters:
    *args : int, float, list, tuple
        Arbitrary number of input arguments. Each argument can be a single value
        or a range represented as a list or tuple of two values.

    Returns:
    list
        A list containing the generated random values or the input values if they
        are already single values. If an input argument is not recognized as a
        value or range, None is appended to the list and an error message is printed.
    """
    results = []

    for input_range in args:

        if isinstance(input_range, (int, float)):
            ## If input is single value it is not randomised as is manually set
            results.append(input_range)
        
        elif isinstance(input_range, (list, tuple)):
            ## If input is a list or tuple then it is considered a range of values and is randomised in that range
            
            if all(isinstance(x, int) for x in input_range):
                ## If all values in the list are ints then the whole list is considered to be a range of ints and an int is returned
                results.append(torch.randint(input_range[0], input_range[1] + 1, (1,)).item())

            elif (isinstance(x, float) for x in input_range):
                ## Else if any single value in the list is a float then function will return a float
                results.append(torch.rand(1).item() * (input_range[1] - input_range[0]) + input_range[0])
            
            else:
                print("Error: input_range_to_random_value() input is not a value or pair of values in recognised format, must be float or int")
                results.append(None)
        else:
            print("Error: input_range_to_random_value() input is not a value or pair of values")
            results.append(None)
    
    return results
