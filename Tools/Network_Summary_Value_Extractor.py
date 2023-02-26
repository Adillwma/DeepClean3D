# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 2022
Net Summary Value Extractors V1.0.0
Author: Adill Al-Ashgar
University of Bristol

# File not found error already returns well defined error message so not including reporting for it?
"""

# Scrape Training Time
def get_training_time(file_path):
    "Time_Scraper V1.0.0"
    with open(file_path, 'r') as f:                                   # Opens the file at the given file path using a with to ensure proper file handline (makes sure file is closed properly)
        for line in f:                                                # Iterates through file line by line
            if 'Training Time:' in line:                              # For each line, checks if the string "Training Time:" is present
                Dstr = line.strip().split(': ')[1].split(' ')[0]  # If it is, it extracts the training time string, which is the value after the colon and before the first space character
                return float(Dstr)                                # Converts the extracted string to a float and returns it
    print('Training Time not found in file')               # If the function reaches the end of the file without finding the training time string, it warns user but does not break for error, the function will return a flag so main body program can implement its own conditions if neded.
    return ("Error")

# Val Loss
def get_valloss_time(file_path):
    "Val_Loss_Scraper V1.0.0"
    with open(file_path, 'r') as f:                                   # Opens the file at the given file path using a with to ensure proper file handline (makes sure file is closed properly)
        for line in f:                                                # Iterates through file line by line
            if 'Val Loss:' in line:                              # For each line, checks if the string "Training Time:" is present
                Dstr = line.strip().split(': ')[1].split(' ')[0]  # If it is, it extracts the training time string, which is the value after the colon and before the first space character
                return float(Dstr)                                # Converts the extracted string to a float and returns it
    print('Val Loss not found in file')               # If the function reaches the end of the file without finding the training time string, it warns user but does not break for error, the function will return a flag so main body program can implement its own conditions if neded.
    return ("Error")


# Train Loss
def get_trainloss_time(file_path):
    "Train_Loss_Scraper V1.0.0"
    with open(file_path, 'r') as f:                                   # Opens the file at the given file path using a with to ensure proper file handline (makes sure file is closed properly)
        for line in f:                                                # Iterates through file line by line
            if 'Train Loss:' in line:                              # For each line, checks if the string "Training Time:" is present
                Dstr = line.strip().split(': ')[1].split(' ')[0]  # If it is, it extracts the training time string, which is the value after the colon and before the first space character
                return float(Dstr)                                # Converts the extracted string to a float and returns it
    print('Train Loss not found in file')               # If the function reaches the end of the file without finding the training time string, it warns user but does not break for error, the function will return a flag so main body program can implement its own conditions if neded.
    return ("Error")