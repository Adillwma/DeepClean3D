# -*- coding: utf-8 -*-
"""
Fast Dataset Creator V2.0.0
Created on Fri March 3 2023
Author: Adill Al-Ashgar
University of Bristol

### Improvements to be made
# Add a way to have some files rotated some shifted some not without making the input settings to complicated
"""

#%% - User Inputs
xdim = 88                                           # X dimension of image
ydim = 128                                          # Y dimension of image
time_dimension = 100                                # Time dimension of image
sig_pts = 50                                        # Number of signal points to generate
data_set_title = "Dataset 32_X10Ks Sparse SIG50"    # Ouput title of dataset
data_sets_folder =r"N:\Yr 3 Project Datasets\\"     # Folder to save dataset to

#Ada: r"C:\Users\Student\Documents\UNI\Onedrive - University of Bristol\Yr 3 Project\Circular and Spherical Dummy Datasets\\"
#Max: 

### Number of data files of each type to generate  # (~89kb each, which scales linearly so 20,000 ~ 1.75Gb)
number_of_blanks = 0            # Number of blank data files to 
number_of_hemisphere = 0        # Number of hemisphere data files to generate

number_single_x = 10000         # Number of single x data files to generate
number_of_two_x = 0             # Number of two x data files to generate
number_of_three_x = 0           # Number of three x data files to generate
number_of_four_x = 0            # Number of four x data files to generate

number_single_realisitic = 0    # Number of realistic data files to generate
number_of_two_realisitic = 0    # Number of two realistic data files to generate
number_of_three_realisitic = 0  # Number of three realistic data files to generate
number_of_four_realisitic = 0   # Number of four realistic data files to generate

# Shared Settings
shift_positions = True         #If True then the signal positions are shifted by a random amount, otherwise they are not

# X Gen Settings
rotate_x_positions = False     #If True then the x signals are rotated, otherwise they are not
rotate_seperatly = True        #If True then each x signal is rotated by a random amount, otherwise they are all rotated by the same amount (only used if rotate_x_positions is True)

#Realistic Gen Settings
ideal_hit_spread = True        #If True then the hit points spread is idealised as linearly spaced in the pattern, otherwise points are chosen randomly from pattern 

#%% - Dependencies
#External Libraries
import datetime          # Current local time and date

# Local Custom Generators
from Data_Generators.Comb_X_Gen_Full import simp_generator
from Data_Generators.Blank_Data_Image_Generator import generate_blanks
from Data_Generators.Realistic_Gen import multi_real_gen_wrapper
#from Data_Generators.Hemispheres_Gen import 

# Local Dataset Validation Functions
from DC3D_V3.Helper_files.Dataset_distribution_tester_V1 import dataset_distribution_tester
from DC3D_V3.Helper_files.Dataset_Integrity_Check_V1 import dataset_integrity_check

#%% - Program Initalisation
# Combine output folder and dataset information folder
output_dir = data_sets_folder + data_set_title + "\Data\\"
output_dataset_info = data_sets_folder + data_set_title + " Information.txt" 
output_dataset_dist_img = data_sets_folder + data_set_title

#%% - Generators
# Generate Blanks
generate_blanks(xdim=xdim, ydim=ydim, number_of_files=number_of_blanks, output_dir=output_dir)

# Generate X signals
datasplit_values = [number_single_x, number_of_two_x, number_of_three_x, number_of_four_x]
simp_generator(output_dir, datasplit_values, sig_pts, ydim, xdim, time_dimension, shift_positions, rotate_x_positions)    #needs a output dir arg

#Generate Realisitic Signals
realistic_proportions = [number_single_realisitic, number_of_two_realisitic, number_of_three_realisitic, number_of_four_realisitic]
multi_real_gen_wrapper(output_dir, realistic_proportions, signal_points=sig_pts, detector_pixel_dimensions=(ydim, xdim), hit_point='random', ideal=ideal_hit_spread, debug_image_generator=False, shift=shift_positions)

print("Dataset Generation Completed")
#%% - Run Tests
try:
    print("\nVerifying Dataset Integrity...")
    integrity_results = dataset_integrity_check(output_dir, full_test=True, print_output=True)
    
    print("\nVerifying Dataset TOF Distribution...")
    distribution_results = dataset_distribution_tester(output_dir, time_dimension=time_dimension, ignore_zero_vals_on_plot=True, output_image_dir=output_dataset_dist_img)

except:                             #If test itself fails then result is automatically set to fail with unknown cause.
    dataset_tests_fail = True

print("- All Tests Completed -\n")

#%% - Generate Dataset Information txt and save it alongside distribution png 
print("Generating Dataset Details as txt file...")
# Save .txt Encoder/Decoder Network Summary
with open(output_dataset_info, 'w', encoding='utf-8') as output_file:    #utf_8 encoding needed as default (cp1252) unable to write special charecters present in the summary
    
    # Write the local date and time to the file
    TD_now = datetime.datetime.now()         # Get the current local date and time
    output_file.write(f"Date dataset created: {TD_now.strftime('%Y-%m-%d %H:%M:%S')}\n")    
    output_file.write((f"Dataset ID: {data_set_title}\n"))
    
    output_file.write("\nInput Settings:\n")
    output_file.write((f"X dim: {xdim} pixels\n"))
    output_file.write((f"Y dim: {ydim} pixels\n"))
    output_file.write((f"Time dim: {time_dimension}\n"))
    output_file.write((f"Signal Points: {sig_pts}\n"))
    output_file.write((f"Shift on: {shift_positions}\n"))
    output_file.write((f"Rotation on: {rotate_x_positions}\n"))
    output_file.write((f"Rotate Seperatly: {rotate_seperatly}\n"))
    output_file.write((f"Realistic Ideal Hit Spread: {ideal_hit_spread}\n"))

    output_file.write("\nDataset Blend:\n")
    output_file.write((f"Blank Files: {number_of_blanks}\n"))
    output_file.write((f"One Cross: {number_single_x}\n"))
    output_file.write((f"Two Crosses: {number_of_two_x}\n"))
    output_file.write((f"Three Crosses: {number_of_three_x}\n"))
    output_file.write((f"Four Crosses: {number_of_four_x}\n"))
    output_file.write((f"Hemisphere: {number_of_hemisphere}\n"))
    output_file.write((f"Realistic: {number_single_realisitic}\n"))

    output_file.write("\nDataset Test Results:\n")
    for key, value in integrity_results.items():
        output_file.write(f"{key}: {value}\n")
    for key, value in distribution_results.items():
        output_file.write(f"{key}: {value}\n")
print("- Completed -")


#%% - End of Program
#Final success message, also includes data path for easy copy paste to open    
print(f"\nDataset generated successfully.\nSaved in path:\n{output_dir}\n \nIMPORTANT - Remember to change the filename setting next time you run, OR to move this runs files out of the directory to avoid overwriting the saved data!\n")    
    
