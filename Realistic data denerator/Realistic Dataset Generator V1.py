# -*- coding: utf-8 -*-
"""
2D Flattened Data Test Generator v1.0.0
@authors: Adill Al-Ashgar & Max Carter
Created on Fri Dec 12 22:50:59 2022

USER NOTICE!
Simulates Detector Readings.

To run standalone, just configure the variables in the 'User Settings' section 
below and then run full code.
"""

#%% - User settings
#Generator Variables

signal_points_input = 10 #(50,200) #(50,200) #User setting can be a range i.e "range(min,max,increment). If wanting to set a constant value then pass it as both min and max i.e (4,4)
noise_points_input = 0 #(50,100) #(80,100) #(80,100) #If 0 there is no noise added
dataset_size = 1 #Number of individual data plots to generate and save for the dataset
detector_pixel_dimensions = (11*8, 128) #x, y in pixels
time_resoloution = 100 #time aka z axis
centre_ofset_input = (50,50)#(100,400)    #This is maximum displacment from centre for x and then for y (NOT a range)
radius = (10,40) #User setting can be one number i.e (x) or a range in a tuple (min,max)

#Output options
output_type = 1 #0 outputs hit pixel locations, 1 outputs full sensor pixel array including no hit spaces
filename = 'TEST_REALISTIC_Data'
directory = "C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/Yr 3 Project/Circular and Spherical Dummy Datasets/"

#Debugging Options
seeding_value = 0 #Seed for the random number generators, selecting a value here will make the code deterministic, returnign same vlaues from RNG's each time. If set to 0, seeding is turned off
auto_variables_debug_readout = 0 # 0=off, 1=on

debug_visulisations_on = 1 #0 is defualt, if set to 1 the circular and spherical data is plotted for visualisation
seperate_noise_colour = 1 #0 if desire noise same colour as signal, 1 to seperate noise and signal by colour
signal_hit_size = 10 # 1 is default small, 10 is medium, 20 is large, values in between are fine
noise_hit_size = 10 # 1 is default small, 10 is medium, 20 is large, values in between are fine

debug_block_outputs = 1 # 0=off, 1=on
block_output_labeled_data = 0  #this has taken over the setting below 
seperate_block_noise_colour = 1 # 0=off, 1=on
coord_transform_sig_fig = 12    #Setting significant figures for the coordinate transofrms (polar to cartesian, spherical to cartesian), using set amount of sig figures avoids floating point rounding errors 


#%% - Dependencies
import matplotlib.pyplot as plt
import numpy as np
from Realistic_Data_Simulator_V1 import realistic_data_generator

#%% - Helper Functions
def pol2cart(magnitude, angle, significant_figures = 12):                     
    x = abs(magnitude) * np.cos(angle)
    y = abs(magnitude) * np.sin(angle)
    return(round(x, significant_figures), round(y, significant_figures))

def spherical2cartesian(magnitude, angle1, angle2, significant_figures = 12):
    x = magnitude * np.sin(angle1) * np.cos(angle2)
    y = magnitude * np.sin(angle1) * np.sin(angle2)
    z = magnitude * np.cos(angle1)
    return(round(x, significant_figures), round(y, significant_figures), round(z, significant_figures))

#%% - Internal Program Setup 
#Defines the dimensions of the simulated detector
detector_x_lim_low = 0
detector_x_lim_high = detector_pixel_dimensions[0]
detector_y_lim_low = 0
detector_y_lim_high = detector_pixel_dimensions[1]
detector_z_lim_low = 0
detector_z_lim_high =  time_resoloution

#Checks if user selected a seeding for the RNGs
if seeding_value != 0:          #Setting seeding for the random number generators (usefull for debugging, result validation and collaboration)
    np.random.seed(seeding_value)
else:
    np.random.seed()

#Checks if user wants to plot noise in seperate colour to the signal
if seperate_noise_colour == 0:   #Setting colour to plot noise in, either blue (same as the signal colour) or red (for visual distinction)
    noise_colour = "b"
else:
    noise_colour = "r"    

#%% - Automatic Random Variable Selection 
for f in range(0, dataset_size): 
    #Int or range detection for noise and signal inputs
    print("### Dataset run number:", f+1)

    if type(signal_points_input) == int: 
        signal_points = signal_points_input
        sig_select_debug_message = "Signal points given as Int:"
    else: 
        signal_points = np.random.randint(signal_points_input[0], signal_points_input[1])      
        sig_select_debug_message = "Signal points selected from range:"

    if type(radius) == int: 
        radius = radius
        noise_select_debug_message = "Noise points given as Int:"
    else: 
        radius = np.random.randint(radius[0], radius[1])      
        noise_select_debug_message = "Noise points selected from range:"

    if type(noise_points_input) == int: 
        noise_points = noise_points_input
        noise_select_debug_message = "Noise points given as Int:"
    else: 
        noise_points = np.random.randint(noise_points_input[0], noise_points_input[1])      
        noise_select_debug_message = "Noise points selected from range:"

    if type(centre_ofset_input) == int: 
        centre_ofset_x = centre_ofset_input
        centre_ofset_y = centre_ofset_input
        centre_ofset_debug_message = "Centre offset given as Int:"
    else: 
        centre_ofset_x = np.random.randint(-centre_ofset_input[0], centre_ofset_input[0])
        centre_ofset_y = np.random.randint(-centre_ofset_input[1], centre_ofset_input[1])      
        centre_ofset_debug_message = "Centre offset selected from range:"
        
    #Debugging readout for variable selection including automaticly selected values        
    if auto_variables_debug_readout == 1:
        print(sig_select_debug_message,signal_points)    
        print(noise_select_debug_message,noise_points)           
        print("Random seeding:", seeding_value)
        print("Radius setting:", radius)
        print("Detector pixel dimensions:", detector_pixel_dimensions)
        print("Time resoloution", time_resoloution)
        print("Centre ofset x:", centre_ofset_x)
        print("Centre ofset y:", centre_ofset_y,"\n")   

    """ #Attampt to simplify the above. Works well but cant handle integers in any format ive tried i.e 4 or (4) or (4,4)
    signal_points = np.random.randint(signal_points_input[0], signal_points_input[-1])
    noise_points = np.random.randint(noise_points_input[0], noise_points_input[-1])
    print("signal points/noise points", signal_points, noise_points)
    """
    #Run settings for output file titling
    sp = '[sp %s]' % (signal_points)                 #signal_points
    npt = '[npt %s]' % (noise_points)                #noise_points   
    sv = '[sv %s]' % (seeding_value)                 #seeding_value
    rad = '[rad %s]' % (radius)                      #radius
    dpd = '[dpd {} {}]'.format(detector_pixel_dimensions[0], detector_pixel_dimensions[1])   #detector_pixel_dimensions
    tr = '[tr %s]' % (time_resoloution)              #time_resoloution
    cox = '[cox %s]' % (centre_ofset_x)              #Centre ofset X
    coy = '[coy %s]' % (centre_ofset_y)              #Centre ofset Y
    
    run_settings = sp + npt + sv + rad + dpd + tr + cox + coy 


######################################REPLACE BELOW WITH NEW FUNCTION##############################
x_sph_data, x_sph_noise_data, y_sph_data, y_sph_noise_data, z_sph_data, z_sph_noise_data = realistic_data_generator(signal_points, noise_points, seeding_value, detector_pixel_dimensions, time_resoloution, hit_point=1.5, ideal=1)
######################################REPLACE ABOVE WITH NEW FUNCTION THAT OUTPUTS 6 LISTS, XSIGNAL, XNOISE, YSIGNAL, YNOISE, ZSIGNAL, ZNOISE ##############################

#Plots Initially Generated Image if Requested for Debugging
if debug_visulisations_on  == 1:         
    fig = plt.figure()               
    ax = plt.axes(projection='3d')
    ax.scatter(x_sph_data,y_sph_data, z_sph_data, s = signal_hit_size, c = "b") #Plots spherical data in blue
    ax.scatter(x_sph_noise_data,y_sph_noise_data,z_sph_noise_data, s = noise_hit_size, c = noise_colour) #Plots spherical noise in blue or red depending on the user selection of seperate_noise_colour
    ax.set_xlim(detector_x_lim_low, detector_x_lim_high)
    ax.set_ylim(detector_y_lim_low, detector_y_lim_high)
    ax.set_zlim(detector_z_lim_low, detector_z_lim_high)
    plt.show()

#######################################Output Processing#####################################
#Combines noise and signal data into one array
x_sph_output = np.concatenate((x_sph_data, x_sph_noise_data))
y_sph_output = np.concatenate((y_sph_data, y_sph_noise_data))
z_sph_output = np.concatenate((z_sph_data, z_sph_noise_data))
    
#Combines the different dimensions (x, y & z, and labels) into one N x 4 array    
sphere_data = np.vstack((x_sph_output, y_sph_output, z_sph_output)).T       #Signal + Noise Data Output
sphere_data_labels = np.vstack((x_sph_data, y_sph_data, z_sph_data)).T      #Just Signal Data Output

#Randomises the order of the points so that the noise values are not all the last values, just in case the network uses that fact
np.random.shuffle(sphere_data)

#Flattening the 3D data to 2D array + TOF data embedded in hit information, ie NxN array with 0 values representing no hit and TOF values representing hits
pixel_block_3d_flattened = np.zeros((2, detector_pixel_dimensions[1], detector_pixel_dimensions[0]), dtype = np.single)

#Flattening Signal + Noise
for row, _ in enumerate(sphere_data[ :,2]):
    x_coordinate, y_coordinate, TOF = sphere_data[row]
    if 0 <= x_coordinate < detector_pixel_dimensions[0] and 0 <= y_coordinate < detector_pixel_dimensions[1]:
        pixel_block_3d_flattened[0][int(y_coordinate)][int(x_coordinate)] = TOF    

#Flattening Just Signal   
for row, _ in enumerate(sphere_data_labels[ :,2]):                
    labels_x_coordinate, labels_y_coordinate, labels_TOF = sphere_data_labels[row]
    if 0 <= labels_x_coordinate < detector_pixel_dimensions[0] and 0 <= labels_y_coordinate < detector_pixel_dimensions[1]:                
        pixel_block_3d_flattened[1][int(labels_y_coordinate)][int(labels_x_coordinate)] = TOF

#Plots Final Output Image if Requested for Debugging
if debug_block_outputs == 1:
    
    #Select Whether to Plot Signal Only or Signal + Noise
    if block_output_labeled_data == 1:
        label_choice = 1
    else:
        label_choice = 0                            
    
    #Plot
    plt.imshow(pixel_block_3d_flattened[label_choice]) #, origin='lower')  # origin lower flips the order of filling columns to account for the fact numpy arrays are indexed from top left wheras our sensor is indexed from bottom right
    plt.show()

#Saves all data (signal only and signal + noise) to disk and appends all user set variables to the filename
np.save(directory + filename + ' Sphere (pixel block data) %s - Variables = ' % (f+1) + run_settings, pixel_block_3d_flattened)
        
#%% - End of Program
#Final success message, also includes printing the data path for easy copy paste to open the folder the data was just saved into
print("\nDataset generated successfully.\nSaved in path:",directory,"\n \nIMPORTANT - Remember to change the filename setting next time you run OR move this runs files out of the directory to avoid overwriting your data!\n")    
    

