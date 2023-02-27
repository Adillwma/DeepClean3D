# -*- coding: utf-8 -*-
"""
Universal Datset Generator v1.0.0
Author: Adill Al-Ashgar
Created on 23 Feb 2023

USER NOTICE!
x Simulates Detector Readings of Generated Circular and Spherical Signals.

x To run standalone, just configure the variables in the 'User Settings' section 
below and then run full code.
"""

####TO DO LIST:
# Add label data to pixel block out
# Make condition so that 2d and 3d scatter plots dont show if printing black data, use imshow instead?
# Do the signal variations (movement and scaling)
# Fix flipped axis on the imshow plot ??
# Work out how best to deal with the radius parameter
# Turn into fuction so can be imported to other files


#%% - User settings
#Generator Variables
radius = 40 #User setting can be one number i.e (x) or a range in a tuple (min,max)
signal_points_input = (9000,15000) #(50,200) #User setting can be a range i.e "range(min,max,increment). If wanting to set a constant value then pass it as both min and max i.e (4,4)
number_of_signals = (1,4)
noise_points_input = 0#(80,100)  #(80,100) #If 0 there is no noise added
dataset_size = 1 #Number of individual data plots to generate and save for the dataset

detector_pixel_dimensions = (11*8, 128) #x, y in pixels
time_resoloution = 100 #time aka z axis

#Output options
output_type = 1 #0 outputs hit pixel locations, 1 outputs full sensor pixel array including no hit spaces (FOR FUTURE IMPROVEMENT)
filename = 'TEST'
directory = "C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/Yr 3 Project/Circular and Spherical Dummy Datasets/"

#Debugging Options
seeding_value = 0 #Seed for the random number generators, selecting a value here will make the code deterministic, returnign same vlaues from RNG's each time. If set to 0, seeding is turned off
auto_variables_debug_readout = 1 # 0=off, 1=on

debug_visulisations_on = 1 #0 is defualt, if set to 1 the circular and spherical data is plotted for visualisation
seperate_noise_colour = 1 #0 if desire noise same colour as signal, 1 to seperate noise and signal by colour
signal_hit_size = 10 # 1 is default small, 10 is medium, 20 is large, values in between are fine
noise_hit_size = 10 # 1 is default small, 10 is medium, 20 is large, values in between are fine

debug_block_outputs = 1 # 0=off, 1=on
seperate_block_noise_colour = 1 # 0=off, 1=on
coord_transform_sig_fig = 12    #Setting significant figures for the coordinate transofrms (polar to cartesian, spherical to cartesian), using set amount of sig figures avoids floating point rounding errors 


#%% - Dependencies
import matplotlib.pyplot as plt
import numpy as np


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
detector_x_lim_low = 0                                 #SHOULD THIS BECOME 1???
detector_x_lim_high = detector_pixel_dimensions[0]     #OR MAYBE SHOULD THIS BECOME -1???
detector_y_lim_low = 0                                 
detector_y_lim_high = detector_pixel_dimensions[1]
detector_z_lim_low = 0                                 #SHOULD THIS BECOME 1 (PROBABLY YES!???)
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

#%% - COMPUTE?
for f in range(0, dataset_size): 
    #Int or range detection for noise and signal inputs
    print("### Dataset run number:", f+1)

    if type(signal_points_input) == int: 
        signal_points = signal_points_input
        sig_select_debug_message = "Signal points given as Int:"
    else: 
        signal_points = np.random.randint(signal_points_input[0], signal_points_input[1])      
        sig_select_debug_message = "Signal points selected from range:"
        
    if type(noise_points_input) == int: 
        noise_points = noise_points_input
        noise_select_debug_message = "Noise points given as Int:"
    else: 
        noise_points = np.random.randint(noise_points_input[0], noise_points_input[1])      
        noise_select_debug_message = "Noise points selected from range:"
        
    #Debugging readout for variable selection including automaticly selected values        
    if auto_variables_debug_readout == 1:

        print(sig_select_debug_message,signal_points)    
        print(noise_select_debug_message,noise_points)           
        print("Random seeding:", seeding_value)
        print("Radius setting:", radius)
        print("Detector pixel dimensions:", detector_pixel_dimensions)
        print("Time resoloution", time_resoloution,"\n")
    
#%% - Settings output
    #Run settings for output file titling
    sp = '[sp %s]' % (signal_points)                                                         # signal_points
    npt = '[npt %s]' % (noise_points)                                                        # noise_points   
    sv = '[sv %s]' % (seeding_value)                                                         # seeding_value
    rad = '[rad %s]' % (radius)                                                              # radius
    dpd = '[dpd {} {}]'.format(detector_pixel_dimensions[0], detector_pixel_dimensions[1])   # detector_pixel_dimensions
    tr = '[tr %s]' % (time_resoloution)                                                      # time_resoloution
    run_settings = sp + npt + sv + rad + dpd + tr


def Spherical_Signal_Gen():    

    #Data list initialisation
    x_circ_data = []
    y_circ_data = []
    x_circ_noise_data = []
    y_circ_noise_data = []
    
    x_sph_data = []
    y_sph_data = []
    z_sph_data = []
    x_sph_noise_data = []
    y_sph_noise_data = []
    z_sph_noise_data = []
    
    #creates the label data
    signal_labels = np.ones(signal_points)
    noise_labels = np.zeros(noise_points)
    labels_output = np.concatenate((signal_labels, noise_labels))

#%% - Spherical Signal Simulator
 
    #Spherical Data Generator
    for i in range (0,signal_points):    #Generates points data
        angle1 = np.random.uniform(0, 2*np.pi)
        angle2 = np.random.uniform(0, 2*np.pi)
        x, y, z = spherical2cartesian(radius, angle1, angle2, coord_transform_sig_fig)
        x = round(x)
        y = round(y)
        z = round(z)
        x_sph_data.append(x + (detector_pixel_dimensions[0]/2))     
        y_sph_data.append(y + (detector_pixel_dimensions[1]/2)) 
        z_sph_data.append(z + (time_resoloution/2))    
        
    if noise_points > 0:              #Generates noise data
        for i in range (0,noise_points):
            x_noise = np.random.randint(detector_x_lim_low, detector_x_lim_high)
            y_noise = np.random.randint(detector_y_lim_low, detector_y_lim_high)
            z_noise = np.random.randint(detector_z_lim_low, detector_z_lim_high)
            x_sph_noise_data.append(x_noise)    
            y_sph_noise_data.append(y_noise)
            z_sph_noise_data.append(z_noise)
    
    if debug_visulisations_on  == 1:         
        fig = plt.figure()               #Plots spherical data
        ax = plt.axes(projection='3d')
        ax.scatter(x_sph_data,y_sph_data, z_sph_data, s = signal_hit_size, c = "b") #Plots spherical data in blue
        ax.scatter(x_sph_noise_data,y_sph_noise_data,z_sph_noise_data, s = noise_hit_size, c = noise_colour) #Plots spherical noise in blue or red depending on the user selection of seperate_noise_colour
        ax.set_xlim(detector_x_lim_low, detector_x_lim_high)
        ax.set_ylim(detector_y_lim_low, detector_y_lim_high)
        ax.set_zlim(detector_z_lim_low, detector_z_lim_high)
        plt.show()

    ###Output Modules
    #Combines noise and signal data into one array
    x_sph_output = np.concatenate((x_sph_data, x_sph_noise_data))
    y_sph_output = np.concatenate((y_sph_data, y_sph_noise_data))
    z_sph_output = np.concatenate((z_sph_data, z_sph_noise_data))

    #Combines the different dimensions (x, y & z, and labels) into one N x 4 array    
    sphere_data = np.vstack((x_sph_output, y_sph_output, z_sph_output)).T #, labels_output)).T






    generator_data = 

    #Randomises the order of the points so that the noise values are not all the last values, just in case the network uses that fact
    np.random.shuffle(sphere_data)
    
    #Data output to disk
    if output_type == 0:
        np.save(directory + filename + ' Sphere (hits data) %s  - Variables = ' % (f+1) + run_settings, sphere_data)
    
    else:
        pixel_block_3d = np.zeros((1, detector_pixel_dimensions[1],detector_pixel_dimensions[0],time_resoloution),dtype = np.single)
        for idr, r in enumerate(sphere_data[ :,0]):
            c = sphere_data[idr][1]
            depth = sphere_data[idr][2]
            pixel_block_3d[0][int(c)][int(r)][int(depth)] = 1     # -c broke the 3d
        
        if debug_block_outputs == 1:
            hits_3d = np.nonzero(pixel_block_3d)
            x3d = hits_3d[2]
            y3d = hits_3d[1]
            z3d = hits_3d[3]
            
            fig = plt.figure()               #Plots spherical data
            ax = plt.axes(projection='3d')
            ax.scatter(x3d, y3d, z3d)#, s = signal_hit_size, c = "b") #Plots spherical data in blue
            #ax.scatter(x_sph_noise_data,y_sph_noise_data,z_sph_noise_data, s = noise_hit_size, c = noise_colour) #Plots spherical noise in blue or red depending on the user selection of seperate_noise_colour
            ax.set_xlim(0, detector_pixel_dimensions[0])
            ax.set_ylim(0, detector_pixel_dimensions[1])
            ax.set_zlim(0, time_resoloution)
            plt.show()

        np.save(directory + filename + ' Sphere (pixel block data) %s - Variables = ' % (f+1) + run_settings, pixel_block_3d)
                
#%% - End of Program
#Final success message, also includes data path for easy copy paste to open    
print("\nDataset generated successfully.\nSaved in path:",directory,"\n \nIMPORTANT - Remember to change the filename setting next time you run OR move this runs files out of the directory to avoid overwriting your data!")    
    

