# -*- coding: utf-8 -*-
"""
Realistic Data Simulator v1.0.1
@author: Max Carter & Adill Al-Ashgar
Created on Fri Dec 13 06:50:34 2022
"""

"""
Origional Simulator.
This returns hit points and noise points along with the number of signal points.
It is not yet flattened. This is flattened in the Generator function.

It has NO STD. Max z pixel is the longest a photon could take to get to top.
The z axis has pixel widths of t_max / time_resolution
time_resolution is how many z pixels you want.
"""

#%% - Dependencies
import numpy as np
import matplotlib.pyplot as plt
import random
   
#%% - Function
def realistic_data_sim(signal_points, detector_pixel_dimensions=(128,88), time_resoloution=100, hit_point=1.5, ideal=1, debug_image_generator=0, shift = 0):
    '''
    Inputs:
    signal_points = number of points the hit produces (average 30 for realistic photon), N.B. these may not be at critical angle to 60 is maximum number
    noise_points = number of noise points
    detector_pixel_dimensions = pixel dimensions of the detector, x,y in a tuple (88x128)
    time_resoloution = number of discrete points the time dimension is sampled in
    hit_point = The point on the detector the particle makes impact. May be able to make this 2D later (x,y)
    ideal = [default=1] XXXXXXXXXXXXXXXXXXXXX
    debug_image_generator = [default=0] set to 1 to plot the output of this simulator (for debugging purposes)
    
    Returns:
    x_pixel = list of signal x coordinates 
    x_noise = list of noise x coordinates  
    y_pixel = list of signal y coordinates  
    y_noise = list of noise y coordinates  
    z_pixel = list of signal z coordinates  
    z_noise = list of noise z coordinates  
    num_of_signal_points = True number of signal points in the output (as some photons will have left due to passing the critical angle and exiting the block). Could be used to measure error or for the loss function
    '''
    #Width of Quartz Block in meters (x dimension)
    Quartz_width = 0.4

    #Half Width of Quartz Block in meters (x dimension) - BEWARE THIS IS HALF WIDTH, TRUE WIDTH GOES FROM -reflect_x to reflect_x
    reflect_x = Quartz_width/2

    #Creates ideal linearly spaced x data #Uniform points to see the pattern
    min_num = -np.pi
    max_num = np.pi
    linear_x_points = np.linspace(min_num, max_num, signal_points)

    #Creates more realistic randomly spaced x data  #random numbers that are more accurate to data
    random_x_points = [random.uniform(-np.pi, np.pi) for _ in range(signal_points)]

    #Ideal or random? 1 if want ideal or 0 if want random #Here can set to either random numbers or to linear_x_points depending on what you want
    if ideal == 1:
        x = linear_x_points
    elif ideal == 0:
        x = random_x_points

    #Takes x range (either uniform (ideal) or random) and creates the reflections
    x_reflect_points = []
    for i in x:
        while i < -reflect_x or i > reflect_x:
            if i < -reflect_x:
                i = -i - 2 * reflect_x
            elif i > reflect_x:
                i = -i + 2 * reflect_x
        x_reflect_points.append(i)              # this has been checked and works


    #Calulates corresponding y parabola points for each x
    y_points = [np.cos(i) for i in x]

    # join these x reflected x and parabola y points into one list
    conjoined = list(zip(x_reflect_points, y_points))

    # critical angle quarts in radians
    q_crit = 40.49 * np.pi / 180

    # filter list to remove those that arent at critical angle ## FIRST LESS THAN AS COS(PI - Q_CRIT) IS -VE NUMBER 
    angle_filter = [i for i in conjoined if np.cos((np.pi/2) + q_crit) > i[1] or i[1] > np.cos((np.pi/2) - q_crit)] # filters out points less than critical angle

    # ADDING TIME AXIS

    # length of the quartz screen in m
    quartz_length = 1.6     

    # Speed of the cherenkov radiation in m, need to add n for quartz
    particle_speed = 3E8

    # max time particle could take. This allows us to set the max z axis.
    t_max = ((2*quartz_length) / np.cos((np.pi/2)-q_crit)) / particle_speed
   
    #Alternatively, just define each pixel in z axis to be i.e. 0.01ns later, then assign from there (this is probs better)

    time = []
    # for each of the allowed cherenkov particles
    for i in angle_filter:

        # if it moves straight towards the detector
        if i[1] > 0:
            # this is just time = dist / speed formula. dist / cos(angle) is true distance
            time.append((hit_point / i[1]) / particle_speed )

        # if it moves away from the detector and gets reflected back up
        elif i[1] < 0:
            # goes down and back up
            time.append(((2 * quartz_length - hit_point) / abs(i[1])) / particle_speed)

    final = list(zip(angle_filter, time))     # This is list of ((x,y), time) - dont ask me why, its annoying

    # create bins
    # x axis is between the reflectors, and the pixels of the x axis are the first dim of the input dim
    x_idxs = np.digitize([i[0][0] for i in final],np.linspace(-reflect_x, reflect_x, detector_pixel_dimensions[0])) #takes x points, then bins, returns indices

    # -1 to 1 as cos(x)
    y_idxs = np.digitize([i[0][1] for i in final],np.linspace(-1, 1, detector_pixel_dimensions[1]))

    # 0 to t_max as thats the max time a photon can take.
    z_idxs = np.digitize([i[1] for i in final],np.linspace(0, t_max, time_resoloution))   

    # combine all idxs to a list:
    coords = np.column_stack((x_idxs,y_idxs,z_idxs))

    # define flattened array
    flattened_data = np.zeros((detector_pixel_dimensions[0], detector_pixel_dimensions[1]))
    
    # add shift
    if shift == 1:
        coords[:,0] += np.random.randint(-np.round(detector_pixel_dimensions[0] / 2),np.round(detector_pixel_dimensions[0] / 2))
        coords[:,1] += np.random.randint(-np.round(detector_pixel_dimensions[1] / 2),np.round(detector_pixel_dimensions[1] / 2))
        coords[:,2] += np.random.randint(-np.round(time_resoloution/2),np.round(time_resoloution/2))

    # select those that would fall within the bounds of the thing after shifting:
    filtered = np.array([coord for coord in coords if
    (0 <= coord[0] <= detector_pixel_dimensions[0] - 1) and
    (0 <= coord[1] <= detector_pixel_dimensions[1] - 1) and
    (1 <= coord[2] <= time_resoloution)]) 
    
    # add the hits to the zeros array
    for coord in filtered:
        # TOF is the z axis
        TOF = int(coord[2])
        # index is the x and y axis
        flattened_data[int(coord[0])][int(coord[1])] = TOF
    
    #Plots the figure if user requests debugging
    if debug_image_generator == 1:
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(x_idxs, y_idxs, z_idxs)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Time')

        plt.show()

        plt.imshow(flattened_data)
        plt.show()
    
    #Outputs to return to main dataset generator script
    return(flattened_data)

#%% - Testing Driver
#Uncomment line below for testing, make sure to comment out when done to stop it creating plots when dataset generator is running

# realistic_data_sim(signal_points=1000, detector_pixel_dimensions=(128,88), time_resoloution=100, hit_point=1.3, ideal=1, debug_image_generator=1, shift = 1)