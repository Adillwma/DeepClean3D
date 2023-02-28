# -*- coding: utf-8 -*-
"""
Realistic Data Simulator v1.0.1
@author: Max Carter & Adill Al-Ashgar
Created on Thurs Jan 26 2023
"""
"""
The time axis is now defined in set 35ps blocks (time res of TORCH)
The detector also has a standard deviation of 70ps on the time axis.
THIS PRODUCES A SET TIME DIMENSION of 509. 
"""

#%% - Dependencies
import numpy as np
import matplotlib.pyplot as plt
import random

resolution = 35E-12
std = 70E-12

#%% - Function
def realistic_data_sim(signal_points=1000, detector_pixel_dimensions=(128,88), hit_point=1.5, ideal=0, debug_image_generator=0, shift=1):
    '''
    Inputs:
    signal_points = number of points the hit produces (average 30 for realistic photon), N.B. these may not be at critical angle to 60 is maximum number
    detector_pixel_dimensions = pixel dimensions of the detector, x,y in a tuple (88x128)
    hit_point = The point on the detector the particle makes impact. This can be varied to change the shape of the graph.
    ideal = [default=1] produces either perfect circular photon ejection from impact (1) or random (0)
    debug_image_generator = [default=0] set to 1 to plot the output of this simulator (for debugging purposes)
    shift = [default=1] produces ribbon patterns that are shifted by half the axis maximum in either the +ve or -ve direction.
    
    The produced ribbon pattern is then shifted so that the model is not overfit, and flattened to a 2D image to run
    through the AE.

    Returns:
    flattened_data = The produced ribbon pattern is flattened and returned to prepare for the AE.
    The maximum of this flattened data (in flattened z) is given by t_pix_max and this MAX IS CURRENTLY SET AT 509
    However, N.B. The max signal point in the produced dataset (unnoised) is almost always NOT 509.
    '''
    # Width of Quartz Block in meters (physical x dimension - not to be confused with pixel x dimensions)
    # (this is not actual size of quarts, but it forms a good pattern. This is fundamental to the way this
    # pattern has been built so should not be changed)
    Quartz_width = 0.4

    #Half Width of Quartz Block in meters (x dimension) - BEWARE THIS IS HALF WIDTH, TRUE WIDTH GOES FROM -reflect_x to reflect_x
    reflect_x = Quartz_width/2   

    #Ideal or random? 1 if want ideal or 0 if want random #Here can set to either random numbers or to linear_x_points depending on what you want
    if ideal == 1:
        #Creates ideal linearly spaced x data #Uniform points to see the pattern
        linear_x_points = np.linspace(-np.pi, np.pi, signal_points)
        x = linear_x_points
    elif ideal == 0:
        #Creates more realistic randomly spaced x data  #random numbers that are more accurate to data
        random_x_points = [random.uniform(-np.pi, np.pi) for _ in range(signal_points)]
        x = random_x_points

    #Takes x range (either uniform (ideal) or random) and creates the reflections
    x_reflect_points = []
    for i in x:
        while i < -reflect_x or i > reflect_x:
            if i < -reflect_x:
                i = -i - 2 * reflect_x
            elif i > reflect_x:
                i = -i + 2 * reflect_x
        x_reflect_points.append(i)


    #Calulates corresponding y parabola points for each x
    y_points = []
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
    
    # max time particle could take. This allows us to set the max z axis for the noise to take.
    t_max = (((2*quartz_length) / np.cos((np.pi/2)-q_crit)) / particle_speed)
    # in pixels, this will be:
    t_pix_max = t_max // resolution

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

    # this is list of x at idx [0][0], y at [0][1] and z (time) at [1]
    final = list(zip(angle_filter, time))     # This is list of ((x,y), time) - dont ask me why, its annoying

    # create bins
    # x, y go from 0 to dim - 1 respectively here. z has TOF pixel.
    # x axis is between the reflectors, and the pixels of the x axis are the first dim of the input dim
    x_pixel = np.digitize([i[0][0] for i in final],np.linspace(-reflect_x, reflect_x, detector_pixel_dimensions[0])) #takes x points, then bins, returns indices

    # y axis is currently between 1 and -1 from cosine function (which has angle embedded by nature)
    # splits this into pixels in y:
    y_pixel = np.digitize([i[0][1] for i in final],np.linspace(-1, 1, detector_pixel_dimensions[1]))

    # z needs std dev in time added. This is added after the pattern is set.
    # This is put into bins without the digitize for now:
    z_pixel = np.array([np.random.normal(i[1], std)//resolution for i in final]) # i[1] add std to time then give z pixel with //resolution

    # combine them:
    coords = np.column_stack((x_pixel, y_pixel, z_pixel))

    # shift them all a maximum of half the max size if shift = 1:
    if shift == 1:
        coords[:,0] += np.random.randint(-np.round(np.max(x_pixel)/2),np.round(np.max(x_pixel)/2))
        coords[:,1] += np.random.randint(-np.round(np.max(y_pixel)/2),np.round(np.max(y_pixel)/2))
        coords[:,2] += np.random.randint(-np.round(t_pix_max/2),np.round(t_pix_max/2))

    # select those that would fall within the bounds of the thing after shifting:
    filtered = np.array([coord for coord in coords if
    (0 <= coord[0] <= detector_pixel_dimensions[0] - 1) and
    (0 <= coord[1] <= detector_pixel_dimensions[1] - 1) and
    (1 <= coord[2] <= t_pix_max)])  

    #Generates the random noise points (tmax//resolution sets the max pixels in the z axis)
    # x_noise = [random.randint(1, detector_pixel_dimensions[0]) for _ in range(noise_points)]
    # y_noise = [random.randint(1, detector_pixel_dimensions[1]) for _ in range(noise_points)]
    # z_noise = [random.randint(0, t_max//resolution) for _ in range(noise_points)]
    # flattening:
    flattened_data = np.zeros((detector_pixel_dimensions[0], detector_pixel_dimensions[1]))

    # check for if any remaining data is in range:
    if np.size(filtered) == 0:
        print('All data outside range. Empty array returned.')
        return flattened_data
    
    # this continues if there is data
    for point in filtered:
        # TOF is the z axis
        TOF = int(point[2])
        # index is the x and y axis
        flattened_data[int(point[0])][int(point[1])] = TOF

    
    #Plots the figure if user requests debugging
    # N.B time dimensiton is in pixels. x 3E-12 to get time value.
    if debug_image_generator == 1:
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        # the origional ribbon graph
        ax.scatter(x_pixel, y_pixel, z_pixel, label='Unshifted Data')
        # the shifted ribbon graph
        ax.scatter(filtered[:,0], filtered[:,1], filtered[:,2], label='Shifted Data')
        ax.legend()

        # setting limits:
        ax.set_xlim((0, detector_pixel_dimensions[0]))
        ax.set_ylim((0, detector_pixel_dimensions[1]))
        ax.set_zlim((0, t_pix_max))

        # add axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Time')

        plt.show()

        plt.imshow(flattened_data)
        plt.show()
    
    #Outputs to return to main dataset generator script
    return flattened_data

#%% - Testing Driver
#Uncomment line below for testing, make sure to comment out when done to stop it creating plots when dataset generator is running
realistic_data_sim(signal_points=1000, detector_pixel_dimensions=(128,88), hit_point=0.8, ideal=0, debug_image_generator=1)