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

Max z pixel is the longest a photon could take to get to top.
The z axis has pixel widths of t_max / time_resolution
time_resolution is how many z pixels you want.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import math

# change the defaults to change whats printed for debugging
def realistic_data_sim(signal_points = 1000, detector_pixel_dimensions=(128,88), time_resoloution=100, hit_point=1.4, ideal = True, std = False, shift = False, rotate = False, rotate_seperately = False, num_real = 1):
    '''
    Inputs:
    signal_points = number of points the hit produces (average 30 for realistic photon), N.B. these may not be at critical angle to 60 is maximum number
    detector_pixel_dimensions = pixel dimensions of the detector, x,y in a tuple (88x128)
    time_resoloution = number of discrete points the time dimension is sampled in
    hit_point = The point on the detector the particle makes impact in y axis
    ideal = if true makes perfect pattern. if false, random distribution.
    std = if True adds std to the z axis.
    
    Returns:
    A flattened array

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
    if ideal:
        x = linear_x_points
    else:
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
    if std:
        # deviation (dev) is usually 70ps in the z axis:
        dev = 70E-12

        z_idxs = np.digitize([np.random.normal(i[1], dev) for i in final],np.linspace(0, t_max, time_resoloution))

    else: 
        z_idxs = np.digitize([i[1] for i in final],np.linspace(0, t_max, time_resoloution))


    # combine all idxs to a list:
    hits_comb = np.column_stack((x_idxs,y_idxs,z_idxs))

    # define flattened array
    flattened_data = np.zeros((detector_pixel_dimensions[0], detector_pixel_dimensions[1]))
    
    #-------------------------------------------------------------------
    # adding shift, roatation and multi real:

    # assigning this here keeps the rotation angle the same during the loops if rotate == 0 or together:
    angle_rad = math.radians(np.random.randint(0,360))

    # if = 0 its an empty array.
    if num_real == 0:
        for point in hits_comb:
            # TOF is the z axis
            TOF = round(point[2])
            # index is the x and y axis
            flattened_data[round(point[0])][round(point[1])] = TOF

    # loops through num_real:
    else:
        for _ in range(num_real):

            # make a copy of hits_comb to alter:
            new_hits_comb = hits_comb.copy()

            # this rotates the cross if specified, before shifting
            if rotate:
                # sets angle to change every loop only if rotate = seperate:
                if rotate_seperately:
                    # rotation angle in x, y plane
                    angle_rad = math.radians(np.random.randint(0,360))

                # rotate around the center, z axis is zero (no TOF data):
                cent_pt = np.array((round(detector_pixel_dimensions[0]/2),round(detector_pixel_dimensions[1]/2) ,0))

                # move to around (0,0) point, so that we can rotate it.
                new_hits_comb -= cent_pt

                # add the rotation:
                x_rot = new_hits_comb[:,0] * math.cos(angle_rad) - new_hits_comb[:,1] * math.sin(angle_rad)
                y_rot = new_hits_comb[:,0] * math.sin(angle_rad) + new_hits_comb[:,1] * math.cos(angle_rad)
                z_rot = new_hits_comb[:,2]

                new_hits_comb = np.column_stack((x_rot, y_rot, z_rot))

                # move it back to the origional position
                new_hits_comb += cent_pt
            

            # shift individual x by half the max if shift is on
            if shift:
                
                new_hits_comb[:,0] = new_hits_comb[:,0] + np.random.randint(-np.round(detector_pixel_dimensions[0]/2),np.round(detector_pixel_dimensions[0]/2))
                new_hits_comb[:,1] = new_hits_comb[:,1] + np.random.randint(-np.round(detector_pixel_dimensions[1]/2),np.round(detector_pixel_dimensions[1]/2))
                new_hits_comb[:,2] = new_hits_comb[:,2] + np.random.randint(-np.round(time_resoloution/2),np.round(time_resoloution/2))

            
            # select those that would fall within the bounds of the array after rotating and shifting for each loop:
            new_hits_comb = np.array([hit for hit in new_hits_comb if
                (0 <= round(hit[0]) <= detector_pixel_dimensions[0] - 1) and
                (0 <= round(hit[1]) <= detector_pixel_dimensions[1] - 1) and
                (1 <= round(hit[2]) <= time_resoloution)])

            # adds this loops cross:
            for point in new_hits_comb:
                # TOF is the z axis
                TOF = round(point[2])
                # index is the x and y axis
                flattened_data[round(point[0])][round(point[1])] = TOF

            # break the loop as the coordinates will just be overwritten if shift = 0 and (rotate = 0 or together):
            if shift == False and (rotate == False or rotate_seperately == False):
                break

    # add shift
    # if shift:
    #     coords[:,0] += np.random.randint(-np.round(detector_pixel_dimensions[0] / 2),np.round(detector_pixel_dimensions[0] / 2))
    #     coords[:,1] += np.random.randint(-np.round(detector_pixel_dimensions[1] / 2),np.round(detector_pixel_dimensions[1] / 2))
    #     coords[:,2] += np.random.randint(-np.round(time_resoloution/2),np.round(time_resoloution/2))

    # # select those that would fall within the bounds of the thing after shifting:
    # filtered = np.array([coord for coord in coords if
    #     (0 <= round(coord[0]) <= detector_pixel_dimensions[0] - 1) and
    #     (0 <= round(coord[1]) <= detector_pixel_dimensions[1] - 1) and
    #     (1 <= round(coord[2]) <= time_resoloution)]) 
    
    # # add the hits to the zeros array
    # for coord in filtered:
    #     # TOF is the z axis
    #     TOF = round(coord[2])
    #     # index is the x and y axis
    #     flattened_data[round(coord[0])][round(coord[1])] = TOF
    
    #Outputs to return to main dataset generator script
    return(flattened_data)

#%% - Testing Driver
#Uncomment line below for testing, make sure to comment out when done to stop it creating plots when dataset generator is running

array = realistic_data_sim()

print(np.max(array))
plt.imshow(array)
plt.show()