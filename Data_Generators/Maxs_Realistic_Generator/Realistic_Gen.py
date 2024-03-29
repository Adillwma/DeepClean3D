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
from tqdm import tqdm

# this is the resolution and standard deviation of the TORCH detector I believe.
resolution = 35E-12
std = 70E-12

# directory to save to
#directory = r"C:\Users\maxsc\OneDrive - University of Bristol\3rd Year Physics\Project\Autoencoder\2D 3D simple version\Circular and Spherical Dummy Datasets\Realistic Stuff\CombinedTest\Data2/"


def multi_real_gen_wrapper(directory, realistic_proportions, signal_points=1000, detector_pixel_dimensions=(128,88), height=100, hit_point='random', ideal=True, debug_image_generator=True, shift=True):
    """
    quick wrappewr to clean up the fast datset generator and keep it simple
    """
    def realistic_data_gen(directory, dataset_size=5, signal_points=1000, detector_pixel_dimensions=(128,88), height=100, hit_point='random', ideal=True, debug_image_generator=True, shift=True, num = 'random', idx=0):
        """
        This is a generator function that produces, and saves to a specified directory, a number of flattened realistic data images.
        This takes the following inputs:

        dataset_size = Number of different images to produce
        signal_points = Number of points the hit produces (average 30 for realistic impact), 
        N.B. these may not be at critical angle to 60 is maximum number
        detector_pixel_dimensions = pixel dimensions of the detector, x,y in a tuple (128x88) is same as torch.
        hit_point = The point on the detector the particle makes impact. This changes the shape of the graph.
        hit_point can be set to 'random' to produce a different hit point on the quartz each time.
        ideal = [default=True] produces either perfect circular photon ejection from impact (True) or random (False)
        debug_image_generator = [default=False] set to True to plot the output of this simulator (for debugging purposes)
        shift = [default=True] produces ribbon patterns that are shifted by half the axis maximum in either the +ve or -ve direction.
        num = the amount of realistic ribbon patterns to make (can set to 'random' to have random amounts between 0 and 5)

        N.B. The data can be varied in two ways...
        1 - You can change the hit_point argument, which will produce a different pattern for a different impact point
        on the quartz sheet.
        2 - You can set shift to 1 to shift the data around but keep the pattern the same.
        (these can both be done at the same time)
        """

        # define count for number of empty arrays:
        count = 0

        def realistic_data_sim(signal_points, detector_pixel_dimensions, height, hit_point, ideal, debug_image_generator, shift, num):
            
            # do random number of realistic points
            if num == 'random':
                num = random.randint(0,5)
            
            if hit_point == 'random':

                # produce random hit point in the height of the quartz (1.6m):
                hit_point = np.random.random() * 1.6
            
            # Width of Quartz Block in meters (physical x dimension - not to be confused with pixel x dimensions)
            # (this is not actual size of quarts, but it forms a good pattern. This is fundamental to the way this
            # pattern has been built so should not be changed)
            Quartz_width = 0.4

            #Half Width of Quartz Block in meters (x dimension) - BEWARE THIS IS HALF WIDTH, TRUE WIDTH GOES FROM -reflect_x to reflect_x
            reflect_x = Quartz_width/2   

            #Ideal or random? 1 if want ideal or 0 if want random #Here can set to either random numbers or to linear_x_points depending on what you want
            if ideal:
                #Creates ideal linearly spaced x data #Uniform points to see the pattern
                linear_x_points = np.linspace(-np.pi, np.pi, signal_points)
                x = linear_x_points
            else:
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

            # define the zeros flattened data:
            flattened_data = np.zeros(detector_pixel_dimensions)

            # loop for the number of realistic to make:
            for _ in range(num):
                
                # create new coords array
                new_coords = coords.copy()

                # shift them all a maximum of half the max size if shift = 1:
                if shift:
                    new_coords[:,0] += np.random.randint(-round(detector_pixel_dimensions[0] / 2),round(detector_pixel_dimensions[0] / 2))
                    new_coords[:,1] += np.random.randint(-round(detector_pixel_dimensions[1] / 2),round(detector_pixel_dimensions[1] / 2))
                    new_coords[:,2] += np.random.randint(-np.round(t_pix_max/2),np.round(t_pix_max/2))

                # select those that would fall within the bounds of the thing after shifting:
                filtered = np.array([coord for coord in new_coords if
                (0 <= coord[0] <= detector_pixel_dimensions[0] - 1) and
                (0 <= coord[1] <= detector_pixel_dimensions[1] - 1) and
                (1 <= coord[2] <= t_pix_max)])  

                #Generates the random noise points (tmax//resolution sets the max pixels in the z axis)
                # x_noise = [random.randint(1, detector_pixel_dimensions[0]) for _ in range(noise_points)]
                # y_noise = [random.randint(1, detector_pixel_dimensions[1]) for _ in range(noise_points)]
                # z_noise = [random.randint(0, t_max//resolution) for _ in range(noise_points)]
                # flattening:

                # check if not shifted out of range:
                if np.size(filtered) == 0:
                    return flattened_data
                
                # this continues if there is data
                for point in filtered:
                    # TOF is the z axis. Rescale to 100.
                    TOF = round(point[2] * (height / t_pix_max))
                    # index is the x and y axis
                    flattened_data[round(point[0])][round(point[1])] = TOF

            
            #Plots the figure if user requests debugging
            # N.B time dimensiton is in pixels. x 3E-12 to get time value.
            if debug_image_generator:
                
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                
                # the origional ribbon graph
                ax.scatter(x_pixel, y_pixel, z_pixel * (height / t_pix_max), label='Unshifted Data')
                ax.legend()

                # setting limits:
                ax.set_xlim((0, detector_pixel_dimensions[0]))
                ax.set_ylim((0, detector_pixel_dimensions[1]))
                ax.set_zlim((0, height))

                # add axis labels
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Time')

                plt.show()

                plt.imshow(flattened_data)
                plt.show()
            
            #Outputs to return to main dataset generator script
            return flattened_data


        # run the sim
        flattened_data = realistic_data_sim(signal_points, detector_pixel_dimensions, height, hit_point, ideal, debug_image_generator, shift, num)
        
        # count for how many are empty:
        if np.sum(flattened_data) == 0:
            count += 1

        # save
        np.save(directory + 'Realistic (flat pixel block data) ' + str(idx), flattened_data)

    # seperates into the number of Xs specified and their proportions respectively:
    for signals, num_save in enumerate(realistic_proportions):
        print(f"Creating {signals+1}Realistic Signal Images...")
        # define array of all flattened d
        for idx in tqdm(range(num_save), desc="Realistic Image"):

            # define number of crosses
            num_sigs = signals + 1 #(to stop 0 gen)
            flattened_data = realistic_data_gen(directory, num_save, signal_points, detector_pixel_dimensions, height, hit_point, ideal, debug_image_generator, shift, num=num_sigs, idx=idx)

        print(f"Generation of {num_save} {signals+1}Realistic Signal images completed successfully\n")



    # run function with defaults:
    #realistic_data_gen()

