import numpy as np
import matplotlib.pyplot as plt
import random
from Orig_Real_SimV2 import realistic_data_sim

directory = r'C:\Users\maxsc\OneDrive - University of Bristol\3rd Year Physics\Project\Autoencoder\2D 3D simple version\Circular and Spherical Dummy Datasets\Realistic Stuff\ShiftedReal Random H_Point\Data'

def realistic_data_gen(dataset_size=10000, signal_points=1000, detector_pixel_dimensions=(128,88), hit_point='random', ideal=1, debug_image_generator=0, shift=1):
    """
    This is a generator function that produces, and saves to a specified directory, a number of flattened realistic data images.
    This takes the following inputs:
    dataset_size = Number of different images to produce
    Others are defined in the realistic_data_sim function.
    hit_point can be set to 'random' to produce a different hit point on the quartz each time.

    N.B. The data can be varied in two ways...
    1 - You can change the hit_point argument, which will produce a different pattern for a different impact point
    on the quartz sheet.
    2 - You can set shift to 1 to shift the data around but keep the pattern the same.
    (these can both be done at the same time)
    """

    # define count for number of empty arrays:
    count = 0

    for idx in range(dataset_size):
        
        if hit_point == 'random':

            # produce random hit point in the height of the quartz (1.6m):
            hit_point = np.random.random() * 1.6

        flattened_data = realistic_data_sim(signal_points, detector_pixel_dimensions, hit_point, ideal, debug_image_generator, shift)
        
        # count for how many are empty:
        if np.sum(flattened_data) == 0:
            count += 1

        np.save(directory + 'Realistic (flat pixel block data) ' + str(idx), flattened_data)
    
    print('There are ', count, ' empty arrays in the dataset')
    
    print(str(dataset_size) + ' images saved to: ' + directory)


# run function with defaults:
realistic_data_gen()
