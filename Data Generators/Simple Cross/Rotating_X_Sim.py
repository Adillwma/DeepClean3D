
import matplotlib.pyplot as plt
import numpy as np
import math


# N.B. this function rotates points in the x, y axis only. Hence, the angle only needs to be 180 degrees for 128x88:


def simp_simulator(sig_pts = 28, x_dim = 28, y_dim = 28, z_dim = 28, shift=1, rotate = 1):
    """
    This generator function generates crosses across the dimensions of the volume. (seeds to be generalised for non-perfect 28x28).
    It returns a numpy array of only signal points
    (a version that returns the signal, as well as signal and noise in two seperate arrays has been commented
    out as the most basic version of the autoencoder we're trialing adds noise itself)

    sig_pts - number of signal points
    x_dim - x axis dimentions (pixels)
    y_dim - y axis dimentions (pixels)
    z_dim - z axis dimentions (pixels)
    shift - [default = 1] adds shift to the cross to reduce overfitting. when == 0 mean all the same.
    """
    # define min and max of graph (pixels between 0 and 27)
    x_min = 0
    x_max = x_dim - 1     

    y_min = 0
    y_max = y_dim - 1

    # since 0 is how we encode non-hits, and the z axis is the intensity of the hits,
    # the minimum is 1 and the maximum is the maximum.
    z_min = 1
    z_max = z_dim

    # coords of min/max of line 1
    # x1 = (x_min, y_min, z_min)
    # x2 = (x_max, y_max, z_max)

    # coords of min/max of line 2
    # y1 = (x_min, y_max, z_max)
    # y2 = (x_max, y_min, z_min)

    # (the lines will go from x1 to x2 and y1 to y2)
    #--------------------------------

    # line 1 coordinates seperated to x,y,z
    x1_data_points = x_min, x_max
    y1_data_points = y_min, y_max
    z1_data_points = z_min, z_max

    # line 2 coordinates seperated to x,y,z
    x2_data_points = x_min, x_max
    y2_data_points = y_max, y_min
    z2_data_points = z_max, z_min

    #--------------------------------------------
    # for line 1:
    # This sets x, y, z coords for the line
    x1_array = np.linspace(x1_data_points[0], x1_data_points[1], math.floor(sig_pts/2))
    y1_array = np.linspace(y1_data_points[0], y1_data_points[1], math.floor(sig_pts/2))
    z1_array = np.linspace(z1_data_points[0], z1_data_points[1], math.floor(sig_pts/2))

    # joins them all together. 0 to 27 in x, y, 1 to 28 in z.
    L1_comb = np.column_stack((x1_array, y1_array, z1_array))      

    # for line2:
    x2_array = np.linspace(x2_data_points[0], x2_data_points[1], math.floor(sig_pts/2))
    y2_array = np.linspace(y2_data_points[0], y2_data_points[1], math.floor(sig_pts/2))
    z2_array = np.linspace(z2_data_points[0], z2_data_points[1], math.floor(sig_pts/2))

    L2_comb = np.column_stack((x2_array, y2_array, z2_array))

    # make final combined np array
    hits_comb = np.concatenate((L1_comb, L2_comb))

    # --------------------------------------------------------------------
    # adding rotation:

    if rotate == 1:

        # rotation angle in x, y plane
        angle_rad = math.radians(np.random.randint(0,360))
        print(angle_rad)

        # point to rotate around:
        cent_idx = round(len(L1_comb)/2)
        cent_pt = L1_comb[cent_idx]
        # remove TOF info
        cent_pt[2] = 0

        # move to around (0,0) point, so that we can rotate it.
        hits_comb -= cent_pt

        # add the rotation:
        x_rot = hits_comb[:,0] * math.cos(angle_rad) - hits_comb[:,1] * math.sin(angle_rad)
        y_rot = hits_comb[:,0] * math.sin(angle_rad) + hits_comb[:,1] * math.cos(angle_rad)
        z_rot = hits_comb[:,2]

        hits_comb = np.column_stack((x_rot, y_rot, z_rot))

        # move it back to the origional position
        hits_comb += cent_pt



    #-------------------------------------------------------------------
    # adding shift:

    # shift all by half the max
    if shift == 1:
        hits_comb[:,0] += np.random.randint(-np.round(x_max/2),np.round(x_max/2))
        hits_comb[:,1] += np.random.randint(-np.round(y_max/2),np.round(y_max/2))
        hits_comb[:,2] += np.random.randint(-np.round(z_max/2),np.round(z_max/2))

    
    
    # discard those that fall outside of array:
    hits_comb = np.array([hit for hit in hits_comb if
        (x_min <= round(hit[0]) <= x_max) and
        (y_min <= round(hit[1]) <= y_max) and
        (z_min <= round(hit[2]) <= z_max)])

    #-------------------------------------------------------------------


    # flattening the data

    # this creates a 28x28 zeros array  (plus 1 as max is 27.)
    flattened_data = np.zeros((x_dim, y_dim))

    # ADDING NOISE POINTS ARRAY THAT'S BEEN COMMENTED OUT:

    # random noise in x, y, z and join to one list like with hits:
    # random.randint(low, high=None, size=None, dtype=int)
    # noise_pts_x = np.random.randint(0,x_max, n_pts)
    # noise_pts_y = np.random.randint(0,y_max, n_pts)
    # noise_pts_z = np.random.randint(0,z_max, n_pts)
    # noise = np.column_stack((noise_pts_x, noise_pts_y, noise_pts_z))

    # join hits and noise to make list of signal noise (SN) points :
    # SN_pts = np.concatenate((hits_comb, noise))
    
    ###############################################################################

    for point in hits_comb:
        # TOF is the z axis
        TOF = round(point[2])
        # index is the x and y axis
        flattened_data[round(point[0])][round(point[1])] = TOF
    
    # for point in SN_pts:
    #     # TOF is the z axis
    #     TOF = point[2]
    #     # index is the x and y axis
    #     flattened_data[1][int(point[0])][int(point[1])] = TOF

    return flattened_data


# The array has all FLOAT64  as defaut!!!!

#------------------------------------------------------------------
# calling function and plotting results for clean and noisy:

flattened_data = simp_simulator(sig_pts = 200, x_dim = 128, y_dim = 88, z_dim = 28, rotate = 1, shift=1)

# for degubbing perposes:

print(np.shape(flattened_data))

# plot 2d clean data
plt.imshow(flattened_data)
plt.show()
# # plot 2d noisy data
# plt.imshow(flattened_data[1])
# plt.show()

