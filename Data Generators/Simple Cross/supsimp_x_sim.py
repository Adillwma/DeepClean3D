
import matplotlib.pyplot as plt
import numpy as np


def simp_simulator(sig_pts = 28, n_pts = 0, x_dim = 28, y_dim = 28, z_dim = 28):
    """
    This generator function generates crosses across the dimensions of the volume. (seeds to be generalised for non-perfect 28x28).
    It returns an array consisting of two joined arrays:
    1st is the cross with no noise, to calculate loss etc.
    2nd is the cross with noise, thats to be cleaned.
    (The noise is random ever time its called)

    sig_pts - number of signal points
    n_pts - number of noise points
    x_dim - x axis dimentions (pixels)
    y_dim - y axis dimentions (pixels)
    z_dim - z axis dimentions (pixels)

    """
    # define min and max of graph (pixels between 0 and 27)
    x_min = 0
    x_max = x_dim - 1     

    y_min = 0
    y_max = y_dim - 1

    z_min = 0
    z_max = z_dim - 1

    # coords of min/max of line 1
    x1 = (x_min, y_min, z_min)
    x2 = (x_max, y_max, z_max)

    # coords of min/max of line 2
    y1 = (x_min, y_max, z_max)
    y2 = (x_max, y_min, z_min)

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
    # im going to make half of signal points on each line
    # for line 1:
    x1_array = np.linspace(x1_data_points[0], x1_data_points[1], int(sig_pts/2))
    y1_array = np.linspace(y1_data_points[0], y1_data_points[1], int(sig_pts/2))
    z1_array = np.linspace(z1_data_points[0], z1_data_points[1], int(sig_pts/2))

    L1_comb = np.column_stack((x1_array, y1_array, z1_array))      # joins them all together. Should be 28 at each point 0 to 28:
    # print(np.shape(L1_comb))
    # print(L1_comb[0])

    # for line2:
    x2_array = np.linspace(x2_data_points[0], x2_data_points[1], int(sig_pts/2))
    y2_array = np.linspace(y2_data_points[0], y2_data_points[1], int(sig_pts/2))
    z2_array = np.linspace(z2_data_points[0], z2_data_points[1], int(sig_pts/2))

    L2_comb = np.column_stack((x2_array, y2_array, z2_array))      # joins them all together. Should be 28 at each point 0 to 28:
    # print(np.shape(L2_comb))
    # print(L2_comb[1])

    # make final combined np array
    hits_comb = np.concatenate((L1_comb, L2_comb))

    #-------------------------------------------------------------------

    # flattening the data

    # this creates a 28x28 zeros array  (plus 1 as max is 27.)
    flattened_data = np.zeros((2,x_dim, y_dim))

    # random noise in x, y, z
    # random.randint(low, high=None, size=None, dtype=int)
    noise_pts_x = np.random.randint(0,x_max, n_pts)
    noise_pts_y = np.random.randint(0,y_max, n_pts)
    noise_pts_z = np.random.randint(0,z_max, n_pts)

    # join to one list like with hits:
    noise = np.column_stack((noise_pts_x, noise_pts_y, noise_pts_z))
    # and join hits and noise to make list of signal noise (SN) points :
    SN_pts = np.concatenate((hits_comb, noise))
    
    ###############################################################################
    # if setting is Noisy, adds noise, if Clean, no noise.

    for point in hits_comb:
        # TOF is the z axis
        TOF = point[2]
        # index is the x and y axis
        flattened_data[0][int(point[0])][int(point[1])] = TOF
    
    for point in SN_pts:
        # TOF is the z axis
        TOF = point[2]
        # index is the x and y axis
        flattened_data[1][int(point[0])][int(point[1])] = TOF
    
    return flattened_data


#------------------------------------------------------------------
# calling function and plotting results for clean and noisy:

flattened_data = simp_simulator(sig_pts = 28, n_pts=50, x_dim = 28, y_dim = 28, z_dim = 28)

# plot 2d clean data
plt.imshow(flattened_data[0])
plt.show()
# plot 2d noisy data
plt.imshow(flattened_data[1])
plt.show()

