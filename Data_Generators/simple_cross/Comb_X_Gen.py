import numpy as np
import matplotlib.pyplot as plt
import math

"""
This will return float64 array numbers.
MNIST dataset uses xxxxxxxxxx different graphs when the data is downloaded. To prove that its also not the size of the data
that matters, i will generate and test the same amount here:
"""

# ----------------------------------------------------------------------------------
# you can change all the arguments in the function below:
def simp_generator(output_directory, Proportions=[0,0.8,0.2,0,0], sig_pts=200, x_dim=128, y_dim=88, z_dim=100, shift=True, rotate=False):
    """
    This function simply calls the simulator function, and creates and saves the number of simulations with the defined imputs,
    to the directory specified above.
    dataset_size - the number of different flattened data instances to create
    Proportions - This is a list of the proportions to produce the dataset in. Set as 0 if dont want any.
    (First item in list is for 0 Xs, second for 1 X, etx)
    others - defined in the simp_simulator function
    """

    #final_print = 'Saved: \n'
    
    #Converts shit and rotate boolean True/False inputs to 1/0 inputs (could be updated in simp_sim to accept the booleans and remove this line)
    shift = 0
    if shift:
        shift=1    
    
    rotate = 0
    if rotate:
        rotate=1

    # seperates into the number of Xs specified and their proportions respectively:
    for Crosses, num_save in enumerate(Proportions):

        # define array of all flattened d
        for idx in range(num_save):

            # define number of crosses
            num_X = Crosses
            flattened_data = comb_simp_simulator(num_X, sig_pts, x_dim, y_dim, z_dim, shift, rotate)

            np.save(output_directory + 'Flat SimpleX-' + str(x_dim) + 'x' + str(y_dim) + '-' + str(Crosses) + ' Crosses, No' + str(idx), flattened_data)

        #final_print += str(num_save) + ', ' + str(Crosses) + ' Cross Pics\n'
    """    
    print(final_print, '\n')
    print(sum(Proportions),' images saved to: \n' + output_directory)
    """
"""
simp_generator()
"""


def comb_simp_simulator(sig_pts = 100, x_dim = 200, y_dim = 200, z_dim = 100, shift = 1, rotate = 0, num_X = 1):
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
    rotate - rotates Xs (if you want all rotated SEPERATELY call 'seperate', together call 'together', not at all call 0)
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
    # im origionally had only 14 (every other point) on each axis but this didt give proper rounded numbers.
    # so im going to make my life simpler by just having 28...
    # for line 1:
    x1_array = np.linspace(x1_data_points[0], x1_data_points[1], round(sig_pts/2))
    y1_array = np.linspace(y1_data_points[0], y1_data_points[1], round(sig_pts/2))
    z1_array = np.linspace(z1_data_points[0], z1_data_points[1], round(sig_pts/2))

    L1_comb = np.column_stack((x1_array, y1_array, z1_array))      # joins them all together. Should be 28 at each point 0 to 28:
    # print(np.shape(L1_comb))
    # print(L1_comb)

    # for line2:
    x2_array = np.linspace(x2_data_points[0], x2_data_points[1], round(sig_pts/2))
    y2_array = np.linspace(y2_data_points[0], y2_data_points[1], round(sig_pts/2))
    z2_array = np.linspace(z2_data_points[0], z2_data_points[1], round(sig_pts/2))

    L2_comb = np.column_stack((x2_array, y2_array, z2_array))      # joins them all together. Should be 28 at each point 0 to 28:
    # print(np.shape(L2_comb))
    # print(L2_comb[1])

    # make final combined np array
    hits_comb = np.concatenate((L1_comb, L2_comb))

    # define flattened_data thats eventually returned:
    flattened_data = np.zeros((x_dim, y_dim))

    #-------------------------------------------------------------------
    # adding shift, roatation and multi X:

    # assigning this here keeps the rotation angle the same during the loops if rotate == 0 or together:
    angle_rad = math.radians(np.random.randint(0,360))

    # if = 0 its an empty array.
    if num_X == 0:
        for point in hits_comb:
            # TOF is the z axis
            TOF = round(point[2])
            # index is the x and y axis
            flattened_data[round(point[0])][round(point[1])] = TOF

    # loops through num_X:
    else:
        for _ in range(num_X):

            # make a copy of hits_comb to alter:
            new_hits_comb = hits_comb.copy()

            # if == 0 pass without rotating:
            if rotate == 0:
                pass
            # this rotates the cross if specified, before shifting
            else:
                # sets angle to change every loop only if rotate = seperate:
                if rotate == 'seperate':
                    # rotation angle in x, y plane
                    angle_rad = math.radians(np.random.randint(0,360))

                # point to rotate around:
                cent_idx = round(len(L1_comb)/2)
                cent_pt = L1_comb[cent_idx]
                # remove TOF info
                cent_pt[2] = 0

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
            if shift == 1:
                
                new_hits_comb[:,0] = new_hits_comb[:,0] + np.random.randint(-np.round(x_max/2),np.round(x_max/2))
                new_hits_comb[:,1] = new_hits_comb[:,1] + np.random.randint(-np.round(y_max/2),np.round(y_max/2))
                new_hits_comb[:,2] = new_hits_comb[:,2] + np.random.randint(-np.round(z_max/2),np.round(z_max/2))

            
            # select those that would fall within the bounds of the array after rotating and shifting for each loop:
            new_hits_comb = np.array([hit for hit in new_hits_comb if
                (x_min <= round(hit[0]) <= x_max) and
                (y_min <= round(hit[1]) <= y_max) and
                (z_min <= round(hit[2]) <= z_max)])

            # adds this loops cross:
            for point in new_hits_comb:
                # TOF is the z axis
                TOF = round(point[2])
                # index is the x and y axis
                flattened_data[round(point[0])][round(point[1])] = TOF

            # break the loop as the coordinates will just be overwritten if shift = 0 and (rotate = 0 or together):
            if shift == 0 and (rotate == 0 or rotate == 'together'):
                break

    return flattened_data




