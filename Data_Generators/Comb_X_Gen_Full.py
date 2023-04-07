import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

"""
This will return float64 array numbers.
MNIST dataset uses xxxxxxxxxx different graphs when the data is downloaded. To prove that its also not the size of the data
that matters, i will generate and test the same amount here:
"""

# ----------------------------------------------------------------------------------
# you can change all the arguments in the function below:
def simp_generator(output_directory, Proportions=[0,0.8,0.2,0,0], sig_pts=200, x_dim=128, y_dim=88, z_dim=100, shift=True, rotate=False, rotate_seperately=True, std = [2,2,2]):
    """
    This function simply calls the simulator function, and creates and saves the number of simulations with the defined imputs,
    to the directory specified above.
    dataset_size - the number of different flattened data instances to create
    Proportions - This is a list of the proportions to produce the dataset in. Set as 0 if dont want any.
    sig_pts - number of signal points
    x_dim - x axis dimentions (pixels)
    y_dim - y axis dimentions (pixels)
    z_dim - z axis dimentions (pixels)
    shift - [default = 1] adds shift to the cross to reduce overfitting. when == 0 mean all the same.
    (First item in list is for 0 Xs, second for 1 X, etx)
    rotate - rotates Xs (if you want all rotated SEPERATELY call 'seperate', together call 'together', not at all call 0)
    std - the standard deviation in pixels on each noise point on the cross you would like to make - in x, y, z
    """
    
    #Converts shit and rotate boolean True/False inputs to 1/0 inputs (could be updated in simp_sim to accept the booleans and remove this line)
    

    # seperates into the number of Xs specified and their proportions respectively:
    for Crosses, num_save in enumerate(Proportions):
        print(f"Creating {Crosses+1}X Images...")
        # define array of all flattened d
        for idx in tqdm(range(num_save), desc="X Image"):

            # define number of crosses
            num_X = Crosses + 1 #(to stop 0 gen)
            flattened_data = comb_simp_simulator(sig_pts, x_dim, y_dim, z_dim, shift, rotate, rotate_seperately, num_X, std)
            np.save(output_directory + 'Flat SimpleX-' + str(x_dim) + 'x' + str(y_dim) + '-' + str(Crosses+1) + ' Crosses, No' + str(idx), flattened_data)

        print(f"Generation of {num_save} {Crosses+1}X images completed successfully\n")



def comb_simp_simulator(sig_pts = 100, x_dim = 200, y_dim = 200, z_dim = 100, shift = False, rotate = False, rotate_seperately=False, num_X = 1, std = [2,2,2]):
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
    rotate - rotates Xs 
    rotate_seperately - If you want all rotated SEPERATELY call True, TOGETHER call False
    num_X is the number of X to make
    std - the standard deviation in pixels on each noise point on the cross you would like to make - in x, y, z
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

    #%% - adding shift, roatation and multi X:

    # assigning this here keeps the rotation angle the same during the loops if rotate == 0 or together:
    angle_rad = math.radians(np.random.randint(0,360))

    # loops through num_X:
    for _ in range(num_X):
        # make a copy of hits_comb to alter:
        new_hits_comb = hits_comb.copy()

        # this rotates the cross if specified, before shifting
        if rotate:

            # sets angle to change every loop only if rotate = seperate:
            if rotate_seperately:
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
        if shift:
            new_hits_comb[:,0] = new_hits_comb[:,0] + np.random.randint(-np.round(x_max/2),np.round(x_max/2))
            new_hits_comb[:,1] = new_hits_comb[:,1] + np.random.randint(-np.round(y_max/2),np.round(y_max/2))
            new_hits_comb[:,2] = new_hits_comb[:,2] + np.random.randint(-np.round(z_max/2),np.round(z_max/2))

        # here we add an std to all the signal points:
        if type(std) == list:
            # # select those that would fall within the bounds of the array after rotating and shifting for each loop:
            # for hit in new_hits_comb:

            #     shift_x = np.random.normal(0,std[0])
            #     shift_y = np.random.normal(0,std[1])
            #     shift_z = np.random.normal(0,std[2])

                
            #     hit = np.add(hit, [shift_x, shift_y, shift_z])
            #     print(hit)

            #     if (x_min <= round(hit[0]) <= x_max) or (y_min <= round(hit[1]) <= y_max) or (z_min <= round(hit[2]) <= z_max):
            #         del hit
            for idx, val in enumerate(std):

                std_vals = np.random.normal(loc=0, scale=val, size=np.shape(new_hits_comb)[0])

                new_hits_comb[:, idx] += std_vals


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
        if shift == False and (rotate == False or rotate_seperately == False):
            break
    
    plt.imshow(flattened_data)
    plt.show()

    return flattened_data

comb_simp_simulator(sig_pts = 100, x_dim = 200, y_dim = 200, z_dim = 100, shift = False, rotate = False, rotate_seperately=False, num_X = 1, std = [2,2,2])