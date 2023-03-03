import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

# only need to set the directory (dont need to add slash at the end)
output_dir = r"C:\Users\maxsc\OneDrive - University of Bristol\3rd Year Physics\Project\Autoencoder\2D 3D simple version\Circular and Spherical Dummy Datasets\Cross Stuff\Test\Data"


# you can change all the arguments in the function below:
def simp_generator(output_directory, Num_Images=[0,0,2,0,0], sig_pts=200, x_dim=128, y_dim=88, z_dim=100, shift=True, rotate=True, rotate_seperately=False):
    """
    This function simply calls the simulator function, and creates and saves the number of simulations with the defined imputs,
    to the directory specified above.
    output_directory - the path to save the file
    Num_Images - This is a list of the number of images to save. INDEX IS THE num_X variable!
    sig_pts - number of signal points for each X (some will be lost in shift/ rotation)
    x_dim - x axis dimentions (pixels)
    y_dim - y axis dimentions (pixels)
    z_dim - z axis dimentions (pixels)
    shift - [default = 1] adds shift to the cross to reduce overfitting. when == 0 mean all the same.
    (First item in list is for 0 Xs, second for 1 X, etx)
    rotate - rotates Xs 
    rotate_seperately - If you want all rotated SEPERATELY call True, TOGETHER call False
    """

    # the defaults of the following arguments are set above:
    def comb_simp_simulator(sig_pts, x_dim, y_dim, z_dim, shift, rotate, rotate_seperately, num_X):

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

        # assigning this here keeps the rotation angle the same during the loops if rotate == 0 or rotate_seperately = False:
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
            if shift == False and (rotate == False or rotate_seperately == False):
                break

        return flattened_data

    # seperates into the number of Xs specified and their no images respectively:
    for Crosses, num_save in enumerate(Num_Images):


        # will throw error if num_save == 0 as range(0) is undefined
        if num_save != 0:

            # only print this for the ones you actually save
            print(f"Creating {Crosses}X Images...")

            # loop for each image in each X
            for idx in tqdm(range(num_save), desc="X Image"):

                # define number of crosses
                num_X = Crosses #(to stop 0 gen)
                flattened_data = comb_simp_simulator(sig_pts, x_dim, y_dim, z_dim, shift, rotate, rotate_seperately, num_X)
                np.save(output_directory + '/Flat SimpleX-' + str(x_dim) + 'x' + str(y_dim) + '-' + str(Crosses) + 'Crosses, No' + str(idx + 1), flattened_data)

            print(f"Generation of {num_save} {Crosses}X images completed successfully\n")


simp_generator(output_dir)




