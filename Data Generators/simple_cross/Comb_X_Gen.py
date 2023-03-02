import numpy as np
import matplotlib.pyplot as plt
from Comb_X_Sim import comb_simp_simulator

"""
This will return float64 array numbers.
MNIST dataset uses xxxxxxxxxx different graphs when the data is downloaded. To prove that its also not the size of the data
that matters, i will generate and test the same amount here:
"""
# specify where to save the flattened data
directory = r"C:\Users\maxsc\OneDrive - University of Bristol\3rd Year Physics\Project\Autoencoder\2D 3D simple version\Circular and Spherical Dummy Datasets\Cross Stuff\MultiX - 80%1X - 20%2X - 128x88\Data/"


# ----------------------------------------------------------------------------------
# you can change all the arguments in the function below:
def simp_generator(Proportions = [0,0.8,0.2,0,0], dataset_size = 1000, sig_pts = 200, x_dim = 128, y_dim = 88, z_dim = 100, shift = 1, rotate = 0):
    """
    This function simply calls the simulator function, and creates and saves the number of simulations with the defined imputs,
    to the directory specified above.
    dataset_size - the number of different flattened data instances to create
    Proportions - This is a list of the proportions to produce the dataset in. Set as 0 if dont want any.
    (First item in list is for 0 Xs, second for 1 X, etx)
    others - defined in the simp_simulator function
    """

    final_print = 'Saved: \n'

    # seperates into the number of Xs specified and their proportions respectively:
    for Crosses, Proportion in enumerate(Proportions):

        # number to print for each num_X:
        num_save = round(dataset_size * Proportion)

        # define array of all flattened d
        for idx in range(num_save):

            # define number of crosses
            num_X = Crosses
            flattened_data = comb_simp_simulator(num_X, sig_pts, x_dim, y_dim, z_dim, shift, rotate)

            np.save(directory + 'Flat SimpleX-' + str(x_dim) + 'x' + str(y_dim) + '-' + str(Crosses) + ' Crosses, No' + str(idx), flattened_data)

        final_print += str(num_save) + ', ' + str(Crosses) + ' Cross Pics\n'
    
    print(final_print, '\n')
    print(str(dataset_size) + ' images saved to: ' + directory)
    
simp_generator()



