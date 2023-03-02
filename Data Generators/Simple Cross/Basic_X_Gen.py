import numpy as np
import matplotlib.pyplot as plt

from Basic_X_Sim import simp_simulator

"""
This will return float64 array numbers.
MNIST dataset uses xxxxxxxxxx different graphs when the data is downloaded. To prove that its also not the size of the data
that matters, i will generate and test the same amount here:
"""
# specify where to save the flattened data
directory = r"C:\Users\maxsc\OneDrive - University of Bristol\3rd Year Physics\Project\Autoencoder\2D 3D simple version\Circular and Spherical Dummy Datasets\Cross Stuff\BigX 200x200\Data/"


# ----------------------------------------------------------------------------------
# you can change all the arguments in the function below:
def simp_generator(dataset_size = 1000, sig_pts = 600, x_dim = 200, y_dim = 200, z_dim = 100, shift=1):
    """
    This function simply calls the simulator function, and creates and saves the number of simulations with the defined imputs,
    to the directory specified above.
    dataset_size - the number of different flattened data instances to create
    others - defined in the simp_simulator function
    """

    # define array of all flattened d
    for idx in range(dataset_size):
        flattened_data = simp_simulator(sig_pts, x_dim, y_dim, z_dim, shift)

        np.save(directory + 'Simple Cross (flat pixel block data) ' + str(idx), flattened_data)
    
    print(str(dataset_size) + ' images saved to: ' + directory)
    
simp_generator()


