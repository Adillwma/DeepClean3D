import numpy as np
import matplotlib.pyplot as plt

# big_dir = r"C:\Users\maxsc\OneDrive - University of Bristol\3rd Year Physics\Project\Autoencoder\2D 3D simple version\Circular and Spherical Dummy Datasets\Cross Stuff\BigX 200x200\Data\Simple Cross (flat pixel block data) 0.npy"
# big = np.load(big_dir)
# print(np.shape(big), np.max(big), "\n")

arr_dir = r"C:\Users\maxsc\OneDrive - University of Bristol\3rd Year Physics\Project\Autoencoder\2D 3D simple version\Circular and Spherical Dummy Datasets\Cross Stuff\Test\Data\Flat SimpleX-128x88-2Crosses, No1.npy"
arr = np.load(arr_dir)
print(np.shape(arr), np.max(arr))
plt.imshow(arr)
plt.show()

