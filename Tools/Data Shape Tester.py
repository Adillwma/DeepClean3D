import numpy as np
import matplotlib.pyplot as plt

arr_dir = r"C:\Users\maxsc\OneDrive - University of Bristol\3rd Year Physics\Project\Autoencoder\2D 3D simple version\Circular and Spherical Dummy Datasets\Realistic Stuff\RealSetPerfect\Data\Realistic (flat pixel block data) 3718.npy"
arr = np.load(arr_dir)
print(np.shape(arr))


plt.imshow(arr)
plt.show()

