import torch
import numpy as np
import random


# print(torch.tensor(lst))

# a = 1.3
# print(type(a))

# print(random.randint(1,2))

# a = np.array(((1,2,3), (1,2,3)))

# file = "TEST_REALISTIC_Data/"
# dir = "C:/Users/maxsc/OneDrive - University of Bristol/3rd Year Project/Autoencoder/2D 3D simple version/Circular and Spherical Dummy Datasets/"
# np.save(dir + file + 'a', a)
# x = np.load(dir+file+'a.npy')
# # print(x)
# a = [1,2,3,4,-6]

# print(min(a))

# print(np.linspace(0,7,4))
array = np.array([[1, 2], [3, 4], [5, 6]])
first_column = array[:, 0]
print(first_column)