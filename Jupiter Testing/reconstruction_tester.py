import numpy as np
import matplotlib.pyplot as plt

def reconstruct_3D(data, reconstruction_threshold=0.5):
    data_output = []
    for cdx, row in enumerate(data):
        for idx, num in enumerate(row):
            if num > reconstruction_threshold:  #should this be larger than or equal to??? depends how we deal with the 0 slice problem
                data_output.append([cdx,idx,num])
    return np.array(data_output)






time_dimension = 100
reconstruction_threshold = 30

# Creating a 3D numpy array
test_case = np.zeros([128,88,100])

# Creates a diagonal line across the tof direction
for i in range(min(test_case.shape[0], test_case.shape[1])):
    test_case[i,i,i] = int(i % time_dimension)
print(test_case)

# Flattening the array to create a list of coordinates
flat_data = np.sum(test_case, axis=-1)

# Finding non zero vals for plotting
x, y, z = np.nonzero(test_case)
values = test_case[x, y, z]






recon_image = reconstruct_3D(flat_data, reconstruction_threshold)

fig = plt.figure()
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(x, y, z)
ax1.set_zlim(0, time_dimension)
ax1.set_title("Dummy Genrated Input")

ax2 = fig.add_subplot(132)
ax2.imshow(flat_data)

ax3 = fig.add_subplot(133, projection='3d', )
ax3.scatter(recon_image[:,0], recon_image[:,1], recon_image[:,2])
ax3.set_zlim(0, time_dimension)
ax3.set_title("Reconstruction Function")
fig.suptitle("3D Reconstruction Testing")
plt.show()
