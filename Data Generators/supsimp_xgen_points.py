
import matplotlib.pyplot as plt
import numpy as np

# define min and max of graph (pixels between 0 and 27)
x_min = 0
x_max = 27     

y_min = 0
y_max = 27

z_min = 0
z_max = 27

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
# im going to make 28 points on each axis:
# for line 1:
x1_array = np.linspace(x1_data_points[0], x1_data_points[1], 28)
y1_array = np.linspace(y1_data_points[0], y1_data_points[1], 28)
z1_array = np.linspace(z1_data_points[0], z1_data_points[1], 28)

L1_comb = np.column_stack((x1_array, y1_array, z1_array))      # joins them all together. Should be 28 at each point 0 to 28:
print(np.shape(L1_comb))
print(L1_comb[0])

# for line2:
x2_array = np.linspace(x2_data_points[0], x2_data_points[1], 28)
y2_array = np.linspace(y2_data_points[0], y2_data_points[1], 28)
z2_array = np.linspace(z2_data_points[0], z2_data_points[1], 28)

L2_comb = np.column_stack((x2_array, y2_array, z2_array))      # joins them all together. Should be 28 at each point 0 to 28:
print(np.shape(L2_comb))
print(L2_comb[1])

# make final combined np array
hits_comb = np.concatenate((L1_comb, L2_comb))
print(np.shape(hits_comb))

#-------------------------------------------------------------------

# flattening the data

# this creates a 28x28 zeros array  (plus 1 as max is 27.)
flattened_data = np.zeros((x_max + 1, y_max + 1))

for point in hits_comb:
    # TOF is the z axis
    TOF = point[2]
    # index is the x and y axis
    flattened_data[int(point[0])][int(point[1])] = TOF

print(flattened_data)

# plot 2d data
plt.imshow(flattened_data)
plt.show()

#------------------------------------------------------------------

# plot cross

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# # line plotted between min and max of both lines
# ax.plot([x[0] for x in L1_comb], [y[1] for y in L1_comb], [z[2] for z in L1_comb])
# ax.plot([x[0] for x in L2_comb], [y[1] for y in L2_comb], [z[2] for z in L2_comb])
# plt.show()

#------------------------------------------------------------------
