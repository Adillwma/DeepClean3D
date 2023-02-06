
import matplotlib.pyplot as plt
import numpy as np

# define min and max of graph (pixels)
x_min = 0
x_max = 88     

y_min = 0
y_max = 128

z_min = 0
z_max = 100

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

#-------------------------------
# might want to use np.linspace or other to make physical points?


fig = plt.figure()
ax = plt.axes(projection='3d')
# line plotted between min and max of both lines
ax.plot(x1_data_points, y1_data_points, z1_data_points)
ax.plot(x2_data_points, y2_data_points, z2_data_points)
plt.show()

flattened_data = np.zeros(x_max, y_max)
