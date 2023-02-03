
import matplotlib.pyplot as plt
import numpy as np

x_min = 0
x_max = 88     

y_min = 0
y_max = 128

z_min = 0
z_max = 100

x1 = (x_min, y_min, z_min)
x2 = (x_max, y_max, z_max)



y1 = (x_min, y_max, z_max)
y2 = (x_max, y_min, z_min)


x1_data_points = x_min, x_max
y1_data_points = y_min, y_max
z1_data_points = z_min, z_max

x2_data_points = x_min, x_max
y2_data_points = y_max, y_min
z2_data_points = z_max, z_min




fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot(x1_data_points, y1_data_points, z1_data_points)
ax.plot(x2_data_points, y2_data_points, z2_data_points)
plt.show()

flattened_data = np.zeros(x_max, y_max)
