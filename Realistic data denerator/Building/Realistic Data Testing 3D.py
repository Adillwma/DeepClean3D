import numpy as np
import matplotlib.pyplot as plt
import random

def zigzag(x_points):
    y_points = []
    return y_points

reflect_x = 0.1


# this is for a function y = x^2

y_points = []
num_points = 200
max_num = 2*np.pi
step = max_num / num_points
pve_x_points = np.linspace(0,max_num,num_points)
x_reflect_points = [0]

x = step
for i in range(1,len(pve_x_points)):
    x = float(float(f'{x:.{4}g}'))
    x_reflect_points.append(x)
    if (x + step <= reflect_x and x > x_reflect_points[i-1]):
        x += step
    elif (x - step) < - reflect_x:
        x = - reflect_x * 2 + step - x
    elif (x - step >= - reflect_x and x < x_reflect_points[i-1]):
        x -= step
    elif (x + step) > reflect_x:
        x = 2 * reflect_x + step - x

# at the moment this only works for rounded numbers. need it to

x_points = np.append(x_reflect_points, np.negative(x_reflect_points))

pve_y_points = [np.cos(i) for i in pve_x_points]
y_points = np.append(pve_y_points, pve_y_points)
y_points = [i + random.random()*0.03 for i in y_points]

# up till now this is the same as the 2D realistic data
# Now we need to add timing points which will depend on the angle in the y axis, will change like proportional to cos
# to make data more realistic, do circle instead of parabola, deleting points that would be within the critical angle,
# and maintaining the ones that go above and below

conjoined = np.squeeze(np.dstack((x_points, y_points)))           # np.dstack creates a 3D array, with first dimension 1. squeeze just deletes useless dimensions of 1

critial_angle_elim = [i for i in conjoined if np.cos(130.49) < i[1] < np.cos(40.49)]    # critical angle of quartz and air is 40.49


plt.scatter(conjoined[:,0],conjoined[:,1], s=0.1)
plt.show()
