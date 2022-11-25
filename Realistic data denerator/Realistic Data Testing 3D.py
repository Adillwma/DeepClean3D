import numpy as np
import matplotlib.pyplot as plt
import random

def zigzag(x_points):
    y_points = []
    return y_points

reflect_x = 0.1


# this is for a function y = x^2

y_points = []
num_points = 1000
max_num = 1
step = max_num / num_points
pve_x_points = np.linspace(0,max_num,num_points)
x_reflect_points = [0]

x = step
for i in range(1,len(pve_x_points)):
    x = float(float(f'{x:.{4}g}'))
    x_reflect_points.append(x)
    if (x < reflect_x and x > x_reflect_points[i-1]) or x == -reflect_x:
        x += step
    elif (x > - reflect_x and x < x_reflect_points[i-1]) or x == reflect_x:
        x -= step

x_points = np.append(x_reflect_points, np.negative(x_reflect_points))

pve_y_points = [-(i**2) for i in pve_x_points]
y_points = np.append(pve_y_points, pve_y_points)
y_points = [i + random.random()*0.03 for i in y_points]

# up till now this is the same as the 2D realistic data
# Now we need to add timing points which will depend on the angle in the y axis, will change like proportional to cos
# to make data more realistic, do circle instead of parabola, deleting points that would be within the critical angle,
# and maintaining the ones that go above and below



plt.scatter(x_points,y_points, s=0.1)
plt.show()
