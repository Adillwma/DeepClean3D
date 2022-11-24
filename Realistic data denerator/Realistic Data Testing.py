import numpy as np
import matplotlib.pyplot as plt

def zigzag(x_points):
    y_points = []
    return y_points

reflect_x = 0.3


# this is for a function y = x^2

y_points = []
num_points = 10
max_num = 1
step = max_num / num_points
pve_x_points = np.linspace(0,max_num,num_points)
x_reflect_points = [0]

x = step
for i in range(1,len(pve_x_points)):
    x_reflect_points.append(x)
    if x < reflect_x and x > x_reflect_points[i-1]:
        x += step
    elif x > -reflect_x and x < x_reflect_points[i-1]:
        x -= step



        

for i in pve_x_points:
    y_points.append(-(i**2))


plt.plot(x_reflect_points,y_points)
plt.show()
