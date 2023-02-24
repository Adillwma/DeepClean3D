import math
import random


#%% - Rotation Function
def rotate_points(points, angle):
    # convert angle to radians
    angle_rad = math.radians(angle)
    
    # unpack x and y coordinates
    x, y = points
    print(x,y )

    # apply rotation to x and y coordinates
    x_rot = x * math.cos(angle_rad) - y * math.sin(angle_rad)
    y_rot = x * math.sin(angle_rad) + y * math.cos(angle_rad)
    #print(x_rot)

    # add rotated point to list
    rotated_points = (round(x_rot), round(y_rot))   # Rounding makes physical sense becuase althout the photons can be at any point they will be qauntised to a particular pixes activation 

    # return list of rotated points
    return rotated_points


######CHANGE TO X SIM !
for point in hits_comb:
    # TOF is the z axis
    TOF = int(point[2])

    angle = random.uniform(0, 360)
    rotated_x, rotated_y = rotate_points((int(point[0]), int(point[1])), angle)
    print(np.shape(rotated_x))
    print(np.shape(rotated_y))        
    # index is the x and y axis
    flattened_data[rotated_x][rotated_y] = TOF