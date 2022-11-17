# -*- coding: utf-8 -*-
"""
RAW (Spherical Only?) Data Validator V1.0.0
@author: Adill Al-Ashgar
Created on Tue Nov 15 13:11:41 2022

USER NOTICE!
x Loads individual Data files not whole folders, used to check the file saved to 
  disk by the data generator is in the correct formatting.

"""
#%% - Dependencies
import numpy as np
import matplotlib.pyplot as plt

#%% - User Settings
data_path = "C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/Yr 3 Project/Circular and Spherical Dummy Datasets/" #"C:/Users/Student/Desktop/fake im data/"  #"/local/path/to/the/images/"

dataset_title_2d = "TDataset/"
dataset_title_3d = "TDataset/"

name_2d = "TEST Circle"
name_3d = "TEST Sphere2"


#%% - Internal Program Setup
#Data Path Construction
path_2d = data_path + dataset_title_2d + name_2d + ".npy"
path_3d = data_path + dataset_title_3d + name_3d + ".npy"

#Data Load
a2dcheck = np.load(path_2d)
a3dcheck = np.load(path_3d)

#Data Consistency Check 1
a2dcheck = a2dcheck.squeeze()
a3dcheck = a3dcheck.squeeze()
print("a2dcheck batch shape:", a2dcheck.size)
print("a3dcheck batch shape:", a3dcheck.size)

#Data Plotting for Visual Consistency Check
hits_3d = np.nonzero(a3dcheck)
x3d = hits_3d[1]
y3d = hits_3d[0]
z3d = hits_3d[2]

fig = plt.figure()               #Plots spherical data
ax = plt.axes(projection='3d')
ax.scatter(x3d, y3d, z3d)#, s = signal_hit_size, c = "b") #Plots spherical data in blue
#ax.scatter(x_sph_noise_data,y_sph_noise_data,z_sph_noise_data, s = noise_hit_size, c = noise_colour) #Plots spherical noise in blue or red depending on the user selection of seperate_noise_colour
ax.set_xlim(0, 88)
ax.set_ylim(0, 128)
ax.set_zlim(0, 1000)
plt.show()