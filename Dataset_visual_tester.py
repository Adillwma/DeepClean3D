import numpy as np
import matplotlib.pyplot as plt
import os

# Define the folder path where the .npy files are stored
folder_path = r"C:\Users\Student\Documents\UNI\Onedrive - University of Bristol\Yr 3 Project\Circular and Spherical Dummy Datasets\Dataset 16_MultiX\Data"

# Get a list of all the .npy files in the folder
file_list = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

# Sort the list alphabetically to ensure consistent ordering
file_list.sort()

print(len(file_list))

# Define the number of files to load at a time
num_files = 6

# Create a figure with two rows and six columns
fig, axs = plt.subplots(nrows=2, ncols=6, figsize=(12, 6))

# Initialize a counter for the file index
file_index = 0

# Loop through the files and plot them
while file_index < len(file_list):

    # Load the next num_files files
    file_subset = file_list[file_index:file_index+num_files]
    images = [np.load(os.path.join(folder_path, f)) for f in file_subset]

    # Plot each image on the top and bottom row
    for i, image in enumerate(images):
        col_index = i % num_files
        top_ax = axs[0, col_index]
        bottom_ax = axs[1, col_index]
        top_ax.imshow(image)
        bottom_ax.imshow(image)

    # Show the figure and wait for the user to close it
    plt.show()

    # Update the file index
    file_index += num_files