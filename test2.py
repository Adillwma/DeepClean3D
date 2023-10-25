import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def add_noise_points(data, noise_points=100, time_dimension=1000):
    image = data.copy()
    if noise_points > 0:
        #Find dimensions of input image 
        x_dim = image.shape[0]
        y_dim = image.shape[1]

        #Create a list of random x and y coordinates
        x_coords = np.random.randint(0, x_dim, noise_points)   ##### NOTE: Change to using a torch random function instead of a numpy one
        y_coords = np.random.randint(0, y_dim, noise_points)  ##### NOTE: Change to using a torch random function instead of a numpy one

        # Iterate through noise_points number of random pixels to noise
        for i in range(noise_points):

            # Add a random number between recon_threshold and 1 to the pixel 
            image[x_coords[i], y_coords[i]] = np.random.uniform(0, time_dimension)   ##### NOTE: Change to using a torch random function instead of a numpy one

    return image

# Function to create sparse signal from a fully dense signal
def create_sparse_signal(input_image_batch, signal_points=2, linear=False):
    # Take as input a torch tensor in form [batch_size, 1, x_dim, y_dim]]
    # Create a copy of the input image batch
    #convert to torch tensor    
    input_image_batch = torch.from_numpy(input_image_batch)

    image_batch = input_image_batch.clone()

    # add a batch and channel dimension if the input image 
    if len(image_batch.shape) == 2:
        image_batch = image_batch.unsqueeze(0).unsqueeze(0)


    # Flatten the image tensor
    flat_batch = image_batch.view(image_batch.size(0), -1)

    # Count the number of non-zero values in each image
    nz_counts = torch.sum(flat_batch != 0, dim=1)

    # Find the indices of the images that have more non-zero values than signal_points
    sparse_indices = torch.where(nz_counts > signal_points)[0]

    # For each sparse image, randomly select signal_points non-zero values to keep
    for idx in sparse_indices:
        # Find the indices of the non-zero values in the flattened image
        nz_indices = torch.nonzero(flat_batch[idx]).squeeze()

        # Randomly select signal_points non-zero values to keep
        if linear:
            kept_indices = torch.linspace(0, nz_indices.numel() - 1, steps=signal_points).long()
        else:
            kept_indices = torch.randperm(nz_indices.numel())[:signal_points]

        # Zero out all non-selected values
        nonkept_indices = nz_indices[~torch.isin(nz_indices, nz_indices[kept_indices])]
        flat_batch[idx, nonkept_indices] = 0

    # Reshape the flat tensor back into the original shape
    output_image_batch = flat_batch.view_as(image_batch)

    # convert to numpy array
    output_image_batch = output_image_batch.numpy()

    #remove batch and channel dimensions if they were not present in the input image
    if len(input_image_batch.shape) == 2:
        output_image_batch = output_image_batch.squeeze()


    return output_image_batch

# Function to add shift in x, y and ToF to a true signal point due to detector resoloution
def simulate_detector_resolution(input_image_batch, x_std_dev, y_std_dev, tof_std_dev, x_scale, y_scale, time_scale, plot=False):
    """

    """

    # Convert physical values to pixel values
    x_std_dev_pixels = x_std_dev / x_scale
    y_std_dev_pixels = y_std_dev / y_scale
    tof_std_dev_pixels = tof_std_dev / time_scale

    # Take as input a torch tensor in form [batch_size, 1, x_dim, y_dim]]
    # Create a copy of the input image batch
    image_batch_all = input_image_batch.clone()

    for idx, image_batch_andc in enumerate(image_batch_all):
        image_batch = image_batch_andc.squeeze()
        # Assume that the S2 image is stored in a variable called "image_tensor"
        x, y = image_batch.size()

        # For all the values in the tensor that are non zero (all signal points) adda random value drawn from a gaussian distribution with mean of the original value and std dev of ToF_std_dev so simulate ToF resoloution limiting
        image_batch[image_batch != 0] = image_batch[image_batch != 0] + torch.normal(mean=0, std=tof_std_dev_pixels, size=image_batch[image_batch != 0].shape)

        # Generate random values for shifting the x and y indices
        x_shift = torch.normal(mean=0, std=x_std_dev_pixels, size=(x, y), dtype=torch.float32)
        y_shift = torch.normal(mean=0, std=y_std_dev_pixels, size=(x, y), dtype=torch.float32)

        # Create a mask for selecting non-zero values in the image tensor
        mask = image_batch != 0

        # Apply the x and y shifts to the non-zero pixel locations
        new_x_indices = torch.clamp(torch.round(torch.arange(x).unsqueeze(1) + x_shift), 0, x - 1).long()
        new_y_indices = torch.clamp(torch.round(torch.arange(y).unsqueeze(0) + y_shift), 0, y - 1).long()
        shifted_image_tensor = torch.zeros_like(image_batch)
        shifted_image_tensor[new_x_indices[mask], new_y_indices[mask]] = image_batch[mask]

        if plot:
            plt.imshow(shifted_image_tensor, cmap='gray', vmin=0, vmax=100)
            plt.title('S')
            plt.show()

        image_batch_all[idx,0] = shifted_image_tensor
        
    return image_batch_all

def reconstruct_3D(*args):
    results = []
    for data in args:
        data_output = []
        for cdx, row in enumerate(data):
            for idx, num in enumerate(row):
                if num > 0:  
                    data_output.append([cdx, idx, num])
        results.append(np.array(data_output))

    return results

#%% - User Settings
noise_points = 1000
signal_points = 1000 #2000 #30 #200
time_dimension = 1000
show_from_signal_start = False
show_past_signal_end = True
keep_photons = True
follow_time_window = False
loop = True
frame_rate = 30  # Define the frame rate (frames per second)
end_pause_seconds = 4
cmap = mpl.cm.gist_gray
output_dir = r'A:\Users\Ada\GitHub\DeepClean_Repo\Images\\'
output_name = f'{signal_points} scan with {noise_points} noise points. F keep_photons={keep_photons}'
file_path = r'N:\Yr 3 Project Datasets\PDT 1000 FAST\Data\111.npy' # path to npy file






#%% - Data load and preprocess

# load npy file from disk 
data = np.load(file_path)

sparse_data = create_sparse_signal(data, signal_points, linear=False)

noised_data = add_noise_points(sparse_data, noise_points, time_dimension)

data_3d, sparse_data_3d, noised_data_3d = reconstruct_3D(data, sparse_data, noised_data)


# Plot 2D data
"""

import matplotlib.pyplot as plt

# Create a single figure with three subplots side by side
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# First subplot for 'MCP Scan'
axs[0].imshow(data, cmap=cmap)
axs[0].set_title('MCP Scan')
axs[0].axis('off')

# Second subplot for 'Sparse MCP Scan'
axs[1].imshow(sparse_data, cmap=cmap)
axs[1].set_title('Sparse MCP Scan')
axs[1].axis('off')

# Third subplot for 'Noised MCP Scan'
axs[2].imshow(noised_data, cmap=cmap)
axs[2].set_title('Noised MCP Scan')
axs[2].axis('off')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the combined figure with all three plots
#plt.show()

# save the figure to disk
fig.savefig(f'{output_dir}{output_name}.png', dpi=300)

"""



#%% - Prepare anamiation paramters
if show_from_signal_start:
    min_finder = data
else:
    min_finder = noised_data

if show_past_signal_end:
    max_finder = noised_data
else:
    max_finder = data

data_values = noised_data
min_time = min(min_finder[min_finder != 0].flatten())
max_time = max(max_finder.flatten())

interval = 1000 / frame_rate   # Calculate the interval in milliseconds based on the frame rate

#%% - Plot 2D animation
"""
fig, axs = plt.subplots(1, 2, figsize=(20, 12))

def update(frame):
    axs[0].clear()
    axs[1].clear()
    current_time = min_time + frame
    axs[0].imshow(np.where((data_values == current_time), data_values, 0), cmap=cmap)
    axs[1].imshow(np.where((data_values <= current_time), data_values, 0), cmap=cmap)

    # set title for entire figure
    fig.suptitle(f'{signal_points} scan with {noise_points} noise points. keep_photons={keep_photons} \nCurrent Time: {current_time * 50 /1000:.2f} ns', fontsize=16)


    axs[0].set_title(f'MCP Current View')
    axs[1].set_title(f'MCP History')
    axs[0].axis('off')
    axs[1].axis('off')

animation = FuncAnimation(fig, update, frames=max_time - min_time, repeat=loop, blit=False, interval=interval)

# Assign the animation to a variable
anim = animation

# Display the animation
#plt.show()

# Save the animation to disk
#anim.save(f'{output_dir}{output_name}_double.gif', writer='imagemagick', fps=frame_rate)


"""


#%% - Plot 3D animation

# data_3d, sparse_data_3d, noised_data_3d are lists of 3D coordinates (x, y, t)
data_values = noised_data_3d

# Create a single figure with a 3D subplot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))

def update(frame):
    ax.clear()
    current_time = min_time + frame

    # Filter the data for the current time slice
    current_data = [(x, y, t) for x, y, t in data_values if t <= current_time]

    #current_signal = [(x, y, t) for x, y, t in data_3d if t <= current_time]

    current_frame = [(x, y, t) for x, y, t in data_values if t == current_time]

    if current_data:
        xs, ys, ts = zip(*current_data)
        ax.scatter(xs, ys, ts, c='b', alpha=0.3, marker='o', label='MCP Scan')
    """ 
    if current_signal:
        xs, ys, ts = zip(*current_signal)
        ax.scatter(xs, ys, ts, c='r', alpha=0.9, marker='o', label='MCP Signal')
    """
    if current_frame:
        xs, ys, ts = zip(*current_frame)
        ax.scatter(xs, ys, ts, c='r', alpha=0.9, marker='o', label='MCP Scan')




    # Create a meshgrid for the surface plot that  spans the entire x and y axis and then follows the z position of the current time slice
    Z = np.ones_like(X) * current_time

    # plot a surface for the current time slice
    ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.2, label='MCP Scan', linewidth=12)

    # Create a wireframe representation of the surface with red lines
    wireframe = ax.plot_wireframe(X, Y, Z, color='red', linewidth=0.5, rstride=88, cstride=128)


    # set title and plot settings
    ax.set_title(f'MCP Scan at Time: {current_time * 50 / 1000:.2f} ns')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Time (t)')

    ax.set_xlim(0, 128)
    ax.set_ylim(0, 88)

    if follow_time_window == False:
        ax.set_zlim(min_time, max_time)

# Calculate the number of frames for the pause (1 second)
pause_frames = int(frame_rate) * end_pause_seconds  # 'end_pause_seconds' second pause


animation = FuncAnimation(fig, update, frames=int(max_time - min_time) + pause_frames, repeat=loop, interval=interval)

anim = animation  # Assigning the animation to a variable so it persists

# Display the animation
#plt.show()

# Save the animation to disk
anim.save(f'{output_dir}{output_name}_3d.gif', writer='imagemagick', fps=frame_rate)













