import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import os
import torch

def get_wrapped_num(num, width):
    remainder = num % (2 * width)
    return remainder if remainder < width else 2 * width - remainder - 1 

def angle_to_ratio(angle):
    rad = angle * np.pi / 180
    x = int(np.sin(rad) * 100)
    y = int(np.cos(rad) * 100)
    return x, y 

def reconstruct_3D(data):
    indices = np.argwhere(data > 0)
    values = data[data > 0]
    data_output = np.column_stack((indices, values))
    return data_output

def plot_3d(file):
    file3d = reconstruct_3D(file)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(file3d[:, 0], file3d[:, 1], file3d[:, 2], s=2, alpha=1)
    plt.show()

def generate_rdt_data_tensors(num_of_files, angle, up, sideways, x_dim, y_dim, t_dim, show=False, origin=None):
    """
    Function to generate RDT data tensors. 

    Args:
    num_of_files (int): number of files to generate
    angle (int): angle of the data  # NEEDS FIXING, IGNORE FOR NOW
    up (int): up factor
    sideways (int): sideways factor
    x_dim (int): x dimension
    y_dim (int): y dimension
    t_dim (int): t dimension
    show (bool): True/False, shows the generated image
    origin (int, int, int): (x, y, time) origin of the data or 'None' to randomise

    Returns:
    torch.tensor: generated data tensor

    
    """


    file = torch.zeros((num_of_files, 1, y_dim, x_dim), dtype=torch.float64)

    if origin is None:
        # genrate random integers in shape (num_of_files, 1)
        x_origins = torch.randint(0, x_dim, (num_of_files, 1), dtype=torch.int32)
        y_origins = torch.randint(0, y_dim, (num_of_files, 1), dtype=torch.int32)
        t_origins = torch.randint(0, t_dim, (num_of_files, 1), dtype=torch.float64)
    else:
        # fill with the user input origin
        x_origins = torch.full((num_of_files, 1), origin[0], dtype=torch.int32)
        y_origins = torch.full((num_of_files, 1), origin[1], dtype=torch.int32)
        t_origins = torch.full((num_of_files, 1), origin[2], dtype=torch.float64)

    for idx, (origin_x, origin_y, origin_t) in tqdm(enumerate(zip(x_origins, y_origins, t_origins)), total=num_of_files, desc="RDT Image", unit="image", leave=False, colour="green"):
        file[idx, 0, origin_y, origin_x] = origin_t

        nudge_x, nudge_y, i, t_pos = 0, 0, 0, origin_t
        y_pos = origin_y

        while y_pos > 0 and t_pos < t_dim - 1:
            i += 1
            if i % up == 0:
                nudge_x += 1
            if i % sideways == 0:
                nudge_y += 1

            if i % max(up, sideways) == 0:
                y_pos = origin_y - nudge_y
                t_pos += 1
                x1 = get_wrapped_num(origin_x - nudge_x, x_dim)
                x2 = get_wrapped_num(origin_x + nudge_x, x_dim)
                file[idx, 0, y_pos, x1] = t_pos
                file[idx, 0, y_pos, x2] = t_pos

        if show:
            plt.imshow(file[1][0])
            plt.show()
            #plot_3d(file[0][0])

    return file
