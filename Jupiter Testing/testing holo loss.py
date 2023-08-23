# %%
import torch
###

# During processing we want the largest dimesnion to be the one that is virtual, wheras for loss calculation which is important yet trivial on compute we want the largest possible dimensional surface over which to calulate the loss. the smaller of the two dimensions is the one that should be made virtual

def holographic_transform(input_tensor, timesteps, binary_values=True):
    if not isinstance(input_tensor, torch.Tensor):
        raise ValueError("Input tensor should be a torch.Tensor")
    
    if len(input_tensor.shape) != 2:
        raise ValueError("Input tensor should be 2-dimensional")
    
    if timesteps <= 0:
        raise ValueError("Timesteps should be a positive integer")
    
    output_tensor = torch.zeros((input_tensor.shape[0], timesteps), dtype=input_tensor.dtype)
    
    for i in range(input_tensor.shape[0]):
        non_zero_indices = torch.nonzero(input_tensor[i]).squeeze(dim=1)
        for j in non_zero_indices:
            value = input_tensor[i,j]
            quantized_value = int(value * timesteps) - 1 
            print(f"\nInput Coordinate: (y x t), ({i}, {j.item()}, {quantized_value})")
            print(f"Transforms to: (y t x), ({i}, {quantized_value}, {j.item()})\n")

            if binary_values:
                output_tensor[i, quantized_value] = 1.0 #value set to fixed mask rather than embedded x coordinate, will test both ideas
            else:  # BROKEN!!!!
                if value > output_tensor[i, quantized_value]:   # Fix x occlusion mapping, this check is wrong at the moment, it should check if the ?
                    output_tensor[i, quantized_value] = j.item()


    return output_tensor

# Example usage
input_tensor = torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0],
                            [0.9, 1.0, 0.2, 0.0, 0.0]])
timesteps = 10
binary_hologram = True


print(input_tensor)
output_tensor = holographic_transform(input_tensor, timesteps, binary_hologram)
print(output_tensor)

print("\n\n")

import matplotlib.pyplot as plt
plt.imshow(input_tensor)
plt.show()

plt.imshow(output_tensor)
plt.show()





# %%


# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
# During processing we want the largest dimesnion to be the one that is virtual, wheras for loss calculation which is important yet trivial on compute we want the largest possible dimensional surface over which to calulate the loss. the smaller of the two dimensions is the one that should be made virtual

def holographic_transform(input_tensor, timesteps, virtual_dim='y', binary_values=True, debug=False):
    if not isinstance(input_tensor, torch.Tensor):
        raise ValueError("Input tensor should be a torch.Tensor")
    
    if len(input_tensor.shape) != 2:
        raise ValueError("Input tensor should be 2-dimensional")
    
    if timesteps <= 0:
        raise ValueError("Timesteps should be a positive integer")

    
    if virtual_dim not in ['x', 'y']:
        raise ValueError("Virtual dimension should be either 'x' or 'y'")
    
    if virtual_dim == 'x':
        input_tensor = input_tensor.T

    output_tensor = torch.zeros((input_tensor.shape[0], timesteps), dtype=input_tensor.dtype)
    
    for i in range(input_tensor.shape[0]):
        non_zero_indices = torch.nonzero(input_tensor[i]).squeeze(dim=1)
        for j in non_zero_indices:
            value = input_tensor[i,j]
            quantized_value = int(value * timesteps) - 1 

            if debug:
                print(f"\nInput Coordinate: (y x t), ({i}, {j.item()}, {quantized_value})")
                print(f"Transforms to: (y t x), ({i}, {quantized_value}, {j.item()})\n")

            if binary_values:
                output_tensor[i, quantized_value] = 1.0 #value set to fixed mask rather than embedded x coordinate, will test both ideas
            else:  # BROKEN!!!!
                #if value > output_tensor[i, quantized_value]:   # Fix x occlusion mapping, this check is wrong at the moment, it should check if the ?
                output_tensor[i, quantized_value] = j.item()


    return output_tensor

# Example usage
timesteps = 1000
virtual_dim = 'y'
binary_hologram = False
debug = False
input_file_path = r"N:\Yr 3 Project Datasets\PERF VALIDATION SETS\10K 100N 30S\Labels\126.npy .npy"

input_tensor = torch.from_numpy(np.load(input_file_path))
input_tensor = input_tensor / torch.max(input_tensor)

print("shape", input_tensor.shape)
print("T shape", input_tensor.T.shape)
output_tensor = holographic_transform(input_tensor, timesteps, virtual_dim, binary_hologram, debug)    #(input_tensor.T, timesteps, binary_hologram)

plt.imshow(input_tensor)
plt.show()

plt.imshow(output_tensor)
plt.show()





# %%
def reconstruct_3D(data):
    data_output = []
    for cdx, row in enumerate(data):
        for idx, num in enumerate(row):
            if num > 0:  
                data_output.append([cdx,idx,num])
    return np.array(data_output)

# %%
def renormalise_time(tensor, timesteps):
    tensor = tensor * timesteps
    return tensor

# %%




input_array = input_tensor.detach().cpu().numpy()
output_array = output_tensor.detach().cpu().numpy()
print(input_array.shape)
print(output_array.shape)

input_array = renormalise_time(input_array, timesteps)

input_3d = reconstruct_3D(input_array)
output_3d = reconstruct_3D(output_array)
print(input_3d.shape)
print(output_3d.shape)


# Plotting
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(input_3d[:, 0], input_3d[:, 1], input_3d[:, 2], c='b', marker='o', label='Input')

if virtual_dim == 'y':
    ax.scatter(output_3d[:,0], output_3d[:,2], output_3d[:, 1], c='r', marker='o', label='Output')

if virtual_dim == 'x':
    ax.scatter(output_3d[:,2], output_3d[:,0], output_3d[:, 1], c='r', marker='o', label='Output')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.view_init(azim=30, elev=30)
plt.legend(loc='upper left')
plt.show()

# %%



