


import numpy as np
import matplotlib.pyplot as plt
import torch

#%% from 0 - 100
# create numpy arrays
ones_arr = np.ones(101)
range_arr = np.arange(101)

# compute mean squared error and mean absolute error
mse = ((ones_arr - range_arr)**2).mean()
mae = np.abs(ones_arr - range_arr).mean()

# plot mean squared error and mean absolute error
plt.plot(range(101), np.full((101,), mse), label='MSE')
plt.plot(range(101), np.full((101,), mae), label='MAE')
plt.xlabel('Array Index')
plt.ylabel('Error')
plt.legend()
plt.show()


#%% from 0 - 1.0 using floats
# create numpy arrays
ones_arr = 0.01 * np.ones(101)
range_arr = np.arange(0, 1.01, 0.01)

# compute mean squared error and mean absolute error
mse = ((ones_arr - range_arr)**2).mean()
mae = np.abs(ones_arr - range_arr).mean()

# plot mean squared error and mean absolute error
plt.plot(range(101), np.full((101,), mse), label='MSE')
plt.plot(range(101), np.full((101,), mae), label='MAE')
plt.xlabel('Array Index')
plt.ylabel('Error')
plt.legend()
plt.show()


#%% using torch 0 - 100
# create numpy arrays
ones_arr = np.ones(101)
range_arr = np.arange(101)

# create PyTorch tensors
ones_tensor = torch.tensor(ones_arr)
range_tensor = torch.tensor(range_arr)

# compute mean squared error and mean absolute error using PyTorch loss functions
mse_criterion = torch.nn.MSELoss()
mse = mse_criterion(ones_tensor, range_tensor)

mae_criterion = torch.nn.L1Loss()
mae = mae_criterion(ones_tensor, range_tensor)

# plot mean squared error and mean absolute error
plt.plot(range(101), np.full((101,), mse.item()), label='MSE')
plt.plot(range(101), np.full((101,), mae.item()), label='MAE')
plt.xlabel('Array Index')
plt.ylabel('Error')
plt.legend()
plt.show()
range_arr = np.arange(101)

#%% using torch 0 - 1.0

import torch
import numpy as np
import matplotlib.pyplot as plt

# create numpy arrays
ones_arr = 0.01 * np.ones(101)
range_arr = np.arange(0, 1.01, 0.01)

# create PyTorch tensors
ones_tensor = torch.tensor(ones_arr)
range_tensor = torch.tensor(range_arr)

# compute mean squared error and mean absolute error using PyTorch loss functions
mse_criterion = torch.nn.MSELoss()
mse = mse_criterion(ones_tensor, range_tensor)

mae_criterion = torch.nn.L1Loss()
mae = mae_criterion(ones_tensor, range_tensor)

bce_criterion = torch.nn.BCELoss(reduction='none')
bce = bce_criterion(ones_tensor, range_tensor)
bce = bce.numpy()
# plot mean squared error and mean absolute error
plt.plot(range(101), np.full((101,), mse.item()), label='MSE')
plt.plot(range(101), np.full((101,), mae.item()), label='MAE')
plt.plot(range(101), bce, label='BCE')
plt.xlabel('Array Index')
plt.ylabel('Error')
plt.legend()
plt.show()

