import os
import torch
import numpy as np
from functools import partial
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from Helper_files.Data_Degradation_Functions import *
from Helper_files.Helper_Functions import input_range_to_random_value


class CustomDataset(Dataset):
    def __init__(self, dir_path, precision=32, store_full_dataset_in_memory=False):
        self.data_dir = dir_path + "/Data/"
        self.set_dtype(precision)
        self.file_list = os.listdir(self.data_dir)
        self.load_from_memory = store_full_dataset_in_memory
        if store_full_dataset_in_memory:
            self.load_data_into_memory()

    def load_data_into_memory(self):
        self.data = []  # List to hold all the data
        for file_name in tqdm(self.file_list, desc="Loading data into memory", leave=False, unit="files", color="pink"):
            sample = np.load(self.data_dir + file_name)
            sample = self.tensor_transform(sample)
            self.data.append(sample)

    def __getitem__(self, index):      
        if self.load_from_memory:
            sample = self.data[index]
        else:    
            sample = np.load(self.data_dir + self.file_list[index])                              # Loads the 2D image from the path
            sample = self.tensor_transform(sample)                                               # Transform to tesnor of shape [C, H, W] with user selected precision (32f, 64f)
        return sample
    
    def __len__(self):
        return len(self.file_list)

    def np_to_tensor(self, np_array, dtype):
        """
        Convert np array to torch tensor of user selected precision and adds a channel dim. 
        Takes in np array of shape [H, W] and returns torch tensor of shape [C, H, W]
        """
        tensor = torch.tensor(np_array, dtype=dtype)
        tensor = tensor.unsqueeze(0)                                                         # Append channel dimension to begining of tensor
        return(tensor)

    def set_dtype(self, precision):
        if precision == 16:
            dtype = torch.float16
        elif precision == 32:
            dtype = torch.float32
        elif precision == 64:
            dtype = torch.float64
        else:
            raise ValueError("Invalid dataset 'precision' value selected. Please select 16, 32, or 64 which correspond to torch.float16, torch.float32, and torch.float64 respectively.")      
          
        self.tensor_transform = partial(self.np_to_tensor, dtype=dtype) #using functools partial to bundle the args into np_to_tensor to use in custom torch transform using lambda function


class CustomDataLoader(DataLoader):
    def __init__(self, input_signal_settings, physical_scale_parameters, time_dimension, device, preprocess_on_gpu, precision, *args, **kwargs):
        super(CustomDataLoader, self).__init__(*args, **kwargs)
        self.input_signal_settings = input_signal_settings
        self.physical_scale_parameters = physical_scale_parameters
        self.time_dimension = time_dimension
        self.device = device
        self.preprocess_on_gpu = preprocess_on_gpu
        self.precision = precision
        
        
    def __iter__(self):
        return CustomDataLoaderIter(self, self.input_signal_settings, self.physical_scale_parameters, self.time_dimension, self.device, self.preprocess_on_gpu, self.precision)

class CustomDataLoaderIter:
    def __init__(self, loader, input_signal_settings, physical_scale_parameters, time_dimension, device, preprocess_on_gpu, precision):
        self.loader = loader
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.dataset = loader.dataset
        self.index_generator = loader._get_iterator()
        self.sample_iter = iter(self.batch_sampler)
        self.input_signal_settings = input_signal_settings
        self.physical_scale_parameters = physical_scale_parameters
        self.time_dimension = time_dimension
        self.set_dtype(precision)
        self.preprocess_on_gpu = preprocess_on_gpu
        if self.preprocess_on_gpu:
            self.device = device
        else:
            self.device = torch.device("cpu")
    
    def set_dtype(self, precision):
        if precision == 16:
            self.dtype = torch.float16
        elif precision == 32:
            self.dtype = torch.float32
        elif precision == 64:
            self.dtype = torch.float64
        else:
            raise ValueError("Invalid dataloader 'precision' value selected. Please select 16, 32, or 64 which correspond to torch.float16, torch.float32, and torch.float64 respectively.")      
          

    def _custom_processing(self, batch):
        signal_points, x_std_dev, y_std_dev, tof_std_dev, noise_points = self.input_signal_settings
        signal_settings = input_range_to_random_value(signal_points, x_std_dev, y_std_dev, tof_std_dev, noise_points) 
        degraded_batches = signal_degredation(signal_settings, batch, self.physical_scale_parameters, self.time_dimension, self.device, self.dtype)
        return degraded_batches   

    def __iter__(self):
        return self

    def __next__(self):
        indices = next(self.sample_iter)
        batch = self.collate_fn([self.dataset[i] for i in indices])
        batch = batch.to(self.device)

        sparse_output_batch, sparse_and_resolution_limited_batch, noised_sparse_reslimited_batch = self._custom_processing(batch)

        return batch, sparse_output_batch, sparse_and_resolution_limited_batch, noised_sparse_reslimited_batch





# # Test the custom dataset class
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision import transforms
    from torch.utils.data import DataLoader

    time_dimension = 100
    physical_scale_parameters = [0.1, 0.1, 0.1]

    preprocess_on_gpu = True   # Only woirks if cuda gpu is found, else will defulat back to cpu preprocess
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    signal_points = 150 
    noise_points = 20
    x_std_dev = 0
    y_std_dev = 0
    tof_std_dev = 5
    input_signal_settings = [signal_points, x_std_dev, y_std_dev, tof_std_dev, noise_points]

    # Create dataset
    dataset = CustomDataset(dir_path="N:\Yr 3 Project Datasets\Dataset 20_X500", precision=32)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0)
    dataloader2 = CustomDataLoader(input_signal_settings, physical_scale_parameters, time_dimension, device, preprocess_on_gpu, dataset, batch_size=10, shuffle=False, num_workers=0)

    # Get a batch of data
    for sample_batched1, sample_batched2 in zip(dataloader, dataloader2):
        # allclose the results

        print(torch.allclose(sample_batched1, sample_batched2[0]))
        """
        # plot one image form each batch to check they are the same
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(sample_batched1[0,0,:,:])
        plt.subplot(1,2,2)
        plt.imshow(sample_batched2[0][0,0,:,:])
        plt.show()


        # plot the differnt degraded versions of the same image
        plt.figure()
        plt.subplot(1,4,1)
        plt.imshow(sample_batched2[1][0,0,:,:])
        plt.subplot(1,4,1)
        plt.imshow(sample_batched2[1][0,0,:,:])
        plt.subplot(1,4,2)      
        plt.imshow(sample_batched2[2][0,0,:,:])
        plt.subplot(1,4,3)
        plt.imshow(sample_batched2[3][0,0,:,:])
        plt.show()

        """

    print("Finished")