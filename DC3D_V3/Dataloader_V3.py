import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import torch
import os
from tqdm import tqdm
from Helper_files.Data_Degradation_Functions import *
from Helper_files.Helper_Functions import input_range_to_random_value

# Dummy dataset class
class DTM_Dataset(Dataset):
    """
    Disk to Memory Dataset
    Loads data from disk to memory in the form of bundles to reduce i/o operations

    Args:
    data_dir: str, the directory where the data files are stored

    Methods:
    __len__: returns the number of data files

    __getitem__: returns a data file
    
    """
    def __init__(self, data_dir):
        # Get all .npy files in the specified directory
        self.data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pt')]
        self.size = len(self.data_files)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data = torch.load(self.data_files[idx])
        return data
    
class MTN_Dataset(Dataset):
    """
    Memory to Network Dataset
    Loads data from memory to the neural network in the form of individual samples, handling shuffling across the two datasets and iteration bewteen them

    Args:
    large_data_bundles_size: int, the number of individual files in each bundle
    large_batch_size: int, the number of bundles to load into memory at once
    shuffle_data: bool, whether to shuffle the data across both large and small batches

    Methods:
    __len__: returns the number of individual samples in the entire dataset of bundles

    __getitem__: returns a single sample from memory
    """

    def __init__(self, dataset_path_ondisk, large_data_bundles_size, large_batch_size, shuffle_data=False):
        self.large_dataset = DTM_Dataset(dataset_path_ondisk)
        self.large_dataloader = DataLoader(self.large_dataset, large_batch_size, shuffle_data)
        self.large_data_bundles_size = large_data_bundles_size
        self.large_batch_size = large_batch_size
        self.indices = list(range(self.large_data_bundles_size * self.large_batch_size))
        self.shuffle_data = shuffle_data


    def __len__(self):
        return len(self.large_dataset) * self.large_data_bundles_size
    
    def __getitem__(self, idx):
        if idx == 0:
            self.large_dataloader_iterator = iter(self.large_dataloader) # Restarts the large/DTM dataloader iterator

        if idx % (self.large_data_bundles_size * self.large_batch_size) == 0: # If the index is a multiple of the bundle size then load a new bundle of data into memory
            
            #######TRYIN GTO OVERCOME THE HUGE BUMP IN MEMORY WHEN LOADING THE NEW SET? SEEMS LIKE ITS LOADING BEFORE IT CLEARS THE OLD SOMHOW?
            self.data = None # Clear the memory of the old data
            
            # Load a new bundle of data
            self.data = next(self.large_dataloader_iterator)

            # Shuffling across the bundle, without loosing the ability to use logic based on index
            self.shuffled_indices = self.indices.copy()
            if self.shuffle_data:
                random.shuffle(self.shuffled_indices)

        internal_bundle_idx = self.shuffled_indices[idx % (self.large_data_bundles_size * self.large_batch_size)] # Get the index of the data in the bundle

        return self.data[internal_bundle_idx // self.large_data_bundles_size, internal_bundle_idx % self.large_data_bundles_size] 
    
class StackedDatasetLoader(DataLoader):
    """
    Very simple datalaoder with one additional function that perfroms a preprocessing step on the data before returning it to the neural network

    """

    def __init__(self, input_signal_settings, physical_scale_parameters, time_dimension, device, preprocess_on_gpu, precision, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_signal_settings = input_signal_settings
        self.physical_scale_parameters = physical_scale_parameters
        self.time_dimension = time_dimension
        self.preprocess_on_gpu = preprocess_on_gpu
                
        if self.preprocess_on_gpu:                                      # If the user wants to preprocess the data on the GPU
            self.device = device                                        # Set the device to the global processing device, i.e the GPU if one is availble, falling back to CPU if not
        else:
            self.device = torch.device("cpu")                           # If the user does not want to preprocess the data on the GPU, set the device to the CPU for the data pre-processing

        if precision == 16:                                             # Set the precision of the data to be loaded into the neural network
            self.dtype = torch.float16
        elif precision == 32:
            self.dtype = torch.float32
        elif precision == 64:
            self.dtype = torch.float64

    def _custom_processing(self, batch):
        signal_points, x_std_dev, y_std_dev, tof_std_dev, noise_points = self.input_signal_settings  # move directly into next line without breaking out?
        signal_settings = input_range_to_random_value(signal_points, x_std_dev, y_std_dev, tof_std_dev, noise_points) 
        degraded_batches = signal_degredation(signal_settings, batch, self.physical_scale_parameters, self.time_dimension, self.device)
        return degraded_batches   

    def __iter__(self):
        for batch in super().__iter__():
            batch = batch.to(self.dtype)
            if self.preprocess_on_gpu:
                batch = batch.to(self.device)
            degraded_data = self._custom_processing(batch)
            sparse_output_batch, sparse_and_resolution_limited_batch, noised_sparse_reslimited_batch = degraded_data
            yield batch, sparse_output_batch, sparse_and_resolution_limited_batch, noised_sparse_reslimited_batch



# Test the dataloader
if __name__ == "__main__":
    precision = 32
    time_dimension = 1000
    physical_scale_parameters = [0.1, 0.1, 0.1]
    signal_points = 3000 
    noise_points = 2000
    x_std_dev = .5
    y_std_dev = .5
    tof_std_dev = .5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_signal_settings = [signal_points, x_std_dev, y_std_dev, tof_std_dev, noise_points]



    dataset_path_ondisk = r'N:\Yr 3 Project Datasets\[V2]RDT 50KM Fix\Data\\'    # Path to the directory where the data files are stored on disk in bundles
    large_data_bundles_size = 1000     # the number of individual files in each bundle
    number_of_bundles_to_memory = 1 # Essentially the 'Large batch Size' i.e how many bundles to load into memory at once [Must be less than or equal to the number of bundles in the dataset on disk]
    small_batch_size = 100            # Batch Size for loading into the neural net from the in memory bundles               
    shuffle_data = False             # Shuffle the data across both large and small batches



    small_dataset = MTN_Dataset(dataset_path_ondisk, large_data_bundles_size, number_of_bundles_to_memory, shuffle_data)
    small_dataloader = StackedDatasetLoader(input_signal_settings, physical_scale_parameters, time_dimension, device, dataset=small_dataset, batch_size=small_batch_size) # Shuffle is handled by the dataset, set with the variable 'shuffle_data'. It must not be applied here otherwise will induce errors in the logic based on index value

    for idx, data in enumerate(small_dataloader):
        batch, sparse_output_batch, sparse_and_resolution_limited_batch, noised_sparse_reslimited_batch = data
        
        plt.imshow(batch[0][0].cpu().numpy())
        plt.show()
        plt.imshow(sparse_output_batch[0][0].cpu().numpy())
        plt.show()
        plt.imshow(sparse_and_resolution_limited_batch[0][0].cpu().numpy())
        plt.show()
        plt.imshow(noised_sparse_reslimited_batch[0][0].cpu().numpy())
        plt.show()

        print(f'Batch {idx}')

        if idx >= 1:
            break