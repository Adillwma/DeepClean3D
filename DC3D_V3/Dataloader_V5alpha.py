"""
DC3D Dataloader V5
Created by Adill Al-Ashgar 

Custom Dataset and DataLoader classes for the DC3D_V3 project. 
Used to load data from disk and apply custom data degradation functions to the data in the loader. 
Designed for new dataset structure with data stored in bundles of 1000 or 10000 on disk rather than individual files to minimise i/o opperations.
Features tiered memory loading to load bundles to memory and then batches to the netwrok from the memory loaded bundles.
Streamlined Memory to Network Dataloader with custom iterator to call an enitre batch using slicing in one call rather than an individual call for every batch item.
Asynchronous bundle loading to disk to memory to reduce the time spent waiting for data to load and unblock the loading process. 
"""

import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import torch
import os
from tqdm import tqdm
from Helper_files.Data_Degradation_Functions import *
from Helper_files.Helper_Functions import input_range_to_random_value
from Helper_files.ExecutionTimer import Execution_Timer
import matplotlib.pyplot as plt
from sys import getsizeof as sys_getsizeof
import time
import asyncio
import threading
import queue

class AsyncDataLoader:
    def __init__(self, file_paths, timer, device, precision, maxsize_data_queue=1, maxsize_async_queue=1):
        self.timer = timer
        self.file_paths = file_paths    
        self.data_queue = queue.Queue(maxsize=maxsize_data_queue)
        self.async_queue = asyncio.Queue(maxsize=maxsize_async_queue)
        self.loop = asyncio.new_event_loop()
        self.device = device
        self.dtype = {16: torch.float16, 32: torch.float32, 64: torch.float64}.get(precision)

        # Start the asyncio event loop in a new thread
        t = threading.Thread(target=self.start_event_loop, args=(self.loop,), daemon=True)
        t.start()
        
        # Start the transfer data coroutine
        asyncio.run_coroutine_threadsafe(self.transfer_data(), self.loop)
        
        # Start async data loading tasks
        asyncio.run_coroutine_threadsafe(self.load_data(), self.loop)
        
    def start_event_loop(self, loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()
    
    async def load_data(self):
        while True:
            for file_name in self.file_paths:
                # Load the data from disk
                self.timer.record_time(event_name="Load from Disk", event_type="start")
                sample = torch.load(file_name, map_location=self.device)
                self.timer.record_time(event_name="Load from Disk", event_type="stop")
                
                # Convert to the correct precision
                self.timer.record_time(event_name="Async Dtype Conversion", event_type="start")
                sample = sample.to(self.dtype)
                self.timer.record_time(event_name="Async Dtype Conversion", event_type="stop")

                # Add the data into the async queue
                await self.async_queue.put([sample, file_name])
    
    async def transfer_data(self):
        while True:
            data = await self.async_queue.get()
            #self.timer.record_time(event_name="Load to main que", event_type="start")
            self.data_queue.put(data)
            #self.timer.record_time(event_name="Load to main que", event_type="stop")
            self.async_queue.task_done()

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

    def __init__(self, dataset_path_ondisk, large_data_bundles_size, large_batch_size, device, shuffle_data=False, preprocess_on_gpu=True, precision=64, timer=None):
        self.file_paths = [os.path.join(dataset_path_ondisk, f) for f in os.listdir(dataset_path_ondisk) if f.endswith('.pt')]
        self.large_data_bundles_size = large_data_bundles_size
        self.large_batch_size = large_batch_size
        self.indices = list(range(self.large_data_bundles_size * self.large_batch_size))
        self.shuffle_data = shuffle_data
        self.data = None
        self.data_buffer = None
        self.execution_timer = timer
        self.preprocess_on_gpu = preprocess_on_gpu
        self.dtype = {16: torch.float16, 32: torch.float32, 64: torch.float64}.get(precision)
        if self.preprocess_on_gpu:                                     # If the user wants to preprocess the data on the GPU
            self.device = device                                       # Set the device to the global processing device, i.e the GPU if one is availble, falling back to CPU if not

        self.loader = AsyncDataLoader(self.file_paths, self.execution_timer, self.device, precision)

    def __len__(self):
        return len(self.file_paths) * self.large_data_bundles_size
    
    def call_for_bundles(self):
        self.execution_timer.record_time(event_name=f"Waiting on async data load", event_type="start")
        data_loading = True
        while data_loading:
            try:
                self.buffer_data, filename_report = self.loader.data_queue.get(timeout=0.1)  # Adjust timeout for data loading
                #print(f"Data loaded from: {filename_report}")
                data_loading = False
                self.execution_timer.record_time(event_name=f"Waiting on async data load", event_type="stop")
                self.loader.data_queue.task_done()
            except queue.Empty:
                # If the queue is empty, check if the background loading is done
                if not self.loader.loop.is_running():
                    break

        return self.buffer_data
    
    def split_list(self, list):
        result = []
        sublist = []
        for num in list:
            if num == 0 or num % (self.large_data_bundles_size * self.large_batch_size) == 0:
                if sublist:  # Only append if sublist is not empty
                    result.append(sublist)
                sublist = []  # Start a new sublist
                sublist.append(num)
            else:
                sublist.append(num)
        
        if sublist:  # Append the last sublist if not empty
            result.append(sublist)
        
        return result

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
    
        # If a batch of indices is provided
        if isinstance(idx, list):

            # split the indicides list into a list of lists, by splitting the list every time the index is a multiple of the bundle size or a 0
            split_indices = self.split_list(idx)

            semi_batchs = []

            for split_index, idx_list in enumerate(split_indices):                
                if idx_list[0] % (self.large_data_bundles_size * self.large_batch_size) == 0: # If the index is a multiple of the bundle size then load a new bundle of data into memory
                    self.execution_timer.record_time(event_name="Disk to Memory Load", event_type="start")

                    # Load data from the buffer 
                    self.data = self.call_for_bundles()                    

                    # Shuffling across the bundle, without loosing the ability to use logic based on index
                    self.shuffled_indices = self.indices.copy()
                    if self.shuffle_data:
                        random.shuffle(self.shuffled_indices)

                    self.execution_timer.record_time(event_name="Disk to Memory Load", event_type="stop")

                self.execution_timer.record_time(event_name="Memory to Network Load", event_type="start")

                idx_array = np.array(idx_list)

                min_internal_bundle_idx = self.shuffled_indices[idx_array[0] % (self.large_data_bundles_size * self.large_batch_size)] # Get the index of the data in the bundle
                max_internal_bundle_idx = self.shuffled_indices[idx_array[-1] % (self.large_data_bundles_size * self.large_batch_size)] # Get the index of the data in the bundle

                semi_batch = self.data[min_internal_bundle_idx : max_internal_bundle_idx+1]
                semi_batchs.append(semi_batch)

                self.execution_timer.record_time(event_name="Memory to Network Load", event_type="stop")

            self.execution_timer.record_time(event_name="Memory to Network Cat", event_type="start")
            batch = torch.cat(semi_batchs, dim=0)
            self.execution_timer.record_time(event_name="Memory to Network Cat", event_type="stop")
            return batch
    
class StackedDatasetLoader(DataLoader):
    """
    Very simple datalaoder with one additional function that perfroms a preprocessing step on the data before returning it to the neural network

    """

    def __init__(self, input_signal_settings, physical_scale_parameters, time_dimension, device, preprocess_on_gpu, precision, timer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_signal_settings = input_signal_settings
        self.physical_scale_parameters = physical_scale_parameters
        self.time_dimension = time_dimension
        self.preprocess_on_gpu = preprocess_on_gpu
        self.execution_timer = timer
                
        if self.preprocess_on_gpu:                                      # If the user wants to preprocess the data on the GPU
            self.device = device                                        # Set the device to the global processing device, i.e the GPU if one is availble, falling back to CPU if not
        else:
            self.device = torch.device("cpu")                           # If the user does not want to preprocess the data on the GPU, set the device to the CPU for the data pre-processing

    def _custom_processing(self, batch):
        signal_points, x_std_dev, y_std_dev, tof_std_dev, noise_points = self.input_signal_settings  # move directly into next line without breaking out?
        signal_settings = input_range_to_random_value(signal_points, x_std_dev, y_std_dev, tof_std_dev, noise_points) 
        degraded_batches = signal_degredation(signal_settings, batch, self.physical_scale_parameters, self.time_dimension, self.device)
        return degraded_batches   

    def __iter__(self):
        self.sample_iter = iter(self.batch_sampler)
        return self

    def __next__(self):
        batch_indices = next(self.sample_iter)
        batch = self.dataset[batch_indices]
        self.execution_timer.record_time(event_name="Data Degradation", event_type="start")
        degraded_data = self._custom_processing(batch)
        sparse_output_batch, sparse_and_resolution_limited_batch, noised_sparse_reslimited_batch = degraded_data
        self.execution_timer.record_time(event_name="Data Degradation", event_type="stop") 

        return batch, sparse_output_batch, sparse_and_resolution_limited_batch, noised_sparse_reslimited_batch

# Test the dataloader
if __name__ == "__main__":
    time_dimension = 1000
    physical_scale_parameters = [0.1, 0.1, 0.1]
    signal_points = 3000 
    noise_points = 0
    x_std_dev = 0
    y_std_dev = 0
    tof_std_dev = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_signal_settings = [signal_points, x_std_dev, y_std_dev, tof_std_dev, noise_points]

    dataset_path_ondisk = r'N:\Yr 3 Project Datasets\[V3]_RDT_50K\Data\\'    # Path to the directory where the data files are stored on disk in bundles
    large_data_bundles_size = 10000     # the number of individual files in each bundle
    number_of_bundles_to_memory = 1 # Essentially the 'Large batch Size' i.e how many bundles to load into memory at once [Must be less than or equal to the number of bundles in the dataset on disk]
    small_batch_size = 5000            # Batch Size for loading into the neural net from the in memory bundles               
    shuffle_data = False             # Shuffle the data across both large and small batches
    preprocess_on_gpu = True        # Preprocess the data on the GPU if available, otherwise on the CPU
    precision = 64                   # The precision of the data to be loaded into the neural network

    execution_timer = Execution_Timer()

    # Create an instance of AsyncDataLoader

    small_dataset = MTN_Dataset(dataset_path_ondisk, large_data_bundles_size, number_of_bundles_to_memory, device, shuffle_data, preprocess_on_gpu, precision, timer=execution_timer)
    
    batch_sampler = torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(small_dataset), 
                                                  batch_size=small_batch_size, 
                                                  drop_last=False)

    small_dataloader = StackedDatasetLoader(input_signal_settings, physical_scale_parameters, time_dimension, device,  preprocess_on_gpu, precision, dataset=small_dataset, timer=execution_timer, batch_sampler=batch_sampler) # Shuffle is handled by the dataset, set with the variable 'shuffle_data'. It must not be applied here otherwise will induce errors in the logic based on index value

    for epoch in range(5):
        for idx, data in tqdm(enumerate(small_dataloader)):
            execution_timer.record_time(event_name="Small Batch Presented", event_type="start")
            batch, sparse_output_batch, sparse_and_resolution_limited_batch, noised_sparse_reslimited_batch = data
            execution_timer.record_time(event_name="Small Batch Presented", event_type="stop")
            #print(f'Batch dtype: {batch.dtype}, Sparse Output Batch dtype: {sparse_output_batch.dtype}, Sparse and Resolution Limited Batch dtype: {sparse_and_resolution_limited_batch.dtype}, Noised Sparse and Resolution Limited Batch dtype: {noised_sparse_reslimited_batch.dtype}')
            #if idx >= 30:
            #    break
    print('Done...')

    fig = execution_timer.return_plot(dark_mode=True)
    plt.show()

    #execution_timer.return_times()

    #data = execution_timer.return_data()
    #print(data)

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
        




