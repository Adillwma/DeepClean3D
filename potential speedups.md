#




# Streamline dataloader and dataset

# Load all data into system memory




# Load all data into GPU memory 

# set num workers for dataloader

# async data preprocessing


# use pinned memory
If you load your samples in the Dataset on CPU and would like to push it during training to the GPU, you can speed up the host to device transfer by enabling pin_memory.
This lets your DataLoader allocate the samples in page-locked memory, which speeds-up the transfer.
You can find more information on the NVIDIA blog

i.e. : 

    train_loader = torch.utils.data.DataLoader(dataset,  batch_size, pin_memory=True)

# Graph capture using torch.compile to enable JIT
https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/pytorch_2_intro.ipynb 


# Incoperate AMP (Automatic Mixed Precision) into the pipeline

# Enable running nativly at half precision

