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





Freezing Parameters: When you are fine-tuning a pre-trained model and want to freeze certain layers, you can detach the parameters of those layers from the computation graph. This prevents them from being updated during backpropagation. You can detach the parameters by calling the detach() method on the parameter itself, or by using the detach() method on the Module.

    for param in model.parameters():
        param.requires_grad = False

    # Replace the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
    model.fc = nn.Linear(512, 100)

    # Optimize only the classifier
    optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)