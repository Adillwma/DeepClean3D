

# Quick start for Training
Using the settings guide below set the trainer settings to the desired values.
Then run the file DC3D_V3_Trainer 3.py

This section contains the inputs that the user should provide to the program. These inputs are as follows:

    dataset_title: a string representing the title of the dataset that the program will use.
    model_save_name: a string representing the name of the model that the program will save.
    time_dimension: an integer representing the number of time steps in the data.
    reconstruction_threshold: a float representing the threshold for 3D reconstruction; values below this confidence level are discounted. Must be in range 0 > reconstruction_threshold < 1


    ### Example Code

    # Import the Trainer
    from DC3D_V3 import DC3D_V3_Trainer

    # Set the trainer settings
    dataset_title = "my_dataset"
    model_save_name = "my_new_model"
    time_dimension = 128
    reconstruction_threshold = 0.5

    # Run the Trainer
    DC3D_V3_Trainer(dataset_title, model_save_name, time_dimension, reconstruction_threshold)

Training can be exited safely at any time after the initial epoch by pressing Ctrl + C. The model will be saved and can be loaded for deployment or to continue training at a later date.

# DC3D Trainer Settings Manual
If you wish to further customise the training process to fine tune to your specific data and use case then the following is a comprehensive list of the availible settings during training. 

### Hyperparameter Settings

This section contains the hyperparameters that the user can adjust to train the model. These hyperparameters are as follows:

    num_epochs: an integer representing the number of epochs for training.
    batch_size: an integer representing the number of images to pull per batch.
    latent_dim: an integer representing the number of nodes in the latent space, which is the bottleneck layer.
    learning_rate: a float representing the optimizer learning rate.
    optim_w_decay: a float representing the optimizer weight decay for regularization.
    dropout_prob: a float representing the dropout probability.

### Dataset Settings

This section contains the hyperparameters that control how the dataset is split. These hyperparameters are as follows:

    train_test_split_ratio: a float representing the ratio of the dataset to be used for training.
    val_test_split_ratio: a float representing the ratio of the non-training data to be used for validation as opposed to testing.

### Loss Function Settings

This section contains the hyperparameters that control the loss function used in training. These hyperparameters are as follows:

    loss_function_selection: an integer representing the selected loss function; see the program code for the list of options.
    zero_weighting: a float representing the zero weighting for ada_weighted_mse_loss.
    nonzero_weighting: a float representing the nonzero weighting for ada_weighted_mse_loss.
    zeros_loss_choice: an integer representing the selected loss function for zero values in ada_weighted_custom_split_loss.
    nonzero_loss_choice: an integer representing the selected loss function for nonzero values in ada_weighted_custom_split_loss.

### Preprocessing Settings

This section contains the hyperparameters that control image preprocessing. These hyperparameters are as follows:

    signal_points: an integer representing the number of signal points to add.
    noise_points: an integer representing the number of noise points to add.
    x_std_dev: a float representing the standard deviation of the detector's error in the x-axis.
    y_std_dev: a float representing the standard deviation of the detector's error in the y-axis.
    tof_std_dev: a float representing the standard deviation of the detector's error in the time of flight.

### Pretraining Settings

This section contains the hyperparameters that control pretraining. These hyperparameters are as follows:

    start_from_pretrained_model: a boolean representing whether to start from a pretrained model.
    load_pretrained_optimser: a boolean representing whether to load the pretrained optimizer.
    pretrained_model_path: a string representing the path to the saved full state dictionary for pretraining.

### Normalisation Settings

This section contains the hyperparameters that control normalization. These hyperparameters are as follows:

    simple_norm_instead_of_custom: a boolean representing whether to use simple normalization instead of custom normalization.
    all_norm_off: a boolean representing whether to use any input normalization.
    simple_renorm: a boolean representing whether to use simple output renormalization instead of custom output renormal
