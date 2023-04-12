#%% - Dependencies
# External Libraries
import numpy as np 
import matplotlib.pyplot as plt
import torch
import os


pretrained_model_path = "N:/Yr 3 Project Results/D24 10K lr0001 weighted_loss1-1point2 - Training Results/D24 10K lr0001 weighted_loss1-1point2 - Model + Optimiser State Dicts.pth"

input_image_path = "N:\Yr 3 Project Datasets\Dataset 24_X10ks\Data"
reconstruction_threshold = 0.5
time_dimension = 100
latent_dim = 10

# Load one random .npy file from the input directory
input_image_path = os.path.join(input_image_path, np.random.choice(os.listdir(input_image_path)))

# Load a specific file in the input directory
#specificfilename = ""
#input_image_path = os.path.join(input_image_path, specificfilename + ".npy")

### Load image from path 
input_image = np.load(input_image_path)

# Turn input image into tensor and add two extra dimesnions to start of array so shape goes from (x,y) to (1,1,x,y) to represent batch and channel dims
input_image = torch.tensor(input_image)
input_image.double()

#%% - functions
def plotting(new_arr):   #OLD - single image at a time
    # Assume new_arr is your 3D array of size n by m by t_max
    n, m, t_max = new_arr.shape

    # Create a meshgrid of x, y, and z values for the 3D plot
    x, y, z = np.meshgrid(np.arange(m), np.arange(n), np.arange(t_max))

    # Flatten the x, y, and z values and new_arr for plotting
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    new_arr = new_arr.flatten()

    # Plot the 3D scatter plot with the non-zero values in the array set to 1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[new_arr == 1], y[new_arr == 1], z[new_arr == 1], c='r', marker='o', s=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('t')
    plt.show()

def custom_normalisation(data, reconstruction_threshold, time_dimension=100):
    data = ((data / time_dimension) / (1/(1-reconstruction_threshold))) + reconstruction_threshold
    for row in data:   ###REPLACE USING NP.WHERE
        for i, ipt in enumerate(row):
            if ipt == reconstruction_threshold:
                row[i] = 0
    return data

def custom_renormalisation(data, reconstruction_threshold, time_dimension=100):
    data = np.where(data > reconstruction_threshold, ((data - reconstruction_threshold)*(1/(1-reconstruction_threshold)))*(time_dimension), 0)
    return data
#%% - Import and prepare Autoencoder model
from DC3D_V3.Autoencoders.DC3D_Autoencoder_V1 import Encoder, Decoder
encoder = Encoder(encoded_space_dim=latent_dim, fc2_input_dim=128, encoder_debug=False, record_activity=False)
decoder = Decoder(encoded_space_dim=latent_dim, fc2_input_dim=128, decoder_debug=False, record_activity=False)
encoder.double()   
decoder.double()

# load the full state dictionary into memory
full_state_dict = torch.load(pretrained_model_path)

# load the state dictionaries into the models
encoder.load_state_dict(full_state_dict['encoder_state_dict'])
decoder.load_state_dict(full_state_dict['decoder_state_dict'])

encoder.eval()                                   #.eval() is a kind of switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time. For example, Dropouts Layers, BatchNorm Layers etc. You need to turn off them during model evaluation, and .eval() will do it for you. In addition, the common practice for evaluating/validation is using torch.no_grad() in pair with model.eval() to turn off gradients computation
decoder.eval()    

#Following function runs the autoencoder on the input data
def deepclean3(input_image, reconstruction_threshold, time_dimension=100):
    with torch.no_grad():
        norm_image = custom_normalisation(input_image, reconstruction_threshold, time_dimension)
        image_prepared = norm_image.unsqueeze(0).unsqueeze(0)   #Adds two extra dimesnions to start of array so shape goes from (x,y) to (1,1,x,y) to represent batch and channel dims
        rec_image = decoder(encoder(image_prepared))                         #Creates a recovered image (denoised image), by running a noisy image through the encoder and then the output of that through the decoder.
        rec = rec_image.squeeze().numpy()
        print(type(rec))
        rec_image_renorm = custom_renormalisation(rec, reconstruction_threshold, time_dimension)
    return rec_image_renorm


#%% - Driver
recovered_image = deepclean3(input_image, reconstruction_threshold, time_dimension=100)

# Plot the input image and the recovered image beside each other on a mpl 2d plot   
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(input_image)
ax2.imshow(recovered_image)
plt.show()

# Plot the input image and the recovered image beside each other on a mpl 3d plot
plotting(input_image, time_dimension)
plotting(recovered_image, time_dimension)
