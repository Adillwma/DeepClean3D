# -*- coding: utf-8 -*-
"""
Created on Sat Oct 8 2022
DeepClean 2D v0.0.1
@author: Adill Al-Ashgar
"""
#%% - Dependencies
import numpy as np 
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import random 

from Custom_Normalisation_V1 import scale_Ndata as custom_normalisation
#from Custom_Normalisation_V1 import custom_normalisation
from DataLoader_Functions_V2 import initialise_data_loader
from autoencoders.autoencoder_2D_V1 import Encoder, Decoder



def find_np_minmax(data):
    print("NP SHAPE", np.shape(data))
    return(data)


def tensor_datascan(data):
    #tensor_min = np.amin(data.numpy())
    #tensor_max = np.amax(data.numpy())
    #print("MIN", tensor_min, "MAX", tensor_max)
    print("SHAPE tensor: ", np.shape(data.numpy()))
    datascan = data.numpy().flatten()

    data_hit_tensor = datascan[datascan != 0]
    print("TENSOR SCAN HITS",len(data_hit_tensor))
    print(data_hit_tensor)
    plt.hist(datascan,20)
    plt.show()
    return(data)

def numpy_datascan(data):

    #tensor_min = np.amin(data.numpy())
    #tensor_max = np.amax(data.numpy())
    #print("MIN", tensor_min, "MAX", tensor_max)
    print("SHAPE numpy: ", np.shape(data))
    datascan = data.flatten()

    data_hit_numpy = datascan[datascan != 0]
    print("NUMPY SCAN HITS",len(data_hit_numpy))
    print(data_hit_numpy)
    plt.hist(datascan,20)
    plt.show()
    return(data)

#%% - User Inputs
learning_rate = 0.001                       #User controll to set optimiser learning rate(Hyperparameter)
optim_w_decay = 1e-05                       #User controll to set optimiser weight decay (Hyperparameter)
loss_function = torch.nn.MSELoss()          #User controll to set loss function (Hyperparameter)
latent_space_nodes = 5                      #User controll to set number of nodes in the latent space, the bottleneck layer (Hyperparameter)
noise_factor = 0                            #User controll to set the noise factor, a multiplier for the magnitude of noise added. 0 means no noise added, 1 is defualt level of noise added, 10 is 10x default level added (Hyperparameter)
num_epochs = 4                            #User controll to set number of epochs (Hyperparameter)
batch_size = 10                             #Data Loader, number of Images to pull per batch (add a check to make sure the batch size is smaller than the total number of images in the path selected)
reconstruction_threshold = 0.2              #Threshold for 3d reconstruction, values below this confidence level are discounted
seed = 0                                    #0 is default which gives no seeeding to RNG, if the value is not zero then this is used for the RNG seeding for numpy, random, and torch libraries

#%% - Program Settings
print_partial_training_losses = 1           #[default = 1]
print_encoder_debug = 1                     #[default = 1]
print_decoder_debug = 1                     #[default = 1]
debug_noise_function = 0                    #[default = 1]
print_epochs = 2                            #[default = 1] prints every other 'print_epochs' i.e if set to two then at end of every other epoch it will print a test on results
plot_or_save = 0                            #[default = 0] 0 is normal behavior, If set to 1 then saves all end of epoch printouts to disk, if set to 2 then saves outputs whilst also printing for user
outputfig_title = "Test"                    #Must be string, value is used in the titling of the output plots if plot_or_save is selected above
telemetry_on = 1                            #[default = 1]

#%% Dataloading
# - Data Loader User Inputs
dataset_title = "Dataset 1_Realistic"
data_path = "C:/Users/Student/Documents/UNI/Onedrive - University of Bristol/Yr 3 Project/Circular and Spherical Dummy Datasets/" #"C:/Users/Student/Desktop/fake im data/"  #"/local/path/to/the/images/"
time_dimension = 100

# - Advanced Data Loader Settings
debug_loader_batch = 0     #(Default = 0 = [OFF]) //INPUT 0 or 1//   #Setting debug loader batch will print to user the images taken in by the dataoader in this current batch and print the corresponding labels
plot_every_other = 1       #(Default = 1) //MUST BE INTEGER INPUT//  #If debug loader batch is enabled this sets the interval for printing for user, 1 is every single img in the batch, 2 is every other img, 5 is every 5th image etc 
batch_size_protection = 1  #(Default = 1 = [ON]) //INPUT 0 or 1//    #WARNING if turned off, debugging print will cause and exeption due to the index growing too large in the printing loop (img = train_features[i])

# - Data Loader Preparation Transforms 
#####For info on all transforms check out: https://pytorch.org/vision/0.9/transforms.html
train_transforms = transforms.Compose([#transforms.Lambda(numpy_datascan),
                                       transforms.Lambda(custom_normalisation),
                                       #transforms.Lambda(numpy_datascan),
                                       #transforms.Lambda(find_np_minmax),
                                       transforms.ToTensor(),
                                       #transforms.Lambda(custom_normalisation), #transforms.Lambda(function) allows creation of custom transform from a function 
                                       #transforms.Lambda(tensor_datascan),
                                       #transforms.Normalize(),
                                       #transforms.RandomRotation(30),         #Compose is required to chain together multiple transforms in serial 
                                       #transforms.RandomResizedCrop(224),
                                       #transforms.RandomHorizontalFlip(),
                                       #transforms.ToTensor()               #other transforms can be dissabled but to tensor must be left enabled
                                       ]) 

test_transforms = transforms.Compose([#transforms.Lambda(numpy_datascan),
                                      transforms.Lambda(custom_normalisation),
                                      #transforms.Lambda(numpy_datascan),
                                      #transforms.Lambda(find_np_minmax),
                                      #transforms.Resize(255),
                                      #transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      #transforms.Lambda(custom_normalisation),
                                      #transforms.Lambda(tensor_datascan),
 
                                      #transforms.Normalize()
                                      ])

# - Initialise Data Loader
train_loader, test_loader, val_loader, train_dataset, test_dataset, val_datset = initialise_data_loader(dataset_title, data_path, batch_size, train_transforms, test_transforms, debug_loader_batch, plot_every_other, batch_size_protection)

#%% - Classes
### Gaussian Noise Generator Class
class AddGaussianNoise(object):                   #Class generates noise with mean 0 and std deviation 1, (gaussian)
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

#%% - Functions
### Random Noise Generator Function
def add_noise(inputs,noise_factor=0.3, time_dimension=100):
     cNOISE = torch.randn_like(inputs) #* time_dimension
     noise_init = torch.randn_like(inputs)**2 * time_dimension
     noise = torch.clip(cNOISE,0.,100.)
     noisy = inputs# + noise
     if debug_noise_function == 1:
        print("INPUT", torch.min(inputs), torch.max(inputs))
        plt.imshow(inputs[0][0])
        plt.show()
        print("cNOISE", torch.min(cNOISE), torch.max(cNOISE))
        plt.imshow(cNOISE[0][0])
        plt.show()
        print("noise", torch.min(noise), torch.max(noise))
        plt.imshow(noise[0][0])
        plt.show()
        print("noisy", torch.min(noisy), torch.max(noisy))
        plt.imshow(noisy[0][0])
        plt.show()
     return noisy

### Random Noise Generator Function
def add_noise2(inputs, noise_points=0, dimensions=(88,128), time_dimension=100, debug_noise_function=0):
    print("NOISETEST2",np.shape(inputs.numpy()))
    for noise_point in range (0, noise_points+1):
        np_x = np.random.randint(0, dimensions[0]) 
        np_y = np.random.randint(0, dimensions[1]) 
        np_TOF = np.random.randint(0, time_dimension) 
        inputs[np_x][np_y] = np_TOF
     
    if debug_noise_function == 1:
        plt.imshow(inputs[0][0])
        plt.show()
    return inputs




def redimensionalise_time(data, reconstruction_threshold):
    #Reconstruct 3D Image
    x_list = []
    y_list = []
    TOF_list = []
    #print("re-time in: min{}, max{}".format(0, (np.amax(data))))
    for row, row_data in enumerate(data):
        for column, TOF in enumerate(row_data):
            #print("R,C",row,column)
            if TOF != 0 and TOF >= reconstruction_threshold:        #the TOF != 0 is important if user sets a reconstruction threshold of 0
                #Reverse Normalisation
                x_list.append(row)
                y_list.append(column)
                TOF_denorm = (TOF - reconstruction_threshold) * 2 * time_dimension   #the value of 2 needs to be updated to programatic to really be able to modify reconsturction threshold without breaking program 
                TOF_list.append(TOF_denorm)
                #print("TOF",TOF,"denorn",TOF_denorm)
    #print("re-time out: min{}, max{}".format(0, (np.amax(TOF_list))))
    #plt.hist(data, 100)
    #plt.show()
    #plt.hist(TOF_list, 100)
    #plt.show()
    #print(TOF_list)
    return (x_list, y_list, TOF_list)

#3D Reconstruction
def reconstruction_3D(image, time_dimension, reconstruction_threshold, settings, epoch, plot_or_save=0):
    data_hit_numpy = image[image != 0]
    print("IMAGE IN SCAN HITS",len(data_hit_numpy))
    print(data_hit_numpy)
#Remember image comes in in the form y,x not x,y so column and row are flipped in indexing
    #2D Plot Check
    plt.imshow(image)
    plt.title("2D Reconstruction Epoch: %s" %epoch)
    if plot_or_save == 0:
        plt.show()
    else:
        plt.close()
    
    shape = np.shape(image)
    x_list, y_list, z_list = redimensionalise_time(image, reconstruction_threshold)

    #3D Plot
    fig = plt.figure()               #Plots spherical data
    ax = plt.axes(projection='3d')
    ax.scatter(x_list, y_list, z_list)#, s = signal_hit_size, c = "b") #Plots spherical data in blue
    ax.set_xlim(0, shape[0])
    ax.set_ylim(0, shape[1])
    ax.set_zlim(0, time_dimension)
    ax.set_title("3D Reconstruction Epoch: %s" %epoch)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Time")
    if plot_or_save == 0:
        plt.show()
    else:
        Out_Label = 'Output_Graphics/{}, Reconstruction, Epoch {}, {} .png'.format(outputfig_title, epoch, settings) #!!!
        plt.savefig(Out_Label, format='png')        
        plt.close()

###Ploting confidence of each pixel as histogram per epoch with line showing the detection threshold
def belief_telemetry(data, reconstruction_threshold, epoch, settings, plot_or_save=0):
    data2 = data.flatten()

    #Plots histogram showing the confidence level of each pixel being a signal point
    values, bins, bars = plt.hist(data2, 10, histtype='bar')
    plt.axvline(x= reconstruction_threshold, color='red', marker='|', linestyle='dashed', linewidth=2, markersize=12)
    plt.title("Epoch %s" %epoch)
    plt.bar_label(bars, fontsize=10, color='navy') 
    if plot_or_save == 0:
        plt.show()
    else:
        Out_Label = 'Output_Graphics/{}, Confidence Histogram, Epoch {}, {} .png'.format(outputfig_title, epoch, settings) #!!!
        plt.savefig(Out_Label, format='png')        
        plt.close()

    above_threshold = (data2 >= reconstruction_threshold).sum()
    below_threshold = (data2 < reconstruction_threshold).sum()
    return (above_threshold, below_threshold)

def plot_telemetry(telemetry):
    tele = np.array(telemetry)
    #!!! Add labels to lines
    plt.plot(tele[:,0],tele[:,1], color='r', label="Points above threshold") #red = num of points above threshold
    plt.plot(tele[:,0],tele[:,2], color='b', label="Points below threshold") #blue = num of points below threshold
    plt.title("Telemetry over epochs")
    plt.xlabel("Epoch number")
    plt.ylabel("Number of Signal Points")
    plt.legend()
    plt.show()    

###RNG Seeding for Determinism Function
def Determinism_Seeding(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

### Training Function
def train_epoch_den(encoder, decoder, device, dataloader, loss_fn, optimizer,noise_factor=0.3, print_partial_training_losses=print_partial_training_losses):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _ in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        image_noisy = add_noise(image_batch,noise_factor)
        image_batch = image_batch.to(device)
        image_noisy = image_noisy.to(device)    
        # Encode data
        encoded_data = encoder(image_noisy)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if print_partial_training_losses == 1:
            # Print batch loss
            print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())
    return np.mean(train_loss)

### Testing Function
def test_epoch_den(encoder, decoder, device, dataloader, loss_fn,noise_factor=0.3):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            image_noisy = add_noise(image_batch,noise_factor)
            image_noisy = image_noisy.to(device)
            # Encode data
            encoded_data = encoder(image_noisy)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label) 
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data

###Plotting Function
def plot_ae_outputs_den(encoder, decoder, epoch, outputfig_title, time_dimension, reconstruction_threshold, plot_or_save=0, n=10,noise_factor=0.5):       #Defines a function for plotting the output of the autoencoder. And also the input + clean training data? Function takes inputs, 'encoder' and 'decoder' which are expected to be classes (defining the encode and decode nets), 'n' which is the number of ?????Images in the batch????, and 'noise_factor' which is a multiplier for the magnitude of the added noise allowing it to be scaled in intensity.  
    
    #Initialise lists for true and recovered signal point values 
    number_of_true_signal_points = []
    number_of_recovered_signal_points = []
    
    #Start Plotting Results
    plt.figure(figsize=(16,4.5))                                      #Sets the figure size

    for i in range(n):                                                #Runs for loop where 'i' itterates over 'n' total values which range from 0 to n-1
      
      ax = plt.subplot(3,n,i+1)                                      #Creates a number of subplots for the 'Original images??????' i.e the labels. the position of the subplot is i+1 as it falls in the first row
      img = test_dataset[i][0].unsqueeze(0)

      #Determine the number of signal points on the input image (have to change this to take it directly from the embeded val in the datsaset as when addig noise this method will break)   
      int_sig_points = (img >= reconstruction_threshold).sum()
      number_of_true_signal_points.append(int(int_sig_points.numpy()))

      #Following section creates the noised image data drom the original clean labels (images)   
      image_noisy = add_noise(img,noise_factor)     
      image_noisy = image_noisy.to(device)
      
      #Following section sets the autoencoder to evaluation mode rather than training (up till line 'with torch.no_grad()')
      encoder.eval()                                   #.eval() is a kind of switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time. For example, Dropouts Layers, BatchNorm Layers etc. You need to turn off them during model evaluation, and .eval() will do it for you. In addition, the common practice for evaluating/validation is using torch.no_grad() in pair with model.eval() to turn off gradients computation
      decoder.eval()                                   #Simarlary as above
      
      with torch.no_grad():                                               #As mentioned in .eval() comment, the common practice for evaluating/validation is using torch.no_grad() which turns off gradients computation whilst evaluating the model (the opposite of training the model)     
      #Following line runs the autoencoder on the noised data
         rec_img  = decoder(encoder(image_noisy))                        #Creates a recovered image (denoised image), by running a noisy image through the encoder and then the output of that through the decoder.
      
      #Determine the number of signal points on the recovered image 
      int_rec_sig_points = (rec_img >= reconstruction_threshold).sum()      
      number_of_recovered_signal_points.append(int(int_rec_sig_points.numpy()))

      #Following section generates the img plots for the original(labels), noised, and denoised data)
      plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')           #plt.imshow plots an image. The arguments for imshow are, 'image data array' and cmap= which is the colour map. #.squeeze() acts on a tensor and returns a tensor, it removes all dimensions of the tensor that are of length 1, (A×1×B) becomes (AxB) where A and B are greater than 1 #.numpy() creates a numpy array from a tensor #!!! is the .cpu part becuase the code was not made to accept the gpu/cpu check i made????
      ax.get_xaxis().set_visible(False)                                   #Hides the x axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
      ax.get_yaxis().set_visible(False)                                   #Hides the y axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
      if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
        ax.set_title('EPOCH %s \nOriginal images' %(epoch+1))               #When above condition is reached, the plots title is set

      ax = plt.subplot(3, n, i + 1 + n)                                   #Creates a number of subplots for the 'Corrupted images??????' i.e the labels. the position of the subplot is i+1+n as it falls in the second row
      plt.imshow(image_noisy.cpu().squeeze().numpy(), cmap='gist_gray')   #plt.imshow plots an image. The arguments for imshow are, 'image data array' and cmap= which is the colour map. #.squeeze() acts on a tensor and returns a tensor, it removes all dimensions of the tensor that are of length 1, (A×1×B) becomes (AxB) where A and B are greater than 1 #.numpy() creates a numpy array from a tensor #!!! is the .cpu part becuase the code was not made to accept the gpu/cpu check i made????
      ax.get_xaxis().set_visible(False)                                   #Hides the x axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
      ax.get_yaxis().set_visible(False)                                   #Hides the y axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
      if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
        ax.set_title('Corrupted images')                                  #When above condition is reached, the plots title is set

      ax = plt.subplot(3, n, i + 1 + n + n)                               #Creates a number of subplots for the 'Reconstructed images??????' i.e the labels. the position of the subplot is i+1+n+n as it falls in the third row
      plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')       #plt.imshow plots an image. The arguments for imshow are, 'image data array' and cmap= which is the colour map. #.squeeze() acts on a tensor and returns a tensor, it removes all dimensions of the tensor that are of length 1, (A×1×B) becomes (AxB) where A and B are greater than 1 #.numpy() creates a numpy array from a tensor #!!! is the .cpu part becuase the code was not made to accept the gpu/cpu check i made????
      ax.get_xaxis().set_visible(False)                                   #Hides the x axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
      ax.get_yaxis().set_visible(False)                                   #Hides the y axis from showing in the plot as we are plotting images not graphs (we may want to retain axis?)
      if i == n//2:                                                       #n//2 divides n by 2 without any remainder, i.e 6//2=3 and 7//2=3. So this line checks to see if i is equal to half of n without remainder. it will be yes once in the loop. not sure of its use
         ax.set_title('Reconstructed images')                             #When above condition is reached, the plots title is set 
    
    plt.subplots_adjust(left=0.1,              #Adjusts the exact layout of the plots including whwite space round edges
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.1, 
                    hspace=0.3)     
    
    settings = "Settings = [ep {}][bs {}][lr {}][od {}][ls {}][nf {}][ds {}][sd {}]".format(num_epochs, batch_size, learning_rate, optim_w_decay, latent_space_nodes, noise_factor, dataset_title, seed)

    if plot_or_save == 0:
        if (epoch+1) % print_epochs == 0:
            plt.show()                                 #After entire loop is finished, the generated plot is printed to screen
        else:
            plt.close()

    elif plot_or_save == 1:
        Out_Label = 'Output_Graphics/{}, Epoch {}, {} .png'.format(outputfig_title, epoch+1, settings) #!!!
        plt.savefig(Out_Label, format='png')
        plt.close()
        print("\n# SAVED OUTPUT TEST IMAGE TO DISK #\n")    

    if (epoch+1) % print_epochs == 0:        
        ###3D Reconstruction
        in_data = img.cpu().squeeze().numpy()
        rec_data = rec_img.cpu().squeeze().numpy()
        reconstruction_3D(in_data, time_dimension, reconstruction_threshold, settings, epoch+1, plot_or_save)
        reconstruction_3D(rec_data, time_dimension, reconstruction_threshold, settings, epoch+1, plot_or_save)
        
        #Telemetry plots
        if telemetry_on == 1:       #needs ttitles and labels etc added
            above_threshold, below_threshold = belief_telemetry(rec_data, reconstruction_threshold, epoch+1, settings, plot_or_save)   
            telemetry.append([epoch, above_threshold, below_threshold])

    return(number_of_true_signal_points, number_of_recovered_signal_points)

#%% - Program Internal Setup

#Sets rng seeding for repoducability and debugging if user inputs a value for variable 'seeding' other than 0
if seed != 0: 
    Determinism_Seeding(seed)

#%% - Setup model, loss criteria and optimiser    
    
### Define the loss function
loss_fn = loss_function

### Define a learning rate for the optimiser
lr = learning_rate                                     #Just sets the learing rate value from the user inputs pannel at the top

### Set the random seed for reproducible results WHY IS THIS ON?????????????????????? TEST
#torch.manual_seed(seed)              

### Initialize the two networks
d = latent_space_nodes #!!!d is passed to the encoder & decoder in the lines below and represents the encoded space dimension. This is the number of layers the linear stages will shrink to? #!!!

#model = Autoencoder(encoded_space_dim=encoded_space_dim)
encoder = Encoder(encoded_space_dim=d,fc2_input_dim=128, encoder_debug=print_encoder_debug)
decoder = Decoder(encoded_space_dim=d,fc2_input_dim=128, decoder_debug=print_decoder_debug)
params_to_optimize = [{'params': encoder.parameters()} ,{'params': decoder.parameters()}] #Selects what to optimise, 

### Define an optimizer (both for the encoder and the decoder!)
wd = optim_w_decay                                                           #Just sets the weight decay value from the user inputs pannel at the top
optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=wd)

#%% - Compute device check
#Following section checks if a CUDA enabled GPU is available. If found it is selected as the 'device' to perform the tensor opperations. If no CUDA GPU is found the 'device' is set to CPU (much slower) 
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')  #Informs user if running on CPU or GPU - (NVIDIA CUDA)

#Following section moves both the encoder and the decoder to the selected device i.e detected CUDA enabled GPU or to CPU
encoder.to(device)   #Moves encoder to selected device, CPU/GPU
decoder.to(device)   #Moves decoder to selected device, CPU/GPU

#%% - Compute

history_da={'train_loss':[],'val_loss':[]}                   #Just creates a variable called history_da which contains two lists, 'train_loss' and 'val_loss' which are both empty to start with. value are latter appeneded to the two lists by way of history_da['val_loss'].append(x)

telemetry = [[0,0.5,0.5]]  #Initalises the telemetry memory, starting values are 0, 0.5, 0.5 which corrspond to epoch(0), above_threshold(0.5), below_threshold(0.5)

for epoch in range(num_epochs):                              #For loop that iterates over the number of epochs where 'epoch' takes the values (0) to (num_epochs - 1)
    print('EPOCH %d/%d' % (epoch + 1, num_epochs))
    ### Training (use the training function)
    train_loss=train_epoch_den(
                               encoder=encoder, 
                               decoder=decoder, 
                               device=device, 
                               dataloader=train_loader, 
                               loss_fn=loss_fn, 
                               optimizer=optim,
                               noise_factor=noise_factor)
    
    ### Validation (use the testing function)
    val_loss = test_epoch_den(
                              encoder=encoder, 
                              decoder=decoder, 
                              device=device, 
                              dataloader=val_loader, 
                              loss_fn=loss_fn,
                              noise_factor=noise_factor)
    
    # Print Validation_loss and plots at end of each epoch
    history_da['train_loss'].append(train_loss)
    history_da['val_loss'].append(val_loss)
    print('\nEND OF EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}\n'.format(epoch + 1, num_epochs,train_loss,val_loss))     #epoch +1 is to make up for the fact the range spans 0 to epoch-1 but we want to numerate things from 1 upwards for sanity
    number_of_true_signal_points, number_of_recovered_signal_points = plot_ae_outputs_den(encoder, decoder, epoch, outputfig_title,time_dimension, reconstruction_threshold, plot_or_save, n=10, noise_factor=noise_factor)


###Loss function plots
epochs_range = range(1,num_epochs+1)
plt.plot(epochs_range, history_da['train_loss']) 
plt.title("Training loss")   
plt.show()

plt.plot(epochs_range, history_da['train_loss']) 
plt.title("Validation loss") 
plt.show()

if telemetry_on == 1:
    plot_telemetry(telemetry)

#Comparison of true signal points to recovered signal points
print("True signal points",number_of_true_signal_points)
print("Recovered signal points: ",number_of_recovered_signal_points)
