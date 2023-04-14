# -*- coding: utf-8 -*-
"""
AE_Visulisations V2.0.0
Created: 20:56 13 Feb 2023
Author: Adill Al-Ashgar
University of Bristol

### Possible Improvements
# Fix GraphViz
# Enable tracking of differnce plot over training? 
"""

import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch
import pandas as pd
from tqdm import tqdm
from torchviz import make_dot
from PIL import Image
import plotly.express as px
from sklearn.manifold import TSNE

#%% - Differnce between images
####Would be cool to run this similarity function every plotting cycle in the training and then plot over time the number of mathching elemets to see if we ever get any perfect pixel reconstructions
def AE_visual_difference(image, noised_image, cleaned_image, print_text=False):
    """
    Calculates and plots the pixel-wise difference between three images and returns statistics.
    
    Parameters:
    image (ndarray): An image.
    noised_image (ndarray): Same image with noise added.
    cleaned_image (ndarray): The same image now denoised.
    print_text (bool): Whether to print text to console summarizing the differences between the images.
    
    Returns:
    num_diff_noised (Int): Number of different elements between `image` and `noised_image`
    num_same_noised (Int): Number of same elements between `image` and `noised_image`
    num_diff_cleaned (Int): the number of different elements between `image` and `cleaned_image`
    num_same_cleaned (Int): the number of same elements between `image` and `cleaned_image`
    im_diff_noised (ndarray): Pixel-wise difference between `image` and `noised_image`.
    im_diff_cleaned (ndarray): Pixel-wise difference between `image` and `cleaned_image`.
    """
    
    def image_diff(img1, img2):
        """
        Calculates and plots the pixel-wise difference between two images of the same size.
        """
        # Check image sizes are the same
        if img1.shape != img2.shape:
            raise ValueError("The two images must have the same shape.")
        
        # Calculate the pixel-wise difference between the two images
        diff = np.abs(img1- img2)  #if input is torch tensors then need to add .numpy().squeeze().squeeze() which takes from input of [batchsize, channels, x, y] to [x, y]

        # Count the total number of elements
        num_total = diff.size

        # Count the number of elements that are different
        num_diff = np.count_nonzero(diff)
        
        # Calculate number of similar elements 
        num_same = num_total - num_diff

        return diff, num_diff, num_same

    # Calculate and plot the pixel-wise difference between the two images
    im_diff_noised, num_diff_noised, num_same_noised = image_diff(image, noised_image)
    im_diff_cleaned, num_diff_cleaned, num_same_cleaned = image_diff(image, cleaned_image)

    if print_text:
        print("\nNumber of different elements between original and noised:", num_diff_noised, "\nNumber of same elements between original and noised", num_same_noised, "\n")
        print("Number of different elements between original and cleaned:", num_diff_cleaned, "\nNumber of same elements between original and cleaned", num_same_cleaned, "\n")

    return (num_diff_noised, num_same_noised, num_diff_cleaned, num_same_cleaned, im_diff_noised, im_diff_cleaned)

#%% - GraphViz
def Graphwiz_visulisation():
    """
    Using GraphViz: GraphViz is a popular open-source graph visualization software that can be used to visualize the 
    structure of your PyTorch autoencoder network. You can use the torchviz package to generate a GraphViz dot file from 
    your PyTorch model and then use the dot command-line tool to generate a PNG image of the graph.:
    """
    """
    ##### WORKING ######
    # Join the encoder and decoder models
    model = torch.nn.Sequential(encoder, decoder)

    # Generate a dot file from the model
    x = torch.randn(batchsize, 1, 128, 88, dtype=torch.double) # dummy input tensor
    dot = make_dot(model(x), params=dict(model.named_parameters()))

    # Save the dot file
    dot.render('model_graphpp')

    
    ##### NOT WORKING ######
    import os

    # Convert the dot file to a PNG image
    os.system('C:/Program Files/Graphviz/bin/dot -Tpng model_graph.dot -o model_graph.png')

    # Open and display the PNG image then save it to a file
    img = Image.open('model_graph.png')
    img.save('model_graph_saved.png')
    img.show()

    """

#%% - Create new images from the latent space
def Generative_Latent_information_Visulisation(encoder, decoder, latent_dim, device, test_loader):
    """
    Function to visualize autoencoder data by generating random latent vectors and reconstructing 
    corresponding images using the decoder model.

    Parameters:
        encoder (torch.nn.Module): The encoder model.
        decoder (torch.nn.Module): The decoder model.
        latent_dim (int): The dimensionality of the latent space.
        device (str): The device to run the computations on.
        test_loader (torch.utils.data.DataLoader): The data loader containing the test dataset.

    Returns:

    """

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        # calculate mean and std of latent code, generated takining in test images as inputs
        images, labels = next(iter(test_loader))
        images = images.to(device)
        latent = encoder(images)
        latent = latent.cpu()

        mean = latent.mean(dim=0)
        #print(mean)
        std = (latent - mean).pow(2).mean(dim=0).sqrt()
        #print(std)

        # sample latent vectors from the normal distribution
        latent = torch.randn(128, latent_dim)*std + mean

        # reconstruct images from the random latent vectors
        latent = latent.to(device)
        img_recon = decoder(latent)
        img_recon = img_recon.cpu()
        
        return (img_recon)

#%% - Plots for the higher dimensional space
def Reduced_Dimension_Data_Representations(encoder, device, test_dataset, plot_or_save=0):
    """
    Display the input data samples as a XXXXX


    Parameters:
        encoder (torch.nn.Module): The encoder model.
        device (str): The device to run the computations on.
        test_dataset (TYPE???): The test dataset.
        plot_or_save (int, optional): Specifies whether to display the visualization (0) or save it to file (1) or both (2).

    Returns:

    """
    try:
        encoded_samples = []
        for sample in tqdm(test_dataset):
            img = sample[0].unsqueeze(0).to(device)
            label = sample[1]
            # Encode image
            encoder.eval()
            with torch.no_grad():
                encoded_img = encoder(img)
                # Append to list
                encoded_img = encoded_img.flatten().cpu().numpy()
                encoded_sample = {f"Enc. Variable {i}": enc for i, enc in enumerate(encoded_img)}
                encoded_sample['label'] = label
                encoded_samples.append(encoded_sample)
        encoded_samples = pd.DataFrame(encoded_samples)

        ### TSNE of Higher dim
        tsne = TSNE(n_components=2)
        tsne_results = tsne.fit_transform(encoded_samples.drop(['label'],axis=1))

        return(encoded_samples, tsne_results)

    except:
        return(None)

"""
This code is performing the following operations:

    encoded_samples = []: Initializes an empty list called encoded_samples to store the encoded images.

    for sample in tqdm(test_dataset):: Loops through each sample in the test_dataset and uses the tqdm library to display a progress bar for the loop.

    img = sample[0].unsqueeze(0).to(device): Extracts the image from the current sample and converts it to a PyTorch tensor. unsqueeze(0) adds an extra dimension to the tensor, which is required for passing it through the neural network. to(device) moves the tensor to the specified device (e.g., GPU or CPU).

    label = sample[1]: Extracts the label from the current sample.

    encoder.eval(): Sets the neural network called encoder to evaluation mode. This is required to disable dropout and other regularization techniques during inference.

    with torch.no_grad():: Wraps the subsequent code in a with statement to disable gradient calculations, which can save memory and computation time.

    encoded_img = encoder(img): Passes the image through the encoder neural network to obtain a compressed representation of the image. This is called encoding.

    encoded_img = encoded_img.flatten().cpu().numpy(): Flattens the encoded image tensor and converts it to a NumPy array. This is required to store the encoded image in a Pandas DataFrame later.

    encoded_sample = {f"Enc. Variable {i}": enc for i, enc in enumerate(encoded_img)}: Creates a dictionary called encoded_sample with keys "Enc. Variable 0", "Enc. Variable 1", etc., and values corresponding to the encoded image array. This is done to store the encoded image data in a structured format.

    encoded_sample['label'] = label: Adds a key-value pair to the encoded_sample dictionary, where the key is "label" and the value is the corresponding label for the current image.

    encoded_samples.append(encoded_sample): Adds the encoded_sample dictionary to the encoded_samples list.

    encoded_samples = pd.DataFrame(encoded_samples): Converts the encoded_samples list to a Pandas DataFrame for easier manipulation and analysis.

    tsne = TSNE(n_components=2): Initializes a t-SNE object with 2 components. t-SNE is a dimensionality reduction technique that can be used to visualize high-dimensional data in 2D or 3D.

    tsne_results = tsne.fit_transform(encoded_samples.drop(['label'],axis=1)): Applies t-SNE to the encoded image data (i.e., all columns except the "label" column) to obtain 2D coordinates for each image. The resulting tsne_results array contains 2D coordinates for each image, which can be used to visualize the data.

    return(encoded_samples, tsne_results): Returns the encoded image data (encoded_samples) and the t-SNE results (tsne_results) as output from the function.


"""