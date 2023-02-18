
#image = 
#noised_image = 
#cleaned_image = 

def AE_visulisation(image, noised_image, cleaned_image, encoder_model, decoder_model, latent_dim, device, test_loader, test_dataset, batchsize):
    #%% - Differnce between images
    import numpy as np
    import matplotlib.pyplot as plt
    import torchvision
    import torch
    import pandas as pd
    from tqdm import tqdm
    from torchviz import make_dot
    from PIL import Image
    import plotly.express as px


    # Define your PyTorch autoencoder model
    encoder = encoder_model
    decoder = decoder_model


    def plot_image_diff(img1, img2):
        #Calculates and plots the pixel-wise difference between two images of the same size.
        if img1.shape != img2.shape:
            raise ValueError("The two images must have the same shape.")
        
        # Calculate the pixel-wise difference between the two images
        diff = np.abs(img1.squeeze().squeeze() - img2.squeeze().squeeze())
        
        # Plot the difference image as an imshow plot
        plt.imshow(diff, cmap='gray')
        plt.title("Pixel-wise difference between the two images")
        plt.axis('off')
        plt.show()
        
        # Count the number of elements that are different
        num_diff = np.count_nonzero(diff)
        
        return num_diff

    # Calculate and plot the pixel-wise difference between the two images
    num_diff_noised = plot_image_diff(image, noised_image)
    print("Number of different elements:", num_diff_noised)

    num_diff_cleaned = plot_image_diff(image, cleaned_image)
    print("Number of different elements:", num_diff_cleaned)



    #%% - GraphViz
    """
    Using GraphViz: GraphViz is a popular open-source graph visualization software that can be used to visualize the 
    structure of your PyTorch autoencoder network. You can use the torchviz package to generate a GraphViz dot file from 
    your PyTorch model and then use the dot command-line tool to generate a PNG image of the graph. Here's an example 
    code snippet:
    """
    """

    # Join the encoder and decoder models
    model = torch.nn.Sequential(encoder, decoder)

    # Generate a dot file from the model
    x = torch.randn(batchsize, 1, 128, 88, dtype=torch.double) # dummy input tensor
    dot = make_dot(model(x), params=dict(model.named_parameters()))

    # Save the dot file
    dot.render('model_graphpp')

    

    import os

    # Convert the dot file to a PNG image
    os.system('C:/Program Files/Graphviz/bin/dot -Tpng model_graph.dot -o model_graph.png')

    # Open and display the PNG image then save it to a file
    img = Image.open('model_graph.png')
    img.save('model_graph_saved.png')
    img.show()

    """


    #%% - Create new images from the latent space

    def show_image(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        # calculate mean and std of latent code, generated takining in test images as inputs
        images, labels = next(iter(test_loader))
        images = images.to(device)
        latent = encoder(images)
        latent = latent.cpu()

        mean = latent.mean(dim=0)
        print(mean)
        std = (latent - mean).pow(2).mean(dim=0).sqrt()
        print(std)

        # sample latent vectors from the normal distribution
        latent = torch.randn(128, latent_dim)*std + mean

        # reconstruct images from the random latent vectors
        latent = latent.to(device)
        img_recon = decoder(latent)
        img_recon = img_recon.cpu()

        fig, ax = plt.subplots(figsize=(20, 8.5))
        show_image(torchvision.utils.make_grid(img_recon[:100],10,5))
        plt.show()
        
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
    encoded_samples

    fig2 = px.scatter(encoded_samples, x='Enc. Variable 0', y='Enc. Variable 1',
            color=encoded_samples.label.astype(str), opacity=0.7)
    fig2.show()



    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(encoded_samples.drop(['label'],axis=1))
    fig = px.scatter(tsne_results, x=0, y=1,
                    color=encoded_samples.label.astype(str),
                    labels={'0': 'tsne-2d-one', '1': 'tsne-2d-two'})
    fig.show()




