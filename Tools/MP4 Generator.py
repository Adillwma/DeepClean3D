#Author: Adill Al-Ashgar
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np 
from PIL import Image

def create_video_from_pngs(folder_path, frame_rate, output_path):
    # Get a list of all .png files in the directory
    png_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')])

    if not png_files:
        print("Error, no PNG files found in the directory!")
        return None

    def load_png_as_np_array(file_path):
        # Open the PNG file using Pillow
        image = Image.open(file_path)

        # Convert the image to a numpy array
        np_array = np.array(image)

        return np_array

    img = load_png_as_np_array(png_files[0])
    height, width, channels = img.shape

    # First image test
    print("First image filename", png_files[0])
    plt.imshow(img)
    print("Height", height, "Width", width, "Channels", channels)
    
    # Remove the last element in the list   - It is commanly corrupted due to render quit
    png_files = png_files[:-1]

    # Create the video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    # Loop through all .png files and add them to the video
    for png_file in tqdm(png_files):
        img = load_png_as_np_array(png_file)
        video.write(img)

    # Release the video writer and print a success message
    video.release()
    if not Path(output_path).exists():
        print("Error, video could not be created!")
    else:
        print('Video successfully created at', output_path)

#%% - Driver
path = "Frames\AE"                       # ('/path/to/raw/pngs')
path2 = "Frames\CNN"                       # ('/path/to/raw/pngs')
frame_rate = 30                          # Frames per second
output_path = "MP4_Renders"             # ('/path/to/save/MP4')
output_filename = "\AE"                   # Filename for output video
output_filename2 = "\CNN"                   # Filename for output video

#%% - Compute
output = output_path + output_filename + ".MP4"
output2 = output_path + output_filename2 + ".MP4"
create_video_from_pngs(path, frame_rate, output) # Creates the MP4 video

create_video_from_pngs(path2, frame_rate, output2)