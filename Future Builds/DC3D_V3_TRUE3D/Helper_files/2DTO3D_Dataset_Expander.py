import os
import numpy as np

def custom_renormalisation(data, reconstruction_threshold, time_dimension=100):
    data = np.where(data > reconstruction_threshold, ((data - reconstruction_threshold)*(1/(1-reconstruction_threshold)))*(time_dimension), data)
    return data


input_dir = "path/to/input/directory"
output_dir = 


def dataset_2D_to_3D_Expansion(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop over each file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".npy"):
            # Load the data from the file
            filepath = os.path.join(input_dir, filename)
            data = np.load(filepath)

            # Apply the processing function to the data
            processed_data = custom_renormalisation(data, 0.8)

            # Save the processed data to a new file in the output directory
            output_filepath = os.path.join(output_dir, "3D -" + filename)
            np.save(output_filepath, processed_data)