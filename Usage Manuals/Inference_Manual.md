## Quick Start for Denoising
To denoise images they must first be numpy arrays with shape 88 x 128.
Use the DC3D_V3 Denoiser file located in the folder DC3D_V3. To use the file all you need to do is input the file dir to the folder that contains the images to be denoised, and specify the output dir where the resulting denoised images should be saved. Then run the file.

    ### Example Code
    
    # Import the Denoiser
    from DC3D_V3 import DC3D_V3_Denoiser
    
    # Set the input and output directories
    input_dir = "C:/Users/.../input_folder"
    output_dir = "C:/Users/.../output_folder"
    
    # Run the Denoiser
    DC3D_V3_Denoiser(input_dir, output_dir)