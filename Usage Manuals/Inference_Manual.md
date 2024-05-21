
# Installation
## Dependencies
    Numpy >= 1.18.1
    PyTorch >= 1.4.0

## Install from PyPI
```bash
pip install DC3D_Inference
```

## Downloading the Pretrained Models
Link to models: 





# Quick Start for Denoising

## Input Data
### Single Frame Processing
Input expects data of shape 88 x 128. The output will be the same shape as the input.

### Batch Processing
Input expects data of shape B x 88 x 128, where B is the batch size. The output will be the same shape as the input.

### Code Example
To use the denosiier all you need to do is input the file dir to the folder that contains the images to be denoised, and specify the output dir where the resulting denoised images should be saved. Then run the file.

```python
from DC3D_Full import DC3D_Inference

### Example Code

# User Inputs
reconstruction_threshold = 0.5
time_dimension = 1000
latent_dim = 10
model_name =  "10KN_V3"

# Setup Inference Class
dc3d_inference = DC3D_Inference(time_dimension, reconstruction_threshold, latent_dim, model_name)

# Run Inference
recovered_image, masking_rec_image = dc3d_inference.run(input_image_tensor)

```


    