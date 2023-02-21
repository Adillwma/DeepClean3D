"""
### Modal_tester V1
# If the saved model is successfully loaded, the program will print out the contents of the model. 
# If the "AttributeError" occurs, you can try to investigate the saved model file to check whether it 
# contains the expected "Encoder" attribute. You can also check the code used to save the model to ensure 
# that the "Encoder" attribute is included.
"""
import torch
from DC3D_V3.DC3D_Autoencoder_V1 import Encoder, Decoder
encoder, decoder = torch.load("Models/10X_Activation_V1.pth")

print(encoder, decoder)





