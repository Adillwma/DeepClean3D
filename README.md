# DeepClean - [PyTorch Convolutional Neural Network Architecture to Perform Noise Suppression for LHCb Torch Detector]
# Authors: Adill Al-Ashgar & Max Carter
# Physics Department - University of Bristol 


# Motivation / Background

One of the four largest experiments on the LHC, the LHCb's (Large Hadron Collider Beauty) main focus is in investigating
baryon anti-baryon asymmetry, responsible for the matter dominated universe we inhabit today.

As part of the planned upgrade for the LHCb in 2022, a new detector is proposed to be added called TORCH (Time Of internally Reflected Cherenkov light) 
which is closely related  to the PANDA DIRC and Belle TOP detectors, combining timing information with DIRC-style reconstruction, but aiming 
for higher TOF resolution, 10–15 ps (per track). The detecttor is sensetive to incoming photons, it is arranged in a grid format, its output is a list of 
grid coocrdinates which detected photons

A current challenge is in reconstructing detected photons path data, it is a computationally costly process and the data flow is incredible (Xinclude data rates here from the 40Mhz streamX). 
All detections are currently reconstructed. However, once reconstructed the events are filtered down to those that correlated to a particular event, a large 
proportion of the detected photons (Xquote average percentage of noise? or sig to noise ratioX) are noise from other collisions or the detector and electronic subsystem.

Our desire is to reduce the number of signals that require path reconstruction by using a neural network to detect the signal, only passing these 
points on to the reconstruction algorithm saving the computation time spent on reconstructing the noise signals only to throw them away.

This is critical in the efficiency of the processing pipeline as the LHC moves into its high luminosity upgrade where data rates will be further increased.

# Code Contents Page:
01 - Simple Circular & Spherical Dataset Generator
   
02 - Dataset Validator

03 - 2D & 3D DataLoader
   - DataLoader_Functions_V1
   
04 - Convolution Layer Output Size Calculator

05 - 2D Conv Autoencoder - Simple Data
   - DataLoader_Functions_V1
   
06 - 3D Conv Autoencoder - Simple Data
   - DataLoader_Functions_V1


# Code Breakdown:
01 - Simple Circular & Spherical Dataset Generator
Description:

02 - Dataset Validator
Description:


03 - 2D & 3D DataLoader
Description:


   - DataLoader_Functions_V1
Description:



04 - Convolution Layer Output Size Calculator
Description:


05 - 2D Conv Autoencoder - Simple Data
Description:



   - DataLoader_Functions_V1
Description:



06 - 3D Conv Autoencoder - Simple Data
Description:



   - DataLoader_Functions_V1
Description:

