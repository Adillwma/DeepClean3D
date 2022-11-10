# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 20:33:51 2022
Flat Conv2d Input Output Size Helper Function v0.0.1
@author: Adill Al-Ashgar
Notes: can be improved to handle multidimensional arrays not just flat 1d
"""

#Function        
def out_array_size(in_array_size, stride, kernal, padding=0, outputs=1):
    i=1
    uniques=0
    
    if padding != 0:
        in_array_size = in_array_size + padding + padding
    
    while i <= in_array_size - kernal+1: # in_array_size[::stride]:
        #print("start pixel",i)
        i = i + stride  
        uniques = uniques + 1
        #print("uniques",uniques)
    result = uniques
    incomplete = in_array_size - i
    totalsize = result * outputs
    
    print("Finished, output size is:",result,"with a remainder of:",incomplete,"unscanned pixels")
    print("size across total number of outputs is:", totalsize)
    return(result, incomplete)

#Driver
stride = 2
kernal = 3
padding = 1
in_array_size = 128*88
outputs = 8

results = out_array_size(in_array_size, stride, kernal, padding, outputs)     
