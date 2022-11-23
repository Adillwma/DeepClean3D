# -*- coding: utf-8 -*-
"""
ROOT to NUMPY Convertor
@author: Max Carter
Created on Tue Nov 23 2022
"""

#%% - Dependencies
import matplotlib.pyplot as plt
import numpy as np
import uproot


with uproot.open("path/to/dataset.root") as file:
    do_something...

    