#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:39:40 2023

@author: enrico
"""

from PIL import Image
import numpy as np
import torch

from torchvision.io import read_image
import matplotlib.pyplot as plt


img_path = "/home/enrico/Desktop/Wires_Stat/ebr_GenesisF_binary_survey4statistics/DE11_280nm_004.tif"
img = torch.Tensor(np.array(Image.open(img_path)))
m = torch.nn.MaxPool1d((2, 5), stride = (1, 5))

print(img.shape)

output = m(img)
plt.imshow(output)
