#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 13:10:28 2023

@author: enrico
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import copy

img_path = "/home/enrico/Desktop/Wires_Stat/ebr_GenesisF_binary_survey4statistics/DE11_280nm_003.tif"
image = cv2.imread(img_path, 0)
img=image
img = cv2.fastNlMeansDenoising(image[100 : image.shape[0] - 200, 
           0 : image.shape[1]],dst=img,h=30)
cv2.imwrite("input1.tiff", img)

def gauss_smooth(img, save_gsm = False):
    gsm = cv2.GaussianBlur(img, (7, 7), 0)
    if save_gsm:
        cv2.imwrite("gsm.tiff", gsm)
    return gsm

def low_pass_filter(img, kernel_size = 4):
    kernel = np.ones(shape = (kernel_size, kernel_size), dtype = np.float64)/25
    lpf = cv2.filter2D(img, 0, kernel)
    cv2.imwrite("lpf.tiff", lpf)
    return lpf

def hist_equal(img):
    heq = cv2.equalizeHist(img)
    cv2.imwrite("heq.tiff", heq)
    return heq
    
def canny_edge_detect(img, save_ced = False):
    ced = cv2.Canny(img, 70, 150, 1)
    if save_ced:
        cv2.imwrite("ced.tiff", ced)
    return ced

def int_filter(img, low_limit = 90, high_limit = 200):
    ll = np.uint8(low_limit)
    hl = np.uint8(high_limit)
    inf = np.zeros(shape = img.shape[:2], dtype = np.uint8)
    h, w = img.shape[:2]
    for i in range(h):
        for j in range(w):
            if img[i, j] > hl:
                inf[i, j] = hl
            elif img[i, j] < ll:
                inf [i, j] = ll
            else:
                inf[i,j] = img[i, j]
    cv2.imwrite(f"inf_{low_limit}_{high_limit}.tiff", inf)
    return inf

def int_select(img, low_limit = 115, high_limit = 150):
    ll = np.uint8(low_limit)
    hl = np.uint8(high_limit)
    ins = np.zeros(shape = img.shape[:2], dtype = np.uint8)
    h, w = img.shape[:2]
    for i in range(h):
        for j in range(w):
            if img[i, j] > hl:
                continue
            elif img[i, j] < ll:
                continue
            else:
                ins[i,j] = np.uint8(255)
    cv2.imwrite("ins.tiff", ins)
    return ins

def clahe(img):
    cla_he = cv2.createCLAHE(tileGridSize = (8, 8))
    cla = cla_he.apply(img)
    cv2.imwrite("cla.tiff", cla)
    return cla

def invert(img):
    print(img.min(), img.max())
    inv = np.uint8(256 - img)
    print(inv.min(), inv.max(), inv.shape)
    cv2.imwrite("inv.tiff", inv)
    return inv

def close(img, kernel = 20):
    clo = cv2.morphologyEx(img, cv2.MORPH_CLOSE, (kernel, kernel))
    cv2.imwrite("clo.tif", clo)
    return clo

def maxpool_vertical(img):
    im = np.zeros(shape = img.shape[:2], dtype = np.uint8)
    im_size = img.shape
    size = (5, 1)
    stride = (1, 1)
    for i in range(0, im_size[0], stride[1]):
        for j in range(0, im_size[1], stride[0]):
            im[i : i + size[0], 
                j : j + size[1]] = np.max(img[i : i + size[0], 
                                              j : j + size[1]])
    return im

def maxpool(img):
    im = np.zeros(shape = img.shape[:2], dtype = np.uint8)
    im_size = img.shape
    cv2.imwrite("before.tiff", im)
    size = (3, 3)
    stride = (1, 1)
    for i in range(0, im_size[0], stride[0]):
        for j in range(0, im_size[1], stride[1]):
            im[i : i + size[0], 
                j : j + size[1]] = np.max(img[i : i + size[0], 
                                              j : j + size[1]])
    cv2.imwrite("after.tiff", im)
    return im

def minpool(img):
    im = np.zeros(shape = img.shape[:2], dtype = np.uint8)
    im_size = img.shape
    size = (1, 2)
    stride = (1, 1)
    for i in range(0, im_size[0], stride[0]):
        for j in range(0, im_size[1], stride[1]):
            im[i : i + size[0], 
                j : j + size[1]] = np.min(img[i : i + size[0], 
                                              j : j + size[1]])
    return im

def img_mirror(img):
    flp = cv2.flip(img, 1)
    return flp

def divide_img(img):
    im = minpool(img)
    slices = im[0 : im.shape[0] // 3,
                 0 : im.shape[1]]
    print(slices.shape)
    # limits = np.empty((0, 3), np.uint8)
    # for i in range(segment.shape[0]):
    #     for j in range(segment.shape[1]):
    #         print(segment[i, j])
    #         if segment[i, j] > 159:
    #             continue
    #         else:
    #             limits = np.append(limits, np.array([i, j, segment[i, j]]), axis = 0)
    #             #break
    # print(limits)
#    for i in range(3):
#        j = int(i + 1)
#        print(j)
#        im = img[0 : img.shape[0], 
#                 100 + i * img.shape[1] // 3 : j * img.shape[1] // 3]
#        print(f"writing input_segment{j}.tiff")
#        cv2.imwrite(f"input_segment{j}.tiff", im)
#        print(f"writing input_segment{j}_m.tiff")
#        im_m = img_mirror(im)
#        cv2.imwrite(f"input_segment{j}_m.tiff", im_m)
#        print(f"writing output{j}.tiff")
#        im1 = copy.deepcopy(im)
#        cv2.imwrite(f"output{j}.tiff", (hist_equal(minpool(im))))
#        print(f"writing output{j}_m.tiff")
#        cv2.imwrite(f"output{j}_m.tiff", (hist_equal(maxpool(im1))))
#    return None

# divide_img(img)
cv2.imwrite("output.tiff", maxpool(gauss_smooth(img)))