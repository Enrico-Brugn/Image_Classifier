import copy
import numpy as np
from SmartImage import SmartImage
import cv2

def min_pool(img, sz = (5,2), stride = (5,1)):
    im = copy.deepcopy(img)
    im_size = img.shape

    for i in range(0, im_size[0], stride[0]):
        for j in range(0, im_size[1], stride[1]):
            im[i : i + sz[0], j : j + sz[1]] = np.min(im[i : i + sz[0],j : j + sz[1]])
    return im


def max_pool(img, sz = (5,2), stride = (5,1)):
    im = copy.deepcopy(img)
    im_size = img.shape

    for i in range(0, im_size[0], stride[0]):
        for j in range(0, im_size[1], stride[1]):
            im[i : i + sz[0], j : j + sz[1]] = np.max(im[i : i + sz[0],j : j + sz[1]])
    return im

def cut_side(img1 , sides):
    for side in sides:
        if side == 1:
            # cut left
            img1.cut(side)
        if side == 2:
            # cut right
            img1.fliplr()
            img1.cut(side)
            img1.fliplr()
        if side == 3:
            # cut bottom
            img1.rot90(number = 3)
            img1.cut(side)
            img1.rot90()
        if side == 4:
            # cut top
            img1.rot90()
            img1.cut(side)
            img1.rot90(3)
    return img1

def split(img1, num = 3):
    arrays = np.array_split(img1.img, num, axis=1)
    x_sizes = []
    for array in arrays:
        x_sizes.append(array.shape[1])
    #print(f"x_sizes = {x_sizes}")
    smart_image_list = []
    oc = img1.coord
    if num == 11:
        starting_x = oc[1][0]
    else:
        starting_x = oc[0][1]
    for i in range(len(arrays)):
        #print(f"starting_x = {starting_x}")
        new_start = starting_x
        starting_x += x_sizes[i]
        #print(f"ending_x = {starting_x}")
        if num == 11:
            coords_i = np.array([[starting_x, oc[0][1]], 
                                 [new_start, oc[1][1]]])
        else:
            coords_i = np.array([[oc[0][0], new_start], 
                                 [oc[1][0], starting_x]])
        #print(f"coords_i = {coords_i}")
        smart_image_list.append(SmartImage(arrays[i], coords_i, img1.rotation_number))
    if num == 2:
        smart_image_list[1].fliplr()
        smart_image_list[1].assess_coords()
    return smart_image_list

