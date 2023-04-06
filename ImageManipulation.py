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

def cut(smart_im):
    A = cv2.fastNlMeansDenoising(copy.deepcopy(smart_im.img))
    oc = smart_im.coord # original coordinates

    min_pl = min_pool(A)[A.shape[0]//3:2*A.shape[0]//3]
    max_pl = max_pool(A)[A.shape[0]//3:2*A.shape[0]//3]
    
    avg_min = np.min(min_pl, axis=0)
    avg_max = np.max(max_pl, axis=0)
    
    grad_min = np.abs(np.gradient(avg_min))
    grad_max = np.abs(np.gradient(avg_max))

    cut_point_1 = np.argmax(grad_min) + 1
    cut_point_2 = np.argmax(grad_max) + 1
    cut_point = cut_point_1 if cut_point_1>cut_point_2 else cut_point_2

    B = smart_im.img[:, cut_point :]
    new_coords = np.array([oc[0], 
                          [oc[1][0] + cut_point, oc[1][1]]])
    
    smart_img1 = SmartImage(B, new_coords)
    return smart_img1

def cut_side(img1 , sides):
    for side in sides:
        if side == 1:
            # cut left
            img1 = cut(img1)
        if side == 2:
            # cut right
            img1.fliplr()
            img1 = cut(img1)
            img1.fliplr()
        if side == 3:
            # cut bottom
            img1.rot90(3)
            img1 = cut(img1)
            img1.rot90()
        if side == 4:
            # cut top
            img1.rot90()
            img1 = cut(img1)
            img1.rot90(3)
    return img1

def split(img1, num = 3):
    arrays = np.array_split(img1.img, num, axis=1)
    x_sizes = []
    for array in arrays:
        x_sizes.append(array.shape[1])

    smart_image_list = []
    oc = img1.coord
    starting_x = oc[0][1]
    for i in range(len(arrays)):
        new_start = starting_x
        starting_x += x_sizes[i]
        coords_i = np.array([[oc[0][0], new_start], 
                             [oc[1][0], starting_x]])
        smart_image_list.append(SmartImage(arrays[i], coords_i))
    if num == 2:
        smart_image_list[1].fliplr()
    return smart_image_list

