import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import copy
from scipy.stats import mode
from scipy import ndimage
from pathlib import Path
import labelme
from SmartImage import SmartImage 
import ImageManipulation as IM

mainDir = Path("/home/enrico/Desktop/json")
jsonList = list(mainDir.glob('**/*.json'))
tiffNames = [(i.stem + ".tif") for i in jsonList]

images_vector = []

for jsonPath in jsonList:
    label_file = labelme.LabelFile(filename=jsonPath.absolute())
    
    img = labelme.utils.img_data_to_arr(label_file.imageData)
    img = 255 * (img-img.min())/(img.max()-img.min())
    imga = img.astype("uint8")
    img1 = imga[100 : imga.shape[0] - 200, 
                50 : imga.shape[1] - 50]
    
    starting_smart_im = SmartImage(img1, np.array([[100, 50], 
                                                   [img.shape[0] - 200, img.shape[1] - 50]]))
    
    wire_array = IM.cut_side(starting_smart_im, [1, 2, 3, 4])
    
    three_columns = IM.split(wire_array)
    three_columns[0] = IM.cut_side(three_columns[0], [2])
    three_columns[1] = IM.cut_side(three_columns[1], [1,2])
    three_columns[2] = IM.cut_side(three_columns[2], [1])

    six_columns = []

    for column in three_columns:
        six_columns.extend(IM.split(column, 2))

    wire_list = []
    for column in six_columns:
        column.rot90()
        wires = IM.split(column, 11)
        for wire in wires:
            wire.rot90(3)
        wire_list.extend(wires)

    labels = label_file.shapes
    i = 0
    for wire in wire_list:
        i += 1
        wire.setName(label_file.filename, i)
        wire_label = []
        for label in labels:
            points_in_image = []
            for point in label["points"]:
                if wire.contains(point):
                    points_in_image.append(point)
                else:
                    continue
            if len(points_in_image) > 0:
                wire_label.append([label["label"], len(points_in_image)])
            else:
                continue

        if len(wire_label) == 1:
            wire.setLabel(wire_label[0][0])
        elif len(wire_label) > 1:
            for label in wire_label:
                if label[0] == "Wire_Tilted_Perfect" or label[0] == "Wire_Tilted_Defect":
                    wire.setLabel("Wire_Tilted_Defect")
                    break
                elif label[0] == "Wire_Straight_Perfect" or label[0] == "Wire_Straight_Defect":
                    wire.setLabel("Wire_Straight_Defect")
                    break
                else:
                    wire.setLabel("Parassitic")
                    break
        elif len(wire_label) == 0:
            wire.setLabel("Delete")

    filename = label_file.filename
    for wire in wire_list:
        if wire.label == "Delete":
            wire_list.remove(wire)
        else:

            continue

    images_vector.extend(wire_list)

print(images_vector)

img_dict = {}

for image in images_vector:
    cv2.imwrite(image.name, image.img)
    current_wire = {image.name : image.label}
    img_dict.update(current_wire)
    exit