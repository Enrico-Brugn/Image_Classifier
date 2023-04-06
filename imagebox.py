import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import copy
from scipy.stats import mode
from scipy import ndimage
import json
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
    print(label_file.filename)
    img = labelme.utils.img_data_to_arr(label_file.imageData)
    img = 255 * (img-img.min())/(img.max()-img.min())
    imga = img.astype("uint8")
    img1 = imga[100 : imga.shape[0] - 200, 
                50 : imga.shape[1]- 50]
    
    starting_smart_im = SmartImage(img1, np.array([[100, img.shape[0] - 200], 
                                                   [50 , img.shape[1]- 50]]))
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
    
    assert len(wire_list) == 66

    labels = label_file.shapes

    for wire in wire_list:
        wire.setOrigin(label_file.filename)
        wire_label = []
        for label in labels:
            points_in_image = []
            # points_outside_image = []
            for point in label["points"]:
                if wire.contains(point):
                    points_in_image.append(point)
                else:
                    #points_outside_image.append(point)
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
    
    assert len(wire_list) == 66 

    #for wire in wire_list:
    #    if wire.label == "Delete":
    #        wire_list.remove(wire)
    #    else:
    #        continue
    print(wire_list)
    print(str(len(wire_list)))
    plt.imshow(img)
    plt.annotate("Hello", xy=[10,10])
    for wire in wire_list:
        plt.annotate(wire.label, xy = wire.coord[:,0])
        print(wire.coord[:,0])
    plt.savefig("test.png")
    exit()
    images_vector.extend(wire_list)

print(images_vector)