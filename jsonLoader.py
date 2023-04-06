import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
mainDir = Path("/home/enrico/Desktop/json")
import labelme
import io
jsonList = list(mainDir.glob('**/*.json'))
tiffNames = [(i.stem + ".tif") for i in jsonList]
import matplotlib.pyplot as plt


for jsonPath in jsonList:
    # jsonFile = json.loads(jsonPath.read_text(encoding="UTF-8"))
    
    label_file = labelme.LabelFile(filename=jsonPath.absolute())
    img = labelme.utils.img_data_to_arr(label_file.imageData)
    print(img.shape)
    for i in label_file.shapes:
        print(i)
    exit()
    # Split blocks
    # Identify belonging