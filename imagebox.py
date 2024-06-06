# Import necessary libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import csv
from pathlib import Path
import labelme
from SmartImage import SmartImage 
import ImageManipulation as IM
import pandas as pd

# Get the current working directory
pathcurrent = os.getcwd()

# Define the main directory where the labelled data is stored
main_dir = Path(os.path.join(pathcurrent, "Labelled_Data"))
print(f"Label directory: {main_dir}")

# Get a list of all JSON files in the main directory and its subdirectories
jsonList = list(main_dir.glob('**/*.json'))

# Get the names of the corresponding TIFF files
tiffNames = [(i.stem + ".tif") for i in jsonList]

# Initialize an empty list to store the images
images_vector = []

# Define a function to process a JSON file to extract the all the wires
def process_json(label_file):
    """ Isolate the various wires from the array image contained in the LabelMe-genarated JSON file and assign each image a label.
        Args:
            label_file: JSON file from the dataset associated with this code.
    """

    # Convert the image data in the JSON file to an array
    img = labelme.utils.img_data_to_arr(label_file.imageData)
    
    # Normalize the image data to the range [0, 255]
    img = 255 * (img-img.min())/(img.max()-img.min())
    
    # Convert the image data to 8-bit unsigned integers
    imga = img.astype("uint8")
    
    # Crop the image to remove the borders
    img1 = imga[100 : imga.shape[0] - 200, 
                50 : imga.shape[1] - 50]
    
    # Create a SmartImage instance from the cropped image
    starting_smart_im = SmartImage(img1, 
                                   np.array([[100, 50], 
                                             [img.shape[0] - 200, img.shape[1] - 50]]))
    
    # Cut the image into four parts along the sides
    wire_array = IM.cut_side(starting_smart_im, [1, 2, 3, 4])
    
    # Split the image into three columns
    three_columns = IM.split(wire_array)
    
    # Cut the sides of the columns
    three_columns[0] = IM.cut_side(three_columns[0], [2])
    three_columns[1] = IM.cut_side(three_columns[1], [1,2])
    three_columns[2] = IM.cut_side(three_columns[2], [1])

    # Initialize an empty list to store the six columns
    six_columns = []

    # Split each of the three columns into two columns
    for column in three_columns:
        six_columns.extend(IM.split(column, 2))

    # Initialize an empty list to store the wires
    wire_list = []
    
    # Split each of the six columns into eleven wires
    for column in six_columns:
        column.rot90()
        wires = IM.split(column, 11)
        for wire in wires:
            wire.rot90(3)
        wire_list.extend(wires)

    # Get the labels from the JSON file
    labels = label_file.shapes
    
    # Initialize an empty dictionary to store the polygon labels
    polygon_labels = {}
    
    # Initialize a counter
    i = 0
    
    # Assign a name and a label to each wire
    for wire in wire_list:
        i += 1
        wire.setName(label_file.filename, i)
        label, polygon_labels = IM.label_finder(wire, labels, polygon_labels)
        wire.setLabel(label)

    # Extend the list of images with the list of wires
    images_vector.extend(wire_list)

# Process each JSON file in the list
for jsonPath in jsonList:
    label_file = labelme.LabelFile(filename = jsonPath.absolute())
    try:
        process_json(label_file)
    except:
        print(f"there was an error on file {label_file.filename}")
        continue

# Initialize an empty list to store the CSV data
csv_list = []

# Define the path to the directory where the wire images will be saved
path = os.path.join(os.getcwd(), "wires")

# Create the directory if it does not exist
if not os.path.exists(path) and not os.path.isdir(path):
    os.mkdir(path)

# Define a function to save the wire image and add its information to the CSV list
def image_writer(image):
    """ Save the isolated wire image and add its path and label to the csv file
        Args:
            image: SmartImage instance
    """
    wire = os.path.join(path, image.name)
    cv2.imwrite(wire, image.img)
    current_wire = {'image_path' : wire, 'label' : image.label}
    csv_list.append(current_wire)

# Save each image in the list and add its information to the CSV list
for image in images_vector:
    try:
        image_writer(image)
    except:
        print(f"There was and error wile saving image {image.name}")
        continue

# Remove the existing CSV file if it exists
existing_csv = os.path.join(os.getcwd(), "Input_Data.csv")
if os.path.exists(existing_csv):
    os.remove(existing_csv)

# Write the CSV list to a CSV file
with open('Input_Data.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = csv_list[0].keys())
    writer.writeheader()
    writer.writerows(csv_list)

# Read the CSV file into a pandas DataFrame
cvsfile = pd.read_csv('Input_Data.csv')

# Drop the rows with all missing values
csvfile.dropna(axis=0, how='all', inplace=True)

# Save the DataFrame to the CSV file
csvfile.to_csv('Input_Data.csv', index=False)
