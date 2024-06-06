# Import necessary libraries
import numpy as np
from SmartImage import SmartImage
from shapely import Polygon

# Define a function to cut an image along specified sides
def cut_side(img1 , sides):
    """ Cut an image along specified sides:
        Args:
            img1: SmartImage instance.
            sides: list or iterable object containing int objects. Accepted values are 1, 2, 3, and 4.
        Returns:
            img1: SmartImage instance.
    """
    for side in sides:
        if side == 1: # cut left
            img1.cut(side)
        if side == 2: # cut right
            img1.fliplr()
            img1.cut(side)
            img1.fliplr()
        if side == 3: # cut bottom
            img1.rot90(number = 3)
            img1.cut(side)
            img1.rot90()
        if side == 4: # cut top
            img1.rot90()
            img1.cut(side)
            img1.rot90(3)
    return img1

# Define a function to split an image into a specified number of parts
def split(img1, num = 3):
    """ Divides an image into multiple images.
        Args:
            img1: SmartImage instance.
            num: int object. Specifies how many imgaes will result from the cut.
        Returns:
            smart_image_list: list object containing as many SmartImage instances as specified in "num".
    """
    arrays = np.array_split(img1.img, num, axis=1)
    x_sizes = []
    for array in arrays:
        x_sizes.append(array.shape[1])

    smart_image_list = []
    oc = img1.coord

    if num == 11:
        starting_x = oc[1][0]
    else:
        starting_x = oc[0][1]
    for i in range(len(arrays)):
        new_start = starting_x
        starting_x += x_sizes[i]

        if num == 11:
            coords_i = np.array([[starting_x, oc[0][1]], 
                                 [new_start, oc[1][1]]])
        else:
            coords_i = np.array([[oc[0][0], new_start], 
                                 [oc[1][0], starting_x]])

        smart_image_list.append(SmartImage(arrays[i], coords_i, img1.rotation_number))
    
    if num == 2:
        smart_image_list[1].fliplr()
        smart_image_list[1].assess_coords()
    return smart_image_list

# Define a function to generate four coordinates for a rectangle
def generate_four_coordinate(rectangle, source="SmartImage"):
    """ Generates a list of 4 coordinate pairs from the two edges defining a rectangle.
        Args:
            rectangle: numpy array containing two coordinate pairs.
            source: string object with value "SmartImage" or "label" to take into account the different coordinate order in the SmartImage object and the label file.
        Returns
            coords: tuple object containing four tuple objects each containing two int objects representing the coordinate pairs
    """
    if source == "SmartImage":
        xs = rectangle[:,1]
        ys = rectangle[:,0]
    elif source == "label":
        rectangle_array = np.array(rectangle, dtype=int)
        ys = rectangle_array[:,1]
        xs = rectangle_array[:,0]
    else:
        exit()
    x_high, x_low = xs.max(), xs.min()
    y_high, y_low = ys.max(), ys.min()
    coords = ((x_low, y_low), (x_low, y_high), (x_high, y_high), (x_high, y_low))
    return coords

# Define a function to generate a polygon label couple
def generate_poly_label(label):
    """ Creates a dictionary containing a Polygon instance as key and the associated label string from one of the labels in the JSON file
        Args:
            label: single label from the JSON file from the dataset associated with this code
        Returns:
            poly_label: Dictionary object with a single key:value pair.
    """
    if label["shape_type"] == "rectangle":
        label_polygon = Polygon(shell = generate_four_coordinate(label["points"], source = "label"))
        poly_label = {label_polygon: f'{label["label"]}'}
        return poly_label
    elif label["shape_type"] == "polygon":
        polygon_array = np.array(label["points"], dtype=int)
        label_polygon = Polygon(shell = polygon_array)
        poly_label = {label_polygon: f'{label["label"]}'}
        return poly_label
    else:
        raise AssertionError

# Define a function to find the label for a wire
def label_finder(wire, labels, polygon_labels):
    """ Function to discerne which label to assign to an SmartImage instance.
        Args:
            wire: SmartImage instance.
            labels: labels portion of the JSON file contained in the dataset paired with this code
            polygon_labels: dictionary object
        Returns:
            final_label: string object, one of the labels used in the paired dataset.
            polygon_labels: dictionary object containing the Polygon istances created with the Shapely package from the label coordinates in the Json file as keys and their associated label strings as values.
    """
    wire_label = []
    poly_labels = polygon_labels
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
        return wire_label[0][0], poly_labels
    elif len(wire_label) == 0:
        label_delete = "Null"
        return label_delete, poly_labels
    elif len(wire_label) > 1:
        wire_polygon = Polygon(shell=generate_four_coordinate(wire.coord))
        matching_labels = {}
        if len(poly_labels) == 0:
            for label in labels:
                try:
                    generated_poly_label = generate_poly_label(label)
                except:
                    print(f'Invalid shape {label["shape_type"]} (expected "rectangle" or "polygon") for function "generate_poly_label" while processing wire {wire.name}.')
                poly_labels.update(generated_poly_label)
        for poly, label in poly_labels.items():
            if wire_polygon.intersects(poly):
                try:
                    intersection = wire_polygon.intersection(poly)
                except:
                    print(f"Polygon {poly} with label {label} in wire {wire.name} has an invalid geometry and was skipped.")
                    continue
                area = intersection.area
                matching_labels.update({area : label})
            else:
                continue
        max_area = float(0.0)
        final_label = None

        for area, label in matching_labels.items():
            if area > max_area:
                max_area = area
                final_label = label
        return final_label, poly_labels
