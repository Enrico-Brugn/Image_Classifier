# Import necessary libraries
from torch.utils.data.dataset import Dataset
import pandas as pd
import torchvision.transforms as T
import cv2

# Define a custom dataset class
class WireDataset(Dataset):
    def __init__(self, input_csv):
        # Define a mapping for the labels
        mapping = {'Parasitic': 0, 'Parassitic' : 0,'Wire_Straight_Defect': 1, 'Wire_Straight_Perfect': 2 , 'Wire_Tilted_Defect' : 3, 'Wire_Tilted_Perfect' : 4, 'Null' : 5}
        
        # Read the input CSV file
        csv = pd.read_csv(input_csv, dtype="string")
        
        # Assign the image paths and labels to instance variables
        self.input = csv.image_path
        self.output = csv.label

        # Remove the rows with 'Delete'
        inds = self.output.index[self.output=='Delete'].to_list()
        self.output = self.output.drop(inds)
        self.input = self.input.drop(inds).values
        
        # Replace the string labels with their corresponding numerical values
        self.output = self.output.replace(to_replace = mapping).values
        
        # Define the image size and the transformation pipeline
        self.image_size = (178,55)
        self.transform = T.Compose([T.ToPILImage(), T.Resize(self.image_size), T.ToTensor()])

    def __getitem__(self, index):
        # Load the image at the given index in grayscale format
        img = cv2.imread(self.input[index], cv2.IMREAD_GRAYSCALE)
        
        # Apply the transformations to the image
        input = self.transform(img)
        
        # Return the transformed image and its corresponding label
        return input, self.output[index]
    
    def __len__(self):
        # Return the total number of images in the dataset
        return len(self.input)