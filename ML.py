# Import necessary libraries
import torch.nn as nn
import torch.nn.functional as F

# Define a custom neural network class
class Net(nn.Module):
    def __init__(self, kernel = (3,6,5)):
        # Call the parent constructor
        super().__init__()
        
        # Initialize a dimension variable
        self.dim = 0

        # Define the layers of the neural network
        self.layers = nn.Sequential(
            # First convolutional layer with 1 input channel, 6 output channels, and a kernel size of 5
            nn.Conv2d(1, 6, 5),
            
            # Apply a ReLU activation function
            nn.ReLU(),
            
            # Apply a max pooling over an input signal composed of several input planes with a kernel size of 2 and stride of 2
            nn.MaxPool2d(2, 2),
            
            # Second convolutional layer with 6 input channels, 16 output channels, and a kernel size of 5
            nn.Conv2d(6, 16, 5),
            
            # Apply a ReLU activation function
            nn.ReLU(),
            
            # Apply a max pooling over an input signal composed of several input planes with a kernel size of 2 and stride of 2
            nn.MaxPool2d(2, 2),
            
            # Flatten the tensor into a single dimension
            nn.Flatten(),
            
            # Fully connected layer with 6560 input features and 120 output features
            nn.Linear(6560, 120),
            
            # Apply a ReLU activation function
            nn.ReLU(),
            
            # Fully connected layer with 120 input features and 84 output features
            nn.Linear(120, 84),
            
            # Apply a ReLU activation function
            nn.ReLU(),
            
            # Fully connected layer with 84 input features and 6 output features
            nn.Linear(84, 6)
        )

    def forward(self, x):
        # Define the forward pass of the neural network
        return self.layers(x)
