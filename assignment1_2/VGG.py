import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class VGG(nn.Module):
    """
    A custom neural network architecture implemented using PyTorch.
    This class inherits from nn.Module, which is the base class for all neural network modules in PyTorch.
    """
    
    def __init__(self, input_channels, num_classes):
        """
        Initialize the network architecture.
        
        Parameters:
        - input_channels (int): Number of input channels (e.g., 3 for RGB images, 1 for grayscale)
        - num_classes (int): Number of output classes for classification
        
        TODO:
        1. Call the parent class constructor using super().__init__()
        2. Define the network layers as class attributes:
           - Convolutional layers (nn.Conv2d)
           - Pooling layers (nn.MaxPool2d)
           - Batch normalization layers (nn.BatchNorm2d)
           - Fully connected layers (nn.Linear)
        3. Define activation functions (e.g., ReLU)
        4. Consider adding dropout layers (nn.Dropout) for regularization
        """
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size= 3, padding=1, stride=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv4 = nn.Conv2d(256, 512, kernel_size = 3, padding = 1, stride = 1)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv5 = nn.Conv2d(512, 512, kernel_size = 3, padding = 1, stride = 1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(512, 512, kernel_size = 3, padding = 1, stride = 1)
        self.relu6 = nn.ReLU()
        self.maxpool6 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.relu7 = nn.ReLU()
        self.dropout1 = nn.Dropout(p = 0.5)

        self.fc2 = nn.Linear(4096, 4096)
        self.relu8 = nn.ReLU()
        self.dropout2 = nn.Dropout(p = 0.5)

        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        """
        Define the forward pass of the network.
        
        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, num_classes)
        
        TODO:
        1. Implement the forward pass using the layers defined in __init__
        2. Process the input through convolutional layers
        3. Apply activation functions after each conv layer
        4. Apply pooling layers as needed
        5. Flatten the tensor before fully connected layers
        6. Pass through fully connected layers
        7. Return the final output
        """
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.maxpool6(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu7(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu8(x)
        x = self.dropout2(x)

        x = self.fc3(x)

        return x



class ImageDataset(Dataset):
    """
    Custom Dataset for loading image data from CSV file.
    The CSV file should contain image pixel values and labels.
    """
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (str): Path to the csv file with image data
            transform (callable, optional): Optional transform to be applied on a sample
        """
        # Read the CSV file
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
        # Separate features (images) and labels
        self.images = self.data.iloc[:, 2:].values  # Skip id and label columns
        self.labels = self.data.iloc[:, 1].values   # Labels are in second column
        
        # Dictionary for class names
        self.class_map = {
            0: 'Airplane', 1: 'Automobile', 2: 'Bird', 3: 'Cat',
            4: 'Deer', 5: 'Dog', 6: 'Frog', 7: 'Horse',
            8: 'Ship', 9: 'Truck'
        }

    def __len__(self):
        """Returns the size of the dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a tuple (image, label) for the given index
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # First reshape to (32, 32, 3) for proper image layout
        image = self.images[idx].reshape(32, 32, 3).astype(np.uint8)
        label = self.labels[idx]

        # Convert to tensor and transpose to (3, 32, 32) for PyTorch
        image = torch.from_numpy(image).float() / 255.0
        image = image.permute(2, 0, 1)  # Change from (H,W,C) to (C,H,W)
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)

        return image, label

    def show_sample(self, idx):
        """
        Display an image from the dataset
        Args:
            idx (int): Index of the image to display
        """
        image, label = self.__getitem__(idx)
        
        # Convert from (3, 32, 32) back to (32, 32, 3) for plotting
        image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        plt.figure(figsize=(4, 4))
        plt.imshow(image_np)
        plt.title(f'Class: {self.class_map[label]}')
        plt.axis('off')
        plt.savefig(f'sample_{idx}.png')
    


if __name__ == "__main__":
    vgg = VGG(input_channels=3, num_classes=10)
    summary(vgg, (3, 224, 224))