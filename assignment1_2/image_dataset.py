import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Example usage
if __name__ == "__main__":
    # Create dataset instance
    dataset = ImageDataset('./competition_data/train.csv')
    
    print(f"Dataset size: {len(dataset)}")
    
    # Show a few random samples
    import random
    print("\nDisplaying 3 random samples from the dataset:")
    for _ in range(3):
        idx = random.randint(0, len(dataset)-1)
        dataset.show_sample(idx)
        print(f"Image index: {idx}") 