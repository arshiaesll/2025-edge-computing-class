import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from contextlib import nullcontext
import uuid


class VGG(nn.Module):
    """
    A custom neural network architecture implemented using PyTorch.
    This class inherits from nn.Module, which is the base class for all neural network modules in PyTorch.
    """
    
    def __init__(self, input_channels, num_classes):
        """
        Initialize the network architecture.
        """
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size = 3, padding = 1, stride = 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25)
        )

        # self.block3 = nn.Sequential(
        #     nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )

        # self.block4 = nn.Sequential(
        #     nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),    
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        # self.block5 = nn.Sequential(
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),    
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes)  # Direct mapping from channels to classes
        )


    def forward(self, x):
        """
        Define the forward pass of the network. 
        """
        x = self.block1(x)
        x = self.block2(x)
        # x = self.block3(x)
        # x = self.block4(x)
        # x = self.block5(x)
        x = self.classifier(x)
        return x



class ImageDataset(Dataset):
    """
    Custom Dataset for loading image data from CSV file.
    The CSV file should contain image pixel values and labels.
    """
    def __init__(self, csv_file, transform=None, test=False):
        """
        Args:
            csv_file (str): Path to the csv file with image data
            transform (callable, optional): Optional transform to be applied on a sample
            test (bool): Whether this is test data (no labels)
        """
        # Read the CSV file
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.test = test
        
        # Separate features (images) and labels
        if not test:
            self.images = self.data.iloc[:, 2:].values  # Skip id and label columns
            self.labels = self.data.iloc[:, 1].values   # Labels are in second column
        else:
            self.images = self.data.iloc[:, 1:].values  # Skip id column
        
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
        For test data, returns (image, 0) as a dummy label
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # First reshape to (32, 32, 3) for proper image layout
        image = self.images[idx].reshape(32, 32, 3).astype(np.uint8)
        
        # Get label if not test data, otherwise use dummy label
        label = self.labels[idx] if not self.test else 0

        # Convert to tensor and transpose to (3, 32, 32) for PyTorch
        image = torch.from_numpy(image).float() / 255.0
        image = image.permute(2, 0, 1)  # Change from (H,W,C) to (C,H,W)
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)

        return image, label

    def save_sample(self, idx):
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
        # plt.savefig(f'sample_{idx}.png')
        plt.imsave(f'sample_{idx}.png', image_np)


# Implement the datasets for train and test

class ModelManager:
    def __init__(self):
        print("Initializing Model")
        self.model = VGG(input_channels=3, num_classes=10)
        print("Initializing Train Dataset")
        self.train_dataset = ImageDataset('./competition_data/train.csv')
        self.batch_size = 32
        self.lr = 0.0001
        self.epochs = 100
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Device selection logic
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("Using CUDA GPU")
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("Using Apple Metal (MPS)")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")
            
        # Move model to the selected device
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        self.val_dataset = ImageDataset('./competition_data/val.csv')
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        os.makedirs('./models', exist_ok=True)
        self.model_save_path = './models'

    def train_model(self):
        train_lossses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        train_total_steps = len(self.train_dataloader)
        val_total_steps = len(self.val_dataloader)

        for epoch in range(self.epochs):
            self.model.train(True)

            train_loss, train_acc = self.train_validate(epoch, self.epochs, train_total_steps)
            val_loss, val_acc = self.train_validate(epoch, self.epochs, val_total_steps, validation=True)
            print("--------------------------------")
            print(f"End of Epoch {epoch + 1} / {self.epochs}")
            print (f"Epoch {epoch + 1} / {self.epochs} --> Val Loss: {val_loss} Val Acc: {val_acc}")
            print("--------------------------------")
            self.save_best_model(val_acc, val_accuracies, epoch)
            train_lossses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)


        return train_lossses, train_accuracies, val_losses, val_accuracies

    def train_validate(self, epoch: int, total_epochs: int, total_steps: int, validation : bool = False):
        
        running_acc = 0.0
        running_loss = 0.0
        
        # Set model to eval mode during validation
        self.model.train(not validation)
        
        context = torch.no_grad() if validation else nullcontext()
        with context:
            for i, (images, labels) in enumerate(self.val_dataloader if validation else self.train_dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Only zero gradients and do backward pass during training
                if not validation:
                    self.optimizer.zero_grad()
                
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                accuracy = self.compute_accuracy(outputs, labels)

                # Only do backward pass and optimization during training
                if not validation:
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item()
                running_acc += accuracy

                if (i + 1) % 100 == 0:
                    print(
                        f"{'Validation' if validation else 'Training'} --> " +
                        f"Epoch {epoch + 1} / {total_epochs} " +
                        f"Step {i + 1} / {total_steps} " +
                        f"Loss: {running_loss / (i+1):.4f} " +
                        f"Accuracy: {running_acc / (i+1):.4f}"
                    )
            
            running_loss = running_loss / total_steps
            running_acc = running_acc / total_steps
            return running_loss, running_acc


    def save_best_model(self, val_acc, val_accuracies, epoch):
        
        model_name = f'Model_MiniVGG_{epoch + 1}_{str(uuid.uuid4())[:4]}.pt'
        if len(val_accuracies) == 0:
            pass
        elif val_acc > max(val_accuracies):
            torch.save(self.model.state_dict(), os.path.join(self.model_save_path, model_name)) 
            with open('summary.txt', 'a') as f:
                f.write(f'{model_name},{val_acc:.4f}\n')



    def compute_accuracy(self, outputs, labels):
        predictions = torch.argmax(outputs, dim=1)

        num_predictions = len(predictions)
        num_incorrect = torch.count_nonzero(predictions - labels)

        accuracy = (num_predictions - num_incorrect) / num_predictions
        return accuracy




if __name__ == "__main__":
    manager = ModelManager()
    train_lossses, train_accuracies, val_losses, val_accuracies = manager.train_model()
    # vgg = VGG(input_channels=3, num_classes=10)
    # dataset = ImageDataset('./competition_data/train.csv')
    # dataset.save_sample(0)
    # summary(vgg, (3, 224, 224))