"""
This script creates a custom dataset with Albumentations for image augmentations and tensor conversion. 
It is used to load both the training data and the testing data.
"""

import torch
import numpy as np
from torchvision import datasets
from albumentations.pytorch import ToTensorV2
from PIL import Image
import albumentations as A

# Define Albumentations transformations for training
transform_train = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=40, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3),
    A.MotionBlur(blur_limit=5, p=0.3),
    A.RandomShadow(p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    ToTensorV2(),
])

# Define Albumentations transformations for testing
transform_test = A.Compose([
    ToTensorV2(),
])

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, transform=None):
        self.data = datasets.ImageFolder(root=image_folder)
        self.transform = transform

    def __len__(self):
        return len(self.data.samples)

    def __getitem__(self, idx):
        path, label = self.data.samples[idx]
        image = Image.open(path)
        
        if self.transform:
            image = self.transform(image=np.array(image))['image']
        
        image = image.float()  # Convert to float
        return image, label

def get_train_loader(image_folder, transform=transform_train):
    train_data = CustomDataset(image_folder=image_folder, transform=transform)
    return train_data

def get_test_loader(image_folder, transform=transform_test):
    test_data = CustomDataset(image_folder=image_folder, transform=transform)
    return test_data
