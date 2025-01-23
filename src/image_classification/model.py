import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # First convolutional layer (3 input channels for RGB images, 32 output channels)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # BatchNorm layer
        # Second convolutional layer (32 input channels, 64 output channels)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # BatchNorm layer
        # Third convolutional layer (64 input channels, 128 output channels)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # BatchNorm layer

        # Max pooling layer to reduce spatial dimensions by half
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layer, input size is adjusted based on 256x256 image input
        self.fc1 = nn.Linear(128 * 32 * 32, 512)  # Output size after pooling is 128x32x32
        self.fc2 = nn.Linear(512, 14)  # Output layer with 14 units for classification

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)  # Dropout with 50% probability

    def forward(self, x):
        # Apply first convolution, batch normalization, ReLU, and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Apply second convolution, batch normalization, ReLU, and pooling
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Apply third convolution, batch normalization, ReLU, and pooling
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten the output from the convolutional layers
        x = x.view(-1, 128 * 32 * 32)  # Flatten 128x32x32 into a 1D tensor

        # Pass the flattened tensor through the fully connected layers
        x = F.relu(self.fc1(x))

        # Apply dropout to the fully connected layer
        x = self.dropout(x)

        x = self.fc2(x)  # Output layer with 14 units for classification

        return x  # Output predictions for class probabilities
