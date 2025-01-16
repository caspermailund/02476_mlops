import torch
import torch.optim as optim
from image_classification.model import SimpleCNN

# Reading the hyperparameters and paths from the configuration file
def test_training():

    # Initialize the model
    model = SimpleCNN()

    # Loss function (Cross Entropy Loss for multi-class classification)
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer (Adam optimizer, often a good default choice)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    optimizer.zero_grad()

    # Forward pass
    outputs = model(torch.randn([1, 3, 256, 256]))
    labels = torch.zeros([1, 14])
    labels[0, 0] = 1
    loss = criterion(outputs, labels)
    assert loss != 0
