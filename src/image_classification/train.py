import torch
import torch.optim as optim
from model import SimpleCNN
from dataloader import get_train_loader
from tqdm import tqdm

# Initialize the model
model = SimpleCNN()

# Loss function (Cross Entropy Loss for multi-class classification)
criterion = torch.nn.CrossEntropyLoss()

# Optimizer (Adam optimizer, often a good default choice)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load the training data using the DataLoader
train_loader = get_train_loader(image_folder='../../data/raw/pc_parts_train')

# List to store the loss values for each epoch
losses = []

# Training loop
epochs = 15
for epoch in range(epochs):
    avg_loss = 0.0  # Initialize running average loss for the epoch
    with tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as pbar:
        for iteration, (images, labels) in enumerate(pbar):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Update running average loss
            avg_loss += (loss.detach().item() - avg_loss) / (iteration + 1)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update the progress bar with the average loss so far
            pbar.set_postfix(loss=avg_loss)

    # After each epoch, print the average loss for that epoch
    print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss}')
    
    # Append the loss to the list of losses
    losses.append(avg_loss)

# Save the trained model
torch.save(model.state_dict(), '../../models/pc_parts_cnn.pth')

# Save the loss values to a text file
with open('../../models/losses.txt', 'w') as f:
    for loss in losses:
        f.write(f"{loss}\n")
