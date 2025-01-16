import torch
import torch.optim as optim
from model import SimpleCNN
from dataloader import get_train_loader
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import os

# Reading the hyperparameters and paths from the configuration file
@hydra.main(config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig):

    # Initialize the model
    model = SimpleCNN()

    # Loss function (Cross Entropy Loss for multi-class classification)
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer (Adam optimizer, often a good default choice)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # Get the path to the data folder from the config
    image_folder = os.path.abspath(cfg.data_folder)  

    # Load the training data using the DataLoader with the batch size from the config
    train_loader = get_train_loader(image_folder=image_folder)
    train_loader = torch.utils.data.DataLoader(train_loader, batch_size=cfg.batch_size, shuffle=True)

    # List to store the loss values for each epoch
    losses = []

    # Training loop
    for epoch in range(cfg.epochs):
        avg_loss = 0.0  # Initialize running average loss for the epoch
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg.epochs}', unit='batch') as pbar:
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
        print(f'Epoch {epoch+1}/{cfg.epochs}, Average Loss: {avg_loss}')
        
        # Append the loss to the list of losses
        losses.append(avg_loss)

    # Save the trained model using the path from the config file
    model_save_path = os.path.abspath(cfg.model_save_path)  
    torch.save(model.state_dict(), model_save_path)

    # Save the loss values to a text file
    with open('../../models/losses.txt', 'w') as f:
        for loss in losses:
            f.write(f"{loss}\n")

# Run the main function
if __name__ == "__main__":
    main()
