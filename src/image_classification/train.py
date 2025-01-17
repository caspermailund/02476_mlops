"""
This script trains a convolutional neural network (CNN).

The training script uses Hydra to load hyperparameters and paths from a configuration file. 
Loguru is used for logging the model initialization, hyperparameters, and the average loss of each epoch.
Weights and Biases (W&B) also logs hyperparameters and the training progress. Furthermore, W&B saves the model as an artifact.
"""

import torch
import torch.optim as optim
from model import SimpleCNN
from dataloader import get_train_loader
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import os
from loguru import logger
import wandb

# Reading the hyperparameters and paths from the configuration file
@hydra.main(config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig):

    # Initialize W&B
    wandb.init(
        project=cfg.wandb_project_name,
        config={
            "learning_rate": cfg.learning_rate,
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "model": "SimpleCNN",
        },
    )

    # Configure logging
    log_file_path = os.path.abspath(cfg.logging_train_path)
    logger.add(log_file_path, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

    # Initialize the model
    model = SimpleCNN()
    logger.info(f"Model initialized: {model}")

    # Loss function (Cross Entropy Loss for multi-class classification)
    criterion = torch.nn.CrossEntropyLoss()
    logger.info(f"Loss function: {criterion}")

    # Optimizer (Adam optimizer)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    logger.info(f"Optimizer: {optimizer}, Learning Rate: {cfg.learning_rate}")

    # Get the path to the data folder from the config
    image_folder = os.path.abspath(cfg.data_folder)

    # Load the training data using the DataLoader with the batch size from the config
    train_loader = get_train_loader(image_folder=image_folder)
    train_loader = torch.utils.data.DataLoader(train_loader, batch_size=cfg.batch_size, shuffle=True)

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

                # Log loss to W&B
                wandb.log({"epoch": epoch + 1, "loss": loss.detach().item()})

                # Update the progress bar with the average loss so far
                pbar.set_postfix(loss=avg_loss)

        # After each epoch, log the average loss for that epoch
        logger.info(f"Epoch {epoch+1}/{cfg.epochs}, Average Loss: {avg_loss}")

    # Save the trained model as an artifact
    try:
        # Get the model save path from the config
        model_path = os.path.abspath(cfg.model_save_path)
        
        # Save the model state to a temporary file
        torch.save(model.state_dict(), model_path)

        # Log model as a W&B artifact
        artifact = wandb.Artifact("simple_cnn_model", type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)

        # Remove the temporary file after logging to W&B
        os.remove(model_path)

        logger.info("Model saved as W&B artifact successfully.")
    except Exception as e:
        logger.error(f"Error saving model as W&B artifact: {e}")

    # Finish W&B run
    wandb.finish()

# Run the main function
if __name__ == "__main__":
    main()
