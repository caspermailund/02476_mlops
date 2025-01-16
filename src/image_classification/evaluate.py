import torch
from model import SimpleCNN
from dataloader import get_test_loader
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import os

device = torch.device('cpu')

# Load the trained model
@hydra.main(config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig):
    # Initialize the model
    model = SimpleCNN()
    model.load_state_dict(torch.load(cfg.model_save_path, map_location=device))
    model.to(device)
    model.eval()

    # Load the test data
    test_loader = get_test_loader(image_folder=os.path.abspath(cfg.test_data_folder))
    test_loader = torch.utils.data.DataLoader(test_loader, batch_size=cfg.batch_size, shuffle=False)

    # Lists for true and predicted labels
    all_labels = []
    all_predictions = []

    # Evaluation loop
    with torch.no_grad():
        with tqdm(test_loader, desc="Evaluating", unit="batch") as pbar:
            for images, labels in pbar:
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                _, predictions = torch.max(outputs, 1)  # Find the class with the highest score

                # Collect labels and predictions
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Generate a classification report
    class_report = classification_report(all_labels, all_predictions)
    print("\nClassification Report:\n")
    print(class_report)

    # Save results
    evaluation_save_path = os.path.abspath(cfg.evaluation_save_path)  # Save path from config
    with open(evaluation_save_path, 'w') as f:
        f.write(f'Accuracy: {accuracy * 100:.2f}%\n\n')
        f.write("Classification Report:\n")
        f.write(class_report)

# Run the main function
if __name__ == "__main__":
    main()
