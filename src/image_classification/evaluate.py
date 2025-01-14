import torch
from model import SimpleCNN
from dataloader import get_test_loader
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm 

# Brug CPU
device = torch.device('cpu')

# Load the trained model
model = SimpleCNN()
model.load_state_dict(torch.load('../../models/pc_parts_cnn.pth', map_location=device))
model.to(device)
model.eval() 

# Load the test data
test_loader = get_test_loader(image_folder='../../data/raw/pc_parts_test')

# Lists for true and predicted labels
all_labels = []
all_predictions = []

# Evaluation loop med fremdriftsbjælke
with torch.no_grad():
    with tqdm(test_loader, desc="Evaluating", unit="batch") as pbar:
        for images, labels in pbar:
            # Flyt billeder og labels til CPU
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)  # Find klassen med højeste score

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

# Save results (optional)
with open('../../models/evaluation_report.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy * 100:.2f}%\n\n')
    f.write("Classification Report:\n")
    f.write(class_report)
