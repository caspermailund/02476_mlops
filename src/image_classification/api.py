from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms
from src.image_classification.model import SimpleCNN  # Import your model class

app = FastAPI()

# Define the model path
model_path = "/Users/sepidehsoleimani/Documents/DTU/Spring_25/02476/02476_mlops/trained_model.pth"

# Load the trained model
model = SimpleCNN()  # Initialize the model
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Use model_path variable here
model.eval()  # Set the model to evaluation mode

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize for pretrained models
])

@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    """
    Classify an uploaded image and return the predicted class.
    """
    try:
        # Read and preprocess the image
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")  # Ensure the image is in RGB mode
        image = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension

        # Classification
        with torch.no_grad():
            output = model(image)  # Get model predictions
            _, predicted = torch.max(output, 1)  # Get the predicted class index

        predicted_class = predicted.item()  # Convert to a Python integer
        return {"predicted_class": predicted_class}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def read_root():
    """
    Root endpoint to check the API status.
    """
    return {"message": "Welcome to the Hardware Classification API"}
