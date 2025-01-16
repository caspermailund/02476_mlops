from fastapi import FastAPI, File, UploadFile
import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load the trained model (ensure model is in eval mode)
model = torch.load("model.pth")
model.eval()

# Define image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Inference function
def predict(image: Image.Image):
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image_tensor)
    _, predicted_class = torch.max(outputs, 1)
    return predicted_class.item()

# Define endpoint for image upload and prediction
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    # Read the uploaded image file
    image_data = await file.read()
    image = Image.open(BytesIO(image_data))

    # Predict the class of the image
    class_idx = predict(image)

    return {"predicted_class": class_idx}
