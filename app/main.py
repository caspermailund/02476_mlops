from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import io
import torch.nn.functional as F
from src.image_classification.model import SimpleCNN

# Initialize the FastAPI app
app = FastAPI()

# Load the model
MODEL_PATH = "models/simple_cnn.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

@app.get("/")
def read_root():
    """Health check endpoint."""
    return {"message": "API is running and ready for inference"}

@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    """Endpoint for image classification."""
    try:
        # Read and process the uploaded image
        image_data = await image.read()  # Read the file content
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")  # Open the image
        tensor_image = transform(pil_image).unsqueeze(0).to(device)  # Transform and add batch dimension

        # Perform inference
        with torch.no_grad():
            outputs = model(tensor_image)
            probabilities = F.softmax(outputs[0], dim=0)
            predicted_class = probabilities.argmax().item()

        # Send response
        return JSONResponse(content={
            "predicted_class": predicted_class,
            "probabilities": probabilities.tolist()
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
