from fastapi import FastAPI

# Create a FastAPI app instance
app = FastAPI()

# Define a basic route to test the API
@app.get("/")
def read_root():
    return {"message": "Welcome to the API!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
