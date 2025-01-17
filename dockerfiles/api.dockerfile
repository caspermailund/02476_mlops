# Use a specific version of Python as the base image
FROM python:3.12.7-slim AS base

# Install build dependencies and clean up to reduce image size
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy project files into the container
COPY src src/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

# Install project dependencies from requirements.txt
RUN pip install -r requirements.txt --no-cache-dir --verbose

# Install the project itself (in case you want to package it as a module)
RUN pip install . --no-deps --no-cache-dir --verbose

# Expose port 8000 for the FastAPI app
EXPOSE 8000

# Start the FastAPI application using Uvicorn
ENTRYPOINT ["uvicorn", "src/image_classification/api:app", "--host", "0.0.0.0", "--port", "8000"]
