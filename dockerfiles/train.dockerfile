# Use the official slim Python image as a base
FROM python:3.12-slim AS base

# Install necessary system dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy dependency files first to leverage Docker layer caching
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY pyproject.toml pyproject.toml

# Install project dependencies
RUN pip install --no-cache-dir -r requirements.txt --verbose

# Install the project itself
RUN pip install . --no-deps --no-cache-dir --verbose

# Copy the rest of the application code
COPY src/ src/
COPY README.md README.md

# Specify the default entrypoint for the container
ENTRYPOINT ["python", "-u", "src/image_classification/train.py"]
