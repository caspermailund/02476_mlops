# Use a specific version of Python slim image
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

# Install production dependencies
RUN pip install --no-cache-dir -r requirements.txt --verbose

# Install the project itself
RUN pip install . --no-deps --no-cache-dir --verbose

# Copy the application source code
COPY src/ src/
COPY README.md README.md

# Expose the port your API runs on
EXPOSE 8000

# Set the entrypoint to run the API using uvicorn
ENTRYPOINT ["uvicorn", "src.image_classification.api:app", "--host", "0.0.0.0", "--port", "8000"]
