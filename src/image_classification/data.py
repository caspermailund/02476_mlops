""" 
This script retrieves the PC Parts dataset, extracts its contents from a zip file, and organizes the data into training and testing sets. 
The dataset is split into 80% training data and 20% testing data. 
The resulting files are stored in dedicated 'train' and 'test' directories. 
"""

import subprocess
import os
import zipfile
import logging
from sklearn.model_selection import train_test_split
import shutil

# Configure logging
logging.basicConfig(filename='../../logs/train_test_split_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths
download_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/raw")
train_dir = os.path.join(download_path, "pc_parts_train")
test_dir = os.path.join(download_path, "pc_parts_test")
pc_parts_dir = os.path.join(download_path, "pc_parts")

# Download dataset (if needed)
subprocess.run(["kaggle", "datasets", "download", "asaniczka/pc-parts-images-dataset-classification", "-p", download_path])

# Extract dataset (if needed)
zip_path = os.path.join(download_path, 'pc-parts-images-dataset-classification.zip')
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(download_path)

# Create train/test directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

total_train_files = 0
total_test_files = 0

# Iterate through each category folder
for category in os.listdir(pc_parts_dir):
    category_path = os.path.join(pc_parts_dir, category)
    
    if os.path.isdir(category_path):
        # Get all .jpg files in the category folder
        image_files = [f for f in os.listdir(category_path) if f.endswith('.jpg')]
        
        # Split the images into train and test
        train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)
        
        # Create category subdirectories in train and test directories
        train_category_dir = os.path.join(train_dir, category)
        test_category_dir = os.path.join(test_dir, category)
        
        os.makedirs(train_category_dir, exist_ok=True)
        os.makedirs(test_category_dir, exist_ok=True)
        
        # Move the files to the respective directories
        for file in train_files:
            shutil.copy(os.path.join(category_path, file), os.path.join(train_category_dir, file))
        
        for file in test_files:
            shutil.copy(os.path.join(category_path, file), os.path.join(test_category_dir, file))
        
        # Update counters
        total_train_files += len(train_files)
        total_test_files += len(test_files)

# Log the split
logging.info(f"Total training files: {total_train_files}")
logging.info(f"Total testing files: {total_test_files}")
