import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from image_classification.data import split_data, get_paths


@patch("subprocess.run")
@patch("zipfile.ZipFile")
@patch("image_classification.data.get_paths")
def test_split_data(mock_get_paths, mock_zipfile, mock_subprocess):
    """
    Test the split_data function by mocking external dependencies.
    """
    # Mock the Kaggle download subprocess
    mock_subprocess.return_value = MagicMock()

    # Create a temporary directory to simulate the dataset structure
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock paths for the raw dataset
        mock_paths = {
            "download_path": temp_dir,
            "train_dir": os.path.join(temp_dir, "train"),
            "test_dir": os.path.join(temp_dir, "test"),
            "pc_parts_dir": os.path.join(temp_dir, "pc_parts"),
        }
        os.makedirs(mock_paths["pc_parts_dir"])
        mock_get_paths.return_value = mock_paths

        # Create mock category directories and files
        categories = ["CPU", "GPU", "Motherboard"]
        for category in categories:
            category_dir = os.path.join(mock_paths["pc_parts_dir"], category)
            os.makedirs(category_dir)
            for i in range(10):  # Create 10 mock files per category
                with open(os.path.join(category_dir, f"{category}_{i}.jpg"), "w") as f:
                    f.write("mock image content")

        # Mock the zipfile extraction process
        mock_zipfile.return_value.__enter__.return_value.extractall = MagicMock()

        # Call the function
        split_data(test_size=0.2, random_state=42)

        # Check that the train and test directories were created
        assert os.path.exists(mock_paths["train_dir"])
        assert os.path.exists(mock_paths["test_dir"])

        # Check that files were split correctly
        total_train_files = sum(len(files) for _, _, files in os.walk(mock_paths["train_dir"]))
        total_test_files = sum(len(files) for _, _, files in os.walk(mock_paths["test_dir"]))

        assert total_train_files == 24  # 80% of 30 files
        assert total_test_files == 6  # 20% of 30 files
