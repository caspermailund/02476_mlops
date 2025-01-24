import pytest
import torch
from PIL import Image
from src.image_classification.dataloader import (
    CustomDataset,
    get_train_loader,
    get_test_loader,
    transform_train,
    transform_test,
)


@pytest.fixture
def mock_data_dir(tmp_path):
    """
    Creates a temporary mock dataset directory structure:
    mock/
        class_3/
            mock_image.jpg
        mock_label.txt
    """
    # Create directory structure
    data_dir = tmp_path / "mock"
    class_dir = data_dir / "class_3"
    class_dir.mkdir(parents=True)

    # Add a mock image (256x256 RGB)
    mock_image_path = class_dir / "mock_image.jpg"
    mock_image = Image.new("RGB", (256, 256), color=(255, 0, 0))  # Create a red square image
    mock_image.save(mock_image_path)

    # Add a mock label file (not used directly, but simulates dataset structure)
    (data_dir / "mock_label.txt").write_text("mock_image.jpg,class_3")

    return str(data_dir)


def test_custom_dataset_initialization(mock_data_dir):
    """Test that the CustomDataset initializes properly."""
    dataset = CustomDataset(image_folder=mock_data_dir)
    assert len(dataset) == 1  # Should contain one image
    assert "class_3" in dataset.data.class_to_idx  # Ensure class_3 is in the class mapping
    assert len(dataset.data.class_to_idx) == 1  # Ensure there are 14 classes


def test_custom_dataset_getitem(mock_data_dir):
    """Test that the __getitem__ method works as expected."""
    # Pass the test transformation explicitly
    dataset = CustomDataset(image_folder=mock_data_dir, transform=transform_test)
    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)  # Ensure image is a tensor
    assert image.dtype == torch.float32  # Ensure it's of type float
    assert image.shape == (3, 256, 256)  # Ensure dimensions are 256x256 RGB
    assert label == dataset.data.class_to_idx["class_3"]  # Ensure the label is correctly mapped


def test_custom_dataset_train_transform(mock_data_dir):
    """Test that train transformations are applied correctly."""
    dataset = CustomDataset(image_folder=mock_data_dir, transform=transform_train)
    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)  # Image should be a tensor
    assert image.dtype == torch.float32  # Ensure it's of type float
    assert image.shape == (3, 256, 256)  # Ensure dimensions are 256x256 RGB
    assert label == dataset.data.class_to_idx["class_3"]  # Ensure the label is correctly mapped


def test_custom_dataset_test_transform(mock_data_dir):
    """Test that test transformations are applied correctly."""
    dataset = CustomDataset(image_folder=mock_data_dir, transform=transform_test)
    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)  # Image should be a tensor
    assert image.dtype == torch.float32  # Ensure it's of type float
    assert image.shape == (3, 256, 256)  # Ensure dimensions are 256x256 RGB
    assert label == dataset.data.class_to_idx["class_3"]  # Ensure the label is correctly mapped


def test_get_train_loader(mock_data_dir):
    """Test the get_train_loader function."""
    train_loader = get_train_loader(image_folder=mock_data_dir)
    assert len(train_loader) == 1  # Only one image in mock dataset
    image, label = train_loader[0]

    assert isinstance(image, torch.Tensor)  # Ensure image is a tensor
    assert image.shape == (3, 256, 256)  # Ensure dimensions are 256x256 RGB
    assert label == train_loader.data.class_to_idx["class_3"]  # Ensure label matches class_3


def test_get_test_loader(mock_data_dir):
    """Test the get_test_loader function."""
    test_loader = get_test_loader(image_folder=mock_data_dir)
    assert len(test_loader) == 1  # Only one image in mock dataset
    image, label = test_loader[0]

    assert isinstance(image, torch.Tensor)  # Ensure image is a tensor
    assert image.shape == (3, 256, 256)  # Ensure dimensions are 256x256 RGB
    assert label == test_loader.data.class_to_idx["class_3"]  # Ensure label matches class_3
