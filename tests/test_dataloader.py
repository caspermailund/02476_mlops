import pytest
import torch
from src.image_classification.dataloader import CustomDataset, get_train_loader, get_test_loader, transform_train, \
    transform_test


@pytest.fixture
def mock_data_dir():
    """
    Returns the path to the permanently available mock dataset directory.
    """
    return "data/mock/"


def test_custom_dataset_initialization(mock_data_dir):
    """Test that the CustomDataset initializes properly."""
    dataset = CustomDataset(image_folder=mock_data_dir)
    assert len(dataset) == 14  # Should contain 14 images
    assert len(dataset.data.class_to_idx) == 14  # Ensure there are 14 classes
    for i in range(14):
        assert f"class_{i}" in dataset.data.class_to_idx  # Ensure all classes are recognized


def test_custom_dataset_getitem(mock_data_dir):
    """Test that the __getitem__ method works as expected."""
    dataset = CustomDataset(image_folder=mock_data_dir, transform=transform_test)
    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)  # Ensure image is a tensor
    assert image.dtype == torch.float32  # Ensure it's of type float
    assert image.shape == (3, 256, 256)  # Ensure dimensions are 256x256 RGB
    assert label in range(14)  # Label should be between 0 and 13


def test_custom_dataset_train_transform(mock_data_dir):
    """Test that train transformations are applied correctly."""
    dataset = CustomDataset(image_folder=mock_data_dir, transform=transform_train)
    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)  # Image should be a tensor
    assert image.dtype == torch.float32  # Ensure it's of type float
    assert image.shape == (3, 256, 256)  # Ensure dimensions are 256x256 RGB
    assert label in range(14)  # Label should be between 0 and 13


def test_custom_dataset_test_transform(mock_data_dir):
    """Test that test transformations are applied correctly."""
    dataset = CustomDataset(image_folder=mock_data_dir, transform=transform_test)
    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)  # Image should be a tensor
    assert image.dtype == torch.float32  # Ensure it's of type float
    assert image.shape == (3, 256, 256)  # Ensure dimensions are 256x256 RGB
    assert label in range(14)  # Label should be between 0 and 13


def test_get_train_loader(mock_data_dir):
    """Test the get_train_loader function."""
    train_loader = get_train_loader(image_folder=mock_data_dir)
    assert len(train_loader) == 14  # Should contain 14 images
    for i, (image, label) in enumerate(train_loader):
        assert isinstance(image, torch.Tensor)  # Ensure image is a tensor
        assert image.shape == (3, 256, 256)  # Ensure dimensions are 256x256 RGB
        assert label in range(14)  # Label should be between 0 and 13


def test_get_test_loader(mock_data_dir):
    """Test the get_test_loader function."""
    test_loader = get_test_loader(image_folder=mock_data_dir)
    assert len(test_loader) == 14  # Should contain 14 images
    for i, (image, label) in enumerate(test_loader):
        assert isinstance(image, torch.Tensor)  # Ensure image is a tensor
        assert image.shape == (3, 256, 256)  # Ensure dimensions are 256x256 RGB
        assert label in range(14)  # Label should be between 0 and 13
