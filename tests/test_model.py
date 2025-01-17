import torch
from image_classification.model import SimpleCNN


def test_model():
    model = SimpleCNN()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    assert y.shape == (1, 14)
