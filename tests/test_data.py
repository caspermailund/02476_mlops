import numpy as np
from PIL import Image
import torch
import io
from image_classification.dataloader import CustomDataset, transform_test

class InMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, num_classes=14, image_size=(256, 256), transform=None):
        self.transform = transform
        self.samples = []
        self.classes = [f"class_{i}" for i in range(num_classes)]

        # Generate one image per class using Pillow
        for label in range(num_classes):
            # Create a blank RGB image using Pillow
            image = Image.new("RGB", image_size, (255, 255, 255))  # White image
            image_file = io.BytesIO()
            image.save(image_file, format="JPEG")
            image_file.seek(0)  # Reset file pointer for reading
            self.samples.append((image_file, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_file, label = self.samples[idx]
        image = Image.open(image_file)
        if self.transform:
            image = self.transform(image=np.array(image))["image"]
        return image.float(), label


def test_in_memory_dataset():
    dataset = InMemoryDataset(transform=transform_test)

    # Check dataset length
    assert len(dataset) == 14, "Dataset should contain 14 samples"

    # Check sample shapes and labels
    for i in range(len(dataset)):
        image, label = dataset[i]
        assert image.shape == (3, 256, 256), f"Image shape mismatch: {image.shape}"
        assert 0 <= label < 14, f"Invalid label: {label}"

    print("All tests passed!")


if __name__ == "__main__":
    test_in_memory_dataset()
