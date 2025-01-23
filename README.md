# Image Classification of Computer Hardware Components
A project by Group 2 in 02476 Machine Learning Operations at DTU.

## Project overview
This project focuses on classifying images of various computer hardware components using a Convolutional Neural Network (CNN). To optimize the dataset for better results, the third-party package Albumentations is used for image augmentation. The dataset used for this project is the PC Parts Images Dataset from Kaggle, which contains images of different computer components (e.g., motherboards, RAM, graphics cards, etc.).

## Data description
The dataset initially used for this project is the [PC Parts Images Dataset](https://www.kaggle.com/datasets/asaniczka/pc-parts-images-dataset-classification?select=pc_parts) from Kaggle, consisting of 3,279 images categorized into 14 distinct classes:

- cables
- case
- cpu
- gpu
- hdd
- headset
- keyboard
- microphone
- monitor
- motherboard
- mouse
- ram
- speakers
- webcam

Each class contains between 142 and 298 images. All images has a resolution of 256x256 pixels and formatted in the JPG file type.

## Albumentations (third-party package)
The third-party package used in this project is [Albumentations](https://github.com/albumentations-team/albumentations), which is an image augmentation library written in Python. Albumentation has +70 augmentations that can be applied to the training data, involving transformations such as JPEG compression, blurring, grayscaling, channel shuffling, RGB switching, and more. The documentation can be found [here](https://albumentations.ai/docs/).

## Model
The network is a CNN for image classification with three convolutional layers, each followed by batch normalization and ReLU activation. The filters increase from 32 in the first layer to 64 in the second and 128 in the third. Max pooling is applied after each convolution to reduce spatial dimensions. The output is flattened and passed through two fully connected layers, with 512 units in the first and 14 units in the second for classification. Dropout is used for regularization.

## Project structure
The directory structure of the project is based on the [mlops_template](https://github.com/SkafteNicki/mlops_template) with slight modifications:
```txt
├── .dvc/
│   ├── .gitignore
│   └── config
├── .github/
│   ├── dependabot.yaml
│   └── workflows/
├── configs/
│   ├── .gitkeep
│   └── config.yaml
├── data/
│   ├── processed/
│   └── raw/
├── dockerfiles/
│   ├── api.dockerfile
│   └── train.dockerfile
├── docs/
│   ├── README.md
│   ├── mkdocs.yaml
│   └── source/
├── logs/
│   ├── train_test_split_log.txt
│   └── training_log.txt
├── models/
│   └── .gitkeep
├── notebooks/
│   ├── .gitkeep
│   └── pictures_and_labels.ipynb
├── reports/
│   ├── .gitkeep
│   ├── README.md
│   ├── figures/
│   └── report.py
├── src/
│   └── image_classification/
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   ├── test_model.py
│   └── test_training.py
├── .coveragerc
├── .dvcignore
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── README.md
├── cloudbuild.yaml
├── pyproject.toml
├── requirements.txt
├── requirements_dev.txt
└── tasks.py
```
