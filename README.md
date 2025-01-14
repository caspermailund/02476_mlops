# Image classification of computer hardware components
A project by Group 2 in 02476 Machine Learning Operations at DTU.

## Project overview
This project focuses on classifying images of various computer hardware components using a simple simple convolutional neural network (CNN). To optimize the dataset for better results, the third-party package Albumentations is used for image augmentation. The dataset used for this project is the PC Parts Images Dataset from Kaggle, which contains images of different computer components (e.g., motherboards, RAM, graphics cards, etc.).

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

The total size of the dataset is approximately 36 MB, where each image has a resolution of 256x256 pixels and formatted in the JPG file type. This consistency allows for simple application of data augmentation techniques to the images, creating a larger and more diverse datasets. 
Each class contains between 142 and 298 images, providing a relatively balanced representation of PC parts which is needed for training a robust model and reduces the models likelihood of bias towards a specific class. This dataset is therefore, very suitable for a multi class classification task. 

## Albumentations (third-party package)
The third-party package used in this project is [Albumentations](https://github.com/albumentations-team/albumentations), which is an image augmentation library written in Python. Albumentation has +70 augmentations that can be applied to the training data, involving transformations such as JPEG compression, blurring, grayscaling, channel shuffling, RGB switching, and more. The documentation can be found [here](https://albumentations.ai/docs/). 

## Models
Our primary focus will be on developing a CNN for the image classification task. The CNN will include layers for feature extraction (convolutional and pooling layers) and fully connected layers for classification. To optimize performance, we will experiment with the network architecture and hyperparameter settings using the tools provided in 02476. If needed, we might explore transfer learning using pre-trained models such as MobileNet or ResNet and fine-tuning them. An alternative experiment might involve training two identical models, one leveraging Albumentations and the other not, and comparing their results to assess Albumentations overall contribution to performance.

## Project structure
The directory structure of the project is based on the [mlops_template](https://github.com/SkafteNicki/mlops_template). It looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```
