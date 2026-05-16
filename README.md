# Handwritten Digit Recognition using CNN (MNIST)
## Project Overview

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset.

The main goal of the project is to compare the performance of two optimizers:

- Adam Optimizer
- SGD Optimizer

The notebook includes:
- Data preprocessing
- Dataset splitting
- Building CNN architecture
- Training and validation
- Model evaluation
- Accuracy and loss visualization
- Final comparison between both experiments


## Google Colab Notebook

[Open in Colab](https://colab.research.google.com/drive/13ZtZCrl6KoZcZZqRr0E3yXjXycZNi34R?usp=sharing)
# Dataset Information
## MNIST Dataset

The MNIST dataset contains grayscale handwritten digit images.

Dataset Details:
- 60,000 training images
- 10,000 testing images
- Image size: 28×28
- 10 classes (digits from 0 to 9)

## Dataset Source

- MNIST Official Website:
  http://yann.lecun.com/exdb/mnist/

- PyTorch MNIST Documentation:
  https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html

The dataset is downloaded automatically using torchvision.datasets.MNIST.

# Libraries Used

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing

The preprocessing pipeline includes:

## 1. Resize Images

transforms.Resize((28, 28))

All images are resized to 28×28.


## 2. Convert Images to Tensor
transforms.ToTensor()
This converts image pixels into tensors.


## 3. Normalize Images

transforms.Normalize((0.5,), (0.5,))

Normalization helps the model train faster and improves stability.

## Transform Pipeline


transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataset Splitting

The training dataset is divided into:
- 80% Training Data
- 20% Validation Data

Using:

random_split(dataset, [train_size, val_size])

# DataLoader Configuration


batch_size = 32


Three dataloaders are created:
- train_loader
- val_loader
- test_loader

# CNN Architecture
## Model Architecture

Input Image : 1 × 28× 28

Conv Block 1:
- Conv2D (32 filters)
- BatchNorm
- ReLU
- MaxPooling

Conv Block 2:
- Conv2D (64 filters)
- BatchNorm
- ReLU
- MaxPooling

Conv Block 3:
- Conv2D (128 filters)
- BatchNorm
- ReLU
- MaxPooling

Flatten Layer

Dropout Layer (0.5)

Fully Connected Layer:
- 256 neurons

Output Layer:
- 10 neurons

## CNN Model

class SimpleCNN(nn.Module):

# Techniques Used

- Convolutional Neural Networks (CNN)
- Batch Normalization
- Dropout
- ReLU Activation
- MaxPooling
- CrossEntropyLoss

# Optimizers Used

## Experiment 1 — Adam Optimizer


optim.Adam(model.parameters(), lr=0.001)

Adam provides adaptive learning rates and fast convergence.

## Experiment 2 — SGD Optimizer

optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

Momentum was used with SGD to improve convergence stability and training performance.

# Training Configuration
epochs = 7
batch_size = 32
learning_rate = 0.001


During training, the notebook stores:
- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy


# Evaluation Phase

Both models are evaluated on the test dataset using:
model.eval()
Metrics calculated:
- Test Accuracy
- Final Loss

Testing is performed using:

with torch.no_grad():

# Visualization

The notebook plots:

## 1. Loss Curves
- Adam Train Loss
- Adam Validation Loss
- SGD Train Loss
- SGD Validation Loss

## 2. Accuracy Curves
- Adam Train Accuracy
- Adam Validation Accuracy
- SGD Train Accuracy
- SGD Validation Accuracy

Using Matplotlib.

# Final Comparison

At the end of the notebook, both optimizers are compared based on:
- Accuracy
- Final Loss

Example:

| Model | Accuracy | Loss |
CNN + Adam	98.97%	0.0307
CNN + SGD	98.94%	0.0303

# Key Concepts Used

- Deep Learning
- Convolutional Neural Networks (CNN)
- Batch Normalization
- Dropout
- ReLU Activation
- MaxPooling
- Cross Entropy Loss
- Adam Optimizer
- SGD Optimizer
- Training & Validation
- PyTorch

# How to Run the Project

## 1. Open the notebook in Google Colab or Jupyter Notebook

## 2. Install required libraries if needed

```bash
pip install torch torchvision matplotlib pandas
```

## 3. Run all notebook cells sequentially

The notebook will:
- Download the dataset
- Train the CNN model
- Validate the model
- Test the model
- Generate graphs
- Print final comparison results


# Future Improvements

- Add ResNet model
- Add AlexNet architecture
- Add confusion matrix visualization
- Use data augmentation
- Tune hyperparameters

