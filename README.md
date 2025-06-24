# DEEP-LEARNING-PROJECT
*COMPANY*: CODETECH IT SOLUTIONS

*NAME*: ANKEPALLI SHIVA SUPRAJA

*INTERN ID*: CT08DN1734

*DOMAIN*: DATA SCIENCE

*DURATION*: 8 WEEKS

*MENTOR*: NEELA SANTHOSH KUMAR

## Overview:

To build, train, and evaluate a CNN model that classifies images from the CIFAR-10 dataset into one of 10 categories like airplane, car, bird, etc.

This project builds a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify images from the CIFAR-10 dataset, which contains 60,000 32x32 color images across 10 different categories.

## Dataset Used: CIFAR-10:

*Training Set*: 50,000 images

*Test Set*: 10,000 images

*Image Size*: 32x32 pixels, 3 color channels (RGB)

*Classes*:

    airplane

    automobile

    bird

    cat

    deer

    dog

    frog

    horse

    ship

    truck

## Model Architecture:

A sequential CNN model with the following layers:

1.Conv2D (32 filters, 3x3) – feature extraction

2.MaxPooling2D (2x2) – reduces spatial size

3.Conv2D (64 filters, 3x3) – deeper features

4.MaxPooling2D (2x2) – downsampling

5.Conv2D (64 filters, 3x3) – more features

6.Flatten – converts 2D to 1D

7.Dense (64 units, ReLU) – fully connected

8.Dense (10 units, Softmax) – output layer for 10 classes

## Model Compilation and Training:

->Loss Function: sparse_categorical_crossentropy

->Optimizer: adam

->Metrics: accuracy

->Epochs: 10

->Batch Size: 64

->Validation: Done on the test set during training

## Evaluation and Visualization:

1.The model is evaluated on the test set for final accuracy.

2.Accuracy and loss graphs are plotted over epochs to show learning progress.

3.Sample predictions are shown for the first 10 test images, with:

->Green titles: correct predictions

->Red titles: incorrect predictions

## Goal of this project:

To demonstrate how a basic CNN can learn to classify real-world images into predefined categories, and to visualize both performance metrics and actual predictions.

## Required Libraries:

Make sure you have the following Python packages installed:

     tensorflow

     numpy

     matplotlib

You can install them all at once using:

    pip install tensorflow numpy matplotlib

### requirements.txt

tensorflow==2.15.0     # or latest stable version

numpy

matplotlib

Then install using:

     pip install -r requirements.txt

## How to Run the Code:

1.Install the above libraries and dependencies

2.Run the script:

    python main.py

This will:

Download the CIFAR-10 dataset

Train the CNN model

Show training progress

Evaluate the model

Display example predictions

3.Output:

Training and validation accuracy/loss graphs

Test accuracy

Visual predictions for first 10 test images

## Applications:

This project forms the foundation for:

1.Object detection systems

2.Autonomous vehicles (vision)

3.Mobile image classification apps

4.AI-based quality control















