CIFAR-10 CNN Image Classification Experiment
---------------------------------------------
Overview:
---------
This project implements a Convolutional Neural Network (CNN) from scratch using TensorFlow and Keras to classify images from the CIFAR-10 dataset into 10 distinct categories. The repository includes code for data loading, model architecture building, training over 20 epochs, evaluation metrics computation, and feature map/confusion matrix visualizations.

Dataset:
--------
Dataset: CIFAR-10
Classes: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
Split: 50,000 training images and 10,000 test images (evenly distributed, 5,000 per class)
Image Resolution: 32x32 pixels, RGB color channels

Model Architecture:
--------------------
Conv2D Layer 1: 32 filters, 3x3 kernel, ReLU activation
MaxPooling2D Layer 1
Conv2D Layer 2: 64 filters, 3x3 kernel, ReLU activation
MaxPooling2D Layer 2
Flatten Layer
Dense Layer: 64 units, ReLU activation
Output Layer: 10 units, Softmax activation
Total Parameters: 167,562

Requirements:
-------------
Python 3.x
TensorFlow / Keras
Matplotlib

NumPy

Scikit-learn
