# Comparative Study of Deep CNN Architectures Using Transfer Learning
**Course:** CS3807 Deep Learning Laboratory (Semester V)[cite: 1]  
**Institution:** Shiv Nadar University Chennai[cite: 1]  

---

## Overview
This repository contains the implementation for Experiment 4, focusing on the study and comparison of classic and modern Convolutional Neural Network (CNN) architectures[cite: 1]. It demonstrates feature extraction and fine-tuning techniques on the CIFAR-10 dataset using pre-trained models[cite: 1].

---

## Objectives
- Study the structural evolution of CNN architectures (LeNet-5, AlexNet, VGG16, GoogleNet, ResNet50)[cite: 1].
- Implement transfer learning with ImageNet pre-trained weights[cite: 1].
- Perform fine-tuning on top convolutional layers[cite: 1].
- Evaluate multi-class classification performance using standard metrics and confusion matrices[cite: 1].

---

## Dataset
- **Name:** CIFAR-10[cite: 1]
- **Dimensions:** 32 x 32 x 3[cite: 1]
- **Train / Test Split:** 50,000 / 10,000 images[cite: 1]
- **Classes (10):** Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck[cite: 1]

---

## Experimental Workflow
1. **Dataset Preparation:** Load CIFAR-10 and normalize pixel values to the [0, 1] range[cite: 1].
2. **Transfer Learning Setup:** Load the pre-trained base model without top classification layers, append GlobalAveragePooling2D, Dense ReLU, and Softmax output layers[cite: 1].
3. **Feature Extraction:** Freeze the convolutional base and train only the classification head[cite: 1].
4. **Fine-Tuning:** Unfreeze selected top convolutional blocks and re-train using a lower learning rate[cite: 1].
5. **Model Evaluation:** Generate Precision, Recall, F1-score classification reports, training/validation loss and accuracy curves, and confusion matrices[cite: 1].

---

## Dependencies
- Python 3.x
- TensorFlow / Keras[cite: 1]
- NumPy
- Matplotlib
- Scikit-Learn