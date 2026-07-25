# Multi-Layer Perceptron 
The experiment focuses on image classification using Multi-Layer Perceptrons (MLP) on the Fashion-MNIST dataset, hyperparameter optimization using RandomizedSearchCV, and a demonstration of solving the XOR problem using single-layer vs. multi-layer architectures.

## Overview
1. Preprocessing and flattening 28x28 grayscale images for neural network input.
2. Building and training a baseline MLP model in TensorFlow/Keras.
3. Automated hyperparameter tuning using SciKeras and RandomizedSearchCV with 5-fold cross-validation.
4. Performance evaluation and comparison between baseline and optimized models.
5. Analysis of why Single-Layer Perceptrons fail on non-linearly separable data (XOR gate) and how MLPs resolve the issue.

## Dataset
- **Name:** Fashion-MNIST
- **Training Set:** 60,000 images
- **Test Set:** 10,000 images
- **Input Size:** 28x28 pixels (flattened to 784 inputs)
- **Output Classes:** 10 categories (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)

## Setup and Preprocessing
- Reshaped 28x28 image matrices into 784-dimensional feature vectors.
- Scaled pixel values from [0, 255] to [0, 1].
- Converted class labels into one-hot encoded vectors.

## Model Configurations

### Baseline Model
- **Input:** 784 units
- **Hidden Layers:** Dense (128, ReLU) -> Dense (64, ReLU)
- **Output:** Dense (10, Softmax)
- **Optimizer:** Adam (learning rate = 0.001)
- **Loss:** Categorical Crossentropy
- **Training:** 20 epochs, batch size 32

### Optimized Model (Selected via RandomizedSearchCV)
- **Hidden Layers:** 2 layers
- **Neurons per Layer:** 256
- **Activation Function:** Sigmoid
- **Optimizer:** Adam (learning rate = 0.001)
- **Batch Size:** 32
- **Epochs:** 30
- **Dropout Rate:** 0.0

## Key Results
- **Baseline Test Accuracy:** 88.57%
- **Optimized Test Accuracy:** 89.12%
- **Validation Loss:** Decreased from ~0.318 in baseline to ~0.305 in the optimized model, demonstrating improved stability and reduced overfitting.

## Additional Task: XOR Logic Gate
- **Single-Layer Perceptron (SLP):** Fails to converge because XOR inputs (0,0), (0,1), (1,0), (1,1) are not linearly separable by a single decision boundary in 2D space. The weight updates oscillate continuously without reaching zero error.
- **Multi-Layer Perceptron (MLP):** Successfully constructs a non-linear decision boundary by using hidden neurons to decompose the problem into linear sub-regions (combining OR and NAND logic), achieving 100% classification accuracy.

## Dependencies
- Python 3.8+
- TensorFlow / Keras
- SciKeras
- Scikit-learn
- NumPy
- Matplotlib
- Seaborn

## How to Run
1. Clone the repository:
   git clone https://github.com/un-shreeya/DeepLearning_Lab.git

2. Install required packages:
   pip install tensorflow scikeras scikit-learn numpy matplotlib seaborn

3. Open and execute the notebook in Google Colab or Jupyter Notebook.
