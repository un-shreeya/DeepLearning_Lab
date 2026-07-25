# Banknote Authentication using Single Layer Perceptron
This repository contains the implementation of a Single Layer Perceptron built from scratch in Python to classify banknotes as authentic or forged using the UCI Banknote Authentication dataset.

## Project Overview
The objective of this lab is to understand the mechanics of a single artificial neuron, train it using Rosenblatt's perceptron learning rule, and analyze its performance on a binary classification task.

Key tasks completed:
* Exploratory data analysis (boxplots, feature histograms, and correlation heatmaps).
* Feature normalization using MinMaxScaler.
* Single Layer Perceptron implementation from scratch.
* Benchmarking performance against Scikit-Learn's Perceptron module.
* Evaluating the effect of different learning rates (0.001, 0.01, 0.1) on convergence.

## Key Findings and Observations
* Feature Distributions: Skewness has a very wide spread across the dataset, whereas Curtosis contains several upper outliers up to 18. Normalizing features brings all metrics to a common scale, preventing features with larger values from dominating gradient updates.
* Linear Separability: The model achieves roughly 98.5% classification accuracy on the test set. This confirms that the Banknote dataset is largely linearly separable using combinations of features like Variance and Skewness.
* Impact of Learning Rate: When using zero-initialized weights and a hard step activation function, varying the learning rate across 0.001, 0.01, and 0.1 results in identical error trajectories. This happens because changing the learning rate scales the weight updates proportionally without altering the sign of the dot product or the position of the decision boundary.
* Model Comparison: The scratch perceptron implementation produces results identical to Scikit-Learn's built-in Perceptron module when evaluated under the same initialization and split conditions.

## Dependencies
* Python 3.x
* numpy
* pandas
* matplotlib
* seaborn
* scikit-learn

Install the required packages using:
pip install numpy pandas matplotlib seaborn scikit-learn

## How to Run
1. Place dataset.csv in the root directory.
2. Run the main script:
python banknote_perceptron.py
