# CS3807 Deep Learning Laboratory - Experiment 5
Comprehensive Study of CNN Training, Regularization, Optimization, Hyperparameter Tuning, Transfer Learning, and Cross-Validation

## Overview
This repository contains the code, analysis, and final report for Experiment 5 of the B.Tech Artificial Intelligence and Data Science curriculum at Shiv Nadar University Chennai. The experiment explores deep learning workflows using a MobileNetV2 architecture trained on the Oxford-IIIT Pet dataset.

## Objectives
- Investigate the impact of various weight initialization strategies on convergence and training stability.
- Analyze overfitting dynamics and evaluate regularization techniques (Dropout, L2 Regularization, and Batch Normalization).
- Compare the performance of multiple optimization algorithms (SGD, Momentum, RMSprop, and Adam).
- Perform hyperparameter tuning on learning rates, batch sizes, and architectural settings.
- Implement transfer learning and fine-tuning strategies utilizing a pretrained MobileNetV2 base.
- Execute 5-fold cross-validation for model selection and evaluate final generalization performance on an independent test set.

## Dataset
- **Oxford-IIIT Pet Dataset:** Consists of RGB images spanning 37 different pet breeds. All images are resized to 224x224x3 to match the input requirements of MobileNetV2.

## Experimental Structure
1. **Weight Initialization:** Evaluation of zero, random, Xavier/Glorot, and He initializations.
2. **Regularization Analysis:** Monitoring training and validation curves to identify and mitigate overfitting.
3. **Batch Normalization:** Mathematical formulation and empirical evaluation of mini-batch normalization layers.
4. **Optimizers:** Comparative benchmarking of SGD, Momentum, RMSprop, and Adam.
5. **Hyperparameter Tuning:** Systematic analysis of learning rates, batch sizes, and dropout configurations.
6. **Transfer Learning & Fine-Tuning:** Contrasting strict feature extraction against unfreezing upper network layers with a reduced learning rate.
7. **5-Fold Cross-Validation:** Robust model selection across configurations C1 through C4.
8. **Final Evaluation:** Comprehensive performance analysis on the untouched test set using confusion matrices and error analysis.
9. **Additional Exercise:** Supplementary evaluation of alternative hyperparameter configurations (Configurations C1 and C4) against the primary selected model (C3).

## Requirements
- Python 3.x
- PyTorch / TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
