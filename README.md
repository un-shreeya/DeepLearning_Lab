# CS3807 — Deep Learning Laboratory
## Experiment 6: End-to-End Study of RNN, LSTM, and GRU for Sequence Learning and Video Understanding
---

## 1. Project Overview

This repository contains the complete implementation, experimental benchmarking, and analysis for **Experiment 6**. The objective of this lab is to establish an end-to-end understanding of recurrent sequence architectures by building, evaluating, and comparing **Vanilla RNN**, **LSTM**, and **GRU** models across three core tasks:

1. **Inertial Sequence Classification:** Human Activity Recognition (HAR) using multi-channel sensor time series from the UCI HAR Dataset.
2. **Video Action Understanding:** Spatial feature extraction using a pretrained CNN (MobileNetV2) coupled with recurrent sequence models (CNN–LSTM / CNN–GRU).
3. **Sequence-to-Sequence Modeling:** Sequence reversal using an Encoder–Decoder architecture with Teacher Forcing.

---

## 2. Objectives & Learning Outcomes

- **Tensor Representation:** Formatting sequential data into standard (N, T, F) dimensional tensors.
- **Recurrent Dynamics:** Implementing Vanilla RNN, LSTM, and GRU architectures under controlled experimental conditions.
- **Theoretical Foundations:** Analyzing Backpropagation Through Time (BPTT), vanishing/exploding gradients, and long-term dependency limits.
- **Hybrid Pipelines:** Extracting spatial embeddings from video frames using CNNs and modeling frame transitions with recurrent units.
- **Seq2Seq Frameworks:** Implementing Encoder–Decoder models and contrasting **Token Accuracy** vs. **Sequence Accuracy**.
- **Model Trade-Offs:** Evaluating performance vs. computational efficiency (Parameter Count vs. Memory vs. Macro F1-score).

---

## Datasets & Experimental Setup

### Primary Dataset: UCI Human Activity Recognition (HAR)
- **Classes (6):** `WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`.
- **Input Tensor Structure:** X in R^(N x 128 x 9)
  - N: Number of sequence windows (1500–3000 subset for lab execution)
  - T = 128: Temporal measurements per window (at 50Hz)
  - F = 9: Tri-axial Body Acceleration, Tri-axial Gyroscope, Tri-axial Total Acceleration
- **Data Split:** 70% Training, 15% Validation, 15% Testing (Strictly untouched test set during model selection).

### Secondary Dataset: UCF101 (Video Subset)
- **Classes (3–5):** `Basketball`, `Biking`, `Walking`, `Running`, `TennisSwing`.
- **Pipeline:** Sample 10 uniform frames per video -> Resize to (224 x 224 x 3) -> Pass through frozen MobileNetV2 (Global Average Pooling) -> Yields feature sequence (B, 10, 1280) -> LSTM/GRU Classifier.

---

## Getting Started

### Prerequisites
Python 3.9+ along with the required dependencies:

pip install numpy pandas matplotlib scikit-learn tensorflow torch torchvision

---

## Summary of Results & Visualizations

The script automatically executes and saves **Plots 1 to 9** directly as `.eps` vector graphics in the `outputs/plots/` folder:

| Plot ID | Description |
| :--- | :--- |
| **Plot 1** | Temporal sensor signals across time steps (T=128) for distinct activity classes. |
| **Plots 2 & 3** | Training and validation Loss & Accuracy curves for Vanilla RNN, LSTM, and GRU. |
| **Plot 4** | 3-Panel Confusion Matrices comparing class-level errors for RNN, LSTM, and GRU. |
| **Plot 5** | Model Performance Comparison bar chart (Accuracy, Macro F1, and Parameter Count). |
| **Plot 6** | Sequence Length Sensitivity Study (T in {32, 64, 128} vs. Test Macro F1). |
| **Plot 7** | Representative grid of sampled video frames (10 frames per video). |
| **Plots 8 & 9** | Training curves and Confusion Matrix for the CNN–LSTM/GRU Video Classifier. |

---

## References
1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
2. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation.
3. Cho, K., et al. (2014). *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation*. EMNLP.
4. Anguita, D., et al. (2013). *A Public Domain Dataset for Human Activity Recognition Using Smartphones*. ESANN.
5. Soomro, K., Zamir, A. R., & Shah, M. (2012). *UCF101: A Dataset of 101 Human Actions Classes From Videos in the Wild*.
