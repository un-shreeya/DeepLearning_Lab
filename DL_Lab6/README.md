# Experiment 6: End-to-End Study of RNN, LSTM, and GRU for Sequence Learning and Video Understanding

---

## 1. Overview & Objectives

This experiment provides an end-to-end practical understanding of sequential deep learning architectures across three primary tasks:

1. **Sensor-based Sequence Classification:** Building and comparing **Vanilla RNN**, **LSTM**, and **GRU** models using raw inertial signals from smartphone sensors.
2. **Video Action Recognition:** Extracting spatial features using a pretrained CNN (**MobileNetV2**) and capturing temporal dependencies across frames using **LSTM** and **GRU**.
3. **Sequence-to-Sequence (Seq2Seq) Learning:** Constructing an **Encoder-Decoder architecture** on a synthetic sequence reversal task to understand context vector transmission and token vs. sequence-level evaluation metrics.

---

## 2. Learning Outcomes

By completing this lab, you will learn to:
* Format sequential data into standard **$(N, T, F)$** tensors (Batch Size, Sequence Length, Features).
* Implement and train Vanilla RNN, LSTM, and GRU architectures using TensorFlow/Keras.
* Analyze BPTT (Backpropagation Through Time), vanishing/exploding gradients, and gate mechanics (Forget, Input, Output, Reset, Update gates).
* Implement feature extraction pipelines using pretrained CNNs (e.g., MobileNetV2) for video frame sequences.
* Evaluate sequential models using macro-averaged metrics (Precision, Recall, F1-Score) and confusion matrices.
* Understand the distinction between **Token Accuracy** and **Sequence Accuracy** in Seq2Seq models.

---

## 3. Datasets & System Setup

### Datasets Used
1. **UCI Human Activity Recognition (HAR) Dataset:**
   * **Source:** UCI Machine Learning Repository
   * **Input Shape:** $(N, 128, 9)$ representing 128 time steps across 9 raw inertial signal channels (3-axis body acceleration, 3-axis gyroscope, 3-axis total acceleration).
   * **Classes (6):** `WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`.
2. **UCF101 Subset (Video Classification):**
   * **Subset Selection:** 3–5 action classes (e.g., `Basketball`, `Biking`, `JumpRope`, `TennisSwing`, `WalkingWithDog`).
   * **Input Shape:** 10 uniformly sampled frames per video resized to $224 \times 224 \times 3$.
3. **Synthetic Reversal Dataset (Seq2Seq):**
   * **Task:** Sequence reversal (e.g., `[1, 4, 7, 2]` $\rightarrow$ `[2, 7, 4, 1]`).

---

## 4. Overall Pipeline Architecture
[ Sensor Signals / Video Frames ]
│
▼
[ Data Preprocessing / CNN Feature Extraction ]
│
▼
[ Tensor Shape: (N, T, F) ]
│
▼
[ Recurrent Layer: RNN / LSTM / GRU (32 units) ]
│
▼
[ Dropout (0.2) + Dense (16 ReLU) ]
│
▼
[ Dense (Softmax Output) ] ──► [ Activity / Action Prediction ]

---

## 5. Experimental Tasks & Workflow

### Task 1: Human Activity Recognition (UCI HAR)
1. **Preprocessing:**
   * Load 9 raw inertial signal channels.
   * Format input into tensors of shape $(N, 128, 9)$.
   * Split data into 70% Train, 15% Validation, and 15% Test partitions.
   * Standardize features using mean and variance calculated strictly from the training set.
2. **Model Architectures:**
   * **Vanilla RNN:** `SimpleRNN(32)` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Dense(16, relu)` $\rightarrow$ `Dense(6, softmax)`
   * **LSTM:** `LSTM(32)` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Dense(16, relu)` $\rightarrow$ `Dense(6, softmax)`
   * **GRU:** `GRU(32)` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Dense(16, relu)` $\rightarrow$ `Dense(6, softmax)`
3. **Training Setup:**
   * **Optimizer:** Adam ($\text{lr} = 10^{-3}$)
   * **Loss:** Sparse Categorical Cross-Entropy
   * **Batch Size:** 32
   * **Epochs:** 30
4. **Sequence Length Study:**
   * Evaluate model behavior across truncated sequence lengths: $T \in \{32, 64, 128\}$.

### Task 2: Video Understanding (CNN + RNN)
1. Extract 10 uniform frames per video clip ($224 \times 224 \times 3$).
2. Pass frames through a frozen **MobileNetV2** (with classification head removed) to extract spatial feature vectors of dimension $D$.
3. Pass frame feature sequences $(B, 10, D)$ through `LSTM(32)` or `GRU(32)` layers to predict video action labels.

### Task 3: Sequence-to-Sequence Learning
1. Implement an **Encoder-Decoder LSTM** framework for integer sequence reversal.
2. Evaluate performance using both **Token Accuracy** and **Sequence Accuracy**.

---
### Required Plots
* **Plot 1:** Temporal Sensor Signal vs. Time (for 3 representative channels across activities).
* **Plots 2 & 3:** Training vs. Validation Loss and Accuracy curves (RNN, LSTM, GRU).
* **Plot 4:** Confusion Matrices for RNN, LSTM, and GRU on HAR test set.
* **Plot 5:** Comparative Bar Plot (Accuracy, Macro F1, Training Time).
* **Plot 6:** Sequence Length ($T \in \{32, 64, 128\}$) vs. Test F1-Score.
* **Plots 7, 8, & 9:** Video sample frames, training/validation curves, and video confusion matrices.
---
