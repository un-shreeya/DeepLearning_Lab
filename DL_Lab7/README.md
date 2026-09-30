# CS3807 – Deep Learning Laboratory
## Experiment 7: End-to-End Study of Autoencoders, Convolutional Autoencoders, Denoising Autoencoders, and Variational Autoencoders

### 1. Overview
The primary objective is to build, train, evaluate, and compare four key autoencoder architectures using the MNIST Handwritten Digit dataset:
1. Fully Connected Autoencoder (FC-AE)
2. Convolutional Autoencoder (CAE)
3. Denoising Convolutional Autoencoder (DAE)
4. Variational Autoencoder (VAE)

The experiment covers dataset preprocessing, architectural construction, custom loss formulation, metric-based evaluation, latent space analysis, denoising performance under controlled noise, and generative sampling via reparameterisation.

---

### 2. Dataset Setup
* **Dataset:** MNIST Handwritten Digit Database
* **Sample Allocation:** 10,000 training images and 2,000 test images (as specified by the laboratory protocol).
* **Preprocessing:**
  * Pixel normalization: Rescaled pixel values from [0, 255] to [0.0, 1.0].
  * Structural formatting:
    * Fully Connected AE: Flattened input vectors of size 784 (28 x 28).
    * Convolutional Models & VAE: Spatial tensor dimensions of 28 x 28 x 1.

---

### 3. Repository Architecture & File Structure

```text
.
├── data/
│   └── mnist_subset.npz             # Local storage for training/test splits
├── models/
│   ├── fc_autoencoder.py            # Fully Connected Autoencoder implementation
│   ├── conv_autoencoder.py          # Convolutional Autoencoder implementation
│   ├── denoising_autoencoder.py     # Denoising CAE pipeline with noise injection
│   └── variational_autoencoder.py   # VAE with reparameterization trick & custom loss
├── utils/
│   ├── data_loader.py               # Preprocessing and noise generation functions
│   └── evaluation.py                # Reconstruction metrics (MSE, MAE, SSIM)
├── outputs/
│   ├── saved_models/                # Trained model weight files
│   └── plots/                       # Generated comparison plots and figures
├── main.py                          # Main execution pipeline running all experiments
├── requirements.txt                 # Dependencies and environment config
└── README.md                        # Project documentation
```
---

### 4. Implemented Architectures

#### 4.1 Fully Connected Autoencoder
* **Encoder:** Input (784) -> Dense (128, ReLU) -> Dense (32, ReLU) -> Latent Bottleneck (16, ReLU)
* **Decoder:** Dense (32, ReLU) -> Dense (128, ReLU) -> Output (784, Sigmoid)
* **Loss & Optimizer:** Binary Cross-Entropy / Adam (lr = 10^-3, Batch Size = 128, Epochs = 20)

#### 4.2 Convolutional Autoencoder
* **Encoder:** Input (28 x 28 x 1) -> Conv2D (32, 3x3, ReLU) -> MaxPool2D (2x2) -> Conv2D (64, 3x3, ReLU) -> MaxPool2D (2x2) -> Conv2D (64, 3x3, ReLU)
* **Decoder:** UpSampling2D (2x2) -> Conv2D (32, 3x3, ReLU) -> UpSampling2D (2x2) -> Conv2D (1, 3x3, Sigmoid)

#### 4.3 Denoising Convolutional Autoencoder
* **Corruptions Applied:**
  * Gaussian Noise: sigma in {0.1, 0.2, 0.3}
  * Salt-and-Pepper Noise: corruption probability p in {0.05, 0.10, 0.20}
* **Target:** Original uncorrupted clean image.

#### 4.4 Variational Autoencoder (VAE)
* **Probabilistic Formulation:** Maps input x to latent mean mu in R^2 and log-variance log(sigma^2) in R^2.
* **Reparameterization Trick:** z = mu + sigma * epsilon, where epsilon ~ N(0, I).
* **Loss Function:** Reconstruction Loss (Binary Cross-Entropy) + KL Divergence.

---

#### Environment 
Python 3.8+ , required packages:
```bash
pip install tensorflow numpy matplotlib scikit-image
```

6. Key Experimental Deliverables & Analysis Covered
i) Reconstruction Quality: Evaluation of image reconstruction across models using Mean Squared Error (MSE), Mean Absolute Error (MAE), and Structural Similarity Index (SSIM).
ii) Spatial Feature Preservation: Quantitative and qualitative comparison showing how convolutional layers outperform fully connected layers by preserving 2D spatial relationships.
iii) Robustness to Noise: Analysis of the denoising CAE's ability to strip Gaussian and salt-and-pepper noise at varying corruption intensities.
iv) Latent Space Exploration: 2D VAE latent space mapping, class clustering inspection, random generative sampling from N(0, I), and continuous latent space interpolation.
v) Bottleneck Impact Study: Analysis of reconstruction fidelity trade-offs across bottleneck sizes dz in {2, 8, 16, 32}.
