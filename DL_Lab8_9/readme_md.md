# Deep Learning Lab: Effect of Attention Type and Attention Position in a Pretrained ResNet50
## 📌 Project Overview

This experiment provides an  evaluation of various attention mechanisms integrated at different feature hierarchy stages within a pretrained **ResNet50** backbone. The evaluation uses the **TensorFlow Flowers dataset** (5 classes) to measure performance trade-offs across quantitative metrics, computational complexity, and qualitative interpretability.

---

## 🔬 Experimental Setup

- **Backbone Architecture:** Pretrained ResNet50
- **Dataset:** TensorFlow Flowers (3,670 images, 5 classes: Daisy, Dandelion, Roses, Sunflowers, Tulips)
- **Data Splits:** 70% Train, 15% Validation, 15% Test
- **Input Resolution:** $224 \times 224 \times 3$
- **Training Setup:** Adam Optimizer, Categorical Cross-Entropy, Early Stopping, Cosine Annealing Learning Rate Schedule

### Attention Variants & Placement Stages
1. **Attention Types:**
   - Channel Attention (SE Block)
   - Spatial Attention
   - Channel-Spatial Combined Attention (CBAM)
   - Coordinate Attention (CA)
   - Self-Attention (Multi-Head Self-Attention)
2. **Insertion Positions:**
   - $P_2$: Early Stage (After Stage 2 / $conv3\_x$)
   - $P_3$: Mid Stage (After Stage 3 / $conv4\_x$)
   - $P_4$: Late Stage (After Stage 4 / $conv5\_x$)

---

## 📊 Evaluation Dimensions

The evaluation framework consists of three core analytical pillars:

1. **Quantitative Analysis:**
   - Classification Accuracy
   - Macro Precision, Recall, and F1-Score
   - Per-Class Confusion Matrices

2. **Computational Complexity Analysis:**
   - Total & Trainable Parameter Count
   - FLOPs / Memory Footprint
   - Average Inference Time per Sample (ms)
   - Accuracy vs. Parameter Trade-offs

3. **Qualitative Interpretability:**
   - Channel-wise Weight Distributions
   - Spatial Attention Map Overlays
   - Multi-Stage Attention Heatmap Evolution ($P_2 \rightarrow P_3 \rightarrow P_4$)
   - Comparative Grad-CAM Analysis

---
