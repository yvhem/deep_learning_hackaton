# Graph Classification under Noisy Labels

## Overview

This repository contains our solution for the **Graph Classification with Noisy Labels Hackathon**, hosted by the University of Rome Sapienza. The goal of the competition is to build a robust model capable of accurately classifying graphs across multiple noisy datasets.

## Method Summary

We developed a modular and noise-aware graph classification pipeline consisting of the following components.

### Model Architecture
<p align="center">
  <img src="https://github.com/user-attachments/assets/7a3aeaf3-9903-4f02-b42e-1a3436c3436d" alt="uWuModel" width="600"/>
</p>

#### Input Projection

- If input node features `x` are missing, they are initialized as tensors of zeros with a single feature dimension.
- The input is then projected into a learned embedding space through a linear layer.

#### Node Feature Initialization

- Ensures consistency and allows the model to learn meaningful representations even in the absence of raw node features, since graphs have features only on the edges.

#### Model Backbone

The model leverages a stack of GNN layers that combine local and global structural information:

- **2 × GINConvE layers**  
  Modified GIN layers with:
  - Edge-gated message passing
  - Dedicated edge encoders
  - LayerNorm and dropout
  - Residual connections for gradient stability

- **1 × TransformerConvE layer**  
  Enables the model to capture long-range dependencies via multi-head attention while incorporating encoded edge features.

- **2 × GINConvE layers (post-Transformer)**  
  Deepens the architecture to refine graph-level representations, preserving robustness to label noise.

#### Readout

- A **global mean pooling** operation aggregates node embeddings into graph-level representations.

#### Classifier Head

- A compact multilayer perceptron (MLP) with dropout and LeakyReLU activation performs the final classification.


### Training

#### Custom Loss: `NoisyCrossEntropyLoss`

- **Conditional Noise Injection**  
  Label noise is only applied during training.

- **Random Noise Application**  
  For a given batch, noise is introduced with a specified probability. A noise mask is applied to flip a subset of labels to randomly selected classes.

- **Purpose**  
  This approach improves the model’s generalization by reducing overfitting to mislabeled samples.


#### Adaptive Learning Rate

- **Learning Rate Scheduling**  
  The learning rate is adapted during training using a `ReduceLROnPlateau` scheduler based on validation F1-score.
