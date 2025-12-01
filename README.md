# Project Solomonoff

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hydra](https://img.shields.io/badge/Config-Hydra-89b8cd?style=for-the-badge&logo=python&logoColor=white)](https://hydra.cc/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

> *"All inductive inference is just finding the shortest program that produces the data."* — Ray Solomonoff

## 📄 Abstract

**Project Solomonoff** is a research framework investigating the compression of knowledge,testing the hypothesis that the weight manifolds of Large Language Models are highly compressible.

---

## ⚙️ Methodology: Full-Weight Synthesis via Low-Rank Upscaling

Instead of storing static weight matrices ($W_{static} \in \mathbb{R}^{d_{out} \times d_{in}}$), we implement a **Generative Hypernetwork** that compresses a model to latent space and upscale on infrence. Since currently generating $d \times d$ parameters directly is computationally intractable, we generate **Low-Rank Factors** ($U, V$) and upscale them via matrix multiplication to approximate the target distribution.

### The HyperNET Architecture

The system functions as a **Neural Decompressor**:
1.  **Compression:** The "Knowledge" of the model is compressed into the Hypernetwork parameters ($\theta_H$).
2.  **Decompression:** For a given layer, the Hypernetwork projects a layer-specific embedding ($z_l$) into factors $U$ and $V$.
3.  **Reconstruction:** The full weight matrix is reconstructed as $W_{approx} = U \times V$.

$$W_{approx} \approx W_{target}$$

This is done on the assumtion that the ith layer depends on the layer before it and so on, as such a recurrent process of feeding in hidden state can lead to coherent generations of a full model.

### The System Diagram
