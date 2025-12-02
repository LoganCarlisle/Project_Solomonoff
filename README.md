# Project Solomonoff

[![Status](https://img.shields.io/badge/Status-Active_Research-blue)]() [![Phase](https://img.shields.io/badge/Current_Phase-Neural_Compression-brightgreen)]() [![Model](https://img.shields.io/badge/Target-GPT--2_(124M)-red)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hydra](https://img.shields.io/badge/Config-Hydra-89b8cd?style=for-the-badge&logo=python&logoColor=white)](https://hydra.cc/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
> ** Research Preview:** This repository is currently in active development. The **Neural Compression** pipeline is functional, achieving loss descent on 100m+ param models.

**Project Solomonoff** is a Hypernetwork framework designed to scale **Hypernets** to the foundation model era. It treats model parameters as sequential signal data, utilizing a postition invariant weight encoder,RNN **Mamba-SSM** backbone, and a **Coordinate-Aware Diffusion Transformer** to compress and reconstruct functional weights for 100M+ parameter architectures.

---

##  Research Roadmap

###  Phase 1: Architecture & Compression (Completed)
The core infrastructure for handling massive, variable-sized weight spaces is built and verified.
- [x] **Patch-Based Tokenization:** Lossless conversion of arbitrary 2D weight matrices (e.g., $50k \times 768$) into fixed-size patch sequences.
- [x] **Permutation-Invariant Encoder:** Implementation of Set Transformers to capture layer topology independent of input order.
- [x] **Memory-Efficient Training:** Implementation of Truncated Backpropagation Through Time (TBPTT) and Gradient Checkpointing to fit 148-layer training on consumer hardware.
- [x] **Signal Preservation:** Solved "Signal Propagation Collapse" via local activation stability losses, ensuring reconstructed weights maintain unit variance.

###  Phase 2: Generative Synthesis (In Progress)
Transitioning from encoding existing models to fabricating novel models from latent noise.
- [ ] **Experiment on various Architectures** Test leading SOTA Architetures for the Hypernet Model
- [ ] **Pure Noise Generation:** Validating functional Perplexity on weights generated autoregressively without Teacher Forcing.
- [ ] **Latent Space Analysis:** Investigating the smooth interpolation between different model checkpoints (e.g., interpolated weights between two fine-tunes). More focused on scaling though for now.
- [ ] **Local Task Distillation:** Refining the auxiliary loss engine to enforce functional correctness (activations) during the generation process.
- [ ] **Real Task Loss Backpropagation:** Implementing memory-efficient gradient flow from validation Perplexity back to the Hypernetwork.


### 🔭 Phase 3: Scaling & Optimization (Future)
- [ ] **Scale to 1B+ Parameters:** Optimizing the pipeline for Llama/Pythia scale models.
- [ ] **Scale and generalize across architetures** Train on multiple models and archietures to investigate zero shot training.



> *"All inductive inference is just finding the shortest program that produces the data."* — Ray Solomonoff

## 📄 Abstract

**Project Solomonoff** is a research framework investigating the compression of knowledge,testing the hypothesis that the weight manifolds of Large Language Models are highly compressible.

---

##  Methodology: Full-Weight Synthesis via Low-Rank Upscaling

Instead of storing static weight matrices ($W_{static} \in \mathbb{R}^{d_{out} \times d_{in}}$), we implement a **Generative Hypernetwork** that compresses a model to latent space and upscale on infrence. Since currently generating $d \times d$ parameters directly is computationally intractable, we generate **Low-Rank Factors** ($U, V$) and upscale them via matrix multiplication to approximate the target distribution.

### The HyperNET Architecture

The system functions as a **Neural Decompressor**:
1.  **Compression:** The "Knowledge" of the model is compressed into the Hypernetwork parameters ($\theta_H$).
2.  **Decompression:** For a given layer, the Hypernetwork projects a layer-specific embedding ($z_l$) into factors $U$ and $V$.
3.  **Reconstruction:** The full weight matrix is reconstructed as $W_{approx} = U \times V$.

$$W_{approx} \approx W_{target}$$

This is done on the assumtion that the ith layer depends on the layer before it and so on, as such a recurrent process of feeding in hidden state can lead to coherent generations of a full model.

### The System Diagram
