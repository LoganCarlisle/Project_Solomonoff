# Project Solomonoff

[![Status](https://img.shields.io/badge/Status-Active_Research-blue)]() 
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hydra](https://img.shields.io/badge/Config-Hydra-89b8cd?style=for-the-badge&logo=python&logoColor=white)](https://hydra.cc/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](./LICENSE)
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

##  References & Prior Work

This project builds upon foundational research in Neural Functional Networks (NFNs)(NFN not much yet), Hypernetworks, and Generative Sequence Modeling.

### Neural Functional Networks (NFNs) & Set Learning
* **Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks** *Lee et al. (ICML 2019)* Proposed the linear-complexity attention mechanisms (ISAB and PMA) used in Solomonoff's encoder to process variable-sized sets of weight patches efficiently.  
  [\[ArXiv\]](https://arxiv.org/abs/1810.00825)

* **Permutation Equivariant Neural Functionals** *Zhou et al. (NeurIPS 2023)* Defined the "Weight Space" learning problem and introduced Permutation Equivariant layers for processing neural network parameters as functional representations.  
  [\[ArXiv\]](https://arxiv.org/abs/2302.14040)

* **Neural Functional Transformers** *Zhou et al. (NeurIPS 2023)* Introduced attention-based architectures for weight spaces, treating neurons as sets of columns to capture topology independent of permutation.  
  [\[Paper\]](https://proceedings.neurips.cc/paper_files/paper/2023/file/f4757db82a02eea015670ecca605d5cc-Paper-Conference.pdf)

### Hypernetworks & Architecture Search
* **HyperNetworks** *Ha et al. (ICLR 2017)* The GOAT! The original paper that introduced HyperNets a all time paper but couldve came up with a cooler name for HyperNets imo(still love this paper).
  [\[ArXiv\]](https://arxiv.org/abs/1609.09106)

* **Weight Agnostic Neural Networks** *Gaier & Ha (NeurIPS 2019)* Demonstrated that architecture search can find networks that function with shared random weights, decoupling topology from specific parameters.  
  [\[ArXiv\]](https://arxiv.org/abs/1906.04358)

### Generative Backbones
* **Scalable Diffusion Models with Transformers (DiT)** *Peebles & Xie (ICCV 2023)* Replaced the U-Net with Transformers for diffusion, enabling the "Patching" strategy used in Solomonoff to handle variable-size tensors.  
  [\[ArXiv\]](https://arxiv.org/abs/2212.09748)

* **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** *Gu & Dao (2023)* Introduced the SSM backbone used here for memory-efficient autoregressive generation of deep layer sequences.  
  [\[ArXiv\]](https://arxiv.org/abs/2312.00752)

### Parameter Efficiency
* **LoRA: Low-Rank Adaptation of Large Language Models** *Hu et al. (ICLR 2022)* Proposed the low-rank decomposition ($W = BA$) that this project targets for efficient compression and generation.  
  [\[ArXiv\]](https://arxiv.org/abs/2106.09685)
