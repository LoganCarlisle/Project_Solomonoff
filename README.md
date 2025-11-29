# Project Solomonoff

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hydra](https://img.shields.io/badge/Config-Hydra-89b8cd?style=for-the-badge&logo=python&logoColor=white)](https://hydra.cc/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

> *"All inductive inference is just finding the shortest program that produces the data."* — Ray Solomonoff

## 📄 Abstract

**Project Solomonoff** is a research framework investigating the compression of knowledge

---

## ⚙️ Methodology

The architecture decouples the **Policy Generator** (Hypernetwork) from the **Policy Executor** (Frozen Backbone).

```mermaid
graph LR
    subgraph "Compression Layer (The Generator)"
        Input[Task Embedding z] -->|Input| HyperNet{HyperNetwork}
        HyperNet -->|Synthesizes| Weights[LoRA Matrices A & B]
    end

    subgraph "Inference Layer (The Agent)"
        Weights -->|Stateless Injection| LLM[Frozen Transformer]
        Query[User Input] --> LLM
        LLM --> Output[Prediction]
    end
    
    style HyperNet fill:#eee,stroke:#333,stroke-width:2px
    style LLM fill:#fff,stroke:#333,stroke-dasharray: 5 5
