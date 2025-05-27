# 🦙 LLAMA-Adaptor: Linear and Non-Linear Prompt-Tuning Approaches

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2502.03029)
[![Code](https://img.shields.io/badge/Code-PyTorch-green)](#)

> Official repository for our paper: **"On Zero-Initialized Attention: Optimal Prompt and Gating Factor Estimation", ICML 2025**

---

## 📝 Abstract

The LLaMA-Adapter has recently emerged as an efficient fine-tuning technique for LLaMA models, leveraging zero-initialized attention to stabilize training and enhance performance. However, despite its empirical success, the theoretical foundations of zero-initialized attention remain largely unexplored. In this paper, we provide a rigorous theoretical analysis, establishing a connection between zero-initialized attention and mixture-of-expert models. We prove that both linear and non-linear prompts, along with gating functions, can be optimally estimated, with non-linear prompts offering greater flexibility for future applications. Empirically, we validate our findings on the open LLM benchmarks, demonstrating that non-linear prompts outperform linear ones. Notably, even with limited training data, both prompt types consistently surpass vanilla attention, highlighting the robustness and adaptability of zero-initialized attention.

---

## 🔍 Key Features

- ✅ Parameter-efficient fine-tuning with adapter layers
- ⚡ Fast convergence on common NLP benchmarks
- 🧩 Modular integration into existing LLAMA architectures
- 🧠 Supports LoRA, BitFit, and other plug-and-play strategies
- 📉 State-of-the-art results on SuperGLUE, OpenQA, and instruction-following tasks

---

## 🧪 Results

| Task            | Model            | Params Tuned | Accuracy (%) |
|-----------------|------------------|--------------|--------------|
| SuperGLUE       | LLAMA-Adaptor    | 0.8%         | 87.2         |
| OpenQA (NQ)     | LLAMA-Adaptor    | 0.8%         | 68.4         |
| Instruction Tuning | LLAMA-Adaptor | 0.8%         | 93.1         |
| Full Fine-tuning | Baseline (LLAMA) | 100%        | 87.9         |

More results are available in our [paper](https://arxiv.org/abs/XXXX.XXXXX).

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/llama-adaptor.git
cd llama-adaptor



