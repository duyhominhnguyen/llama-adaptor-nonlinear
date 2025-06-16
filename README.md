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

## 📊 Main Experiments

### 🧪 Evaluation Benchmarks

We evaluate our models using the **Open LLM Benchmark Suite** (Beeching et al., 2024), which includes four diverse tasks to assess the generative, reasoning, and factual capabilities of LLMs:

- **ARC (AI2 Reasoning Challenge)** – Easy (`ARC-eas`) and Challenge (`ARC-cha`) subsets [(Clark et al., 2018)]  
  > Evaluates commonsense and scientific reasoning.
  
- **HellaSwag** [(Zellers et al., 2019)]  
  > Tests the model's ability to complete sentences using commonsense inference.
  
- **MMLU (Massive Multitask Language Understanding)** [(Hendrycks et al., 2020)]  
  > Measures knowledge across 57 academic subjects.

- **TruthfulQA** [(Lin et al., 2021)]  
  > Measures whether model outputs are truthful rather than just plausible.

**Evaluation Settings**:
- **Zero-shot** evaluation: ARC, MMLU, and TruthfulQA  
- **10-shot** evaluation: HellaSwag  
  > (n-shot means using n instruction-following examples in the prompt.)

---

### 🏗️ Architectures & Training Setup

We conduct experiments on two LLaMA model sizes:

| Model       | Transformer Layers | Prompt Length (L) | Prompt Layers (K) |
|-------------|--------------------|--------------------|--------------------|
| LLaMA-7B    | 32                 | 10                 | Last 30 layers     |
| LLaMA-13B   | 40                 | 10                 | Last 38 layers     |

**Training Configuration**:
- GPUs: 4 × A100
- Epochs: 5 (including 2 warm-up epochs)
- Total Batch Size: 64
- Learning Rate: 0.009
- Weight Decay: 0.02
- Dataset: [Alpaca (Taori et al., 2023)](https://github.com/tatsu-lab/stanford_alpaca)

**Optimizer**: AdamW  
**Fine-tuning Method**: Zero-Initialized Attention with Linear/Non-Linear Prompts

---
## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/llama-adaptor.git
cd llama-adaptor
```

### Repo Structures
```
llama-adaptor/
├── figures                    # Figures in the paper
├── README.md                  # Project overview and usage
├── LICENSE                    # MIT license
├── requirements.txt           # Required Python packages
├── setup.py                   # Optional: Install as a package
├── configs/                   # YAML or JSON config files
│   ├── train_linear.yaml
│   └── train_nonlinear.yaml
├── scripts/                   # Example shell scripts for training/eval
│   ├── run_train.sh
│   └── run_eval.sh
├── data/                      # Data loading and preprocessing
│   ├── alpaca_loader.py
│   ├── open_llm_utils.py
│   └── ...
├── models/                    # Model architecture and prompt modules
│   ├── llama_adaptor.py
│   ├── linear_prompt.py
│   └── nonlinear_prompt.py
├── train.py                   # Main training entry point
├── eval.py                    # Evaluation script
├── utils/                     # Helper functions: logging, metrics, etc.
│   ├── trainer.py
│   ├── metrics.py
│   └── logger.py
├── checkpoints/               # (Optional) Pretrained or fine-tuned weights
│   └── llama7b_linear.pt
├── results/                   # Evaluation results, logs, etc.
│   └── arc_easy_results.json
└── docs/                      # (Optional) Additional documentation
    └── architecture.png
```

### 2. Training

To fine-tune LLaMA using linear or non-linear prompts:

```bash
# Linear prompt tuning on LLaMA-7B
python train.py \
    --model llama-7b \
    --dataset alpaca \
    --prompt_type linear \
    --prompt_length 10 \
    --insert_layers 30 \
    --batch_size 64 \
    --lr 0.009 \
    --epochs 5 \
    --output_dir outputs/llama7b_linear

# Non-linear prompt tuning on LLaMA-13B
python train.py \
    --model llama-13b \
    --dataset alpaca \
    --prompt_type nonlinear \
    --prompt_length 10 \
    --insert_layers 38 \
    --batch_size 64 \
    --lr 0.009 \
    --epochs 5 \
    --output_dir outputs/llama13b_nonlinear
```
### 2. Inference
```
# Zero-shot evaluation on ARC-easy
python eval.py \
    --model_path outputs/llama7b_linear \
    --task arc-easy \
    --shots 0

# 10-shot evaluation on HellaSwag
python eval.py \
    --model_path outputs/llama13b_nonlinear \
    --task hellaswag \
    --shots 10
```

## 📚 Citation

If you use this codebase or find our work helpful, please consider citing:

```bibtex
@article{diep2025zero,
  title={On Zero-Initialized Attention: Optimal Prompt and Gating Factor Estimation},
  author={Diep, Nghiem T and Nguyen, Huy and Nguyen, Chau and Le, Minh and Nguyen, Duy MH and Sonntag, Daniel and Niepert, Mathias and Ho, Nhat},
  journal={International Conference on Machine Learning (ICML)},
  year={2025}
}



