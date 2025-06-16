# 🦙 LLAMA-Adaptor: Linear and Non-Linear Prompt-Tuning Approaches

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2502.03029)
[![Code](https://img.shields.io/badge/Code-PyTorch-green)](#)

> Official repository for our paper: **"On Zero-Initialized Attention: Optimal Prompt and Gating Factor Estimation", ICML 2025**

---

## 📝 Introduction

The LLaMA-Adapter has recently emerged as an efficient fine-tuning technique for LLaMA models, leveraging zero-initialized attention to stabilize training and enhance performance. However, despite its empirical success, **the theoretical foundations of zero-initialized attention remain largely unexplored**. 

In this paper, we **(i) provide a rigorous theoretical analysis, establishing a connection between zero-initialized attention and mixture-of-expert models**. We **(ii) prove that both linear and non-linear prompts**, along with gating functions, can be optimally estimated, with non-linear prompts offering greater flexibility for future applications. Empirically, we validate our findings on the open LLM benchmarks, demonstrating that non-linear prompts outperform linear ones. Notably, even with limited training data, both prompt types consistently surpass vanilla attention, highlighting the robustness and adaptability of zero-initialized attention.

<p align="center">
<img src="https://github.com/duyhominhnguyen/llama-adaptor-nonlinear/blob/main/figures/fig_1_overview.png" alt="Alt text" width="400"/>
</p>

---

## 🔍 Key Features

- ✅ Parameter-efficient fine-tuning with linear/non-linear adapter layers
- ⚡ Fast convergence on common NLP benchmarks
- 🧩 Modular integration into existing LLAMA architectures
- 🧠 Supports LoRA, BitFit, and other plug-and-play strategies
- 📉 State-of-the-art results on Open LLM Benchmark Suite 

---

## 📊 Main Experiments

### 🧪 Evaluation Benchmarks

We evaluate our models the following benchmarks: ARC, HellaSwag, MMLU, TruthfulQA, which includes four diverse tasks to assess the generative, reasoning, and factual capabilities of LLMs. We reproduce all the results of the baselines in our paper.

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
├── figures                                     # Figures in the paper
├── README.md                                   # Project overview and usage
├── LICENSE                                     # MIT license
├── requirements.txt                            # Required Python packages
├── setup.py                                    # Optional: Install as a package
├── setup.cfg
├── pyproject.toml
├── generate.sh
├── example.py
├── hellaswag_check.py
├── hellaswag_check.sh
├── alpaca_data.json                            # Dataset for fine-tuning the LLaMA-Adapter
├── alpaca_finetuning_v1/                       # Contain code and script for fine-tuning LLaMA-Adapter
│   ├── llama/
|   |   ├── generation.py
|   |   ├── model.py
|   |   └── tokenizer.py
|   ├── util/
|   ├── engine_finetuning.py
|   ├── extract_adapter_from_checkpoint.py
|   ├── finetune_llama.sh
|   ├── finetuning.py
│   └── models_llama_adapter.py
|
├── llama/                                      # Example shell scripts for training/eval
|   ├── generation.py
|   ├── model.py
|   └── tokenizer.py
├── lm-evaluation-harness/                      # Folder Library for evaluating HellaSwag benchmark
├── utils/                                      # Helper functions: quantization.py
│   ├── quantization.py 
└── docs/                                       # (Optional) Additional documentation
    └── architecture.png
```

### 2. Training

To fine-tune LLaMA using linear or non-linear prompts:

```bash
# Linear prompt tuning on LLaMA
cd alpaca_finetuning_v1/
bash finetune_llama.sh
# or
typ_act=identity
typ_gate=random
hid_acti_func=none
hidden_dim=0

torchrun --nnodes=1 --nproc_per_node=4 --master_port=25012 finetuning.py \
    --model Llama7B_adapter \
    --llama_model_path ../LLaMA-${name} \
    --data_path ../alpaca_data.json \
    --adapter_layer ${adapter_layer} \
    --adapter_len 10 \
    --max_seq_len 512 \
    --batch_size ${batch_size} \
    --epochs ${epoch} \
    --warmup_epochs 2 \
    --blr 9e-3 \
    --weight_decay 0.02 \
    --typ_act ${typ_act} \
    --typ_gate ${typ_gate} \
    --hid_acti_func ${hid_acti_func}\
    --hidden_dim ${hidden_dim}\
    --output_dir ./checkpoint_adapter_layer${adapter_layer}_${typ_act}${hidden_dim}${hid_acti_func}_random_init${random_init}_batchsize${batch_size}_epoch${epoch}_${name}_test/

# Non-linear prompt tuning on LLaMA
cd alpaca_finetuning_v1/
bash finetune_llama.sh
# or
typ_act=hypermodel
typ_gate=random
hid_acti_func=relu
hidden_dim=64

torchrun --nnodes=1 --nproc_per_node=4 --master_port=25012 finetuning.py \
    --model Llama7B_adapter \
    --llama_model_path ../LLaMA-${name} \
    --data_path ../alpaca_data.json \
    --adapter_layer ${adapter_layer} \
    --adapter_len 10 \
    --max_seq_len 512 \
    --batch_size ${batch_size} \
    --epochs ${epoch} \
    --warmup_epochs 2 \
    --blr 9e-3 \
    --weight_decay 0.02 \
    --typ_act ${typ_act} \
    --typ_gate ${typ_gate} \
    --hid_acti_func ${hid_acti_func}\
    --hidden_dim ${hidden_dim}\
    --output_dir ./checkpoint_adapter_layer${adapter_layer}_${typ_act}${hidden_dim}${hid_acti_func}_random_init${random_init}_batchsize${batch_size}_epoch${epoch}_${name}_test/
```
### 2. Inference
```bash
# Zero-shot evaluation on ARC
torchrun --nnodes=1 --nproc_per_node=1 --master_port=25010 example.py \
      --typeQues arc \
      --typ_act hypermodel \
      --hid_acti_func relu \
      --random_init False \
      --max_seq_len 2048 \
      --ckpt_dir ./LLaMA-7B\
      --tokenizer_path ./LLaMA-7B/tokenizer.model \
      --adapter_path ./alpaca_finetuning_v1/checkpoint_adapter_layer30_hypermodel64_random_initFalse_batchsize16_epoch5_7B_test/adapter_adapter_len10_layer30_epoch5.pth

# 10-shot evaluation on HellaSwag
bash hellaswag_check.sh
# or
python hellaswag_check.py \
      --ckpt_dir ./LLaMA-7B \
      --adapter_path ./alpaca_finetuning_v1/checkpoint_adapter_layer30_hypermodel64_random_initFalse_batchsize16_epoch5_7B_test/adapter_adapter_len10_layer30_epoch5.pth \
      --typ_act hypermodel \
      --hid_acti_func relu \
      --random_init False
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



