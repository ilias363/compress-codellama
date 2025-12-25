# CodeLLaMA Compression Pipeline

A comprehensive pipeline for compressing CodeLLaMA models through **pruning**, **fine-tuning**, and **quantization**.

## Overview

This project implements a multi-stage compression pipeline:

1. **Wanda Pruning** - Unstructured weight pruning to create sparse models
2. **QLoRA Fine-tuning** - Efficient fine-tuning to recover accuracy after pruning
3. **AWQ Quantization** - 4-bit quantization using llm-compressor for deployment

## Installation

```bash
# Clone the repository
git clone https://github.com/ilias363/compress-codellama.git
cd compress-codellama

# Install dependencies
pip install -r requirements.txt
```

### For Code Evaluation (HumanEval/MBPP)

⚠️ **Important**: Before running code evaluations, you must clone and install the bigcode evaluation harness:

```bash
# Clone bigcode-evaluation-harness
git clone https://github.com/bigcode-project/bigcode-evaluation-harness.git
cd bigcode-evaluation-harness

# Install dependencies
pip install -e .

# Go back to project root
cd ..
```

## Usage

### 1. Prepare Dataset
```bash
python pipeline/prepare_dataset.py -v
```

### 2. Prune Model (Wanda)
```bash
python pipeline/prune_wanda.py -v --model codellama/CodeLlama-7b-Instruct-hf
```

### 3. Fine-tune with QLoRA
```bash
python pipeline/finetune_qlora.py -v --model outputs/models/pruned
```

### 4. Merge LoRA Adapter
```bash
python pipeline/merge_adapter.py -v --base_model outputs/models/pruned --adapter_path outputs/models/finetuned/checkpoint-300/adapter_model
```

### 5. Quantize (AWQ via llm-compressor)
```bash
python pipeline/quantize_llmcompressor.py -v --model outputs/models/merged
```

## Evaluation

### Perplexity (WikiText-2)
```bash
python pipeline/evaluate_perplexity.py -v --model <model_path_or_hf_id>
```

### Code Benchmarks (HumanEval/MBPP)
```bash
# Requires bigcode-evaluation-harness to be installed first!
python pipeline/evaluate_code.py -v --model <model_path_or_hf_id>
```

### Efficiency Metrics
```bash
# Standard efficiency eval
python pipeline/evaluate_efficiency.py -v --model <model_path_or_hf_id>
```

## Configuration

All settings can be configured in `configs/default.json`:

- **Pruning**: sparsity ratio, calibration samples
- **Fine-tuning**: LoRA rank, learning rate, epochs
- **Quantization**: AWQ scheme, calibration settings
- **Evaluation**: batch sizes, sequence lengths

## Project Structure

```
compress-codellama/
├── configs/           # Configuration files
├── datasets/          # Prepared datasets
├── lib/               # Core libraries
│   ├── eval/          # Evaluation functions
│   ├── llmcompressor/ # llm-compressor quantization
│   ├── qlora/         # QLoRA fine-tuning
│   └── wanda/         # Wanda pruning
├── outputs/           # Generated outputs
│   ├── models/        # Saved models
│   ├── stats/         # Evaluation results
│   └── cache/         # Cached files
├── pipeline/          # Main pipeline scripts
└── utils/             # Utility functions
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (16GB+ VRAM recommended)
- Transformers 4.45.0+ (required for llm-compressor)

## Acknowledgements

### 🌟 Special Thanks to Kaggle

**[Kaggle](https://www.kaggle.com/)** deserves the biggest spotlight in this project. Without their **free GPU resources** (including P100 and T4 GPUs), this research would not have been possible. Their commitment to democratizing machine learning by providing free computational resources to researchers and students worldwide is truly remarkable.

### 🔓 The Open Source Community

This project stands on the shoulders of giants:

- **[Hugging Face Transformers](https://github.com/huggingface/transformers)**
- **[vLLM Project & llm-compressor](https://github.com/vllm-project/llm-compressor)**
- **[Wanda Pruning](https://github.com/locuslab/wanda)**
- **[QLoRA](https://github.com/artidoro/qlora)**
- **[Meta AI & CodeLLaMA](https://github.com/meta-llama/codellama)**
- **[BigCode Evaluation Harness](https://github.com/bigcode-project/bigcode-evaluation-harness)**
