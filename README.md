# VLA-Interp: Interpretability for Vision-Language-Action Models

This repository contains tools for training and interpreting Sparse Autoencoders (SAEs) on Vision-Language-Action (VLA) models, specifically Qwen2.5-VL-3B-Instruct fine-tuned on the Libero dataset.

It is built upon a minimal reimplementation of [VLA-0](https://github.com/NVlabs/vla0) using [TRL](https://github.com/huggingface/trl).

## Project Structure

```text
.
├── src/                # Core package source code
│   ├── rv_train/       # Training utilities
│   ├── rv_eval/        # Evaluation utilities
│   └── rv_interp/      # Interpretability and SAE analysis
├── scripts/            # Slurm sbatch and python scripts for various tasks
├── configs/            # Training and model configurations
├── tools/              # Visualization apps and plotting tools
├── data/               # (Git-ignored) Datasets and model outputs
│   ├── activation_outputs/
│   ├── sae_checkpoints/
│   └── ...
├── logs/               # (Git-ignored) Slurm logs and WandB runs
├── docs/               # Documentation and research notes
└── setup.sh            # Environment setup script
```

## Features

- **Minimal VLA Training**: Minimal implementation of VLA-0 (~1,200 LOC) getting ~92% on LIBERO.
- **SAE Training**: Scripts for chunked activation collection and SAE training on VLA layers.
- **Causal Tracing**: Tools to identify the importance of language and vision tokens in action prediction.
- **Feature Visualization**: Interactive Dash app for exploring discovered SAE features and their activations on visual patches and language tokens.
- **Steering**: Evaluate the impact of steering SAE features on robot performance.

## Installation

We recommend using [`uv`](https://docs.astral.sh/uv/) for managing dependencies.

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
# Install LeRobot dependency
GIT_LFS_SKIP_SMUDGE=1 uv pip install git+https://github.com/huggingface/lerobot.git@f39652707caed42a7cd5ab36066da5663b9565eb

# For evaluation
uv pip install -e ".[eval]"

# BYU-Specific Setup (see setup.sh)
./setup.sh
```

## Usage

### Training VLA
```bash
python scripts/train.py --config configs/vla0.yaml
```

### SAE Pipeline
1. **Collect Activations**: `sbatch scripts/collect_activations.sbatch`
2. **Train SAE**: `sbatch scripts/train_sae.sbatch`
3. **Identify Features**: `sbatch scripts/identify_features.sbatch`
4. **Visualize**: `python tools/visualization/app.py`

### Interpretability Analysis
- **Causal Tracing**: `sbatch scripts/causal_tracing.sbatch`
- **Attention Analysis**: `sbatch scripts/attention_analysis.sbatch`

## Acknowledgements

Based on [vla0-trl](docs/VLA0-TRL_README.md) by Ankit Goyal and interpretability techniques inspired by [Anthropic's SAE work](https://transformer-circuits.pub/2023/monosemantic-features/index.html).
