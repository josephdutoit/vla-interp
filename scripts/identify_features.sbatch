#!/bin/bash

#SBATCH --job-name=vla-dist-features
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16         # Increased for 8 GPUs
#SBATCH --gpus=8                   # Total GPUs
#SBATCH --nodes=1                  # Assuming all 8 GPUs are on one node
#SBATCH --mem=128G                 # Increased memory for 8 models/SAEs
#SBATCH --output=logs/slurm-logs/dist_features_%j.out
#SBATCH --qos=matrix

# Determine project root
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

source "$PROJECT_ROOT/setup.sh"
source "$PROJECT_ROOT/.venv/bin/activate"

# Add src to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$PROJECT_ROOT/src"

cd "$PROJECT_ROOT"

set -e

# 1. Path Logic
SAE_PATH="data/data/sae_checkpoints/layer_25_2/sae_l25_final.pt"

if [ ! -f "$SAE_PATH" ]; then
    SAE_PATH=$(ls -t data/data/sae_checkpoints/layer_25_2/sae_l25_e*.pt | head -n 1)
    echo "Using latest checkpoint: $SAE_PATH"
fi

export MASTER_PORT=$(shuf -i 10000-65535 -n 1)
export MASTER_ADDR=$(hostname)

# 2. Launch with torchrun
# --nproc_per_node=8 matches your --gpus=8
echo "Identifying features across 8 GPUs..."
torchrun --nproc_per_node=8 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/identify_features.py \
    --sae_path "$SAE_PATH" \
    --batch_size 8 \
    --expansion_factor 2 \
    --layer_idx 25 \
    --output_dir "data/data/discovered_features/layer_25_2"