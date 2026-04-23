#!/bin/bash
#SBATCH --job-name=attn_analysis
#SBATCH --output=logs/slurm-logs/attn_%j.out
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --qos=matrix

# Determine project root
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Load environment
source "$PROJECT_ROOT/setup.sh"
source "$PROJECT_ROOT/.venv/bin/activate"

# Add src to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$PROJECT_ROOT/src"

# Distributed training parameters
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12356
export CUDA_LAUNCH_BLOCKING=1


# Use srun to launch 8 processes (one per GPU)
cd "$PROJECT_ROOT"

srun python3 scripts/attention_analysis.py \
    --model_path "logs/runs/vla0-trl-test" \
    --batch_size 1 \
    --output_file "data/data/activation_outputs/attention_density.json"
