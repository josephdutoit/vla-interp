#!/bin/bash

#SBATCH --job-name=vla-eval-sae
#SBATCH --time=01:00:00   # walltime
#SBATCH --cpus-per-task=8   # number of processor cores
#SBATCH --gpus=1
#SBATCH --mem=32G   # memory
#SBATCH --output=logs/slurm-logs/eval_sae_%j.out   # %j inserts jobid
#SBATCH --qos=matrix

# Set the max number of threads to use for programs using OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# Determine project root
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
source "$PROJECT_ROOT/setup.sh"
source "$PROJECT_ROOT/.venv/bin/activate"

# Add src to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$PROJECT_ROOT/src"

cd "$PROJECT_ROOT"
python scripts/eval_sae.py --sae_path "data/data/sae_checkpoints/layer_25_2/sae_l25_final.pt" --layer_idx 25 --expansion_factor 2