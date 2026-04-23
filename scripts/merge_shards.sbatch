#!/bin/bash

#SBATCH --job-name=vla-merge-features
#SBATCH --time=12:00:00   # walltime
#SBATCH --cpus-per-task=128  # number of processor cores
#SBATCH --mem=64G   # memory
#SBATCH --output=logs/slurm-logs/merge_features_%j.out   # %j inserts jobid

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
python scripts/merge_shards.py --input_dir data/data/discovered_features/layer_25_2