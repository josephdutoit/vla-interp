#!/bin/bash

#SBATCH --job-name=vla-causal-tracing
#SBATCH --time=12:00:00   # walltime
#SBATCH --cpus-per-task=8   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1
#SBATCH --mem=32G   # memory per CPU core
#SBATCH --output=logs/slurm-logs/causal_tracing_%j.out   # %j inserts jobid
#SBATCH --qos=matrix


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# Determine project root
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
source "$PROJECT_ROOT/setup.sh"
source "$PROJECT_ROOT/.venv/bin/activate"

# Add src to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$PROJECT_ROOT/src"

cd "$PROJECT_ROOT"
python scripts/causal_tracing.py --trace_target "language" --output_path "data/data/tracing_outputs/language_tracing_results_2.parquet"