#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/aip-aspuru/aburger/hip
#SBATCH --time=71:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=8
#SBATCH --job-name=hip-sweep
#SBATCH --output=/project/aip-aspuru/aburger/hip/outslurm/sweep-%j.txt
#SBATCH --error=/project/aip-aspuru/aburger/hip/outslurm/sweep-%j.txt

set -euo pipefail

# Activate environment
source ${PYTHONBIN}/activate
source .env

# Ensure WANDB is available via uv if not globally installed
export WANDB_PROJECT=hip

echo $(date): Job $SLURM_JOB_ID starting sweep agent for one suggestion

# One suggestion per allocation to respect 3-day walltime
uvx --from wandb wandb agent --count 1 "$SWEEP_ID"

echo $(date): Job $SLURM_JOB_ID finished

