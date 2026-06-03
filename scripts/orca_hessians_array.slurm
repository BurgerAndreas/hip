#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/aip-aspuru/aburger/hip
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --array=0-709
#SBATCH --job-name=orca-hess
#SBATCH --output=/home/aburger/scratch/hip/orca_hessians/slurm/%x-%A_%a.txt
#SBATCH --error=/home/aburger/scratch/hip/orca_hessians/slurm/%x-%A_%a.txt

set -euo pipefail

module load orca/6.1.1

export SAMPLE_INDEX="$SLURM_ARRAY_TASK_ID"
export NPROCS="$SLURM_NTASKS"
export ORCA_HESSIAN_ROOT="${ORCA_HESSIAN_ROOT:-$HOME/scratch/hip/orca_hessians}"

bash scripts/run_orca_hessians.bash
