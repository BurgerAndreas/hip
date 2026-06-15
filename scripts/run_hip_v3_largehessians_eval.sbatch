#!/usr/bin/env bash
#SBATCH -A nvr_qualg_lmbm
#SBATCH -J hip_v3_eval
#SBATCH -p batch_singlenode
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH --output=/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip/slurm_logs/hip-v3-largehessians-eval-%j.out

set -euo pipefail

cd /lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

source scripts/setup_cuda.sh

exec .venv/bin/python scripts/eval_orca_hessians_scratch_hip_v3.py \
  --h5-glob '/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip/hf_hip/orca_wb97x_631gd_chno_30_100/h5/*.h5' \
  --ckpt-path ckpt/hip_v3.ckpt \
  --output-dir results_eval_largehessians_orca_hip_v3 \
  --redo
