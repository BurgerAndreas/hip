#!/usr/bin/env bash
#SBATCH -A nvr_qualg_lmbm
#SBATCH -J hip_eqv2_lmdb
#SBATCH -p batch_singlenode
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH --output=/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip/slurm_logs/hip-eqv2-lmdb-%j.out

set -euo pipefail

cd /lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip

export HIP_VENV="$PWD/.venv"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

source scripts/setup_cuda.sh

exec uv run python scripts/compare_eqv2_lmdb.py \
  --dataset data/sample_100.lmdb \
  --checkpoint ../HORM/ckpt/eqv2.ckpt \
  --output-dir results_compare_eqv2_lmdb_hip \
  --max-samples 10
