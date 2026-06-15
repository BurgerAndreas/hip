#!/usr/bin/env bash
#SBATCH -A nvr_qualg_lmbm
#SBATCH -J eqv2_rough_dft
#SBATCH -p batch_singlenode
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH --output=/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip/slurm_logs/eqv2-rough-dft-%j.out

set -euo pipefail

cd /lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip

export HIP_VENV="$PWD/.venv"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

source scripts/setup_cuda.sh

mkdir -p slurm_logs runs/eqv2_roughness_vs_dft

# Per-sample roughness (FD self-inconsistency, force-Jacobian asymmetry) vs autograd-Hessian
# accuracy against the HORM DFT Hessian, across the local 100-sample HORM set.
exec uv run python scripts/eqv2_roughness_vs_dft.py \
  --checkpoint ckpt/eqv2.ckpt \
  --dataset data/sample_100.lmdb \
  --h-grid 1e-4,3e-4,1e-3,3e-3 \
  --output-dir runs/eqv2_roughness_vs_dft \
  --tag eqv2_sample100
