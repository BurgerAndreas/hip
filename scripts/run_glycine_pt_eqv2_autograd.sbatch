#!/usr/bin/env bash
#SBATCH -A nvr_qualg_lmbm
#SBATCH -J gly_eqv2_ag
#SBATCH -p batch_singlenode
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 1-08:00:00
#SBATCH --output=/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip/slurm_logs/glycine-eqv2-autograd-%j.out

set -euo pipefail

REPO=${REPO:-/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip}
N_GRID=${N_GRID:-36}
OUTPUT_DIR=${OUTPUT_DIR:-runs/glycine_pt_eqv2_autograd_n36}

cd "$REPO"

export HIP_VENV="$PWD/.venv"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

source scripts/setup_cuda.sh

exec uv run python scripts/glycine_pt_scan.py \
  --h5 data/transition1x.h5 \
  --checkpoint ckpt/eqv2.ckpt \
  --hessian-method autograd \
  --model-label eqv2_autograd \
  --output-prefix eqv2_autograd \
  --output-dir "$OUTPUT_DIR" \
  --n-grid "$N_GRID" \
  --device cuda
