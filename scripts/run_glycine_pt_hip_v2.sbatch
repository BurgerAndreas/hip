#!/usr/bin/env bash
#SBATCH -A nvr_qualg_lmbm
#SBATCH -J gly_hip_v2
#SBATCH -p batch_singlenode
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH --output=/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip/slurm_logs/glycine-hip-v2-%j.out

set -euo pipefail

REPO=${REPO:-/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip}
N_GRID=${N_GRID:-36}
OUTPUT_DIR=${OUTPUT_DIR:-runs/glycine_pt_scan_n36}

cd "$REPO"

export HIP_VENV="$PWD/.venv"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

source scripts/setup_cuda.sh

exec uv run python scripts/glycine_pt_scan.py \
  --h5 data/transition1x.h5 \
  --checkpoint ckpt/hip_v2.ckpt \
  --hessian-method predict \
  --model-label hip_v2 \
  --output-prefix hip_v2 \
  --output-dir "$OUTPUT_DIR" \
  --n-grid "$N_GRID" \
  --device cuda
