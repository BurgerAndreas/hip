#!/usr/bin/env bash
#SBATCH -A nvr_qualg_lmbm
#SBATCH -J sel_ad_hess
#SBATCH -p batch_singlenode
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH --output=/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip/slurm_logs/selected-ad-hessians-%j.out

set -euo pipefail

cd /lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip

export HIP_VENV="$PWD/.venv"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

source scripts/setup_cuda.sh

mkdir -p slurm_logs

CHECKPOINT=${CHECKPOINT:-ckpt/eqv2_orig.ckpt}
TAG=${TAG:-eqv2_orig}
SCAN_DIR=${SCAN_DIR:-runs/t1x_val_force_spectra_100x2x51}
OUTPUT_DIR=${OUTPUT_DIR:-}

ARGS=(
  --checkpoint "$CHECKPOINT"
  --tag "$TAG"
  --scan-dir "$SCAN_DIR"
  --device cuda
)

if [[ -n "$OUTPUT_DIR" ]]; then
  ARGS+=(--output-dir "$OUTPUT_DIR")
fi

exec uv run python scripts/eval_selected_ad_hessians.py "${ARGS[@]}"
