#!/usr/bin/env bash
#SBATCH -A nvr_qualg_lmbm
#SBATCH -J gly_mep_lncf
#SBATCH -p batch_singlenode
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH --output=/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip/slurm_logs/glycine-mep-leftnet-cf-%j.out

set -euo pipefail

REPO=${REPO:-/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip}
MEP_DIR=${MEP_DIR:-runs/glycine_pt_path_n150}
HORM_DIR=${HORM_DIR:-/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/HORM}
CHECKPOINT=${CHECKPOINT:-$HORM_DIR/ckpt/left.ckpt}

cd "$REPO"

export HIP_VENV="$PWD/.venv"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

source scripts/setup_cuda.sh

exec uv run python scripts/eval_glycine_pt_mep_leftnet_cf.py \
  --mep-dir "$MEP_DIR" \
  --horm-dir "$HORM_DIR" \
  --checkpoint "$CHECKPOINT" \
  --device cuda
