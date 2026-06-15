#!/usr/bin/env bash
#SBATCH -A nvr_qualg_lmbm
#SBATCH -J gly_mep_mlip
#SBATCH -p batch_singlenode
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --output=/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip/slurm_logs/glycine-mep-mlip-%j.out

set -euo pipefail

REPO=${REPO:-/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip}
OUTPUT_DIR=${OUTPUT_DIR:-runs/glycine_pt_mep_73}
TRAJECTORY_XYZ=${TRAJECTORY_XYZ:-runs/glycine_pt_xyzrender/xyz/reaction_path.xyz}
CHECKPOINT=${CHECKPOINT:-ckpt/hip_v2.ckpt}
HESSIAN_METHOD=${HESSIAN_METHOD:-predict}
MODEL_LABEL=${MODEL_LABEL:-hip_v2}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-$MODEL_LABEL}

cd "$REPO"

export HIP_VENV="$PWD/.venv"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

source scripts/setup_cuda.sh

exec uv run python scripts/glycine_pt_mep_hessian_scan.py \
  --h5 data/transition1x.h5 \
  --trajectory-xyz "$TRAJECTORY_XYZ" \
  --output-dir "$OUTPUT_DIR" \
  --checkpoint "$CHECKPOINT" \
  --hessian-method "$HESSIAN_METHOD" \
  --model-label "$MODEL_LABEL" \
  --output-prefix "$OUTPUT_PREFIX" \
  --device cuda
