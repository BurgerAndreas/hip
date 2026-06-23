#!/usr/bin/env bash
#SBATCH -A nvr_qualg_lmbm
#SBATCH -J gly_mep_orig
#SBATCH -p batch_singlenode
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --output=/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip/slurm_logs/glycine-mep-orig-models-%j.out

set -euo pipefail

REPO=${REPO:-/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip}
MEP_DIR=${MEP_DIR:-runs/glycine_pt_mep_145}
HORM_DIR=${HORM_DIR:-/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/HORM}

cd "$REPO"

export HIP_VENV="$PWD/.venv"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

source scripts/setup_cuda.sh

EQV2_OUT="$MEP_DIR/eqv2_orig_autograd_arrays.npz"
LEFTNET_OUT="$MEP_DIR/leftnet_cf_orig_arrays.npz"

if [[ ! -f "$EQV2_OUT" ]]; then
  uv run python scripts/eval_glycine_pt_mep_eqv2.py \
    --mep-dir "$MEP_DIR" \
    --checkpoint ckpt/eqv2_orig.ckpt \
    --hessian-method autograd \
    --output "$EQV2_OUT" \
    --summary-csv "$MEP_DIR/eqv2_orig_autograd_predictions.csv" \
    --device cuda
else
  echo "Skipping EqV2 orig; already present at $EQV2_OUT"
fi

if [[ ! -f "$LEFTNET_OUT" ]]; then
  uv run python scripts/eval_glycine_pt_mep_leftnet_cf.py \
    --mep-dir "$MEP_DIR" \
    --horm-dir "$HORM_DIR" \
    --checkpoint "$HORM_DIR/ckpt/left_orig.ckpt" \
    --output "$LEFTNET_OUT" \
    --summary-csv "$MEP_DIR/leftnet_cf_orig_predictions.csv" \
    --device cuda
else
  echo "Skipping LeftNet-CF orig; already present at $LEFTNET_OUT"
fi
