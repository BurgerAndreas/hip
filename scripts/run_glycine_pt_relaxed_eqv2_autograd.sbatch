#!/usr/bin/env bash
#SBATCH -A nvr_qualg_lmbm
#SBATCH -J gly_rlx_ag
#SBATCH -p batch_singlenode
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 3:30:00
#SBATCH --output=/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip/slurm_logs/glycine-relaxed-eqv2-autograd-%j.out

set -euo pipefail

REPO=${REPO:-/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip}
SCAN_DIR=${SCAN_DIR:-runs/glycine_pt_scan_relaxed}

cd "$REPO"

export HIP_VENV="$PWD/.venv"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

source scripts/setup_cuda.sh

exec uv run python scripts/eval_glycine_pt_relaxed_mlip.py \
  --scan-dir "$SCAN_DIR" \
  --checkpoint ckpt/eqv2.ckpt \
  --hessian-method autograd \
  --model-label eqv2_autograd \
  --output-prefix eqv2_autograd \
  --device cuda
