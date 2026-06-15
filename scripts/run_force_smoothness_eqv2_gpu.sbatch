#!/usr/bin/env bash
#SBATCH -A nvr_qualg_lmbm
#SBATCH -J force_smooth_eqv2
#SBATCH -p batch_singlenode
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH --output=/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip/slurm_logs/force-smooth-eqv2-%j.out

set -euo pipefail

cd /lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip

export HIP_VENV="$PWD/.venv"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

source scripts/setup_cuda.sh

mkdir -p slurm_logs runs/force_smoothness

# Baseline EquiformerV2 (trained without the HIP Hessian loss). Same geometries / sweep as the
# hip_v3 run so the force-field smoothness and autograd-Hessian reliability can be compared
# directly. If eqv2's force field is comparably rough, non-smoothness is intrinsic to the MLIP;
# if hip_v3 is rougher, the Hessian-matching training trades force smoothness for Hessian accuracy.
COMMON_ARGS=(
  --checkpoint ckpt/eqv2.ckpt
  --h5 data/transition1x.h5
  --split test
  --sample-id 0
  --directions lowest_hess random
  --amplitude 0.2
  --n-scan 2001
  --h-min 1e-5
  --h-max 2e-1
  --n-h 41
  --n-h-full 17
  --dtype float32
  --output-dir runs/force_smoothness
)

echo "=== transition_state (saddle) ==="
uv run python scripts/force_smoothness_scan.py "${COMMON_ARGS[@]}" \
  --geometry transition_state \
  --tag eqv2_test_s0_transition_state_ext

echo "=== reactant (minimum, contrast) ==="
uv run python scripts/force_smoothness_scan.py "${COMMON_ARGS[@]}" \
  --geometry reactant \
  --tag eqv2_test_s0_reactant_ext

echo "DONE"
