#!/usr/bin/env bash
#SBATCH -A nvr_qualg_lmbm
#SBATCH -J force_smooth
#SBATCH -p batch_singlenode
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH --output=/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip/slurm_logs/force-smooth-%j.out

set -euo pipefail

cd /lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip

export HIP_VENV="$PWD/.venv"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

source scripts/setup_cuda.sh

mkdir -p slurm_logs runs/force_smoothness

# Extended finite-difference sweep: smaller h_min (probe deeper) in float32. The model's
# SO3/Wigner rotation does not support float64, but float32's noise floor (~eps*|F|/h ~ 2e-4 at
# h=1e-5) is still ~1000x below the observed error. If the FD-vs-autograd error still plateaus
# instead of following the O(h^2) guide down to that floor, that is non-smoothness (not roundoff).
COMMON_ARGS=(
  --checkpoint ckpt/hip_v3.ckpt
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
  --tag hip_v3_test_s0_transition_state_ext

echo "=== reactant (minimum, contrast) ==="
uv run python scripts/force_smoothness_scan.py "${COMMON_ARGS[@]}" \
  --geometry reactant \
  --tag hip_v3_test_s0_reactant_ext

echo "DONE"
