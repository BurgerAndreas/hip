#!/usr/bin/env bash
#SBATCH -A nvr_qualg_lmbm
#SBATCH -J gly_pt_path
#SBATCH -p batch_singlenode
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --output=/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip/slurm_logs/glycine-pt-path-%j.out

set -euo pipefail

REPO=${REPO:-/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip}
OUTPUT_DIR=${OUTPUT_DIR:-runs/glycine_pt_path}
N_DENSE=${N_DENSE:-1000}
N_DFT=${N_DFT:-1000}
WRITE_DFT=${WRITE_DFT:-0}   # 1 = also emit ORCA inputs for the DFT subset
HIP_CKPT=${HIP_CKPT:-ckpt/hip_v2.ckpt}
EQV2_CKPT=${EQV2_CKPT:-ckpt/eqv2.ckpt}

cd "$REPO"

# Never overwrite existing MLIP results.
if [[ -e "$OUTPUT_DIR/path_arrays.npz" ]]; then
  echo "Refusing to overwrite existing results: $OUTPUT_DIR/path_arrays.npz" >&2
  exit 1
fi

if [[ "$WRITE_DFT" == "1" ]]; then
  DFT_ARGS=(--n-dft "$N_DFT")
else
  DFT_ARGS=(--skip-dft-inputs)
fi

export HIP_VENV="$PWD/.venv"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

source scripts/setup_cuda.sh

# Runs EQV2 (autograd) and HIP (direct) on the same dense grid, writes path_arrays.npz
# (full EQV2 Hessians, vib eigenvalues, n_negative, asymmetry). With WRITE_DFT=1 it also
# emits the matching ORCA inputs for the DFT subset.
exec uv run python scripts/glycine_pt_path_scan.py \
  --hip-checkpoint "$HIP_CKPT" \
  --eqv2-checkpoint "$EQV2_CKPT" \
  --output-dir "$OUTPUT_DIR" \
  --n-dense "$N_DENSE" \
  "${DFT_ARGS[@]}" \
  --device cuda
