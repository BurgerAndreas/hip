#!/usr/bin/env bash
#SBATCH -A nvr_qualg_lmbm
#SBATCH -J force_wiggle
#SBATCH -p batch_singlenode
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH --array=0-9
#SBATCH --output=/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip/slurm_logs/force-wiggle-%A_%a.out

set -euo pipefail

cd /lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip

export HIP_VENV="$PWD/.venv"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

source scripts/setup_cuda.sh

CHUNKS=(
  "0,1,2,3,4,5,6,7,8,9"
  "10,11,12,13,14,15,16,17,18,19"
  "20,21,22,23,24,25,26,27,28,29"
  "30,31,32,33,34,35,36,37,38,39"
  "40,41,42,43,44,45,46,47,48,49"
  "50,51,52,53,54,55,56,57,58,59"
  "60,61,62,63,64,65,66,67,68,69"
  "70,71,72,73,74,75,76,77,78,79"
  "80,81,82,83,84,85,86,87,88,89"
  "90,91,92,93,94,95,96,97,98,99"
)

DATASET_INDICES="${CHUNKS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="runs/force_wiggle_gpu/chunk_${SLURM_ARRAY_TASK_ID}"

echo "Running force wiggle diagnostic"
echo "  task=${SLURM_ARRAY_TASK_ID}"
echo "  indices=${DATASET_INDICES}"
echo "  output_dir=${OUTPUT_DIR}"
echo "  cuda_available=$(uv run python - <<'PY'
import torch
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
)"

exec uv run python scripts/force_wiggle_diagnostic.py \
  --checkpoint ckpt/eqv2.ckpt \
  --dataset data/sample_100.lmdb \
  --output-dir "${OUTPUT_DIR}" \
  --dataset-indices "${DATASET_INDICES}" \
  --directions-per-structure 2 \
  --n-t 81 \
  --t-max 0.02 \
  --fd-eps 0.00003,0.0001,0.0003,0.001,0.003,0.01 \
  --seed "$((2400 + SLURM_ARRAY_TASK_ID))" \
  --max-plots 6 \
  --device cuda
