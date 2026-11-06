#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -J hip-pubchem-eval
#SBATCH -p gpubase_l40s_b1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 00:45:00
#SBATCH --array=0-2
#SBATCH --output=/project/aip-aspuru/aburger/hip-neighbors-runs-20261102/pubchem-eval-20261106/logs/eval-%A_%a.out
set -euo pipefail
P=/project/aip-aspuru/aburger
ROOT=$P/hip-pubchem-eval-20261106
RUN=$P/hip-neighbors-runs-20261102
OUTPUT=$RUN/pubchem-eval-20261106
CHECKPOINT=$RUN/short100-20261105-12nG1k/final/$SLURM_ARRAY_TASK_ID/final-epoch-100.ckpt
case "$SLURM_ARRAY_TASK_ID" in
  0) export HIP_EXPECT_BACKBONE=5 HIP_EXPECT_HESSIAN=100 HIP_EXPECT_CONNECTED=true ;;
  1) export HIP_EXPECT_BACKBONE=5 HIP_EXPECT_HESSIAN=5 HIP_EXPECT_CONNECTED=false ;;
  2) export HIP_EXPECT_BACKBONE=12 HIP_EXPECT_HESSIAN=12 HIP_EXPECT_CONNECTED=false ;;
esac
case "${HIP_PUBCHEM_MODE:-full}" in
  smoke) H5_DIR=$OUTPUT/smoke-input; DEST=$OUTPUT/smoke-original/$SLURM_ARRAY_TASK_ID ;;
  full) H5_DIR=$RUN/pubchem-reference/orca_wb97x_631gd_chno_30_100/h5; DEST=$OUTPUT/results/$SLURM_ARRAY_TASK_ID ;;
  *) echo "Unknown evaluation mode" >&2; exit 2 ;;
esac
cd "$ROOT"
export PYTHONPATH="$OUTPUT/python-deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/eval_original_rebuttal.py" \
  --h5-glob "$H5_DIR/*.h5" --ckpt-path "$CHECKPOINT" --output-dir "$DEST"
