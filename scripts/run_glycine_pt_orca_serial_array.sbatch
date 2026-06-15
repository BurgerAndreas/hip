#!/usr/bin/env bash
#SBATCH -A nvr_qualg_lmbm
#SBATCH -J gly_pt_orca1
#SBATCH -p cpu
#SBATCH --array=0-5%3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH -t 2:00:00
#SBATCH --output=/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip/slurm_logs/gly_pt_orca1-%A_%a.out

set -euo pipefail

REPO=${REPO:-/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip}
SCAN_DIR=${SCAN_DIR:-runs/glycine_pt_scan_n36}
TASK_OFFSET=${TASK_OFFSET:-0}
ORCA_DIR=/lustre/fsw/portfolios/nvr/users/anburger/software/orca_6_1_1
MPI_DIR=/cm/shared/apps/openmpi4/gcc/4.1.5
INPUT_LIST=${INPUT_LIST:-"$REPO/$SCAN_DIR/orca_engrad_retry_input_list.txt"}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO/$SCAN_DIR/orca_engrad_retry_outputs"}
WORK_ROOT=${WORK_ROOT:-"$REPO/$SCAN_DIR/orca_engrad_retry_work"}

export PATH="$ORCA_DIR:$MPI_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ORCA_DIR/lib:$MPI_DIR/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$REPO"

INPUT_INDEX=$((SLURM_ARRAY_TASK_ID + TASK_OFFSET))
INPUT=$(sed -n "$((INPUT_INDEX + 1))p" "$INPUT_LIST")
if [[ -z "${INPUT:-}" ]]; then
  echo "No input for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID TASK_OFFSET=$TASK_OFFSET INPUT_INDEX=$INPUT_INDEX" >&2
  exit 0
fi

STEM=$(basename "$INPUT" .inp)
WORK_DIR="$WORK_ROOT/$STEM"
mkdir -p "$OUTPUT_DIR" "$WORK_DIR"
rm -rf "$WORK_DIR"/*
cp "$INPUT" "$WORK_DIR/$STEM.inp"

cd "$WORK_DIR"
echo "[$(date)] Starting serial ORCA for $STEM on $(hostname)"
echo "ORCA: $ORCA_DIR/orca"
"$ORCA_DIR/orca" "$STEM.inp" > "$STEM.out"
echo "[$(date)] Finished ORCA for $STEM"

shopt -s nullglob
cp "$STEM".* "$OUTPUT_DIR"/
