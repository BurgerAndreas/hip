#!/usr/bin/env bash
#SBATCH -A nvr_qualg_lmbm
#SBATCH -J t1x_orca_f
#SBATCH -p cpu
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 6:00:00
#SBATCH --output=/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip/slurm_logs/t1x-orca-force-%A_%a.out

set -euo pipefail

REPO=${REPO:-/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip}
SCAN_DIR=${SCAN_DIR:-runs/t1x_val_force_spectra_100x2x51}
TASK_OFFSET=${TASK_OFFSET:-0}
INPUTS_PER_TASK=${INPUTS_PER_TASK:-1}
SKIP_EXISTING=${SKIP_EXISTING:-1}
ORCA_DIR=${ORCA_DIR:-/lustre/fsw/portfolios/nvr/users/anburger/software/orca_6_1_1}
MPI_DIR=${MPI_DIR:-/cm/shared/apps/openmpi4/gcc/4.1.5}
INPUT_LIST=${INPUT_LIST:-"$REPO/$SCAN_DIR/orca_input_list.txt"}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO/$SCAN_DIR/orca_outputs"}
WORK_ROOT=${WORK_ROOT:-"$REPO/$SCAN_DIR/orca_work"}

export PATH="$ORCA_DIR:$MPI_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ORCA_DIR/lib:$MPI_DIR/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$REPO"

if ldd "$ORCA_DIR/orca_scfgrad_mpi" | grep -q 'not found'; then
  ldd "$ORCA_DIR/orca_scfgrad_mpi" >&2
  exit 1
fi

BASE_INPUT_INDEX=$((TASK_OFFSET + SLURM_ARRAY_TASK_ID * INPUTS_PER_TASK))
for LOCAL_INDEX in $(seq 0 $((INPUTS_PER_TASK - 1))); do
  INPUT_INDEX=$((BASE_INPUT_INDEX + LOCAL_INDEX))
  INPUT=$(sed -n "$((INPUT_INDEX + 1))p" "$INPUT_LIST")
  if [[ -z "${INPUT:-}" ]]; then
    echo "No input for INPUT_INDEX=$INPUT_INDEX; stopping task"
    exit 0
  fi

  STEM=$(basename "$INPUT" .inp)
  FINAL_OUT="$OUTPUT_DIR/$STEM.out"
  if [[ "$SKIP_EXISTING" == "1" && -f "$FINAL_OUT" ]] && grep -q "ORCA TERMINATED NORMALLY" "$FINAL_OUT"; then
    echo "[$(date)] Skipping completed ORCA output for $STEM"
    continue
  fi

  WORK_DIR="$WORK_ROOT/$STEM"
  mkdir -p "$OUTPUT_DIR" "$WORK_DIR"
  rm -rf "$WORK_DIR"/*
  cp "$INPUT" "$WORK_DIR/$STEM.inp"

  cd "$WORK_DIR"
  echo "[$(date)] Starting ORCA EnGrad for $STEM on $(hostname)"
  echo "ORCA: $ORCA_DIR/orca"
  echo "MPI: $MPI_DIR"
  "$ORCA_DIR/orca" "$STEM.inp" > "$STEM.out"
  echo "[$(date)] Finished ORCA EnGrad for $STEM"

  shopt -s nullglob
  cp "$STEM".* "$OUTPUT_DIR"/
  cd "$REPO"
done
