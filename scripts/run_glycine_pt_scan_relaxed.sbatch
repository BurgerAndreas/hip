#!/usr/bin/env bash
#SBATCH -A nvr_qualg_lmbm
#SBATCH -J gly_rlx_gen
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH -t 1:00:00
#SBATCH --output=/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip/slurm_logs/glycine-relaxed-gen-%j.out

set -euo pipefail

REPO=${REPO:-/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip}
OUTPUT_DIR=${OUTPUT_DIR:-runs/glycine_pt_scan_relaxed}
FMAX=${FMAX:-0.1}
ORCA_FREQ_NPROCS=${ORCA_FREQ_NPROCS:-4}
ORCA_ENGRAD_NPROCS=${ORCA_ENGRAD_NPROCS:-4}

cd "$REPO"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

exec uv run python -u scripts/glycine_pt_scan_relaxed.py \
  --output-dir "$OUTPUT_DIR" \
  --fmax "$FMAX" \
  --orca-freq-nprocs "$ORCA_FREQ_NPROCS" \
  --orca-engrad-nprocs "$ORCA_ENGRAD_NPROCS"
