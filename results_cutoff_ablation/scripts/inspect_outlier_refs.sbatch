#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -J hip-pubchem-refcheck
#SBATCH -p gpubase_l40s_b1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 00:05:00
#SBATCH --output=/project/aip-aspuru/aburger/hip-neighbors-runs-20261102/pubchem-eval-20261106/logs/refcheck-%j.out
set -euo pipefail
O=/project/aip-aspuru/aburger/hip-neighbors-runs-20261102/pubchem-eval-20261106
export PYTHONPATH="$O/python-deps${PYTHONPATH:+:$PYTHONPATH}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec /project/aip-aspuru/aburger/hip-neighbors-retrain-20261102/.venv/bin/python "$O/inspect_outlier_refs.py"
