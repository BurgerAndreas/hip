#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -J hip-pubchem-plot
#SBATCH -p gpubase_l40s_b1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 00:05:00
#SBATCH --output=/project/aip-aspuru/aburger/hip-neighbors-runs-20261102/pubchem-eval-20261106/logs/plot-%j.out
set -euo pipefail
O=/project/aip-aspuru/aburger/hip-neighbors-runs-20261102/pubchem-eval-20261106
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR="$O/matplotlib-cache"
exec /project/aip-aspuru/aburger/hip-neighbors-retrain-20261102/.venv/bin/python "$O/plot_log.py"
