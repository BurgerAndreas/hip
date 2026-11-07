#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -J hip-graph-probe
#SBATCH -p gpubase_l40s_b1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 01:30:00
#SBATCH --output=/project/aip-aspuru/aburger/hip-neighbors-runs-20261102/pubchem-eval-20261106/logs/probe-%j.out
set -euo pipefail
ROOT=/project/aip-aspuru/aburger/hip-pubchem-eval-20261106
O=/project/aip-aspuru/aburger/hip-neighbors-runs-20261102/pubchem-eval-20261106
cd $ROOT
export PYTHONPATH="$ROOT:$O/python-deps" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec $ROOT/.venv/bin/python $ROOT/scripts/probe_graph.py
