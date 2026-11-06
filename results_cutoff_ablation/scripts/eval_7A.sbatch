#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -J hip-7A-eval
#SBATCH -p gpubase_l40s_b1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 00:45:00
#SBATCH --output=/project/aip-aspuru/aburger/hip-neighbors-runs-20261102/pubchem-eval-20261106/logs/projected-7A-%j.out
# PubChem eval of the 7 A / 7 A ablation (uncapped neighbours), same evaluator as projected/0-2.
set -euo pipefail
P=/project/aip-aspuru/aburger
ROOT=$P/hip-pubchem-eval-20261106
RUN=$P/hip-neighbors-runs-20261102
OUTPUT=$RUN/pubchem-eval-20261106
CHECKPOINT=$RUN/short100-7A-20261106/final/0/final-epoch-100.ckpt
test -f "$CHECKPOINT" || { echo "7A checkpoint missing: training not finished" >&2; exit 3; }
export HIP_EXPECT_BACKBONE=7 HIP_EXPECT_HESSIAN=7 HIP_EXPECT_CONNECTED=false
cd "$ROOT"
export PYTHONPATH="$OUTPUT/python-deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/eval_projected_pubchem.py" \
  --h5-glob "$RUN/pubchem-reference/orca_wb97x_631gd_chno_30_100/h5/*.h5" --ckpt-path "$CHECKPOINT" \
  --max-neighbors 1000 --output-dir "$OUTPUT/projected/3"
