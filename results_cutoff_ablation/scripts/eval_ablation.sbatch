#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -p gpubase_l40s_b1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 00:45:00
#SBATCH --output=/project/aip-aspuru/aburger/hip-neighbors-runs-20261102/pubchem-eval-20261106/logs/projected-%x-%j.out
# PubChem projected eval of a 100-epoch ablation. Env: NAME, EVROOT, HIP_EXPECT_BACKBONE/HESSIAN/CONNECTED.
set -euo pipefail
P=/project/aip-aspuru/aburger
RUN=$P/hip-neighbors-runs-20261102
OUTPUT=$RUN/pubchem-eval-20261106
CHECKPOINT=$RUN/short100-$NAME/final/0/final-epoch-100.ckpt
test -f "$CHECKPOINT" || { echo "checkpoint missing: training not finished" >&2; exit 3; }
cd "$EVROOT"
export PYTHONPATH="$OUTPUT/python-deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec "$EVROOT/.venv/bin/python" "$EVROOT/scripts/eval_projected_pubchem.py" \
  --h5-glob "$RUN/pubchem-reference/orca_wb97x_631gd_chno_30_100/h5/*.h5" --ckpt-path "$CHECKPOINT" \
  --max-neighbors 1000 --output-dir "$OUTPUT/projected-$NAME"
