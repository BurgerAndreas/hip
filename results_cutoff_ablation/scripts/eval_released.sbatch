#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -J hip-released-eval
#SBATCH -p gpubase_l40s_b1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 00:45:00
#SBATCH --array=0-3
#SBATCH --output=/project/aip-aspuru/aburger/hip-neighbors-runs-20261102/pubchem-eval-20261106/logs/released-%A_%a.out
# Released hip_v2 / hip_v3 (HF 3a0aef9d / 49095693) on PubChem 30-100 atoms, same evaluator, max_neighbors 32 vs 1000.
set -euo pipefail
P=/project/aip-aspuru/aburger
ROOT=$P/hip-pubchem-eval-20261106
OUTPUT=$P/hip-neighbors-runs-20261102/pubchem-eval-20261106
H5_DIR=$P/hip-neighbors-runs-20261102/pubchem-reference/orca_wb97x_631gd_chno_30_100/h5
case "$SLURM_ARRAY_TASK_ID" in
  0) M=hip_v2; N=32 ;; 1) M=hip_v2; N=1000 ;; 2) M=hip_v3; N=32 ;; 3) M=hip_v3; N=1000 ;;
esac
export HIP_EXPECT_BACKBONE=5 HIP_EXPECT_HESSIAN=100 HIP_EXPECT_CONNECTED=true
cd "$ROOT"
export PYTHONPATH="$OUTPUT/python-deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/eval_projected_pubchem.py" \
  --h5-glob "$H5_DIR/*.h5" --ckpt-path "$OUTPUT/released-ckpt/$M.ckpt" --max-neighbors $N \
  --output-dir "$OUTPUT/released/${M}_maxnb$N"
