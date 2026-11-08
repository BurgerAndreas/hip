#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -J hip-released-Hcut
#SBATCH -p gpubase_l40s_b1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH --array=0-4
#SBATCH --output=/project/aip-aspuru/aburger/hip-neighbors-runs-20261102/pubchem-eval-20261106/logs/released-Hcut-%A_%a.out
# Released hip_v2 / hip_v3 on all 710 PubChem molecules with a radius Hessian graph at inference (cutoff sweep follow-up).
set -euo pipefail
P=/project/aip-aspuru/aburger
ROOT=$P/hip-pubchem-eval-20261106
OUTPUT=$P/hip-neighbors-runs-20261102/pubchem-eval-20261106
H5_DIR=$P/hip-neighbors-runs-20261102/pubchem-reference/orca_wb97x_631gd_chno_30_100/h5
case "$SLURM_ARRAY_TASK_ID" in
  0) M=hip_v3; C=6 ;; 1) M=hip_v3; C=7 ;; 2) M=hip_v3; C=9 ;; 3) M=hip_v2; C=4 ;; 4) M=hip_v2; C=6 ;;
esac
export HIP_EXPECT_BACKBONE=5 HIP_EXPECT_HESSIAN=100 HIP_EXPECT_CONNECTED=true HIP_EVAL_HESSIAN_GRAPH=$C
cd "$ROOT"
export PYTHONPATH="$OUTPUT/python-deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/eval_projected_pubchem.py" \
  --h5-glob "$H5_DIR/*.h5" --ckpt-path "$OUTPUT/released-ckpt/$M.ckpt" --max-neighbors 1000 \
  --output-dir "$OUTPUT/released/${M}_maxnb1000_H$C"
