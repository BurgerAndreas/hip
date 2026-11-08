#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -J hip-Hcut-grid
#SBATCH -p gpubase_l40s_b1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 00:15:00
#SBATCH --array=0-44
#SBATCH --output=/project/aip-aspuru/aburger/hip-neighbors-runs-20261102/pubchem-eval-20261106/logs/Hcut-grid-%A_%a.out
# 5 checkpoints x 9 inference Hessian-graph cutoffs on all 710 PubChem molecules (projected evaluator).
set -euo pipefail
P=/project/aip-aspuru/aburger
RUN=$P/hip-neighbors-runs-20261102
ROOT=$P/hip-pubchem-eval-20261106
OUTPUT=$RUN/pubchem-eval-20261106
H5_DIR=$RUN/pubchem-reference/orca_wb97x_631gd_chno_30_100/h5
CUTS=(4 5 6 7 9 12 16 32 full)
MI=$((SLURM_ARRAY_TASK_ID / 9)); C=${CUTS[$((SLURM_ARRAY_TASK_ID % 9))]}
A=$RUN/short100-20261105-12nG1k/final
case $MI in
  0) M=hip_v2;    CK=$OUTPUT/released-ckpt/hip_v2.ckpt; E="5 100 true" ;;
  1) M=hip_v3;    CK=$OUTPUT/released-ckpt/hip_v3.ckpt; E="5 100 true" ;;
  2) M=abl_5_all; CK=$A/0/final-epoch-100.ckpt;          E="5 100 true" ;;
  3) M=abl_5_5;   CK=$A/1/final-epoch-100.ckpt;          E="5 5 false" ;;
  4) M=abl_12_12; CK=$A/2/final-epoch-100.ckpt;          E="12 12 false" ;;
esac
read -r HIP_EXPECT_BACKBONE HIP_EXPECT_HESSIAN HIP_EXPECT_CONNECTED <<< "$E"
export HIP_EXPECT_BACKBONE HIP_EXPECT_HESSIAN HIP_EXPECT_CONNECTED HIP_EVAL_HESSIAN_GRAPH=$C
cd "$ROOT"
export PYTHONPATH="$OUTPUT/python-deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/eval_projected_pubchem.py" \
  --h5-glob "$H5_DIR/*.h5" --ckpt-path "$CK" --max-neighbors 1000 \
  --output-dir "$OUTPUT/cutoff-grid/${M}_H$C"
