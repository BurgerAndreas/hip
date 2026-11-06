#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -p gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 23:59:00
#SBATCH --output=/project/aip-aspuru/aburger/hip-neighbors-runs-20261102/logs/ablation-%x-%j.out
# Generic 100-epoch ablation (same settings as killarney_train_24h_l40s_bz64.sbatch / short100).
# Required env: NAME, CODE (training repo), OVERRIDES (model.* hydra overrides). Auto-resumes across chained jobs.
set -euo pipefail
: "${NAME:?}" "${CODE:?}" "${OVERRIDES:?}"
VENV=/project/aip-aspuru/aburger/hip-neighbors-retrain-20261102/.venv
RUN=/project/aip-aspuru/aburger/hip-neighbors-runs-20261102
OUT=$RUN/short100-$NAME
DATA=/project/aip-aspuru/aburger/.cache/kagglehub/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm/versions/5
HASH_LOG=/project/aip-aspuru/aburger/hip-neighbors-retrain-20261102/run_logs/horm-sha-7171122.out
grep -q "^a404b9ebf466cd84dc94d77230fa51e6b2850b3f4f135e699267aa15474e7500  .*ts1x_hess_train_big.lmdb$" "$HASH_LOG"
grep -q "^aab37592a397f86711433b4509d687b2d5c4c76c481f62a86e3c5b42eb7b8da0  .*ts1x-val.lmdb$" "$HASH_LOG"
if [[ -f "$OUT/final/0/final-epoch-100.ckpt" ]]; then echo "already complete"; exit 0; fi
mkdir -p "$OUT/final/0" "$OUT/wandb" "$OUT/checkpoints"
cd "$CODE"
export HIP_TRAIN_ROOT="$CODE" HIP_RUN="$RUN" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 WANDB_MODE=offline
export HIP_SHORT_OUTPUT="$OUT/final/0" WANDB_DIR="$OUT/wandb"
echo "code=$CODE commit=$(git rev-parse HEAD) name=$NAME overrides=$OVERRIDES seed=20261102"
# shellcheck disable=SC2086
exec "$VENV/bin/python" "$RUN/short100-20261105-12nG1k/short100_train.py" \
  $OVERRIDES model.max_neighbors=1000 \
  optimizer.optimizer=muon optimizer.lr=0.00044 "optimizer.betas=[0.965,0.965]" \
  optimizer.weight_decay=0.00002 optimizer.lr_muon=0.001 optimizer.weight_decay_muon=0.001 \
  training.bz=64 training.bz_val=64 training.num_workers=2 \
  "training.trn_path=$DATA/ts1x_hess_train_big.lmdb" "training.val_path=$DATA/ts1x-val.lmdb" \
  pltrainer.log_every_n_steps=1000 pltrainer.max_epochs=100 pltrainer.limit_train_batches=0.1 pltrainer.limit_val_batches=0.1 \
  ckpt_do_save=true ckpt_resume_auto=true use_wandb=true "run_name=v3_$NAME" \
  "ckpt_base_dir=$OUT/checkpoints" "hydra.run.dir=$OUT/hydra/$SLURM_JOB_ID"
