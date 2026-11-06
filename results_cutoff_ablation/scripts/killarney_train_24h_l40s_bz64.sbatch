#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -J hip-cutoff-train
#SBATCH -p gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 23:59:00
#SBATCH --array=0-2
#SBATCH --output=/project/aip-aspuru/aburger/hip-neighbors-runs-20261102/logs/train-%A_%a.out
set -euo pipefail
ROOT=/project/aip-aspuru/aburger/hip-neighbors-retrain-20261102
RUN=/project/aip-aspuru/aburger/hip-neighbors-runs-20261102
DATA=/project/aip-aspuru/aburger/.cache/kagglehub/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm/versions/5
HASH_LOG="$ROOT/run_logs/horm-sha-7171122.out"
grep -q '^a404b9ebf466cd84dc94d77230fa51e6b2850b3f4f135e699267aa15474e7500  .*ts1x_hess_train_big.lmdb$' "$HASH_LOG"
grep -q '^aab37592a397f86711433b4509d687b2d5c4c76c481f62a86e3c5b42eb7b8da0  .*ts1x-val.lmdb$' "$HASH_LOG"
case "$SLURM_ARRAY_TASK_ID" in
  0) NAME=backbone5_hessian_all; BACKBONE=5.0; HESSIAN=100.0; CONNECTED=true ;;
  1) NAME=backbone5_hessian5; BACKBONE=5.0; HESSIAN=5.0; CONNECTED=false ;;
  2) NAME=backbone12_hessian12; BACKBONE=12.0; HESSIAN=12.0; CONNECTED=false ;;
esac
cd "$ROOT"
export HIP_TRAIN_ROOT="$ROOT" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export WANDB_MODE=offline PYTHONUNBUFFERED=1
echo "commit=$(git rev-parse HEAD) variant=$NAME seed=20261102"
exec "$ROOT/.venv/bin/python" "$RUN/seeded_train.py" \
  "model.max_radius=$BACKBONE" model.max_neighbors=1000 model.cutoff_hessian=100.0 \
  "model.hessian_graph_cutoff=$HESSIAN" "model.fully_connected_hessian=$CONNECTED" \
  optimizer.optimizer=muon optimizer.lr=0.00044 'optimizer.betas=[0.965,0.965]' \
  optimizer.weight_decay=0.00002 optimizer.lr_muon=0.001 optimizer.weight_decay_muon=0.001 \
  training.bz=64 training.bz_val=64 training.num_workers=2 \
  "training.trn_path=$DATA/ts1x_hess_train_big.lmdb" "training.val_path=$DATA/ts1x-val.lmdb" \
  pltrainer.log_every_n_steps=1000 pltrainer.max_epochs=1109 pltrainer.limit_train_batches=0.1 pltrainer.limit_val_batches=0.1 \
  ckpt_do_save=true ckpt_resume_auto=true use_wandb=true "run_name=v3_$NAME" \
  "ckpt_base_dir=../hip-neighbors-runs-20261102/checkpoints" \
  "hydra.run.dir=$RUN/hydra-train-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID"
