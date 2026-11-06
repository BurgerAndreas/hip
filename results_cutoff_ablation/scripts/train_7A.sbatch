#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -J hip-7A-ablation
#SBATCH -p gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 23:59:00
#SBATCH --output=/project/aip-aspuru/aburger/hip-neighbors-runs-20261102/logs/train7A-%j.out
# 7 A backbone / 7 A Hessian ablation; same settings as killarney_train_24h_l40s_bz64.sbatch,
# trained from scratch to epoch 100 (auto-resumes across chained jobs) and saved by short100_train.py.
set -euo pipefail
ROOT=/project/aip-aspuru/aburger/hip-neighbors-retrain-20261102
RUN=/project/aip-aspuru/aburger/hip-neighbors-runs-20261102
OUT=$RUN/short100-7A-20261106
DATA=/project/aip-aspuru/aburger/.cache/kagglehub/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm/versions/5
HASH_LOG="$ROOT/run_logs/horm-sha-7171122.out"
grep -q "^a404b9ebf466cd84dc94d77230fa51e6b2850b3f4f135e699267aa15474e7500  .*ts1x_hess_train_big.lmdb$" "$HASH_LOG"
grep -q "^aab37592a397f86711433b4509d687b2d5c4c76c481f62a86e3c5b42eb7b8da0  .*ts1x-val.lmdb$" "$HASH_LOG"
if [[ -f "$OUT/final/0/final-epoch-100.ckpt" ]]; then echo "already complete"; exit 0; fi
cd "$ROOT"
export HIP_TRAIN_ROOT="$ROOT" HIP_RUN="$RUN" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 WANDB_MODE=offline
export HIP_SHORT_OUTPUT="$OUT/final/0" WANDB_DIR="$OUT/wandb"
echo "commit=$(git rev-parse HEAD) variant=backbone7_hessian7 seed=20261102"
exec "$ROOT/.venv/bin/python" "$OUT/short100_train.py" \
  model.max_radius=7.0 model.max_neighbors=1000 model.cutoff_hessian=100.0 \
  model.hessian_graph_cutoff=7.0 model.fully_connected_hessian=false \
  optimizer.optimizer=muon optimizer.lr=0.00044 "optimizer.betas=[0.965,0.965]" \
  optimizer.weight_decay=0.00002 optimizer.lr_muon=0.001 optimizer.weight_decay_muon=0.001 \
  training.bz=64 training.bz_val=64 training.num_workers=2 \
  "training.trn_path=$DATA/ts1x_hess_train_big.lmdb" "training.val_path=$DATA/ts1x-val.lmdb" \
  pltrainer.log_every_n_steps=1000 pltrainer.max_epochs=100 pltrainer.limit_train_batches=0.1 pltrainer.limit_val_batches=0.1 \
  ckpt_do_save=true ckpt_resume_auto=true use_wandb=true run_name=v3_backbone7_hessian7 \
  "ckpt_base_dir=$OUT/checkpoints" "hydra.run.dir=$OUT/hydra/$SLURM_JOB_ID"
