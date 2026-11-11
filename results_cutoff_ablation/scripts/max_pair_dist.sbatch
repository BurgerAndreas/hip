#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -J t1x-maxdist
#SBATCH -p gpubase_l40s_b1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH --output=/project/aip-aspuru/aburger/hip-neighbors-runs-20261102/pubchem-eval-20261106/logs/t1x-maxdist-%j.out
O=/project/aip-aspuru/aburger/hip-neighbors-runs-20261102/pubchem-eval-20261106
cd /project/aip-aspuru/aburger/hip-neighbors-retrain-20261102
exec .venv/bin/python $O/max_pair_dist.py /project/aip-aspuru/aburger/.cache/kagglehub/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm/versions/5/ts1x_hess_train_big.lmdb
