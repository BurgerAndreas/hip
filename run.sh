#!/bin/bash

source .macevenv/bin/activate

# mace-omol
uv run eval_horm.py --model mace_omol --ckpt extra_large --dataset ts1x-val.lmdb --max_samples 1000 --redo=True

# mace-off
uv run eval_horm.py --model mace_off --ckpt small --dataset ts1x-val.lmdb --max_samples 1000
uv run eval_horm.py --model mace_off --ckpt medium --dataset ts1x-val.lmdb --max_samples 1000
uv run eval_horm.py --model mace_off --ckpt large --dataset ts1x-val.lmdb --max_samples 1000

deactivate
source .fairvenv/bin/activate

# uma
uv run eval_horm.py --model uma --ckpt s --dataset ts1x-val.lmdb --max_samples 1000
# uv run eval_horm.py --model uma --ckpt m --dataset ts1x-val.lmdb --max_samples 1000

# esen
uv run eval_horm.py --model esen --ckpt sm-conserving --dataset ts1x-val.lmdb --max_samples 1000