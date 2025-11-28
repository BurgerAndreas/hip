# Eval on Hessian datasets


## Installation

Install uv (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Env for mace
```bash
# Create virtual environment and install base dependencies
uv venv .macevenv --python 3.12
source .macevenv/bin/activate

uv pip install mace-torch torch==2.6 torch-geometric huggingface-hub plotly[kaleido] wandb pandas ruff
hf auth whoami
# hf auth login

# only if you are on a headless server
# uv run python -c "import plotly.io as pio; pio.get_chrome()"
```

Env for fairchem
```bash
# Create virtual environment and install base dependencies
uv venv .fairvenv --python 3.12
source .fairvenv/bin/activate

uv pip install fairchem-core torch-geometric huggingface-hub plotly[kaleido] wandb pandas ruff
hf auth whoami
# hf auth login

# only if you are on a headless server
# uv run python -c "import plotly.io as pio; pio.get_chrome()"
```

## Run

```bash
source .macevenv/bin/activate

# mace-omol
uv run eval_horm.py --model mace_omol --ckpt extra_large --dataset ts1x-val.lmdb --max_samples 1000

# mace-off
uv run eval_horm.py --model mace_off --ckpt small --dataset ts1x-val.lmdb --max_samples 1000
uv run eval_horm.py --model mace_off --ckpt medium --dataset ts1x-val.lmdb --max_samples 1000
uv run eval_horm.py --model mace_off --ckpt large --dataset ts1x-val.lmdb --max_samples 1000
```

```bash
source .fairvenv/bin/activate

# uma
uv run eval_horm.py --model uma --ckpt s --dataset ts1x-val.lmdb --max_samples 1000
# uv run eval_horm.py --model uma --ckpt m --dataset ts1x-val.lmdb --max_samples 1000

# esen
# direct models are not getting gradients
# uv run eval_horm.py --model esen --ckpt sm-direct --dataset ts1x-val.lmdb --max_samples 1000
uv run eval_horm.py --model esen --ckpt sm-conserving --dataset ts1x-val.lmdb --max_samples 1000
# uv run eval_horm.py --model esen --ckpt md-direct --dataset ts1x-val.lmdb --max_samples 1000
```