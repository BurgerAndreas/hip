#!/usr/bin/env bash
set -euo pipefail

SCRATCH_ROOT="${ORCA_HESSIAN_ROOT:-$HOME/scratch/hip/orca_hessians}"
VENV_PATH="${ORCA_HESSIAN_VENV:-$SCRATCH_ROOT/.venv}"

mkdir -p "$SCRATCH_ROOT" "$SCRATCH_ROOT/slurm"

if [[ ! -d "$VENV_PATH" ]]; then
  uv venv --python 3.11 "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"
uv pip install "numpy>=2,<3" "h5py>=3.11"

python - <<'PY'
import h5py
import numpy as np

print(f"numpy {np.__version__}")
print(f"h5py {h5py.__version__}")
PY
