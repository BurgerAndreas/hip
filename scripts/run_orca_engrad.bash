#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRATCH_ROOT="${ORCA_HESSIAN_ROOT:-$HOME/scratch/hip/orca_hessians}"
VENV_PATH="${ORCA_HESSIAN_VENV:-$SCRATCH_ROOT/.venv}"
NPROCS="${NPROCS:-${SLURM_NTASKS:-${SLURM_CPUS_PER_TASK:-8}}}"
ORCA_MODULE="${ORCA_MODULE:-orca/6.1.1}"

if command -v module >/dev/null 2>&1; then
  module load "$ORCA_MODULE"
fi

if [[ ! -d "$VENV_PATH" ]]; then
  echo "Missing venv: $VENV_PATH" >&2
  echo "Run scripts/setup_orca_hessian_venv.bash first." >&2
  exit 1
fi

source "$VENV_PATH/bin/activate"

if [[ ! -f "$SCRATCH_ROOT/jobs.tsv" ]]; then
  echo "Missing manifest: $SCRATCH_ROOT/jobs.tsv" >&2
  echo "Run scripts/prep_orca_hessian_inputs.py before submitting the array." >&2
  exit 1
fi

if [[ -z "${ORCA_BIN:-}" && -n "${EBROOTORCA:-}" && -x "${EBROOTORCA}/orca" ]]; then
  ORCA_BIN="${EBROOTORCA}/orca"
fi
ORCA_BIN="${ORCA_BIN:-$(command -v orca || true)}"
if [[ -z "$ORCA_BIN" ]]; then
  echo "Could not find ORCA. Set ORCA_BIN or load a working ORCA module." >&2
  exit 1
fi

select_jobs() {
  python - "$SCRATCH_ROOT/jobs.tsv" "${SAMPLE_NAME:-}" "${SAMPLE_INDEX:-}" <<'PY'
import csv
import sys

jobs_tsv, sample_name, sample_index = sys.argv[1:4]
with open(jobs_tsv, newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))

if sample_name:
    rows = [row for row in rows if row["name"] == sample_name]
elif sample_index:
    rows = [rows[int(sample_index)]]

for row in rows:
    print(
        "\t".join(
            row[key]
            for key in (
                "name",
                "workdir",
                "hessian_path",
                "engrad_input_path",
                "engrad_output_path",
                "engrad_path",
            )
        )
    )
PY
}

while IFS=$'\t' read -r name workdir hessian_path engrad_input_path engrad_output_path engrad_path; do
  if [[ -z "$name" ]]; then
    continue
  fi

  if [[ -f "$engrad_path" && "${OVERWRITE:-0}" != "1" ]]; then
    echo "Skipping $name because $engrad_path already exists. Set OVERWRITE=1 to rerun."
  else
    echo "Running ORCA EnGrad for $name in $workdir"
    (cd "$workdir" && "$ORCA_BIN" "$engrad_input_path" > "$engrad_output_path")
  fi

  if [[ ! -f "$engrad_path" ]]; then
    echo "ORCA did not produce a gradient at $engrad_path" >&2
    exit 1
  fi

  if [[ -f "$hessian_path" ]]; then
    python "$REPO_ROOT/scripts/collect_orca_hessian_results.py" \
      --scratch-root "$SCRATCH_ROOT" \
      --sample-name "$name"
  else
    echo "Gradient is present for $name; waiting for Hessian before HDF5 collection."
  fi
done < <(select_jobs)
