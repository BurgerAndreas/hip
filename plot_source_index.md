# Plot Source Index

This file records where to find the plotting code and data for the three figures discussed on 2026-06-15.

## Speed, Memory, And Eigenvalue Scaling

Image contents:
- `Time per Sample (ms)`
- `Peak Memory (MB)`
- `Eigenvalues λ MAE (RGD1)`
- `Eigenvalues λ MAE (PubChem)`

Where to find it:
- Branch: `natcompsci-rebuttal`
- Main plotting script: `plotting/plot_speedmemory_lambda.py`
- Additional PubChem subpanel scripts:
  - `plotting/plot_orca_pubchem_lambda_mae.py`
  - `plotting/plot_orca_pubchem_lambda_subpanel.py`
- Rendered outputs:
  - `plots/speed_memory_lambda_scaling.png`
  - `plots/speed_memory_lambda_scaling_rebuttal.png`
  - `plots/speed_memory_lambda_scaling_seaborn.png`

Data inputs in this repository on `natcompsci-rebuttal`:

Current inputs used by `plotting/plot_speedmemory_lambda.py`:
- `results_speed2/ts1x-val.lmdb_speed_comparison_extended_10_r100.0_rh100.0.csv`
- `results_evalhorm/hesspred_v2_RGD1_predict_metrics.parquet`
- `results_evalhorm/eqv2_RGD1_autograd_metrics.parquet`
- `results_evalhorm/eqv2_orig_RGD1_autograd_metrics.parquet`
- `results_size_eval/eqv2_orig_dft_geometries_autograd_metrics.parquet`
- `results_speed/orca_pubchem_lambda_mae_outliers.csv`
- `results_speed/orca_pubchem_lambda_subpanel_removed_outliers.csv`
- `results_eval_largehessians_orca_hf_horm_eqv2_autograd/metrics.parquet`
- `results_eval_largehessians_orca_hip_v2/metrics.parquet`
- `results_eval_largehessians_orca_hip_v3/metrics.csv`

Parquet metric files now used by related TS1x validation diagnostics:
- `results_evalhorm/eqv2_ts1x-val_autograd_metrics.parquet`
- `results_evalhorm/hip_v2_ts1x-val_predict_metrics.parquet`
- `results_evalhorm/eqv2_orig_ts1xval10k_29148768_ts1x-val_autograd_metrics.parquet`

Useful commands:

```bash
git switch natcompsci-rebuttal
```

```bash
uv run python plotting/plot_speedmemory_lambda.py
```

## Relaxation And ReactBench TS Search

Image contents:
- `Steps to Convergence`
- `Wall Time [s] (Subset)`
- `TS Search (ReactBench)`

Where to find it:
- Repository: `../gad-ff`
- Branch/ref: `origin/main`
- Combined three-panel plotting/generation script: `scripts/second_order_relaxation_pysiyphus.py`
  - Note the filename spelling: `pysiyphus`.
- ReactBench-only plotting script: `scripts/plot_reactbench.py`
- Local copied plotting script: `plotting/plot_steps_walltime_reactbench.py`
- ReactBench-only rendered outputs on `origin/main`:
  - `results_reactbench/plots/reactbench/reactbench.png`
  - `results_reactbench/plots/reactbench/reactbench_lollipop.png`
  - `results_reactbench/plots/reactbench/reactbench_lollipop_square.png`
  - `results_reactbench/plots/reactbench/reactbench_lollipop_wide.png`

Data inputs:
- Local default panel a/b data:
  - `data/reactbench_relaxation/relaxation_results_noiserms0.035.csv`
- Local default panel c data:
  - `data/reactbench_relaxation/reactbench.csv`
- Original panel c data: `../gad-ff` `origin/main:results/reactbench.csv`
- Original panels a/b command from notes:
  - `uv run scripts/second_order_relaxation_pysiyphus.py --max_samples 80 --thresh gau --max_cycles 150 --xyz /ssd/Code/Datastore/t1x/t1x_val_reactant_hessian_100_noiserms0.035.h5`
- Original panel a/b output copied from:
  - `../gad-ff/runs_relaxation/t1x_val_reactant_hessian_100_noiserms0.035_581483_redund_gau_80_pddftFalse_pdpredFalse_pdthresh0/relaxation_results.csv`
- The combined output is written under that run directory as:
  - `runs_relaxation/.../plots/steps_walltime_reactbench_plotly.png`

Important caveat:
- The local script renders from copied CSVs rather than rerunning the old relaxation workflow.
- I still did not find a committed combined `steps_walltime_reactbench_plotly.png` in `../gad-ff`.

Useful commands:

```bash
git -C ../gad-ff show origin/main:scripts/second_order_relaxation_pysiyphus.py
```

```bash
git -C ../gad-ff show origin/main:scripts/plot_reactbench.py
```

```bash
git -C ../gad-ff show origin/main:results/reactbench.csv
```

```bash
uv run python plotting/plot_steps_walltime_reactbench.py
```

## Data Scaling

Image contents:
- `Energy`
- `Force`
- `HIP Hessian`
- `Number of Training Samples`
- `Energy-Force`
- `Energy-Force-Hessian (HIP)`

Where to find it:
- Local plotting script:
  - `plotting/plot_datascaling.py`
- Local plotting support:
  - `hip/colours.py`
- Source plot script refs in this repo:
  - `fd:plotting/plot_datascaling.py`
  - `origin/fd:plotting/plot_datascaling.py`
  - `origin/results:plotting/plot_datascaling.py`
- Source data ref in this repo:
  - `origin/datascaling`
- Rendered output:
  - `plots/datascaling/datascaling_energy_force_hessian.png`

Local data inputs copied from `origin/datascaling`:
- `scaling/wandb_datascaling_loss_energy2.csv`
- `scaling/wandb_datascaling_loss_force2.csv`
- `scaling/wandb_datascaling_loss_hessian2.csv`

Related older/non-final data on `origin/datascaling`:
- `scaling/wandb_datascaling_loss_energy.csv`
- `scaling/wandb_datascaling_loss_force.csv`
- `scaling/wandb_datascaling_loss_hessian.csv`
- `wandb_datascaling.parquet`

Important caveat:
- The script titles panel c as `Hessian`; the shared image title is `HIP Hessian`. The plotted data/legend/axes otherwise match the figure.
- The exact script and CSV inputs were split across branches: the final three-panel assembly script was on `fd`/`origin/results`, while the CSV inputs were on `origin/datascaling`.
- `plotting/plot_datascaling.py` now renders with Matplotlib/Seaborn, using the same visual style as `plotting/plot_speedmemory_lambda.py`.

Useful commands:

```bash
uv run python plotting/plot_datascaling.py
```

```bash
git show origin/datascaling:scaling/wandb_datascaling_loss_energy2.csv
```

```bash
git show origin/datascaling:scaling/wandb_datascaling_loss_force2.csv
```

```bash
git show origin/datascaling:scaling/wandb_datascaling_loss_hessian2.csv
```

