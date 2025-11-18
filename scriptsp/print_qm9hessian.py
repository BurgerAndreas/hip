import pandas as pd
import wandb
from collections import Counter
import re

def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def format_metric(value, decimals: int, bold: bool) -> str:
    text = "-" if value is None else f"{value:.{decimals}f}"
    if bold:
        return f"\\textbf{{ {text} }}"
    return text

"""
goal: print a table like this:
\hline
Solvent & Model & Hessian & Energy $\downarrow$ & Forces $\downarrow$ & Hessian $\downarrow$  & Eigenvalues $\downarrow$ & CosSim $\evec_1$ $\uparrow$ & $\eval_1$ $\downarrow$ \\
 & & & eV/\AA$^2$ & eV/\AA$^2$ & unitless & eV/\AA$^2$ \\
\hline
Vacuum  & E3x & AD & 0.036 & 0.035 & 0.070 & - & - & -  \\
% add other models here
% & EquiformerV2 & AD ...
thf     & E3x & AD & 0.029 & 0.028 & 0.067 & - & - & - \\
% add other models here
Toluene & E3x & AD & 0.032 & 0.029 & 0.062 & - & - & - \\
% add other models here
Water   & E3x & AD & 0.035 & 0.028 & 0.054 & - & - & - \\
% add other models here
    
keep E3x results as is.
"""

TARGET_ROWS = [
    ("AD", "AlphaNet", "autograd"),
    ("AD", "LEFTNet-CF", "autograd"),
    ("AD", "LEFTNet-DF", "autograd"),
    ("AD", "EquiformerV2", "autograd"),
    ("AD", "EquiformerV2 (E-F)", "autograd"),
    ("Predicted", "HIP-EquiformerV2", "predict"),
    ("Predicted", "HIP-EquiformerV2 (end-to-end)", "predict"),
]

METRIC_SPECS = [
    ("energy_mae_per_atom", 3),
    ("forces_mae", 3),
    ("hessian_mae", 3),    
    ("eigval_mae", 3),
    ("eigvec1_cos_eckart", 3),
    ("eigval1_mae_eckart", 3),
    # ("freq_mae_400_4000_dataset", 3), # $\omega$ (4000--400) / cm$^{-1}$
]

api = wandb.Api()
runs = api.runs("andreas-burger/horm", filters={"config.dataset": "RGD1.lmdb"})
print(f"Found {len(runs)} runs")

run_lookup = {}
for run in runs:
    config = run.config
    if config.get("dataset") != "RGD1.lmdb":
        continue
    h_method = config.get("hessian_method")
    if h_method not in {"autograd", "predict"}:
        continue
    key = normalize(run.name)
    run_lookup[key] = run