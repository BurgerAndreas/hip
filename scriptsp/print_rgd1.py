import pandas as pd
import wandb
from collections import Counter
import re

r"""
Goal: print a table like this:
\multirow{2}{*}{Hessian} & \multirow{2}{*}{Model} & Hessian $\downarrow$  & Eigenvalues $\downarrow$ & CosSim $\evec_1$ $\uparrow$ & $\eval_1$ $\downarrow$ & Time $\downarrow$ \\
 & & eV/\AA$^2$ & eV/\AA$^2$ & unitless & eV/\AA$^2$ & ms \\
\hline
\multirow{4}{*}{AD} 
 & AlphaNet & 0.259 & 0.148 & 0.415 & 0.040 & 767.0 \\
 & LEFTNet-CF & 0.226 & 0.130 & 0.244 & 0.015 & 1110.7 \\
 & LEFTNet-DF & 0.304 & 0.142 & 0.290 & 0.013 & 341.3 \\
 & EquiformerV2 & 0.133 & 0.056 & 0.092 & 0.003 & 633.0 \\
 & EquiformerV2 (E-F) & 0.243 & 0.111 & 1.224 & 0.106 & 633.0 \\
\hline
\multirow{1}{*}{Predicted } & HIP-EquiformerV2 & \textbf{ 0.030 } & \textbf{ 0.063 } & \textbf{ 0.982 } & \textbf{ 0.031 } & \textbf{ 38.5 } \\
\multirow{1}{*}{Predicted } & HIP-EquiformerV2 (end-to-end) & \textbf{ 0.030 } & \textbf{ 0.063 } & \textbf{ 0.982 } & \textbf{ 0.031 } & \textbf{ 38.5 } \\
    
Relevant columns:
hessian_mae, eigval_mae, eigvec1_cos_eckart, eigval1_mae_eckart, time

config:
hessian_method=autograd, predict
run.name should match the model name up to capitaliziation. discard runs that do not match the model name.
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
    ("hessian_mae", 3),
    ("eigval_mae", 3),
    ("eigvec1_cos_eckart", 3),
    ("eigval1_mae_eckart", 3),
    ("time", 1),
]


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def format_metric(value, decimals: int, bold: bool) -> str:
    text = "-" if value is None else f"{value:.{decimals}f}"
    if bold:
        return f"\\textbf{{ {text} }}"
    return text


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

records = []
for h_label, model_name, expected_method in TARGET_ROWS:
    key = normalize(model_name)
    run = run_lookup.get(key)
    if run is None:
        raise RuntimeError(f"No run found for model '{model_name}'")
    if run.config.get("hessian_method") != expected_method:
        raise RuntimeError(
            f"Run '{run.name}' has unexpected hessian_method {run.config.get('hessian_method')}"
        )
    summary = run.summary._json_dict
    record = {
        "Hessian": h_label,
        "Model": model_name,
        "bold": expected_method == "predict",
    }
    for metric, _ in METRIC_SPECS:
        record[metric] = summary.get(metric)
    records.append(record)

df = pd.DataFrame(records)
print(df)

group_counts = Counter(row[0] for row in TARGET_ROWS)
group_progress = {label: 0 for label in group_counts}

print()
# print("\\begin{tabular}{llccccc}")
print(
    "\\multirow{2}{*}{Hessian} & \\multirow{2}{*}{Model} & Hessian $\\downarrow$  & Eigenvalues $\\downarrow$ & CosSim $\\evec_1$ $\\uparrow$ & $\\eval_1$ $\\downarrow$ & Time $\\downarrow$ \\\\"
)
print(" &  & eV/\\AA$^2$ & eV/\\AA$^2$ & unitless & eV/\\AA$^2$ & ms \\\\")
print("\\hline")
for record in df.to_dict("records"):
    h_label = record["Hessian"]
    if group_progress[h_label] == 0:
        first_col = f"\\multirow{{{group_counts[h_label]}}}{{*}}{{{h_label}}}"
    else:
        first_col = ""
    metrics_text = [
        format_metric(record[m], decimals, record["bold"])
        for m, decimals in METRIC_SPECS
    ]
    line = " & ".join([first_col, record["Model"], *metrics_text]) + " \\\\"
    print(line)
    group_progress[h_label] += 1
    if group_progress[h_label] == group_counts[h_label]:
        print("\\hline")
# print("\\end{tabular}")
