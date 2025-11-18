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

r"""
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

SOLVENTS = ["Vacuum", "thf", "Toluene", "Water"]

TARGET_MODELS = [
    ("AD", "EquiformerV2", "autograd"),
    ("AD", "EquiformerV2 (E-F)", "autograd"),
    ("AD", "EquiformerV2-CF", "autograd"),
    ("Predicted", "HIP-EquiformerV2", "predict"),
    ("Predicted", "HIP-EquiformerV2 (end-to-end)", "predict"),
]

# E3x hardcoded results from docstring
E3X_RESULTS = {
    "Vacuum": (0.036, 0.035, 0.070),
    "thf": (0.029, 0.028, 0.067),
    "Toluene": (0.032, 0.029, 0.062),
    "Water": (0.035, 0.028, 0.054),
}

METRIC_SPECS = [
    ("energy_mae_per_atom", 3),
    ("forces_mae", 3),
    ("hessian_mae", 3),    
    ("eigval_mae", 3),
    ("eigvec1_cos_eckart", 3),
    ("eigval1_mae_eckart", 3),
]


def extract_solvent(dataset_path: str) -> str:
    """Extract solvent name from dataset path like '...:water' -> 'Water'"""
    if ":" not in dataset_path:
        return None
    solvent = dataset_path.split(":")[-1].lower()
    # Map to proper case
    mapping = {"vacuum": "Vacuum", "thf": "thf", "toluene": "Toluene", "water": "Water"}
    return mapping.get(solvent)


api = wandb.Api()
runs = api.runs("andreas-burger/horm")
print(f"Found {len(runs)} runs")

# Build lookup: (model_key, solvent, method) -> run
run_lookup = {}
for run in runs:
    config = run.config
    dataset = config.get("dataset", "")
    if "qm9" not in dataset.lower():
        continue
    h_method = config.get("hessian_method")
    if h_method not in {"autograd", "predict"}:
        continue
    solvent = extract_solvent(dataset)
    if solvent is None:
        continue
    model_key = normalize(run.name)
    run_lookup[(model_key, solvent, h_method)] = run

# Build target rows: (solvent, hessian_label, model_name, method)
target_rows = []
for solvent in SOLVENTS:
    # Add E3x row
    target_rows.append((solvent, "AD", "E3x", "autograd", True))  # True = is_e3x
    # Add other models
    for h_label, model_name, method in TARGET_MODELS:
        model_key = normalize(model_name)
        if (model_key, solvent, method) in run_lookup:
            target_rows.append((solvent, h_label, model_name, method, False))

records = []
for solvent, h_label, model_name, method, is_e3x in target_rows:
    record = {
        "Solvent": solvent,
        "Model": model_name,
        "Hessian": h_label,
        "bold": method == "predict",
        "is_e3x": is_e3x,
    }
    if is_e3x:
        # Use hardcoded E3x values (only first 3 metrics)
        e3x_vals = E3X_RESULTS[solvent]
        record["energy_mae_per_atom"] = e3x_vals[0]
        record["forces_mae"] = e3x_vals[1]
        record["hessian_mae"] = e3x_vals[2]
        record["eigval_mae"] = None
        record["eigvec1_cos_eckart"] = None
        record["eigval1_mae_eckart"] = None
    else:
        model_key = normalize(model_name)
        run = run_lookup.get((model_key, solvent, method))
        if run is None:
            raise RuntimeError(f"No run found for model '{model_name}', solvent '{solvent}', method '{method}'")
        if run.config.get("hessian_method") != method:
            raise RuntimeError(
                f"Run '{run.name}' has unexpected hessian_method {run.config.get('hessian_method')}"
            )
        summary = run.summary._json_dict
        for metric, _ in METRIC_SPECS:
            record[metric] = summary.get(metric)
    records.append(record)

df = pd.DataFrame(records)
print(df)

# Generate LaTeX table
group_counts = Counter(row[0] for row in target_rows)
group_progress = {label: 0 for label in group_counts}

print()
print("\\begin{tabular}{lllccccc}")
print("Solvent & Model & Hessian & Energy $\\downarrow$ & Forces $\\downarrow$ & Hessian $\\downarrow$  & Eigenvalues $\\downarrow$ & CosSim $\\evec_1$ $\\uparrow$ & $\\eval_1$ $\\downarrow$ \\\\")
print(" & & & eV/\\AA$^2$ & eV/\\AA$^2$ & unitless & eV/\\AA$^2$ \\\\")
print("\\hline")
for record in df.to_dict("records"):
    solvent = record["Solvent"]
    if group_progress[solvent] == 0:
        first_col = f"\\multirow{{{group_counts[solvent]}}}{{*}}{{{solvent}}}"
    else:
        first_col = ""
    
    metrics_text = []
    for m, decimals in METRIC_SPECS:
        val = record.get(m)
        if record["is_e3x"] and m not in ["energy_mae_per_atom", "forces_mae", "hessian_mae"]:
            metrics_text.append("-")
        else:
            metrics_text.append(format_metric(val, decimals, record["bold"]))
    
    line = " & ".join(
        [first_col, record["Model"], record["Hessian"], *metrics_text]
    ) + " \\\\"
    print(line)
    group_progress[solvent] += 1
    if group_progress[solvent] == group_counts[solvent]:
        print("\\hline")
print("\\end{tabular}")