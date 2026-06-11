"""
Plot size_eval metrics for all three models together (no error bars).

Usage:
    uv run scripts/plot_size.py
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RESULTS = {
    "HIP": "results_eval_orca_hf_hip_v3/metrics.csv",
    "AD": "results_size_eval/eqv2_dft_geometries_autograd_metrics.csv",
    "AD (E-F)": "results_size_eval/eqv2_orig_dft_geometries_autograd_metrics.csv",
}

METRICS = [
    # ("energy_mae_per_atom", "Energy MAE / atom [eV]"),
    # ("forces_mae", r"Forces MAE [eV/$\AA$]"),
    ("hessian_mae_ev_a2", r"Hessian MAE [eV/$\AA^2$]"),
    ("hessian_rel_mae", "Hessian relative MAE"),
    ("eckart_eigval_mae_ev_a2", r"Eckart $\lambda$ MAE [eV/$\AA^2$]"),
    ("eckart_lowest_eigval_mae_ev_a2", r"Eckart $\lambda_1$ MAE [eV/$\AA^2$]"),
    # ("freq_mae_400_4000", r"Freq MAE 400–4000 [cm$^{-1}$]"),
    # ("asymmetry_mae", "Asymmetry MAE"),
    ("time_ms", "Time [ms]"),
    ("memory_mb", "Memory [MB]"),
]

COLOURS = {
    "HIP": "#ae5a41",
    "AD": "#295c7e",
    "AD (E-F)": "#5a5255",
}

VARIANTS = [
    {"suffix": "", "exclude": []},
    {"suffix": "_no_ef", "exclude": ["AD (E-F)"]},
]

if __name__ == "__main__":
    plot_dir = "plots/size_eval"
    os.makedirs(plot_dir, exist_ok=True)

    dfs = {}
    for label, path in RESULTS.items():
        if not os.path.exists(path):
            print(f"Skipping {label}: {path} not found")
            continue
        dfs[label] = pd.read_csv(path)
        if "status" in dfs[label].columns:
            dfs[label] = dfs[label][dfs[label]["status"] == "ok"].copy()
        print(f"Loaded {label}: {path} ({len(dfs[label])} samples)")

    sns.set_theme(style="whitegrid", context="poster")

    for variant in VARIANTS:
        for col, ylabel in METRICS:
            fig, ax = plt.subplots(figsize=(8, 8))

            for label, df in dfs.items():
                if label in variant["exclude"]:
                    continue
                if col not in df.columns:
                    continue
                grouped = df.groupby("natoms")[col].mean().reset_index()
                ax.plot(
                    grouped["natoms"],
                    grouped[col],
                    marker="o",
                    markersize=4,
                    linewidth=2,
                    label=label,
                    color=COLOURS[label],
                )

            ax.set_xlabel("Number of Atoms")
            ax.set_ylabel(ylabel)
            legend = ax.legend()
            legend.set_title("")
            legend.get_frame().set_edgecolor("none")
            legend.get_frame().set_alpha(1.0)
            plt.tight_layout(pad=0.0)

            plot_path = f"{plot_dir}/size_{col}{variant['suffix']}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"Saved {plot_path}")
            plt.close()
