"""
Plot size_eval metrics for all three models together (no error bars).

Usage:
    uv run scripts/plot_size.py
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = {
    # "HIP": "results_eval_largehessians_orca_hip_v3/metrics.parquet",
    "HIP": "results_eval_largehessians_orca_hip_v2/metrics.parquet",
    "AD": "results_eval_largehessians_orca_hf_horm_eqv2_autograd/metrics.parquet",
    "AD (E-F)": "results_size_eval/eqv2_orig_dft_geometries_autograd_metrics.parquet",
}

OUTLIER_METRIC = "hessian_mae_ev_a2"
OUTLIER_MODIFIED_Z_THRESHOLD = 10.0

METRICS = [
    # ("energy_mae_per_atom", "Energy MAE / atom [eV]"),
    # ("forces_mae", r"Forces MAE [eV/$\AA$]"),
    ("hessian_mae_ev_a2", r"Hessian MAE [eV/$\AA^2$]"),
    ("hessian_rel_mae", "Hessian relative MAE"),
    ("eckart_eigval_mae_ev_a2", r"Eckart $\lambda$ MAE [eV/$\AA^2$]"),
    ("eckart_lowest_eigval_mae_ev_a2", r"Eckart $\lambda_1$ MAE [eV/$\AA^2$]"),
    # ("freq_mae_400_4000", r"Freq MAE 400-4000 [cm$^{-1}$]"),
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

SPREADS = [
    {"suffix": "", "column": None},
    {"suffix": "_std", "column": "std"},
    {"suffix": "_se", "column": "se"},
]


def modified_z_outliers(df, metric, threshold):
    if metric not in df.columns or "natoms" not in df.columns:
        return pd.Series(False, index=df.index)

    values = pd.to_numeric(df[metric], errors="coerce")
    medians = values.groupby(df["natoms"]).transform("median")
    abs_deviation = (values - medians).abs()
    mads = abs_deviation.groupby(df["natoms"]).transform("median")
    modified_z = 0.6745 * (values - medians) / mads
    return modified_z.abs() > threshold


def remove_outlier_samples(dfs, output_dir):
    outlier_rows = []
    outlier_sample_names = set()

    for label, df in dfs.items():
        mask = modified_z_outliers(
            df, OUTLIER_METRIC, OUTLIER_MODIFIED_Z_THRESHOLD
        )
        if not mask.any():
            continue

        label_outliers = df[mask].copy()
        label_outliers.insert(0, "label", label)
        outlier_rows.append(label_outliers)

        if "sample_name" in label_outliers.columns:
            outlier_sample_names.update(label_outliers["sample_name"].dropna())

    if not outlier_rows:
        print("No outliers removed")
        return dfs

    outliers = pd.concat(outlier_rows, ignore_index=True)
    outliers_path = os.path.join(output_dir, "removed_outliers.csv")
    outliers.to_csv(outliers_path, index=False)

    print(
        f"Removing {len(outlier_sample_names)} outlier samples identified by "
        f"per-natoms {OUTLIER_METRIC} modified z-score > "
        f"{OUTLIER_MODIFIED_Z_THRESHOLD}:"
    )
    for sample_name in sorted(outlier_sample_names):
        print(f"  {sample_name}")
    print(f"Saved removed outlier rows to {outliers_path}")

    filtered = {}
    for label, df in dfs.items():
        if "sample_name" not in df.columns:
            filtered[label] = df
            continue
        filtered[label] = df[~df["sample_name"].isin(outlier_sample_names)].copy()
        print(f"Filtered {label}: {len(df)} -> {len(filtered[label])} samples")

    return filtered


def plot_metric(dfs, variant, col, ylabel, spread, plot_dir):
    fig, ax = plt.subplots(figsize=(8, 8))

    for label, df in dfs.items():
        if label in variant["exclude"]:
            continue
        if col not in df.columns:
            continue

        grouped = (
            df.groupby("natoms")[col]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("natoms")
        )
        grouped["std"] = grouped["std"].fillna(0.0)
        grouped["se"] = grouped["std"] / (grouped["count"] ** 0.5)

        x = grouped["natoms"]
        y = grouped["mean"]
        color = COLOURS[label]
        ax.plot(
            x,
            y,
            marker="o",
            markersize=4,
            linewidth=2,
            label=label,
            color=color,
        )

        spread_column = spread["column"]
        if spread_column is not None:
            spread_values = grouped[spread_column]
            ax.fill_between(
                x,
                y - spread_values,
                y + spread_values,
                color=color,
                alpha=0.18,
                linewidth=0,
            )

    ax.set_xlabel("Number of Atoms")
    ax.set_ylabel(ylabel)
    legend = ax.legend()
    legend.set_title("")
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_alpha(1.0)
    plt.tight_layout(pad=0.0)

    plot_path = (
        f"{plot_dir}/size_{col}{spread['suffix']}{variant['suffix']}.png"
    )
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved {plot_path}")
    plt.close()


if __name__ == "__main__":
    plot_dir = "plots/size_eval"
    os.makedirs(plot_dir, exist_ok=True)

    dfs = {}
    for label, path in RESULTS.items():
        if not os.path.exists(path):
            print(f"Skipping {label}: {path} not found")
            continue
        dfs[label] = pd.read_parquet(path)
        if "status" in dfs[label].columns:
            dfs[label] = dfs[label][dfs[label]["status"] == "ok"].copy()
        print(f"Loaded {label}: {path} ({len(dfs[label])} samples)")

    dfs = remove_outlier_samples(dfs, plot_dir)

    plt.rcParams.update(
        {
            "axes.grid": True,
            "axes.labelsize": 20,
            "font.size": 18,
            "legend.fontsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
        }
    )

    for variant in VARIANTS:
        for col, ylabel in METRICS:
            for spread in SPREADS:
                plot_metric(dfs, variant, col, ylabel, spread, plot_dir)
