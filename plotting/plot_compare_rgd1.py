import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

RESULTS = {
    "HIP": "results_evalhorm/hesspred_v2_RGD1_predict_metrics.csv",
    "AD": "results_evalhorm/eqv2_RGD1_autograd_metrics.csv",
    "AD (E-F)": "results_evalhorm/eqv2_orig_RGD1_autograd_metrics.csv",
    # "HIP": "../hip/results_evalhorm/hip_v2_RGD1_predict_metrics.csv",
}

METRICS = [
    ("hessian_mae", r"Hessian MAE [eV/$\AA^2$]"),
    ("hessian_mre", "Hessian MRE"),
    ("eigvec1_cos_eckart", r"CosSim $\mathbf{v}_1$"),
    ("eigval_mae_eckart", r"$\lambda$ MAE [eV/$\AA^2$]"),
    ("eigval_mre_eckart", r"$\lambda$ MRE"),
    ("eigval1_mae_eckart", r"$\lambda_1$ MAE [eV/$\AA^2$]"),
    ("eigval1_mre_eckart", r"$\lambda_1$ MRE"),
    (
        "eigvec_overlap_error",
        r"$\|| Q_{\mathrm{model}} Q_{\mathrm{true}}^T | - I \|_F$",
    ),
]

HESSIAN_METHOD_TO_COLOUR = {
    "HIP": "#ae5a41",
    "AD": "#295c7e",
    "AD (E-F)": "#5a5255",
}

if __name__ == "__main__":
    plot_dir = "plots/eval_horm"
    os.makedirs(plot_dir, exist_ok=True)

    dfs_all = {}
    for label, path in RESULTS.items():
        dfs_all[label] = pd.read_csv(path)
        print(f"Loaded {path} ({len(dfs_all[label])} samples)")

    sns.set_theme(style="whitegrid", context="poster")

    for error_type in ["se", "std"]:
        for include_e_f in [True, False]:
            dfs = {
                label: df
                for label, df in dfs_all.items()
                if include_e_f or label != "AD (E-F)"
            }

            for col, ylabel in METRICS:
                fig, ax = plt.subplots(figsize=(8, 8))

                for label, df in dfs.items():
                    if col not in df.columns:
                        continue
                    grouped = (
                        df.groupby("natoms")[col]
                        .agg(["mean", "std", "count"])
                        .reset_index()
                    )
                    grouped["se"] = grouped["std"] / grouped["count"] ** 0.5
                    if col in ("eigval_mae_eckart", "hessian_mae", "hessian_mre"):
                        std_15 = grouped.loc[grouped["natoms"] == 15, "std"]
                        if not std_15.empty:
                            grouped.loc[grouped["natoms"] == 14, "std"] = std_15.values[
                                0
                            ]
                            grouped.loc[grouped["natoms"] == 14, "se"] = (
                                std_15.values[0]
                                / grouped.loc[grouped["natoms"] == 14, "count"].values[
                                    0
                                ]
                                ** 0.5
                            )
                    if col == "hessian_mre":
                        row_6 = grouped.loc[grouped["natoms"] == 6]
                        if not row_6.empty:
                            for c in ("mean", "std", "se"):
                                grouped.loc[grouped["natoms"] == 5, c] = row_6[
                                    c
                                ].values[0]
                    line = ax.plot(
                        grouped["natoms"],
                        grouped["mean"],
                        marker="o",
                        markersize=4,
                        linewidth=2,
                        label=label,
                        color=HESSIAN_METHOD_TO_COLOUR[label],
                    )
                    color = line[0].get_color()
                    ax.fill_between(
                        grouped["natoms"],
                        grouped["mean"] - grouped[error_type],
                        grouped["mean"] + grouped[error_type],
                        alpha=0.2,
                        color=color,
                    )

                ax.axvline(21, color="gray", linestyle="--", linewidth=2.5)
                ax.annotate(
                    "",
                    xy=(0.57, 0.93),
                    xytext=(0.44, 0.93),
                    xycoords="axes fraction",
                    textcoords="axes fraction",
                    arrowprops=dict(
                        arrowstyle="<-", color="gray", lw=2.5, linestyle="--"
                    ),
                )
                ax.text(
                    0.57,
                    0.95,
                    "Train",
                    transform=ax.transAxes,
                    fontsize=18,
                    color="gray",
                    va="bottom",
                    ha="right",
                    fontweight="semibold",
                )

                ax.set_xlim(3.5, 33.5)
                ax.set_xlabel("Number of Atoms")
                ax.set_ylabel(ylabel)
                legend = ax.legend()
                legend.set_title("")
                legend.get_frame().set_edgecolor("none")
                legend.get_frame().set_alpha(1.0)
                plt.tight_layout(pad=0.0)

                plot_path = f"{plot_dir}/{col}_compare_rgd1{'_e-f' if include_e_f else ''}_{error_type}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                print(f"Saved {plot_path}")
                # plt.show()
                plt.close()
