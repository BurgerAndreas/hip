import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402

from plot_style import (  # noqa: E402
    HESSIAN_METHOD_TO_COLOUR,
    LINE_WIDTH,
    MARKER_SIZE,
    PLOTLY_FONT_COLOR as PLOT_FONT_COLOR,
    apply_plot_style,
    finish_axis,
)
PANEL_LABEL_SIZE = 18


EIGVAL_MAE_COLUMNS = (
    "eckart_eigval_mae_ev_a2",
    "eigval_mae_eckart",
)
OUTLIER_METRIC_COLUMNS = ("hessian_mae_ev_a2", "hessian_mae")
OUTLIER_MODIFIED_Z_THRESHOLD = 10.0
STD_BAND_ALPHA = 0.24
SPEED_STD_BAND_ALPHA = 0.16


def _color_for_method(method):
    method_lower = str(method).lower()
    if method_lower in HESSIAN_METHOD_TO_COLOUR:
        return HESSIAN_METHOD_TO_COLOUR[method_lower]
    if "finite_difference_bz1" in method_lower:
        return HESSIAN_METHOD_TO_COLOUR["finite_difference_bz1"]
    if "finite_difference_bz32" in method_lower:
        return HESSIAN_METHOD_TO_COLOUR["finite_difference_bz32"]
    return HESSIAN_METHOD_TO_COLOUR.get(method_lower, "#cfcfcf")


def _dash_for_memory(method):
    method_lower = str(method).lower()
    if method_lower == "prediction":
        return ""
    if method_lower == "forward_pass":
        return "dot"
    return "solid"


def _display_name(method):
    method_lower = str(method).lower()
    if method_lower == "prediction":
        return "HIP Hessians (ours)"
    if method_lower == "autograd":
        return "AD Hessians (direct force)"
    if method_lower == "autograd_conservative":
        return "AD Hessians (conservative)"
    if method_lower == "forward_pass":
        return "Forward Pass"
    if "finite_difference_bz1" in method_lower:
        return "FD Hessians"
    return str(method)


def _modified_z_outliers(df, metric, threshold):
    if metric not in df.columns or "natoms" not in df.columns:
        return pd.Series(False, index=df.index)

    values = pd.to_numeric(df[metric], errors="coerce")
    medians = values.groupby(df["natoms"]).transform("median")
    abs_deviation = (values - medians).abs()
    mads = abs_deviation.groupby(df["natoms"]).transform("median")
    modified_z = 0.6745 * (values - medians) / mads
    return modified_z.abs() > threshold


def _outlier_metric_for_frame(df):
    return next((metric for metric in OUTLIER_METRIC_COLUMNS if metric in df.columns), None)


def _outlier_key_for_frame(df):
    return next((key for key in ("sample_name", "sample_idx") if key in df.columns), None)


def _remove_outlier_samples(dfs, output_path=None, dataset_name="PubChem"):
    outlier_rows = []
    outlier_sample_ids = {}
    outlier_metrics = set()

    for label, df in dfs.items():
        outlier_metric = _outlier_metric_for_frame(df)
        if outlier_metric is None:
            continue

        mask = _modified_z_outliers(
            df,
            outlier_metric,
            OUTLIER_MODIFIED_Z_THRESHOLD,
        )
        if not mask.any():
            continue

        outlier_metrics.add(outlier_metric)
        label_outliers = df[mask].copy()
        label_outliers.insert(0, "label", label)
        outlier_rows.append(label_outliers)

        outlier_key = _outlier_key_for_frame(label_outliers)
        if outlier_key is not None:
            outlier_sample_ids.setdefault(outlier_key, set()).update(
                label_outliers[outlier_key].dropna()
            )

    if not outlier_rows:
        print(f"No {dataset_name} outliers removed")
        return dfs

    outliers = pd.concat(outlier_rows, ignore_index=True)
    if output_path is not None:
        outliers_path = Path(output_path)
        outliers_path.parent.mkdir(parents=True, exist_ok=True)
        outliers.to_csv(outliers_path, index=False)
        print(f"Saved removed {dataset_name} outlier rows to {outliers_path}")

    print(
        f"Removing {sum(len(ids) for ids in outlier_sample_ids.values())} "
        f"{dataset_name} outlier samples identified by per-natoms "
        f"{'/'.join(sorted(outlier_metrics))} modified z-score > "
        f"{OUTLIER_MODIFIED_Z_THRESHOLD}"
    )

    filtered = {}
    for label, df in dfs.items():
        outlier_key = _outlier_key_for_frame(df)
        if outlier_key is None or outlier_key not in outlier_sample_ids:
            filtered[label] = df
            continue
        filtered[label] = df[~df[outlier_key].isin(outlier_sample_ids[outlier_key])].copy()
        print(f"Filtered {dataset_name} {label}: {len(df)} -> {len(filtered[label])} samples")

    return filtered


def _eigval_curve_from_frame(df, csv_path):
    metric_col = next((col for col in EIGVAL_MAE_COLUMNS if col in df.columns), None)
    if "natoms" not in df.columns or metric_col is None:
        raise ValueError(
            f"Missing required columns in {csv_path}. Need 'natoms' and one of "
            f"{list(EIGVAL_MAE_COLUMNS)}."
        )
    return df.groupby("natoms")[metric_col].mean().sort_index()


def _eigval_curve_stats_from_frame(df, metrics_path):
    metric_col = next((col for col in EIGVAL_MAE_COLUMNS if col in df.columns), None)
    if "natoms" not in df.columns or metric_col is None:
        raise ValueError(
            f"Missing required columns in {metrics_path}. Need 'natoms' and one of "
            f"{list(EIGVAL_MAE_COLUMNS)}."
        )
    stats = df.groupby("natoms")[metric_col].agg(["mean", "std"]).sort_index()
    stats["std"] = stats["std"].fillna(0.0)
    return stats


def _load_eigval_curve(metrics_path):
    print(f"Loading {metrics_path}")
    df = pd.read_parquet(metrics_path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    return _eigval_curve_stats_from_frame(df, metrics_path)


def _load_curves(lambda_results):
    curves = {}
    for label, metrics_path in lambda_results.items():
        if not Path(metrics_path).exists():
            print(f"Skipping {label}: missing {metrics_path}")
            continue
        curves[label] = _load_eigval_curve(metrics_path)
    return curves


def _load_curves_with_outlier_filter(
    lambda_results,
    outliers_output_path=None,
    dataset_name="PubChem",
):
    frames = {}
    paths = {}
    for label, metrics_path in lambda_results.items():
        if not Path(metrics_path).exists():
            print(f"Skipping {label}: missing {metrics_path}")
            continue
        print(f"Loading {metrics_path}")
        df = pd.read_parquet(metrics_path)
        if "status" in df.columns:
            df = df[df["status"] == "ok"].copy()
        frames[label] = df.reset_index(drop=True)
        paths[label] = metrics_path

    frames = _remove_outlier_samples(
        frames,
        output_path=outliers_output_path,
        dataset_name=dataset_name,
    )
    return {
        label: _eigval_curve_stats_from_frame(df, paths[label])
        for label, df in frames.items()
    }


def _outliers_output_path(output_path):
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}_removed_outliers.csv")


def _rgd1_outliers_output_path(output_path):
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}_rgd1_removed_outliers.csv")


def _load_speed_tables(speed_csv):
    print(f"Loading {speed_csv}")
    speed_df = pd.read_csv(speed_csv)
    avg_times = speed_df.groupby(["n_atoms", "method"])["time"].mean().unstack()
    std_times = speed_df.groupby(["n_atoms", "method"])["time"].std().fillna(0.0).unstack()
    avg_memory = speed_df.groupby(["n_atoms", "method"])["memory"].mean().unstack()
    std_memory = speed_df.groupby(["n_atoms", "method"])["memory"].std().fillna(0.0).unstack()

    methods = [
        "autograd",
        "autograd_conservative",
        "finite_difference_bz1",
        "forward_pass",
        "prediction",
    ]
    methods = [m for m in methods if m in avg_times.columns]
    return avg_times[methods], std_times[methods], avg_memory[methods], std_memory[methods]


def make_plot_seaborn(
    speed_csv,
    rgd1_lambda_results,
    pubchem_lambda_results,
    output_path,
    ymin_time=0.0,
    ymax_time=3.7,
    ymin_memory=0.0,
    ymax_memory=2.1,
    dpi=250,
):
    apply_plot_style()

    avg_times, std_times, avg_memory, std_memory = _load_speed_tables(speed_csv)
    avg_times = avg_times / 1000.0
    std_times = std_times / 1000.0
    avg_memory = avg_memory / 1000.0
    std_memory = std_memory / 1000.0
    rgd1_lambda_curves = _load_curves_with_outlier_filter(
        rgd1_lambda_results,
        outliers_output_path=_rgd1_outliers_output_path(output_path),
        dataset_name="RGD1",
    )
    pubchem_lambda_curves = _load_curves_with_outlier_filter(
        pubchem_lambda_results,
        outliers_output_path=_outliers_output_path(output_path),
        dataset_name="PubChem",
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), layout="constrained")
    fig.set_constrained_layout_pads(hspace=0.06)
    ax_time, ax_memory, ax_rgd1, ax_pubchem = axes.ravel()

    for method in avg_times.columns:
        color = _color_for_method(method)
        ax_time.fill_between(
            avg_times.index,
            avg_times[method] - std_times[method],
            avg_times[method] + std_times[method],
            color=color,
            alpha=SPEED_STD_BAND_ALPHA,
            linewidth=0,
            zorder=2,
        )
        ax_time.plot(
            avg_times.index,
            avg_times[method],
            marker="D" if method == "prediction" else "o",
            linestyle="--" if method == "prediction" else "-",
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label=_display_name(method),
            color=color,
            zorder=3,
        )

    for method in avg_memory.columns:
        color = _color_for_method(method)
        linestyle = ":" if _dash_for_memory(method) == "dot" else "-"
        ax_memory.fill_between(
            avg_memory.index,
            avg_memory[method] - std_memory[method],
            avg_memory[method] + std_memory[method],
            color=color,
            alpha=SPEED_STD_BAND_ALPHA,
            linewidth=0,
            zorder=2,
        )
        if method == "prediction":
            ax_memory.plot(
                avg_memory.index,
                avg_memory[method],
                marker="D",
                linestyle="None",
                markersize=MARKER_SIZE,
                color=color,
                zorder=3,
            )
        else:
            ax_memory.plot(
                avg_memory.index,
                avg_memory[method],
                marker="o",
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
                linestyle=linestyle,
                color=color,
                zorder=3,
            )

    label_to_method = {"HIP": "prediction", "AD": "autograd"}
    for label in ["HIP", "AD"]:
        if label in rgd1_lambda_curves:
            color = _color_for_method(label_to_method[label])
            stats = rgd1_lambda_curves[label]
            x_values = stats.index
            mean = stats["mean"]
            std = stats["std"]
            ax_rgd1.fill_between(
                x_values,
                mean - std,
                mean + std,
                color=color,
                alpha=STD_BAND_ALPHA,
                linewidth=0,
                zorder=2,
            )
            ax_rgd1.plot(
                x_values,
                mean,
                marker="D" if label == "HIP" else "o",
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
                color=color,
                zorder=3,
            )
        if label in pubchem_lambda_curves:
            color = _color_for_method(label_to_method[label])
            stats = pubchem_lambda_curves[label]
            x_values = stats.index
            mean = stats["mean"]
            std = stats["std"]
            ax_pubchem.fill_between(
                x_values,
                mean - std,
                mean + std,
                color=color,
                alpha=STD_BAND_ALPHA,
                linewidth=0,
                zorder=2,
            )
            ax_pubchem.plot(
                x_values,
                mean,
                marker="D" if label == "HIP" else "o",
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
                color=color,
                zorder=3,
            )

    ax_rgd1.axvline(22, linewidth=1.4, linestyle="--", color="gray", zorder=2)
    train_annotation_transform = offset_copy(
        ax_rgd1.get_xaxis_transform(),
        fig=fig,
        x=10,
        y=-14,
        units="dots",
    )
    ax_rgd1.annotate(
        "Train",
        xy=(16.0, 0.95),
        xycoords=train_annotation_transform,
        xytext=(18.0, 0.95),
        textcoords=train_annotation_transform,
        arrowprops={
            "arrowstyle": "-|>",
            "edgecolor": "gray",
            "facecolor": "gray",
            "mutation_scale": 12,
        },
        color="gray",
        fontsize=matplotlib.rcParams["font.size"],
        va="center",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2},
        zorder=2,
    )

    titles = (
        "Time",
        "Memory",
        "RGD1",
        "PubChem",
    )
    ylabels = (
        "Time per Sample [s]",
        "Peak Memory [GB]",
        r"Eigenvalue $\lambda$ MAE [$\mathrm{eV}\,\AA^{-2}$]",
        r"Eigenvalue $\lambda$ MAE [$\mathrm{eV}\,\AA^{-2}$]",
    )
    for panel_label, ax, title, ylabel in zip("abcd", axes.ravel(), titles, ylabels):
        if panel_label not in "ab":
            ax.set_title(title)
            ax.set_xlabel("Number of Atoms")
        ax.set_ylabel(ylabel)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", which="both", length=0)
        finish_axis(ax)
        panel_label_transform = offset_copy(
            ax.transAxes,
            fig=fig,
            x=-28,
            y=2,
            units="dots",
        )
        ax.text(
            0,
            1,
            panel_label,
            transform=panel_label_transform,
            fontsize=PANEL_LABEL_SIZE,
            fontfamily="DejaVu Sans",
            fontweight="bold",
            color=PLOT_FONT_COLOR,
            va="bottom",
            ha="right",
        )

    ax_time.set_ylim(ymin_time, ymax_time)
    ax_memory.set_ylim(ymin_memory, ymax_memory)
    # ax_rgd1.set_ylim(bottom=0.0)
    ax_rgd1.set_ylim(bottom=0.0, top=0.345)
    ax_pubchem.set_ylim(bottom=0.0, top=9.8)
    ax_time.set_xlim(4.5, 21.5)
    ax_memory.set_xlim(4.5, 21.5)
    ax_time.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax_memory.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    if rgd1_lambda_curves:
        max_n = max(n for stats in rgd1_lambda_curves.values() for n in stats.index)
        ax_rgd1.set_xlim(9.95, max_n + 0.5)
        ax_rgd1.set_xticks([15, 20, 25, 30])
    if pubchem_lambda_curves:
        ns = [n for stats in pubchem_lambda_curves.values() for n in stats.index]
        ax_pubchem.set_xlim(min(ns) - 0.5, max(ns) + 0.5)
    ax_pubchem.yaxis.set_label_coords(-0.11, 0.5)

    handles, labels = ax_time.get_legend_handles_labels()
    legend = ax_memory.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(-0.02, 1.02),
        bbox_transform=offset_copy(ax_memory.transAxes, fig=fig, x=-1, y=2, units="dots"),
        frameon=True,
        edgecolor="none",
        fontsize=12.5,
        labelcolor=PLOT_FONT_COLOR,
    )
    legend.set_zorder(1)
    legend.get_frame().set_alpha(0.75)
    legend.get_frame().set_zorder(1)
    for artist in (*legend.legend_handles, *legend.get_texts()):
        artist.set_zorder(4)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Plot saved to {output_path}")


def make_plot(
    speed_csv,
    rgd1_lambda_results,
    pubchem_lambda_results,
    output_path,
    ymin_time=0.0,
    ymax_time=3.7,
    ymin_memory=0.0,
    ymax_memory=2.1,
    dpi=250,
):
    make_plot_seaborn(
        speed_csv=speed_csv,
        rgd1_lambda_results=rgd1_lambda_results,
        pubchem_lambda_results=pubchem_lambda_results,
        output_path=output_path,
        ymin_time=ymin_time,
        ymax_time=ymax_time,
        ymin_memory=ymin_memory,
        ymax_memory=ymax_memory,
        dpi=dpi,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone 2x2 speed/memory/lambda plot")
    parser.add_argument(
        "--speed_csv",
        type=str,
        default="results_speed2/ts1x-val.lmdb_speed_comparison_extended_10_r100.0_rh100.0.csv",
    )
    parser.add_argument(
        "--rgd1_hip_metrics_parquet",
        type=str,
        default="results_evalhorm/hesspred_v2_RGD1_predict_metrics.parquet",
    )
    parser.add_argument(
        "--rgd1_ad_metrics_parquet",
        type=str,
        default="results_evalhorm/eqv2_RGD1_autograd_metrics.parquet",
    )
    parser.add_argument(
        "--rgd1_ad_ef_metrics_parquet",
        type=str,
        default="results_evalhorm/eqv2_orig_RGD1_autograd_metrics.parquet",
    )
    parser.add_argument(
        "--pubchem_hip_metrics_parquet",
        type=str,
        # default="results_size_eval/
        # hesspred_v2_dft_geometries_pr
        # edict_metrics.csv",
        default="results_eval_largehessians_orca_hip_v2/metrics.parquet",
    )
    parser.add_argument(
        "--pubchem_ad_metrics_parquet",
        type=str,
        # default="results_size_eval/
        # eqv2_dft_geometries_autograd_
        # metrics.csv",
        default="results_eval_largehessians_orca_hf_horm_eqv2_autograd/metrics.parquet",
    )
    parser.add_argument(
        "--pubchem_ad_ef_metrics_parquet",
        type=str,
        default="results_size_eval/eqv2_orig_dft_geometries_autograd_metrics.parquet",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results_speed2/speed_memory_lambda_scaling_eval_horm_colours.png",
    )
    parser.add_argument("--ymin_time", type=float, default=0.0)
    parser.add_argument("--ymax_time", type=float, default=3.7)
    parser.add_argument("--ymin_memory", type=float, default=0.0)
    parser.add_argument("--ymax_memory", type=float, default=2.1)
    parser.add_argument("--dpi", type=int, default=250)
    args = parser.parse_args()

    make_plot(
        speed_csv=args.speed_csv,
        rgd1_lambda_results={
            "HIP": args.rgd1_hip_metrics_parquet,
            "AD": args.rgd1_ad_metrics_parquet,
            "AD (E-F)": args.rgd1_ad_ef_metrics_parquet,
        },
        pubchem_lambda_results={
            "HIP": args.pubchem_hip_metrics_parquet,
            "AD": args.pubchem_ad_metrics_parquet,
            "AD (E-F)": args.pubchem_ad_ef_metrics_parquet,
        },
        output_path=args.output,
        ymin_time=args.ymin_time,
        ymax_time=args.ymax_time,
        ymin_memory=args.ymin_memory,
        ymax_memory=args.ymax_memory,
        dpi=args.dpi,
    )
