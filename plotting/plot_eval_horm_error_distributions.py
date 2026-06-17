#!/usr/bin/env python
"""Plot eval_horm error distributions and correlations.

Defaults target the 10k TS1x evals for AD Hessians and HIP v2
predicted Hessians.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter  # noqa: E402
import seaborn as sns  # noqa: E402

from plot_style import AD_COLOR, ACCENT_COLOR, HIP_COLOR, SUCCESS_COLOR, finish_axis, model_color, model_palette

matplotlib.rcParams.update(
    {
        "xtick.major.size": 0.9,
        "xtick.minor.size": 0.9,
        "ytick.major.size": 0.9,
        "ytick.minor.size": 0.9,
    }
)


DEFAULT_RESULTS = (
    ("AD", Path("results_evalhorm/eqv2_ts1x-val_autograd_metrics.parquet")),
    ("HIP", Path("results_evalhorm/hip_v2_ts1x-val_predict_metrics.parquet")),
    ("AD (no H)", Path("results_evalhorm/eqv2_orig_ts1xval10k_29148768_ts1x-val_autograd_metrics.parquet")),
)

ENERGY_METRICS = (
    ("energy_error", r"Energy MAE [$\mathrm{eV}$]"),
)

PRIMARY_METRICS = (
    ("forces_error", r"Force MAE [$\mathrm{eV}/\AA$]"),
    ("hessian_error", r"Hessian MAE [$\mathrm{eV}/\AA^2$]"),
)

EXTRA_METRICS = (
    ("energy_difference", r"Signed energy difference [$\mathrm{eV}$]"),
    ("force_l1_error", r"Force L1 error [$\mathrm{eV}/\AA$]"),
    ("force_l2_error", r"Force L2 error [$\mathrm{eV}/\AA$]"),
    ("force_cos_error", "Force cosine error [unitless]"),
)

DFT_COMPARISONS = (
    (
        "energy",
        "energy_true",
        "energy_model",
        r"Energy [$\mathrm{eV}$]",
        False,
        True,
    ),
    (
        "force_norm",
        "force_true_norm",
        "force_model_norm",
        r"Force norm [$\mathrm{eV}/\AA$]",
        True,
        False,
    ),
    (
        "hessian_fro_norm",
        "hessian_true_fro_norm",
        "hessian_model_fro_norm",
        r"Hessian Frobenius norm [$\mathrm{eV}/\AA^2$]",
        True,
        False,
    ),
)

EIGEN_DIAGNOSTIC_METRICS = (
    ("eigval1_mae_eckart", r"First eigenvalue MAE [$\mathrm{eV}/\AA^2$]"),
    ("eigval_mae_eckart", r"Eigenvalue MAE [$\mathrm{eV}/\AA^2$]"),
    ("eigvec1_mae_eckart", "First eigenvector MAE"),
    ("eigvec1_cos_error_eckart", r"First eigenvector cosine error [$1-\cos$]"),
)

STRUCTURE_GROUPS = (
    ("true_neg_num", "True negative modes"),
    ("true_is_ts", "True TS"),
    ("true_is_minima", "True minimum"),
    ("true_is_ts_order2", "True order-2 TS"),
    ("neg_num_agree_ad", "AD neg modes agree"),
    ("neg_num_agree_hip", "HIP neg modes agree"),
)

SCATTER_KWARGS = {
    "s": 8,
    "alpha": 0.18,
    "linewidths": 0,
    "rasterized": True,
}

COMPARISON_COLOR_FIELDS = (
    ("natoms", "Number of Atoms"),
    ("true_neg_num", "true negative-mode count"),
)
TRUE_NEG_COLORBAR_TICKS = [0, 1, 2]
TRUE_NEG_COLORBAR_TICK_LABELS = ["0", "1", ">1"]
TRUE_NEG_COLORMAP = matplotlib.colors.ListedColormap(["#440154", "#21918C", "#FDE725"])
TRUE_NEG_NORM = matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], TRUE_NEG_COLORMAP.N)


@dataclass
class ResultTable:
    label: str
    path: Path
    df: pd.DataFrame


def safe_label(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip().lower())
    return safe.strip("_") or "model"


def short_model_label(label: str) -> str:
    normalized = label.casefold()
    if "hip" in normalized:
        return "HIP"
    if "autograd" in normalized or "eqv2" in normalized or "ad" in normalized:
        return "AD"
    return label


def merged_column(merged: pd.DataFrame, column: str, key: str) -> str | None:
    suffixed = f"{column}_{key}"
    if suffixed in merged.columns:
        return suffixed
    if column in merged.columns:
        return column
    return None


def add_suffix_to_path(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def comparison_color_values(merged: pd.DataFrame, column: str, *keys: str) -> np.ndarray | None:
    for key in keys:
        color_col = merged_column(merged, column, key)
        if color_col is not None:
            values = pd.to_numeric(merged[color_col], errors="coerce").to_numpy(dtype=float)
            if column == "true_neg_num":
                values = values.copy()
                values[values > 1] = 2
            return values
    return None


def result_color_values(result: ResultTable, column: str) -> np.ndarray | None:
    if column not in result.df.columns:
        return None
    values = pd.to_numeric(result.df[column], errors="coerce").to_numpy(dtype=float)
    if column == "true_neg_num":
        values = values.copy()
        values[values > 1] = 2
    return values


def comparison_colorbar_settings(
    color_column: str | None,
    color_values: np.ndarray | None,
) -> tuple[matplotlib.colors.Normalize | None, str | matplotlib.colors.Colormap, list[int] | None, list[str] | None]:
    if color_column == "true_neg_num":
        return TRUE_NEG_NORM, TRUE_NEG_COLORMAP, TRUE_NEG_COLORBAR_TICKS, TRUE_NEG_COLORBAR_TICK_LABELS

    if color_values is None:
        return None, "viridis", None, None

    finite_color = color_values[np.isfinite(color_values)]
    if not len(finite_color):
        return None, "viridis", None, None

    return (
        matplotlib.colors.Normalize(vmin=float(finite_color.min()), vmax=float(finite_color.max())),
        "viridis",
        None,
        None,
    )


def style_colorbar_ticks(cbar, discrete: bool) -> None:
    if discrete:
        cbar.ax.tick_params(which="both", length=0, width=0)
    else:
        cbar.ax.tick_params(which="both", length=2.5, width=0.8)


def scatter_comparison(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    use_log: bool,
    color: str,
    color_values: np.ndarray | None = None,
    color_label: str | None = None,
    norm: matplotlib.colors.Normalize | None = None,
    cmap: str | matplotlib.colors.Colormap = "viridis",
    colorbar_ticks: list[int] | None = None,
    colorbar_ticklabels: list[str] | None = None,
    add_colorbar: bool = True,
) -> None:
    if color_values is None:
        sns.scatterplot(x=x, y=y, ax=ax, color=color, legend=False, **SCATTER_KWARGS)
        return

    points = ax.scatter(
        x,
        y,
        c=color_values,
        cmap=cmap,
        norm=norm,
        s=SCATTER_KWARGS["s"],
        alpha=0.35,
        linewidths=SCATTER_KWARGS["linewidths"],
        rasterized=SCATTER_KWARGS["rasterized"],
    )
    if not add_colorbar:
        return
    cbar = ax.figure.colorbar(points, ax=ax, pad=0.01)
    cbar.outline.set_visible(False)
    style_colorbar_ticks(cbar, discrete=colorbar_ticklabels == TRUE_NEG_COLORBAR_TICK_LABELS)
    if colorbar_ticks is not None:
        cbar.set_ticks(colorbar_ticks)
    if colorbar_ticklabels is not None:
        cbar.set_ticklabels(colorbar_ticklabels)
    if color_label is not None:
        cbar.set_label(color_label)


def comparison_values(
    merged: pd.DataFrame,
    metric: str,
    left_key: str,
    right_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    if metric == "eigvec1_cos_error_eckart":
        left_col = merged_column(merged, "eigvec1_cos_eckart", left_key)
        right_col = merged_column(merged, "eigvec1_cos_eckart", right_key)
        if left_col is None or right_col is None:
            return np.array([]), np.array([])
        x = 1.0 - pd.to_numeric(merged[left_col], errors="coerce").to_numpy(dtype=float)
        y = 1.0 - pd.to_numeric(merged[right_col], errors="coerce").to_numpy(dtype=float)
        return np.clip(x, 0.0, None), np.clip(y, 0.0, None)

    x_col = merged_column(merged, metric, left_key)
    y_col = merged_column(merged, metric, right_key)
    if x_col is None or y_col is None:
        return np.array([]), np.array([])
    x = pd.to_numeric(merged[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(merged[y_col], errors="coerce").to_numpy(dtype=float)
    return x, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        nargs=2,
        metavar=("LABEL", "PARQUET"),
        default=[],
        help=(
            "Result Parquet file to plot. May be repeated. Defaults to the EqV2 autograd "
            "and HIP eval_horm result files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/eval_horm/error_distributions"),
    )
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable log10 transforms for nonnegative error histograms and log scaling for correlation axes.",
    )
    return parser.parse_args()


def load_results(args: argparse.Namespace) -> list[ResultTable]:
    result_specs = args.result or [(label, str(path)) for label, path in DEFAULT_RESULTS]
    results = []
    required = {"dataset_idx", *(metric for metric, _ in ENERGY_METRICS), *(metric for metric, _ in PRIMARY_METRICS)}

    for label, path_str in result_specs:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_parquet(path)
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        results.append(ResultTable(label=label, path=path, df=df))

    return results


def finite_values(df: pd.DataFrame, metric: str) -> np.ndarray:
    values = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def format_log10_hist_axis(ax: plt.Axes) -> None:
    xmin, xmax = ax.get_xlim()
    major_exponents = np.arange(np.floor(xmin), np.ceil(xmax) + 1, dtype=int)
    major_ticks = major_exponents[(major_exponents >= xmin) & (major_exponents <= xmax)]
    minor_ticks = []
    for exponent in major_exponents:
        minor_ticks.extend(np.log10(np.arange(2, 10, dtype=float)) + exponent)
    minor_ticks = [tick for tick in minor_ticks if xmin <= tick <= xmax]

    ax.xaxis.set_major_locator(FixedLocator(major_ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([rf"$10^{{{exponent}}}$" for exponent in major_ticks]))
    ax.xaxis.set_minor_locator(FixedLocator(minor_ticks))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="minor", axis="x", color="#F1F1F1", linewidth=0.7, alpha=0.85)


def plot_histograms(
    results: list[ResultTable],
    metrics: tuple[tuple[str, str], ...],
    output_path: Path,
    bins: int,
    use_log: bool,
) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    axes = np.atleast_1d(axes)
    palette = model_palette(tuple(result.label for result in results))

    for ax, (metric, metric_label) in zip(axes, metrics):
        values_by_model = [finite_values(result.df, metric) for result in results]
        plot_log_error = use_log and metric != "energy_difference"

        for result, values in zip(results, values_by_model):
            if plot_log_error:
                values = values[values > 0]
                values = np.log10(values)
            if len(values) == 0:
                continue
            sns.histplot(
                x=values,
                ax=ax,
                bins=bins,
                stat="density",
                element="step",
                fill=False,
                linewidth=1.8,
                color=palette[result.label],
                label=result.label,
            )

        ax.set_xlabel(metric_label)
        ax.set_ylabel("Density")
        finish_axis(ax)
        if plot_log_error:
            format_log10_hist_axis(ax)

    axes[-1].legend(loc="upper left", frameon=True, edgecolor="none")
    fig.tight_layout(pad=0.01)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_within_model_correlations(
    results: list[ResultTable],
    output_path: Path,
    use_log: bool,
    dpi: int,
) -> None:
    metric_pairs = (
        (
            "energy_error",
            "forces_error",
            r"Energy MAE [$\mathrm{eV}$]",
            r"Force MAE [$\mathrm{eV}/\AA$]",
        ),
        (
            "energy_error",
            "hessian_error",
            r"Energy MAE [$\mathrm{eV}$]",
            r"Hessian MAE [$\mathrm{eV}/\AA^2$]",
        ),
        (
            "forces_error",
            "hessian_error",
            r"Force MAE [$\mathrm{eV}/\AA$]",
            r"Hessian MAE [$\mathrm{eV}/\AA^2$]",
        ),
    )
    fig, axes = plt.subplots(
        len(results),
        len(metric_pairs),
        figsize=(5 * len(metric_pairs), 4 * len(results)),
        squeeze=False,
    )

    for row_idx, result in enumerate(results):
        for col_idx, (x_metric, y_metric, x_label, y_label) in enumerate(metric_pairs):
            ax = axes[row_idx, col_idx]
            x = finite_values(result.df, x_metric)
            y = finite_values(result.df, y_metric)
            n = min(len(x), len(y))
            x = x[:n]
            y = y[:n]
            keep = np.isfinite(x) & np.isfinite(y)
            if use_log:
                keep &= (x > 0) & (y > 0)
            x = x[keep]
            y = y[keep]

            if len(x):
                sns.scatterplot(x=x, y=y, ax=ax, color=model_color(result.label, AD_COLOR), legend=False, **SCATTER_KWARGS)
                corr = np.corrcoef(np.log10(x + 1e-12), np.log10(y + 1e-12))[0, 1]
                ax.text(
                    0.03,
                    0.97,
                    f"log-r={corr:.2f}",
                    transform=ax.transAxes,
                    va="top",
                    color=matplotlib.rcParams["xtick.color"],
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                )

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            if use_log:
                ax.set_xscale("log")
                ax.set_yscale("log")
            finish_axis(ax)

    fig.tight_layout(pad=0.01)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_force_hessian_scatter_by_model(
    results: list[ResultTable],
    output_path: Path,
    use_log: bool,
    dpi: int,
    color_column: str | None = None,
    color_label: str | None = None,
) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.2), squeeze=False, layout="constrained")
    axes = axes[0]
    plot_values: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]] = []

    for result in results:
        x = pd.to_numeric(result.df["forces_error"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(result.df["hessian_error"], errors="coerce").to_numpy(dtype=float)
        color_all = result_color_values(result, color_column) if color_column is not None else None
        if color_column is not None and color_all is None:
            print(f"Skipping {output_path}: missing color column {color_column} for {result.label}")
            return
        keep = np.isfinite(x) & np.isfinite(y)
        if use_log:
            keep &= (x > 0) & (y > 0)
        if color_all is not None:
            keep &= np.isfinite(color_all)
        plot_values.append((x[keep], y[keep], color_all[keep] if color_all is not None else None))

    all_x = np.concatenate([x for x, _, _ in plot_values if len(x)])
    all_y = np.concatenate([y for _, y, _ in plot_values if len(y)])
    color_value_arrays = [c for _, _, c in plot_values if c is not None and len(c)]
    all_color_values = np.concatenate(color_value_arrays) if color_value_arrays else None
    xlim = (float(all_x.min()), float(all_x.max())) if len(all_x) else None
    ylim = (float(all_y.min()), float(all_y.max())) if len(all_y) else None
    color_norm, color_cmap, colorbar_ticks, colorbar_ticklabels = comparison_colorbar_settings(
        color_column,
        all_color_values,
    )
    color_mappable = None

    for ax, result, (x, y, c) in zip(axes, results, plot_values):
        if len(x):
            if c is None:
                sns.scatterplot(x=x, y=y, ax=ax, color=model_color(result.label, AD_COLOR), legend=False, **SCATTER_KWARGS)
            else:
                color_mappable = ax.scatter(
                    x,
                    y,
                    c=c,
                    cmap=color_cmap,
                    norm=color_norm,
                    s=SCATTER_KWARGS["s"],
                    alpha=0.35,
                    linewidths=SCATTER_KWARGS["linewidths"],
                    rasterized=SCATTER_KWARGS["rasterized"],
                )
            corr_x = np.log10(x) if use_log else x
            corr_y = np.log10(y) if use_log else y
            corr = np.corrcoef(corr_x, corr_y)[0, 1]
            ax.text(
                0.03,
                0.97,
                f"{'log-' if use_log else ''}r={corr:.2f}",
                transform=ax.transAxes,
                va="top",
                color=matplotlib.rcParams["xtick.color"],
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )

        ax.set_title(result.label)
        ax.set_xlabel(r"Force MAE [$\mathrm{eV}/\AA$]")
        ax.set_ylabel(r"Hessian MAE [$\mathrm{eV}/\AA^2$]")
        if use_log:
            ax.set_xscale("log")
            ax.set_yscale("log")
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        finish_axis(ax)

    if color_mappable is not None:
        cbar = fig.colorbar(color_mappable, ax=axes, pad=0.01, fraction=0.035)
        cbar.outline.set_visible(False)
        style_colorbar_ticks(cbar, discrete=color_column == "true_neg_num")
        if colorbar_ticks is not None:
            cbar.set_ticks(colorbar_ticks)
        if colorbar_ticklabels is not None:
            cbar.set_ticklabels(colorbar_ticklabels)
        if color_label is not None:
            cbar.set_label(color_label)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison_correlations(
    results: list[ResultTable],
    output_path: Path,
    use_log: bool,
    dpi: int,
    color_column: str | None = None,
    color_label: str | None = None,
) -> None:
    if len(results) != 2:
        return

    left, right = results
    left_key = safe_label(left.label)
    right_key = safe_label(right.label)
    merged = left.df.merge(
        right.df,
        on="dataset_idx",
        suffixes=(f"_{left_key}", f"_{right_key}"),
    )
    color_all = comparison_color_values(merged, color_column, left_key, right_key) if color_column is not None else None
    if color_column is not None and color_all is None:
        print(f"Skipping {output_path}: missing color column {color_column}")
        return
    color_norm, color_cmap, colorbar_ticks, colorbar_ticklabels = comparison_colorbar_settings(
        color_column,
        color_all,
    )

    comparison_metrics = PRIMARY_METRICS
    fig, axes = plt.subplots(
        1,
        len(comparison_metrics),
        figsize=(5.6 * len(comparison_metrics), 5.2),
        layout="constrained",
    )
    axes = np.atleast_1d(axes)
    color_mappable = None
    for ax, (metric, metric_label) in zip(axes, comparison_metrics):
        x_col = f"{metric}_{left_key}"
        y_col = f"{metric}_{right_key}"
        x = pd.to_numeric(merged[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(merged[y_col], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(x) & np.isfinite(y)
        if use_log:
            keep &= (x > 0) & (y > 0)
        if color_all is not None:
            keep &= np.isfinite(color_all)
        x = x[keep]
        y = y[keep]
        c = color_all[keep] if color_all is not None else None

        if len(x):
            scatter_comparison(
                ax,
                x,
                y,
                use_log=use_log,
                color=ACCENT_COLOR,
                color_values=c,
                color_label=color_label,
                norm=color_norm,
                cmap=color_cmap,
                colorbar_ticks=colorbar_ticks,
                colorbar_ticklabels=colorbar_ticklabels,
                add_colorbar=color_norm is None,
            )
            if color_norm is not None and c is not None:
                color_mappable = matplotlib.cm.ScalarMappable(norm=color_norm, cmap=color_cmap)
            corr = np.corrcoef(np.log10(x + 1e-12), np.log10(y + 1e-12))[0, 1]
            lo = min(x.min(), y.min())
            hi = max(x.max(), y.max())
            ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, alpha=0.7)
            ax.text(
                0.03,
                0.97,
                f"log-r={corr:.2f}",
                transform=ax.transAxes,
                va="top",
                color=matplotlib.rcParams["xtick.color"],
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )

        ax.set_xlabel(f"{short_model_label(left.label)} {metric_label}")
        ax.set_ylabel(f"{short_model_label(right.label)} {metric_label}")
        if use_log:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_box_aspect(1)
        finish_axis(ax)

    if color_mappable is not None:
        cbar = fig.colorbar(color_mappable, ax=axes, pad=0.01, fraction=0.035, shrink=0.89)
        cbar.outline.set_visible(False)
        style_colorbar_ticks(cbar, discrete=color_column == "true_neg_num")
        if colorbar_ticks is not None:
            cbar.set_ticks(colorbar_ticks)
        if colorbar_ticklabels is not None:
            cbar.set_ticklabels(colorbar_ticklabels)
        if color_label is not None:
            cbar.set_label(color_label)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_single_model_comparison(
    merged: pd.DataFrame,
    left_label: str,
    right_label: str,
    metric: str,
    metric_label: str,
    output_path: Path,
    use_log: bool,
    dpi: int,
    color_column: str | None = None,
    color_label: str | None = None,
) -> None:
    left_key = safe_label(left_label)
    right_key = safe_label(right_label)
    left_short = short_model_label(left_label)
    right_short = short_model_label(right_label)
    x_col = f"{metric}_{left_key}"
    y_col = f"{metric}_{right_key}"
    x = pd.to_numeric(merged[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(merged[y_col], errors="coerce").to_numpy(dtype=float)
    color_all = comparison_color_values(merged, color_column, left_key, right_key) if color_column is not None else None
    if color_column is not None and color_all is None:
        print(f"Skipping {output_path}: missing color column {color_column}")
        return
    color_norm, color_cmap, colorbar_ticks, colorbar_ticklabels = comparison_colorbar_settings(
        color_column,
        color_all,
    )
    keep = np.isfinite(x) & np.isfinite(y)
    if use_log:
        keep &= (x > 0) & (y > 0)
    if color_all is not None:
        keep &= np.isfinite(color_all)
    x = x[keep]
    y = y[keep]
    c = color_all[keep] if color_all is not None else None

    fig, ax = plt.subplots(figsize=(5.5, 5))
    if len(x):
        scatter_comparison(
            ax,
            x,
            y,
            use_log=use_log,
            color=AD_COLOR,
            color_values=c,
            color_label=color_label,
            norm=color_norm,
            cmap=color_cmap,
            colorbar_ticks=colorbar_ticks,
            colorbar_ticklabels=colorbar_ticklabels,
        )
        corr_values_x = np.log10(x) if use_log else x
        corr_values_y = np.log10(y) if use_log else y
        corr = np.corrcoef(corr_values_x, corr_values_y)[0, 1]
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color="0.25", linewidth=1.0, alpha=0.8)
        ax.text(
            0.03,
            0.97,
            f"{'log-' if use_log else ''}r={corr:.2f}",
            transform=ax.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    ax.set_title(metric_label)
    ax.set_xlabel(f"{left_short} {metric_label}")
    ax.set_ylabel(f"{right_short} {metric_label}")
    if use_log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    finish_axis(ax)
    fig.tight_layout(pad=0.01)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_single_model_ratio(
    merged: pd.DataFrame,
    left_label: str,
    right_label: str,
    metric: str,
    metric_label: str,
    output_path: Path,
    use_log: bool,
    dpi: int,
) -> None:
    left_key = safe_label(left_label)
    right_key = safe_label(right_label)
    left_short = short_model_label(left_label)
    right_short = short_model_label(right_label)
    x_col = f"{metric}_{left_key}"
    y_col = f"{metric}_{right_key}"
    x = pd.to_numeric(merged[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(merged[y_col], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(x) & np.isfinite(y) & (x != 0)
    if use_log:
        keep &= (x > 0) & (y > 0)
    x = x[keep]
    ratio = y[keep] / x
    keep_ratio = np.isfinite(ratio)
    if use_log:
        keep_ratio &= ratio > 0
    x = x[keep_ratio]
    ratio = ratio[keep_ratio]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    if len(x):
        sns.scatterplot(x=x, y=ratio, ax=ax, color=ACCENT_COLOR, legend=False, **SCATTER_KWARGS)
        median_ratio = float(np.median(ratio))
        ax.axhline(1.0, color="0.25", linewidth=1.0, alpha=0.8)
        ax.text(
            0.03,
            0.97,
            f"median ratio={median_ratio:.3g}",
            transform=ax.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    ax.set_title(f"{right_short} / {left_short}")
    ax.set_xlabel(f"{left_short} {metric_label}")
    ax.set_ylabel(f"{right_short} / {left_short} {metric_label}")
    if use_log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    finish_axis(ax)
    fig.tight_layout(pad=0.01)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_separate_model_comparisons(
    results: list[ResultTable],
    output_dir: Path,
    use_log: bool,
    dpi: int,
) -> None:
    if len(results) != 2:
        return

    left, right = results
    merged = left.df.merge(
        right.df,
        on="dataset_idx",
        suffixes=(f"_{safe_label(left.label)}", f"_{safe_label(right.label)}"),
    )
    for metric, metric_label in PRIMARY_METRICS:
        comparison_path = output_dir / f"{metric}.png"
        plot_single_model_comparison(
            merged=merged,
            left_label=left.label,
            right_label=right.label,
            metric=metric,
            metric_label=metric_label,
            output_path=comparison_path,
            use_log=use_log,
            dpi=dpi,
        )
        for color_column, color_label in COMPARISON_COLOR_FIELDS:
            plot_single_model_comparison(
                merged=merged,
                left_label=left.label,
                right_label=right.label,
                metric=metric,
                metric_label=metric_label,
                output_path=add_suffix_to_path(comparison_path, f"by_{color_column}"),
                use_log=use_log,
                dpi=dpi,
                color_column=color_column,
                color_label=color_label,
            )
        plot_single_model_ratio(
            merged=merged,
            left_label=left.label,
            right_label=right.label,
            metric=metric,
            metric_label=metric_label,
            output_path=output_dir / f"{metric}_ratio.png",
            use_log=use_log,
            dpi=dpi,
        )


def plot_dft_comparison(
    results: list[ResultTable],
    output_path: Path,
    stem: str,
    true_col: str,
    pred_col: str,
    metric_label: str,
    use_log_axes: bool,
    offset_correct: bool,
    dpi: int,
) -> bool:
    if not all({true_col, pred_col}.issubset(result.df.columns) for result in results):
        print(f"Skipping {stem}: missing {true_col}/{pred_col} columns")
        return False

    fig, axes = plt.subplots(1, len(results), figsize=(5.5 * len(results), 5), squeeze=False)
    axes = axes[0]
    for ax, result in zip(axes, results):
        x = pd.to_numeric(result.df[true_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(result.df[pred_col], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(x) & np.isfinite(y)
        if use_log_axes:
            keep &= (x > 0) & (y > 0)
        x = x[keep]
        y = y[keep]

        offset = 0.0
        if offset_correct and len(x):
            offset = float(np.median(y - x))
            y = y - offset

        if len(x):
            sns.scatterplot(x=x, y=y, ax=ax, color=model_color(result.label, AD_COLOR), legend=False, **SCATTER_KWARGS)
            corr_x = np.log10(x) if use_log_axes else x
            corr_y = np.log10(y) if use_log_axes else y
            corr = np.corrcoef(corr_x, corr_y)[0, 1]
            lo = min(x.min(), y.min())
            hi = max(x.max(), y.max())
            ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, alpha=0.7)
            ax.text(
                0.03,
                0.97,
                f"{'log-' if use_log_axes else ''}r={corr:.2f}",
                transform=ax.transAxes,
                va="top",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
            if offset_correct:
                ax.text(
                    0.03,
                    0.88,
                    f"median offset={offset:.3g}",
                    transform=ax.transAxes,
                    va="top",
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                )

        ax.set_title(result.label)
        ax.set_xlabel(f"DFT {metric_label}")
        y_label = f"{result.label} {metric_label}"
        if offset_correct:
            y_label += " (median-offset corrected)"
        ax.set_ylabel(y_label)
        if use_log_axes:
            ax.set_xscale("log")
            ax.set_yscale("log")
        finish_axis(ax)

    fig.tight_layout(pad=0.01)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_dft_comparisons(
    results: list[ResultTable],
    output_dir: Path,
    use_log: bool,
    dpi: int,
) -> None:
    wrote_any = False
    for stem, true_col, pred_col, metric_label, use_log_axes, offset_correct in DFT_COMPARISONS:
        wrote_any |= plot_dft_comparison(
            results=results,
            output_path=output_dir / f"dft_comparison_{stem}.png",
            stem=stem,
            true_col=true_col,
            pred_col=pred_col,
            metric_label=metric_label,
            use_log_axes=use_log and use_log_axes,
            offset_correct=offset_correct,
            dpi=dpi,
        )
    if wrote_any:
        print("Wrote DFT-vs-model comparison plots")


def plot_force_diagnostics(
    results: list[ResultTable],
    output_path: Path,
    use_log: bool,
    dpi: int,
) -> bool:
    required = {
        "force_true_norm",
        "force_model_norm",
        "force_l2_error",
        "force_cos_error",
    }
    if not all(required.issubset(result.df.columns) for result in results):
        print(f"Skipping force diagnostics: missing columns {sorted(required)}")
        return False

    panels = (
        (
            "force_true_norm",
            "force_model_norm",
            r"DFT force norm [$\mathrm{eV}/\AA$]",
            r"Model force norm [$\mathrm{eV}/\AA$]",
            True,
            True,
            True,
        ),
        (
            "force_true_norm",
            "force_l2_error",
            r"DFT force norm [$\mathrm{eV}/\AA$]",
            r"Force L2 error [$\mathrm{eV}/\AA$]",
            True,
            True,
            False,
        ),
        (
            "force_true_norm",
            "force_cos_error",
            r"DFT force norm [$\mathrm{eV}/\AA$]",
            "Force cosine error [unitless]",
            True,
            True,
            False,
        ),
    )

    fig, axes = plt.subplots(
        len(results),
        len(panels),
        figsize=(5.5 * len(panels), 4.5 * len(results)),
        squeeze=False,
    )

    for row_idx, result in enumerate(results):
        for col_idx, (x_col, y_col, x_label, y_label, log_x, log_y, diagonal) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            x = pd.to_numeric(result.df[x_col], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(result.df[y_col], errors="coerce").to_numpy(dtype=float)
            keep = np.isfinite(x) & np.isfinite(y)
            if use_log and log_x:
                keep &= x > 0
            if use_log and log_y:
                keep &= y > 0
            x = x[keep]
            y = y[keep]

            if len(x):
                sns.scatterplot(x=x, y=y, ax=ax, color=SUCCESS_COLOR, legend=False, **SCATTER_KWARGS)
                if diagonal:
                    lo = min(x.min(), y.min())
                    hi = max(x.max(), y.max())
                    ax.plot([lo, hi], [lo, hi], color="0.25", linewidth=1.0, alpha=0.8)
                corr_x = np.log10(x) if use_log and log_x else x
                corr_y = np.log10(y) if use_log and log_y else y
                corr = np.corrcoef(corr_x, corr_y)[0, 1]
                ax.text(
                    0.03,
                    0.97,
                    f"{'log-' if use_log and (log_x or log_y) else ''}r={corr:.2f}",
                    transform=ax.transAxes,
                    va="top",
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                )

            ax.set_title(result.label)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            if use_log and log_x:
                ax.set_xscale("log")
            if use_log and log_y:
                ax.set_yscale("log")
            finish_axis(ax)

    fig.tight_layout(pad=0.01)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def comparison_ratio_with_mask(
    merged: pd.DataFrame,
    metric: str,
    left_key: str,
    right_key: str,
    use_log: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y = comparison_values(merged, metric, left_key, right_key)
    if len(x) == 0 or len(y) == 0:
        empty_mask = np.zeros(len(merged), dtype=bool)
        return np.array([]), np.array([]), np.array([]), empty_mask

    keep = np.isfinite(x) & np.isfinite(y) & (x != 0)
    if use_log:
        keep &= (x > 0) & (y > 0)
    ratio = np.full_like(x, np.nan, dtype=float)
    ratio[keep] = y[keep] / x[keep]
    keep &= np.isfinite(ratio)
    if use_log:
        keep &= ratio > 0
    return x[keep], y[keep], ratio[keep], keep


def plot_eigen_scatter(
    merged: pd.DataFrame,
    left_label: str,
    right_label: str,
    metric: str,
    metric_label: str,
    output_path: Path,
    use_log: bool,
    dpi: int,
) -> None:
    left_key = safe_label(left_label)
    right_key = safe_label(right_label)
    x, y = comparison_values(merged, metric, left_key, right_key)
    keep = np.isfinite(x) & np.isfinite(y)
    if use_log:
        keep &= (x > 0) & (y > 0)
    x = x[keep]
    y = y[keep]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    if len(x):
        sns.scatterplot(x=x, y=y, ax=ax, color=AD_COLOR, legend=False, **SCATTER_KWARGS)
        corr_x = np.log10(x) if use_log else x
        corr_y = np.log10(y) if use_log else y
        corr = np.corrcoef(corr_x, corr_y)[0, 1]
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color="0.25", linewidth=1.0, alpha=0.8)
        ax.text(
            0.03,
            0.97,
            f"{'log-' if use_log else ''}r={corr:.2f}",
            transform=ax.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    ax.set_title(metric_label)
    ax.set_xlabel(f"AD {metric_label}")
    ax.set_ylabel(f"HIP {metric_label}")
    if use_log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    finish_axis(ax)
    fig.tight_layout(pad=0.01)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_eigen_ratio(
    merged: pd.DataFrame,
    left_label: str,
    right_label: str,
    metric: str,
    metric_label: str,
    output_path: Path,
    use_log: bool,
    dpi: int,
) -> None:
    left_key = safe_label(left_label)
    right_key = safe_label(right_label)
    x, _, ratio, _ = comparison_ratio_with_mask(merged, metric, left_key, right_key, use_log)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    if len(x):
        sns.scatterplot(x=x, y=ratio, ax=ax, color=ACCENT_COLOR, legend=False, **SCATTER_KWARGS)
        ax.axhline(1.0, color="0.25", linewidth=1.0, alpha=0.8)
        ax.text(
            0.03,
            0.97,
            f"median={np.median(ratio):.3g}\nHIP better={np.mean(ratio < 1.0):.1%}",
            transform=ax.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    ax.set_title(f"HIP / AD: {metric_label}")
    ax.set_xlabel(f"AD {metric_label}")
    ax.set_ylabel("HIP / AD")
    if use_log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    finish_axis(ax)
    fig.tight_layout(pad=0.01)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def structure_group_values(
    merged: pd.DataFrame,
    group: str,
    left_key: str,
    right_key: str,
) -> np.ndarray | None:
    if group == "neg_num_agree_ad":
        column = merged_column(merged, "neg_num_agree", left_key)
    elif group == "neg_num_agree_hip":
        column = merged_column(merged, "neg_num_agree", right_key)
    else:
        column = merged_column(merged, group, left_key)
    if column is None:
        return None
    return pd.to_numeric(merged[column], errors="coerce").to_numpy(dtype=float)


def grouped_masks(values: np.ndarray, group: str) -> tuple[list[str], list[np.ndarray]]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return [], []
    if group == "true_neg_num":
        labels = ["0", "1", "2", "3+"]
        masks = [values == 0, values == 1, values == 2, values >= 3]
        return labels, masks
    unique = sorted(np.unique(finite.astype(int)))
    labels = [str(value) for value in unique]
    masks = [values == value for value in unique]
    return labels, masks


def plot_eigen_ratio_by_structure(
    merged: pd.DataFrame,
    left_label: str,
    right_label: str,
    metric: str,
    metric_label: str,
    output_path: Path,
    use_log: bool,
    dpi: int,
) -> None:
    left_key = safe_label(left_label)
    right_key = safe_label(right_label)
    _, _, ratio, keep = comparison_ratio_with_mask(merged, metric, left_key, right_key, use_log)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), squeeze=False)
    for ax, (group, title) in zip(axes.ravel(), STRUCTURE_GROUPS):
        values = structure_group_values(merged, group, left_key, right_key)
        if values is None or len(ratio) == 0:
            ax.set_axis_off()
            continue
        values = values[keep]
        labels, masks = grouped_masks(values, group)
        data = [ratio[mask] for mask in masks if np.any(mask)]
        labels = [label for label, mask in zip(labels, masks) if np.any(mask)]
        if not data:
            ax.set_axis_off()
            continue
        sns.boxplot(data=data, ax=ax, showfliers=False, color=HIP_COLOR)
        ax.set_xticks(range(len(labels)), labels=labels)
        ax.axhline(1.0, color="0.25", linewidth=1.0, alpha=0.8)
        ax.set_title(title)
        ax.set_yscale("log")
        finish_axis(ax)

    axes[0, 0].set_ylabel("HIP / AD")
    axes[1, 0].set_ylabel("HIP / AD")
    fig.suptitle(f"HIP / AD by structure: {metric_label}", y=0.995)
    fig.tight_layout(pad=0.01)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_negative_mode_confusion(
    merged: pd.DataFrame,
    left_label: str,
    right_label: str,
    output_path: Path,
    dpi: int,
) -> None:
    left_key = safe_label(left_label)
    right_key = safe_label(right_label)
    true_col = merged_column(merged, "true_neg_num", left_key)
    left_model_col = merged_column(merged, "model_neg_num", left_key)
    right_model_col = merged_column(merged, "model_neg_num", right_key)
    if true_col is None or left_model_col is None or right_model_col is None:
        return

    true = pd.to_numeric(merged[true_col], errors="coerce").to_numpy(dtype=float)
    model_values = (
        ("AD", pd.to_numeric(merged[left_model_col], errors="coerce").to_numpy(dtype=float)),
        ("HIP", pd.to_numeric(merged[right_model_col], errors="coerce").to_numpy(dtype=float)),
    )
    labels = ["0", "1", "2", "3+"]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), squeeze=False)
    for ax, (title, pred) in zip(axes[0], model_values):
        keep = np.isfinite(true) & np.isfinite(pred)
        true_clipped = np.clip(true[keep].astype(int), 0, 3)
        pred_clipped = np.clip(pred[keep].astype(int), 0, 3)
        counts = np.zeros((4, 4), dtype=int)
        for true_value, pred_value in zip(true_clipped, pred_clipped):
            counts[true_value, pred_value] += 1
        image = ax.imshow(counts, cmap="Blues")
        for row in range(4):
            for col in range(4):
                ax.text(col, row, str(counts[row, col]), ha="center", va="center")
        ax.set_title(title)
        ax.set_xlabel("Predicted negative modes")
        ax.set_ylabel("True negative modes")
        ax.set_xticks(range(4), labels=labels)
        ax.set_yticks(range(4), labels=labels)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout(pad=0.01)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_negative_mode_outcomes(
    merged: pd.DataFrame,
    left_label: str,
    right_label: str,
    output_path: Path,
    dpi: int,
) -> None:
    left_key = safe_label(left_label)
    right_key = safe_label(right_label)
    left_col = merged_column(merged, "neg_num_agree", left_key)
    right_col = merged_column(merged, "neg_num_agree", right_key)
    if left_col is None or right_col is None:
        return

    ad_correct = pd.to_numeric(merged[left_col], errors="coerce").to_numpy(dtype=float) == 1
    hip_correct = pd.to_numeric(merged[right_col], errors="coerce").to_numpy(dtype=float) == 1
    labels = ["Both correct", "AD only", "HIP only", "Both wrong"]
    counts = [
        int(np.sum(ad_correct & hip_correct)),
        int(np.sum(ad_correct & ~hip_correct)),
        int(np.sum(~ad_correct & hip_correct)),
        int(np.sum(~ad_correct & ~hip_correct)),
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.barplot(x=labels, y=counts, ax=ax, palette=[SUCCESS_COLOR, AD_COLOR, HIP_COLOR, ACCENT_COLOR], hue=labels, legend=False)
    ax.set_ylabel("Samples")
    ax.set_title("Negative-mode classification outcomes")
    ax.tick_params(axis="x", rotation=20)
    finish_axis(ax)
    fig.tight_layout(pad=0.01)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_eigen_worst_cases(
    merged: pd.DataFrame,
    left_label: str,
    right_label: str,
    metric: str,
    metric_label: str,
    output_path: Path,
    use_log: bool,
    dpi: int,
    top_n: int = 20,
) -> None:
    left_key = safe_label(left_label)
    right_key = safe_label(right_label)
    _, _, ratio, keep = comparison_ratio_with_mask(merged, metric, left_key, right_key, use_log)
    if len(ratio) == 0:
        return
    improvement = 1.0 / ratio
    top_positions = np.argsort(improvement)[-top_n:][::-1]
    original_indices = np.flatnonzero(keep)[top_positions]

    true_col = merged_column(merged, "true_neg_num", left_key)
    left_model_col = merged_column(merged, "model_neg_num", left_key)
    right_model_col = merged_column(merged, "model_neg_num", right_key)
    natoms_col = merged_column(merged, "natoms", left_key)
    tick_labels = []
    for row_idx in original_indices:
        row = merged.iloc[row_idx]
        parts = [f"id={int(row['dataset_idx'])}"]
        if natoms_col is not None:
            parts.append(f"N={int(row[natoms_col])}")
        if true_col is not None and left_model_col is not None and right_model_col is not None:
            parts.append(
                f"neg={int(row[true_col])}/{int(row[left_model_col])}/{int(row[right_model_col])}"
            )
        tick_labels.append("\n".join(parts))

    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(top_positions)), 4.5))
    sns.barplot(x=list(range(len(top_positions))), y=improvement[top_positions], ax=ax, color=HIP_COLOR)
    ax.axhline(1.0, color="0.25", linewidth=1.0, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xticks(range(len(top_positions)), labels=tick_labels, rotation=60, ha="right")
    ax.set_ylabel("AD / HIP")
    ax.set_title(f"Largest AD/HIP error ratios: {metric_label}")
    finish_axis(ax)
    fig.tight_layout(pad=0.01)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_eigen_diagnostics(
    results: list[ResultTable],
    output_dir: Path,
    use_log: bool,
    dpi: int,
) -> None:
    if len(results) != 2:
        return

    left, right = results
    left_key = safe_label(left.label)
    right_key = safe_label(right.label)
    merged = left.df.merge(
        right.df,
        on="dataset_idx",
        suffixes=(f"_{left_key}", f"_{right_key}"),
    )
    eigen_dir = output_dir / "eigen_diagnostics"
    eigen_dir.mkdir(parents=True, exist_ok=True)

    for metric, metric_label in EIGEN_DIAGNOSTIC_METRICS:
        x, y = comparison_values(merged, metric, left_key, right_key)
        if len(x) == 0 or len(y) == 0:
            print(f"Skipping {metric}: missing columns")
            continue
        plot_eigen_scatter(
            merged,
            left.label,
            right.label,
            metric,
            metric_label,
            eigen_dir / f"{metric}_comparison.png",
            use_log=use_log,
            dpi=dpi,
        )
        plot_eigen_ratio(
            merged,
            left.label,
            right.label,
            metric,
            metric_label,
            eigen_dir / f"{metric}_ratio.png",
            use_log=use_log,
            dpi=dpi,
        )
        plot_eigen_ratio_by_structure(
            merged,
            left.label,
            right.label,
            metric,
            metric_label,
            eigen_dir / f"{metric}_ratio_by_structure.png",
            use_log=use_log,
            dpi=dpi,
        )
        plot_eigen_worst_cases(
            merged,
            left.label,
            right.label,
            metric,
            metric_label,
            eigen_dir / f"{metric}_worst_ad_cases.png",
            use_log=use_log,
            dpi=dpi,
        )

    plot_negative_mode_confusion(
        merged,
        left.label,
        right.label,
        eigen_dir / "negative_mode_confusion.png",
        dpi=dpi,
    )
    plot_negative_mode_outcomes(
        merged,
        left.label,
        right.label,
        eigen_dir / "negative_mode_outcomes.png",
        dpi=dpi,
    )
    print(f"Wrote eigen diagnostics to {eigen_dir}")


def main() -> None:
    args = parse_args()
    results = load_results(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    use_log = not args.no_log

    plot_histograms(
        results,
        ENERGY_METRICS,
        args.output_dir / "energy_error_histogram.png",
        bins=args.bins,
        use_log=use_log,
    )
    plot_histograms(
        results,
        PRIMARY_METRICS,
        args.output_dir / "primary_error_histograms.png",
        bins=args.bins,
        use_log=use_log,
    )
    plot_histograms(
        results,
        PRIMARY_METRICS,
        args.output_dir / "primary_error_histograms_larger_bins.png",
        bins=max(1, args.bins // 2),
        use_log=use_log,
    )
    available_extra_metrics = tuple(
        (metric, label)
        for metric, label in EXTRA_METRICS
        if all(metric in result.df.columns for result in results)
    )
    if available_extra_metrics:
        plot_histograms(
            results,
            available_extra_metrics,
            args.output_dir / "extra_error_histograms.png",
            bins=args.bins,
            use_log=use_log,
        )
    plot_within_model_correlations(
        results,
        args.output_dir / "within_model_error_correlations.png",
        use_log=use_log,
        dpi=args.dpi,
    )
    plot_force_hessian_scatter_by_model(
        results,
        args.output_dir / "force_hessian_scatter_by_model.png",
        use_log=use_log,
        dpi=args.dpi,
    )
    force_hessian_scatter_path = args.output_dir / "force_hessian_scatter_by_model.png"
    two_panel_results = results[:2]
    for color_column, color_label in COMPARISON_COLOR_FIELDS:
        plot_force_hessian_scatter_by_model(
            results,
            add_suffix_to_path(force_hessian_scatter_path, f"by_{color_column}"),
            use_log=use_log,
            dpi=args.dpi,
            color_column=color_column,
            color_label=color_label,
        )
        if len(two_panel_results) == 2:
            plot_force_hessian_scatter_by_model(
                two_panel_results,
                add_suffix_to_path(force_hessian_scatter_path, f"ad_hip_by_{color_column}"),
                use_log=use_log,
                dpi=args.dpi,
                color_column=color_column,
                color_label=color_label,
            )
    comparison_correlations_path = args.output_dir / "error_correlations.png"
    plot_model_comparison_correlations(
        results,
        comparison_correlations_path,
        use_log=use_log,
        dpi=args.dpi,
    )
    for color_column, color_label in COMPARISON_COLOR_FIELDS:
        plot_model_comparison_correlations(
            results,
            add_suffix_to_path(comparison_correlations_path, f"by_{color_column}"),
            use_log=use_log,
            dpi=args.dpi,
            color_column=color_column,
            color_label=color_label,
        )
    plot_separate_model_comparisons(
        results,
        args.output_dir,
        use_log=use_log,
        dpi=args.dpi,
    )
    plot_dft_comparisons(
        results,
        args.output_dir,
        use_log=use_log,
        dpi=args.dpi,
    )
    if plot_force_diagnostics(
        results,
        args.output_dir / "force_diagnostics.png",
        use_log=use_log,
        dpi=args.dpi,
    ):
        print("Wrote force diagnostics plot")
    plot_eigen_diagnostics(
        results,
        args.output_dir,
        use_log=use_log,
        dpi=args.dpi,
    )

    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
