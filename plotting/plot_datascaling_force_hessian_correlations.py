#!/usr/bin/env python
"""Plot data scaling above force/Hessian correlation panels."""
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
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402
import seaborn as sns  # noqa: E402


AD_COLOR = "#5e859e"
HIP_COLOR = "#d96001"
AD_NO_H_COLOR = "#837d80"
PLOT_FONT_COLOR = "#2F4565"
EF_COLOR = "#837d80"
LINE_WIDTH = 4.4
MARKER_SIZE = 11.0
PANEL_LABEL_SIZE = 18

DEFAULT_DATA_DIR = Path("scaling")
DEFAULT_OUTPUT = Path("plots/datascaling/datascaling_force_hessian_correlations.png")
DEFAULT_RESULTS = (
    ("AD", Path("results_evalhorm/eqv2_ts1x-val_autograd_metrics.parquet")),
    ("HIP", Path("results_evalhorm/hip_v2_ts1x-val_predict_metrics.parquet")),
    ("AD (no H)", Path("results_evalhorm/eqv2_orig_ts1xval10k_29148768_ts1x-val_autograd_metrics.parquet")),
)
LOSS_CONFIGS = (
    ("Energy", "Loss E", "wandb_datascaling_loss_energy2.csv"),
    ("Force", "Loss F", "wandb_datascaling_loss_force2.csv"),
    ("Hessian", "MAE Hessian", "wandb_datascaling_loss_hessian2.csv"),
)
METRIC_TITLES = {
    "Energy": r"Energy MAE [$\mathrm{eV}$]",
    "Force": r"Force MAE [$\mathrm{eV}\,\AA^{-1}$]",
    "Hessian": r"Hessian MAE [$\mathrm{eV}\,\AA^{-2}$]",
}
TRAINING_TYPE_LABELS = {
    True: "Energy-Force",
    False: "HIP",
}
TRAINING_TYPE_ORDER = ["Energy-Force", "HIP"]
TRAINING_TYPE_PALETTE = {
    "Energy-Force": EF_COLOR,
    "HIP": HIP_COLOR,
}
TRAINING_TYPE_MARKERS = {
    "Energy-Force": "o",
    "HIP": "D",
}
EXCLUDED_TRAINING_SIZES = {20000.0, 200000.0}
SCATTER_KWARGS = {
    "s": 8,
    "alpha": 0.18,
    "linewidths": 0,
    "rasterized": True,
}


@dataclass
class ResultTable:
    label: str
    path: Path
    df: pd.DataFrame


def apply_plot_style() -> None:
    sns.set_theme(
        context="talk",
        style="whitegrid",
        palette=[AD_COLOR, HIP_COLOR, AD_NO_H_COLOR],
        rc={
            "axes.edgecolor": "#E6E6E6",
            "axes.labelcolor": PLOT_FONT_COLOR,
            "axes.labelsize": 17,
            "axes.linewidth": 1.1,
            "axes.titlecolor": PLOT_FONT_COLOR,
            "axes.titlesize": 20,
            "figure.facecolor": "white",
            "font.family": "sans-serif",
            "grid.color": "#E9E9E9",
            "grid.linewidth": 1.0,
            "legend.edgecolor": "none",
            "legend.fontsize": 14,
            "legend.frameon": True,
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,
            "savefig.facecolor": "white",
            "xtick.color": PLOT_FONT_COLOR,
            "xtick.labelsize": 14,
            "ytick.color": PLOT_FONT_COLOR,
            "ytick.labelsize": 14,
        },
    )
    matplotlib.rcParams.update(
        {
            "text.color": PLOT_FONT_COLOR,
            "xtick.major.size": 0.9,
            "xtick.minor.size": 0.9,
            "ytick.major.size": 0.9,
            "ytick.minor.size": 0.9,
        }
    )


def finish_axis(ax: matplotlib.axes.Axes, *, legend: bool = False) -> None:
    ax.grid(True, color="#E9E9E9", linewidth=1.0)
    if ax.get_xscale() == "log":
        ax.minorticks_on()
        ax.grid(True, which="minor", axis="x", color="#F1F1F1", linewidth=0.7, alpha=0.85)
    if ax.get_yscale() == "log":
        ax.minorticks_on()
        ax.grid(True, which="minor", axis="y", color="#F1F1F1", linewidth=0.7, alpha=0.85)
    sns.despine(ax=ax, trim=False)
    if legend:
        ax.legend(frameon=True, edgecolor="none")


def dataset_size_from_method(method_name: str) -> float | None:
    match = re.match(r"^([0-9.]+(?:e[+-]?[0-9]+)?)", method_name)
    return float(match.group(1)) if match else None


def load_loss_stats(data_dir: Path) -> dict[str, pd.DataFrame]:
    stats_by_metric = {}
    for human_name, losstype, filename in LOSS_CONFIGS:
        df = pd.read_csv(data_dir / filename)
        rows = []
        for col in [col for col in df.columns if col.endswith(f"val-{losstype}")]:
            clean_data = df[col].dropna()
            if clean_data.empty:
                continue

            method_name = col.replace(f" - val-{losstype}", "")
            is_ef = method_name.endswith("EF")
            rows.append(
                {
                    "Method": method_name,
                    "Min_Value": clean_data.min(),
                    "Training Type": TRAINING_TYPE_LABELS[is_ef],
                    "Dataset size": dataset_size_from_method(method_name),
                }
            )
        stats_by_metric[human_name] = pd.DataFrame(rows)
    return stats_by_metric


def plot_datascaling_metric(
    ax: plt.Axes,
    data: pd.DataFrame,
    title: str,
    *,
    legend: bool,
) -> None:
    plot_data = data.dropna(subset=["Dataset size"])
    plot_data = plot_data[~plot_data["Dataset size"].isin(EXCLUDED_TRAINING_SIZES)]
    plot_data = plot_data.sort_values(["Training Type", "Dataset size"])
    sns.lineplot(
        data=plot_data,
        x="Dataset size",
        y="Min_Value",
        hue="Training Type",
        hue_order=TRAINING_TYPE_ORDER,
        style="Training Type",
        style_order=TRAINING_TYPE_ORDER,
        palette=TRAINING_TYPE_PALETTE,
        markers=TRAINING_TYPE_MARKERS,
        dashes=False,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        ax=ax,
        legend=legend,
        zorder=3,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Number of Training Samples")
    ax.set_ylabel(METRIC_TITLES[title])
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10, labelOnlyBase=True))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="both", which="both", length=0)
    finish_axis(ax)
    legend_obj = ax.get_legend()
    if legend_obj is not None:
        legend_obj.set_title(None)
        legend_obj.get_frame().set_alpha(0.75)
        legend_obj.get_frame().set_linewidth(0)
        for text in legend_obj.get_texts():
            text.set_fontsize(matplotlib.rcParams["font.size"])


def load_results(result_specs: list[tuple[str, Path]]) -> list[ResultTable]:
    results = []
    required = {"dataset_idx", "forces_error", "hessian_error"}
    for label, path in result_specs:
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_parquet(path)
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        results.append(ResultTable(label=label, path=path, df=df))
    return results


def model_color(label: str) -> str:
    if label == "HIP":
        return HIP_COLOR
    if label == "AD (no H)":
        return AD_NO_H_COLOR
    return AD_COLOR


def result_color_values(result: ResultTable, column: str | None) -> np.ndarray | None:
    if column is None:
        return None
    if column not in result.df.columns:
        raise ValueError(f"{result.path} is missing color column {column!r}")
    return pd.to_numeric(result.df[column], errors="coerce").to_numpy(dtype=float)


def collect_force_hessian_values(
    results: list[ResultTable],
    *,
    color_column: str | None,
    use_log: bool,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray | None]], np.ndarray | None]:
    plot_values = []
    color_arrays = []
    for result in results:
        x = pd.to_numeric(result.df["forces_error"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(result.df["hessian_error"], errors="coerce").to_numpy(dtype=float)
        c = result_color_values(result, color_column)
        keep = np.isfinite(x) & np.isfinite(y)
        if use_log:
            keep &= (x > 0) & (y > 0)
        if c is not None:
            keep &= np.isfinite(c)
        kept_c = c[keep] if c is not None else None
        if kept_c is not None:
            color_arrays.append(kept_c)
        plot_values.append((x[keep], y[keep], kept_c))
    all_color_values = np.concatenate(color_arrays) if color_arrays else None
    return plot_values, all_color_values


def plot_force_hessian_scatter_row(
    axes: list[plt.Axes],
    results: list[ResultTable],
    *,
    cax: plt.Axes,
    color_column: str | None,
    color_label: str | None,
    use_log: bool,
) -> None:
    plot_values, all_color_values = collect_force_hessian_values(
        results,
        color_column=color_column,
        use_log=use_log,
    )
    all_x = np.concatenate([x for x, _, _ in plot_values if len(x)])
    all_y = np.concatenate([y for _, y, _ in plot_values if len(y)])
    xlim = (float(all_x.min()), float(all_x.max()))
    ylim = (float(all_y.min()), float(all_y.max()))
    color_norm = None
    color_mappable = None
    if all_color_values is not None and len(all_color_values):
        color_norm = matplotlib.colors.Normalize(
            vmin=float(all_color_values.min()),
            vmax=float(all_color_values.max()),
        )

    for ax, result, (x, y, c) in zip(axes, results, plot_values, strict=True):
        if c is None:
            sns.scatterplot(x=x, y=y, ax=ax, color=model_color(result.label), legend=False, **SCATTER_KWARGS)
        else:
            color_mappable = ax.scatter(
                x,
                y,
                c=c,
                cmap="viridis",
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
            0.91,
            f"{'log-' if use_log else ''}r={corr:.2f}",
            transform=ax.transAxes,
            va="top",
            color=PLOT_FONT_COLOR,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        ax.set_title(result.label)
        ax.set_xlabel(r"Force MAE [$\mathrm{eV}/\AA$]")
        ax.set_ylabel(r"Hessian MAE [$\mathrm{eV}/\AA^2$]")
        if use_log:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        finish_axis(ax)

    if color_mappable is not None:
        cbar = axes[0].figure.colorbar(color_mappable, cax=cax)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(which="both", length=2.5, width=0.8)
        if color_label is not None:
            cbar.set_label(color_label)
    else:
        cax.set_visible(False)


def add_panel_labels(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    for panel_label, ax in zip("abcdef", axes, strict=True):
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


def make_plot(
    data_dir: Path = DEFAULT_DATA_DIR,
    result_specs: list[tuple[str, Path]] | None = None,
    output_path: Path = DEFAULT_OUTPUT,
    *,
    color_column: str | None = "natoms",
    color_label: str | None = "Number of Atoms",
    use_log: bool = True,
    dpi: int = 250,
) -> None:
    apply_plot_style()
    result_specs = list(DEFAULT_RESULTS) if result_specs is None else result_specs
    stats_by_metric = load_loss_stats(data_dir)
    results = load_results(result_specs)

    fig = plt.figure(figsize=(15, 8.6), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.06, hspace=0.05)
    spec = fig.add_gridspec(
        2,
        4,
        width_ratios=(1, 1, 1, 0.045),
        height_ratios=(0.86, 1.0),
    )
    top_axes = [fig.add_subplot(spec[0, idx]) for idx in range(3)]
    bottom_axes = [fig.add_subplot(spec[1, idx]) for idx in range(3)]
    fig.add_subplot(spec[0, 3]).set_visible(False)
    cax = fig.add_subplot(spec[1, 3])

    for ax, title in zip(top_axes, ("Energy", "Force", "Hessian"), strict=True):
        plot_datascaling_metric(
            ax,
            stats_by_metric[title],
            title,
            legend=title == "Energy",
        )
    plot_force_hessian_scatter_row(
        bottom_axes,
        results,
        cax=cax,
        color_column=color_column,
        color_label=color_label,
        use_log=use_log,
    )
    add_panel_labels(fig, [*top_axes, *bottom_axes])
    for idx, ax in enumerate(top_axes):
        if idx != 1:
            ax.set_xlabel("")
    for idx, ax in enumerate(bottom_axes):
        if idx != 1:
            ax.set_xlabel("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument(
        "--result",
        action="append",
        nargs=2,
        metavar=("LABEL", "PARQUET"),
        default=[],
        help="Result Parquet file to plot. May be repeated.",
    )
    parser.add_argument("--color-column", default="natoms")
    parser.add_argument("--color-label", default="Number of Atoms")
    parser.add_argument("--no-log", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_plot(
        data_dir=args.data_dir,
        result_specs=[(label, Path(path)) for label, path in args.result] or None,
        output_path=args.output,
        color_column=args.color_column,
        color_label=args.color_label,
        use_log=not args.no_log,
        dpi=args.dpi,
    )
