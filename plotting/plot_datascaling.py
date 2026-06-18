from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter
from matplotlib.transforms import offset_copy

from plot_style import (
    ANNOTATION_BOLD_FONT_SIZE,
    AXES_FONT_SIZE,
    AXES_TITLE_FONT_SIZE,
    HESSIAN_METHOD_TO_COLOUR,
    LEGEND_FONT_SIZE,
    TITLE_FONT_SIZE,
)

PLOT_FONT_COLOR = "#2a3f5f"
PLOT_FONT_FAMILY = ("Open Sans", "Arial", "Helvetica", "DejaVu Sans", "sans-serif")
TICK_LABEL_FONT_SIZE = AXES_FONT_SIZE - 2
EXCLUDED_TRAINING_SIZES = {20000.0, 200000.0}
LOSS_CONFIGS = (
    ("Energy", "Loss E", "wandb_datascaling_loss_energy2.csv"),
    ("Force", "Loss F", "wandb_datascaling_loss_force2.csv"),
    ("Hessian", "MAE Hessian", "wandb_datascaling_loss_hessian2.csv"),
)
METRIC_TITLES = {
    "Energy": r"Energy MAE [$\mathrm{eV}$]",
    "Force": r"Force MAE [$\mathrm{eV}\,\mathrm{\AA}^{-1}$]",
    "Hessian": r"Hessian MAE [$\mathrm{eV}\,\mathrm{\AA}^{-2}$]",
}
TRAINING_TYPE_LABELS = {
    True: "Energy-Force",
    False: "HIP",
}
TRAINING_TYPE_ORDER = ["Energy-Force", "HIP"]
TRAINING_TYPE_PALETTE = {
    "Energy-Force": HESSIAN_METHOD_TO_COLOUR["ef"],
    "HIP": HESSIAN_METHOD_TO_COLOUR.get(
        "prediction",
        HESSIAN_METHOD_TO_COLOUR["prediction"],
    ),
}
TRAINING_TYPE_MARKERS = {
    "Energy-Force": "o",
    "HIP": "D",
}


def _set_plot_style() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font_family = next(
        (
            family
            for family in PLOT_FONT_FAMILY
            if family == "sans-serif" or family in available_fonts
        ),
        "sans-serif",
    )
    sns.set_theme(
        style="whitegrid",
        context="paper",
        rc={
            "font.family": font_family,
            "text.color": PLOT_FONT_COLOR,
            "axes.labelcolor": PLOT_FONT_COLOR,
            "axes.titlecolor": PLOT_FONT_COLOR,
            "xtick.color": PLOT_FONT_COLOR,
            "ytick.color": PLOT_FONT_COLOR,
            "legend.labelcolor": PLOT_FONT_COLOR,
            "grid.color": "#e6e6e6",
        },
    )


def _dataset_size_from_method(method_name: str) -> float | None:
    match = re.match(r"^([0-9.]+(?:e[+-]?[0-9]+)?)", method_name)
    return float(match.group(1)) if match else None


def _load_loss_stats(data_dir: Path, output_dir: Path) -> dict[str, pd.DataFrame]:
    stats_by_metric = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for human_name, losstype, filename in LOSS_CONFIGS:
        csv_path = data_dir / filename
        df = pd.read_csv(csv_path)
        print(f"Loaded {csv_path} with shape {df.shape}")

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
                    "Last_Value": clean_data.iloc[-1],
                    "Min_Value": clean_data.min(),
                    "Max_Value": clean_data.max(),
                    "ef": is_ef,
                    "Training Type": TRAINING_TYPE_LABELS[is_ef],
                    "Dataset size": _dataset_size_from_method(method_name),
                }
            )

        stats = pd.DataFrame(rows).set_index("Method")
        stats.to_csv(output_dir / f"loss_{human_name.lower()}.csv", index=False)
        stats_by_metric[human_name] = stats
        print(f"Saved filtered {human_name.lower()} data to {output_dir}")
        print(stats)

    return stats_by_metric


def _plot_metric(
    ax,
    data: pd.DataFrame,
    title: str,
    *,
    legend: bool = False,
    xlabel: bool = True,
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
        linewidth=2,
        markersize=4,
        ax=ax,
        legend=legend,
        zorder=3,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(METRIC_TITLES[title], fontsize=TITLE_FONT_SIZE - 2)
    ax.set_xlabel(
        "Number of Training Samples" if xlabel else "",
        fontsize=AXES_TITLE_FONT_SIZE,
    )
    ax.set_ylabel("")
    ax.set_axisbelow(True)
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10, labelOnlyBase=True))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, zorder=0)
    ax.grid(True, which="minor", axis="x", color="#f1f1f1", linewidth=0.7, alpha=0.85)
    ax.grid(True, which="minor", axis="y", color="#f1f1f1", linewidth=0.7, alpha=0.85)
    ax.tick_params(axis="both", which="both", length=0, labelsize=TICK_LABEL_FONT_SIZE)
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_obj = ax.get_legend()
    if legend_obj is not None:
        legend_obj.set_title(None)
        legend_obj.set_zorder(1)
        legend_obj.get_frame().set_alpha(0.6)
        legend_obj.get_frame().set_linewidth(0)
        legend_obj.get_frame().set_edgecolor("none")
        for text in legend_obj.get_texts():
            text.set_fontsize(LEGEND_FONT_SIZE - 2)


def _add_panel_labels(fig, axes) -> None:
    for panel_label, ax in zip("abc", axes):
        panel_label_transform = offset_copy(
            ax.transAxes,
            fig=fig,
            x=44,
            y=-26,
            units="dots",
        )
        ax.text(
            -0.08,
            1.06,
            panel_label,
            transform=panel_label_transform,
            fontsize=ANNOTATION_BOLD_FONT_SIZE - 2,
            fontfamily="DejaVu Sans",
            fontweight="bold",
            color=PLOT_FONT_COLOR,
            va="bottom",
            ha="right",
        )


def make_plot(
    data_dir: str | Path = "scaling",
    output_dir: str | Path = "plots/datascaling",
) -> None:
    _set_plot_style()

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    stats_by_metric = _load_loss_stats(data_dir, output_dir)

    for human_name, stats in stats_by_metric.items():
        fig, ax = plt.subplots(figsize=(3.2, 2.8))
        _plot_metric(ax, stats, human_name, legend=True)
        fig.tight_layout(pad=0.01)
        output_path = output_dir / f"log_log_{human_name.lower()}_mae.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)
        print(f"Saved {output_path}")

    fig, axes = plt.subplots(1, 3, figsize=(8, 2.7))
    for ax, title in zip(axes, ("Energy", "Force", "Hessian")):
        _plot_metric(
            ax,
            stats_by_metric[title],
            title,
            legend=title == "Energy",
            xlabel=title == "Force",
        )

    _add_panel_labels(fig, axes)
    fig.tight_layout(pad=0.01)
    fig.subplots_adjust(wspace=0.24)

    output_path = output_dir / "datascaling_energy_force_hessian.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot data-scaling loss curves.")
    parser.add_argument("--data-dir", default="scaling")
    parser.add_argument("--output-dir", default="plots/datascaling")
    args = parser.parse_args()
    make_plot(data_dir=args.data_dir, output_dir=args.output_dir)
