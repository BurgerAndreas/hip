"""Shared Seaborn styling for repository plotting scripts."""
from __future__ import annotations

from collections.abc import Mapping

import matplotlib
import seaborn as sns


AD_COLOR = "#2F6B8F"
HIP_COLOR = "#E76F00"
DFT_COLOR = "#2B2B2B"
GUIDE_COLOR = "#8A8A8A"
ACCENT_COLOR = "#B75DAE"
SUCCESS_COLOR = "#5AA469"
LINE_WIDTH = 2.2
THIN_LINE_WIDTH = 1.6
GUIDE_LINE_WIDTH = 1.4
MARKER_SIZE = 5.5
SMALL_MARKER_SIZE = 4.5

MODEL_COLORS: dict[str, str] = {
    "AD": AD_COLOR,
    "AD Hessians": AD_COLOR,
    "EQV2": AD_COLOR,
    "EQV2 AD": AD_COLOR,
    "Autograd": AD_COLOR,
    "DFT": DFT_COLOR,
    "HIP": HIP_COLOR,
    "HIP Hessians": HIP_COLOR,
}


def apply_plot_style() -> None:
    sns.set_theme(
        context="talk",
        style="whitegrid",
        palette=[AD_COLOR, HIP_COLOR, SUCCESS_COLOR, ACCENT_COLOR, DFT_COLOR],
        rc={
            "axes.edgecolor": "#E6E6E6",
            "axes.labelcolor": "#2F4565",
            "axes.labelsize": 17,
            "axes.linewidth": 1.1,
            "axes.titlecolor": "#2F4565",
            "axes.titlesize": 20,
            "figure.facecolor": "white",
            "font.family": "sans-serif",
            "grid.color": "#E9E9E9",
            "grid.linewidth": 1.0,
            "legend.frameon": False,
            "legend.fontsize": 14,
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,
            "savefig.facecolor": "white",
            "xtick.color": "#2F4565",
            "xtick.labelsize": 14,
            "ytick.color": "#2F4565",
            "ytick.labelsize": 14,
        },
    )


def model_color(label: str | None, fallback: str | None = None) -> str | None:
    if label is None:
        return fallback
    normalized = label.casefold()
    if "hip" in normalized:
        return HIP_COLOR
    if "ad" in normalized or "eqv2" in normalized or "autograd" in normalized:
        return AD_COLOR
    if "dft" in normalized or "orca" in normalized:
        return DFT_COLOR
    return MODEL_COLORS.get(label, fallback)


def model_palette(labels: list[str] | tuple[str, ...] | Mapping[str, object]) -> dict[str, str]:
    keys = list(labels.keys()) if isinstance(labels, Mapping) else list(labels)
    fallback = sns.color_palette("deep", n_colors=max(len(keys), 1)).as_hex()
    return {label: model_color(label, fallback[idx]) for idx, label in enumerate(keys)}


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
        ax.legend(frameon=False)


apply_plot_style()
