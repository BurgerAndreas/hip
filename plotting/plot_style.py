"""Shared Seaborn styling for repository plotting scripts."""
from __future__ import annotations

from collections.abc import Mapping

import matplotlib
import seaborn as sns
# import plotly.colors

AD_COLOR = "#2F6B8F"
HIP_COLOR = "#E76F00"
DFT_COLOR = "#2B2B2B"
GUIDE_COLOR = "#8A8A8A"
ACCENT_COLOR = "#B75DAE"
SUCCESS_COLOR = "#5AA469"
EQV2_FORCE_COLOR = "#295c7e"
AD_NO_H_COLOR = "#5a5255"
EQV2_NO_H_FORCE_COLOR = AD_NO_H_COLOR
HIP_FORCE_COLOR = "#ffb482"
LEFTNET_CF_FORCE_COLOR = "#4a8a72"
LEFTNET_DF_FORCE_COLOR = "#8A3F3F" # #8A3F3F #743737
LINE_WIDTH = 2.2
THIN_LINE_WIDTH = 1.6
GUIDE_LINE_WIDTH = 1.4
MARKER_SIZE = 5.5
SMALL_MARKER_SIZE = 4.5

MODEL_COLORS: dict[str, str] = {
    "AD": AD_COLOR,
    "AD (no H)": AD_NO_H_COLOR,
    "AD Hessians": AD_COLOR,
    "EQV2": EQV2_FORCE_COLOR,
    "EQV2 AD": EQV2_FORCE_COLOR,
    "EqV2": EQV2_FORCE_COLOR,
    "EqV2 (no H)": EQV2_NO_H_FORCE_COLOR,
    "Autograd": AD_COLOR,
    "DFT": DFT_COLOR,
    "HIP": HIP_COLOR,
    "HIP Hessians": HIP_COLOR,
    "LeftNet-CF": LEFTNET_CF_FORCE_COLOR,
    "LeftNet-CF (no H)": LEFTNET_CF_FORCE_COLOR,
    "LeftNet-DF": LEFTNET_DF_FORCE_COLOR,
    "LeftNet-DF (no H)": LEFTNET_DF_FORCE_COLOR,
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
            "legend.frameon": True,
            "legend.edgecolor": "none",
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
    if label in MODEL_COLORS:
        return MODEL_COLORS[label]
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
        ax.legend(frameon=True, edgecolor="none")


apply_plot_style()


##########################################################
# Old

# SNSPALETTE = sns.color_palette("pastel", 10).as_hex()
# # ['#a1c9f4', '#ffb482', '#8de5a1', '#ff9f9b', '#d0bbff', '#debb9b', '#fab0e4', '#cfcfcf', '#fffea3', '#b9f2f0']


# PLOTLY_DEFAULT_COLOURS = plotly.colors.qualitative.Plotly
# # ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']


# COLOUR_LIST = [
#     "#1b85b8",
#     "#89CFF0",
#     "#68c4af",
#     "#a8e6cf",
#     "#dcedc1",
#     # "#f6cf71",
#     # "#d96002",
#     "#fedd8d",
#     "#ffd3b6",
#     # "#ffa8c6",
#     "#ffbad2",
#     "#ffaaa5",
#     "#ff8b94",
#     # dimmer backup colours
#     "#cfcbc5",
#     "#d6c8e8",
#     "#b8d6ec",
#     "#295c7e",
#     "#444f97",
#     "#b5e2da",
#     "#95b3c0",
#     "#656a95",
#     "#db95a6",
#     "#5a5255",
#     "#559e83",
#     "#ae5a41",
#     "#c3cb71",
# ]

# METHOD_TO_COLOUR = {
#     "alphanet": "#444f97",  # "#ffaaa5",
#     "leftnet": "#68c4af", # "#559e83" #3f7763 #4a8a72
#     "leftnet-df": "#a8e6cf", # "#68c4af" # #8A3F3F #743737

#     "mace": "#cfcbc5",
#     "eqv2": "#89CFF0",  # "#b8d6ec", #89CFF0
#     "hesspred": "#f6cf71",
# }
# # autograd is red
# # HESSIAN_METHOD_TO_COLOUR = {
# #     "predict": "#295c7e",
# #     "autograd": "#db95a6",
# # }
# # HESSIAN_METHOD_TO_COLOUR = {
# #     "predict": "#1b85b8",
# #     "autograd": "#db95a6",
# # }
# # HESSIAN_METHOD_TO_COLOUR = {
# #     "predict": "#295c7e",
# #     "autograd": "#ae5a41",
# # }
# # brighter colours
# # HESSIAN_METHOD_TO_COLOUR = {
# #     "predict": "#68c4af",
# #     "autograd": "#db95a6",
# # }
# # HESSIAN_METHOD_TO_COLOUR = {
# #     "predict": "#ffb482",
# #     "autograd": "#cfcfcf",
# # }
# # our method with signalling colours
# # HESSIAN_METHOD_TO_COLOUR = {
# #     "predict": "#ae5a41",
# #     "autograd": "#295c7e",
# # }
# HESSIAN_METHOD_TO_COLOUR = {
#     "predict": "#ffb482",
#     "prediction_fc": "#68c4af",
#     "autograd": "#295c7e",
#     "ef": "#5a5255",  # #636EFA # #6B4423 #4A2C1A
# }

# HESSIAN_METHOD_TO_COLOUR["prediction"] = HESSIAN_METHOD_TO_COLOUR["predict"]
# HESSIAN_METHOD_TO_COLOUR["learned"] = HESSIAN_METHOD_TO_COLOUR["predict"]

# # Relaxations
# OPTIM_TO_COLOUR = {
#     "firstorder": "#295c7e",
#     "bfgs": "#636EFA",
#     "secondorder": "#db95a6",
# }
# OPTIM_TO_COLOUR["First-Order"] = OPTIM_TO_COLOUR["firstorder"]
# OPTIM_TO_COLOUR["Second-Order"] = OPTIM_TO_COLOUR["secondorder"]
# OPTIM_TO_COLOUR["Quasi-Second-Order"] = OPTIM_TO_COLOUR["bfgs"]
# OPTIM_TO_COLOUR["No Hessian"] = OPTIM_TO_COLOUR["firstorder"]
# OPTIM_TO_COLOUR["No Hessians"] = OPTIM_TO_COLOUR["firstorder"]
# OPTIM_TO_COLOUR["Hessian Free"] = OPTIM_TO_COLOUR["firstorder"]
# OPTIM_TO_COLOUR["Quasi-Hessian"] = OPTIM_TO_COLOUR["bfgs"]
# OPTIM_TO_COLOUR["Hessian"] = OPTIM_TO_COLOUR["secondorder"]


# ANNOTATION_FONT_SIZE = 16
# ANNOTATION_BOLD_FONT_SIZE = 18
# AXES_FONT_SIZE = 14
# AXES_TITLE_FONT_SIZE = 16
# LEGEND_FONT_SIZE = 16
# TITLE_FONT_SIZE = 20
