"""Shared Seaborn styling for repository plotting scripts."""
from __future__ import annotations

from collections.abc import Mapping

import matplotlib
import seaborn as sns
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter
# import plotly.colors

AD_COLOR = "#96b0c0" # #96b0c0 #8a9da8 "#5e859e" "#295c7e" # Plotly: "#2F6B8F"
HIP_COLOR = "#d96001" # Plotly: "#E76F00" Old: "#ffb482"
AD_NO_H_COLOR = "#837d80" # "#5a5255" 
DFT_COLOR = "#2B2B2B"
GUIDE_COLOR = "#8A8A8A"
ACCENT_COLOR = "#B75DAE"
SUCCESS_COLOR = "#5AA469"
EQV2_FORCE_COLOR = AD_COLOR
FORWARD_PASS_COLOR = "#8ed3c3" # #68c4af
# FD Hessians / finite difference bz1: #5a5255
# Finite difference bz32, if present: #ff8b94
# Fallback/unknown method: #cfcfcf
# Annotation/train line uses named gray (#808080)
EQV2_NO_H_FORCE_COLOR = AD_NO_H_COLOR
HIP_FORCE_COLOR = HIP_COLOR
LEFTNET_CF_FORCE_COLOR = "#8b9f98" # #89a198 #8b9f98 #8cb5a5 #7cab99 #9bbfb1 #659C87 "#4a8a72"
LEFTNET_DF_FORCE_COLOR = "#B98C8C" # #9C5C5C #B98C8C #A76F6F "#8A3F3F" # #8A3F3F #743737
PLOTLY_FONT_COLOR = "#2F4565" # #2a3f5f

#0A0A0A`
#26302E
#2C3A37
#374945
#3E4B48
#445754
#4C5754
#5E6864
#68716E
#6C6E6F
#7D8682
#8C8B86
#A6ADA9
#B9C0BC
#C3C7C2
#CDD2CD
#CFD3CE
#D6D6D6
#E2DFD9
#E9E9E9
#F5F5F5
#FCFCFB


LINE_WIDTH = 2.2
THIN_LINE_WIDTH = 1.6
GUIDE_LINE_WIDTH = 1.4
MARKER_SIZE = 5.5
SMALL_MARKER_SIZE = 4.5

ANNOTATION_BOLD_FONT_SIZE = 18
ANNOTATION_FONT_SIZE = 14
AXES_FONT_SIZE = 12
AXES_TITLE_FONT_SIZE = 13
LEGEND_FONT_SIZE = 12
TITLE_FONT_SIZE = 16

PLOT_FONT_FAMILY = ("Open Sans", "Arial", "Helvetica", "DejaVu Sans", "sans-serif")

HESSIAN_METHOD_TO_COLOUR = {
    "autograd": AD_COLOR,  # Alternate: "#a1c9f4"
    "autograd_conservative": "#cfcfcf", # "#b482c8",
    "forward_pass": FORWARD_PASS_COLOR,
    "finite_difference_bz1": "#aea9ab",
    "finite_difference_bz32": "#ffa8af",
    "prediction": HIP_COLOR,
    "ef": AD_NO_H_COLOR,
    "hessian_approx": "#4a8a72",
    "leftnet_df": LEFTNET_CF_FORCE_COLOR,
    "leftnet_cf": LEFTNET_DF_FORCE_COLOR,
    "alphanet": "#cfcfcf",
}
HESSIAN_METHOD_TO_COLOUR["predict"] = HESSIAN_METHOD_TO_COLOUR["prediction"]
HESSIAN_METHOD_TO_COLOUR["learned"] = HESSIAN_METHOD_TO_COLOUR["prediction"]
HESSIAN_METHOD_TO_COLOUR["hip"] = HESSIAN_METHOD_TO_COLOUR["prediction"]

OPTIM_TO_COLOUR = {
    "firstorder": "#295c7e",
    "bfgs": "#636EFA",
    "secondorder": "#db95a6",
}
OPTIM_TO_COLOUR["First-Order"] = OPTIM_TO_COLOUR["firstorder"]
OPTIM_TO_COLOUR["Second-Order"] = OPTIM_TO_COLOUR["secondorder"]
OPTIM_TO_COLOUR["Quasi-Second-Order"] = OPTIM_TO_COLOUR["bfgs"]
OPTIM_TO_COLOUR["No Hessian"] = OPTIM_TO_COLOUR["firstorder"]
OPTIM_TO_COLOUR["No Hessians"] = OPTIM_TO_COLOUR["firstorder"]
OPTIM_TO_COLOUR["Hessian Free"] = OPTIM_TO_COLOUR["firstorder"]
OPTIM_TO_COLOUR["Quasi-Hessian"] = OPTIM_TO_COLOUR["bfgs"]
OPTIM_TO_COLOUR["Hessian"] = OPTIM_TO_COLOUR["secondorder"]

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
            "axes.labelcolor": PLOTLY_FONT_COLOR,
            "axes.labelsize": 17,
            "axes.linewidth": 1.1,
            "axes.titlecolor": PLOTLY_FONT_COLOR,
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
            "xtick.color": PLOTLY_FONT_COLOR,
            "xtick.labelsize": 14,
            "ytick.color": PLOTLY_FONT_COLOR,
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


def configure_log_axes(ax: matplotlib.axes.Axes) -> None:
    for axis_name in ("x", "y"):
        scale = ax.get_xscale() if axis_name == "x" else ax.get_yscale()
        if scale != "log":
            continue
        axis = getattr(ax, f"{axis_name}axis")
        axis.set_major_locator(LogLocator(base=10))
        axis.set_major_formatter(LogFormatterMathtext(base=10, labelOnlyBase=True))
        axis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
        axis.set_minor_formatter(NullFormatter())


def apply_invisible_ticks(ax: matplotlib.axes.Axes) -> None:
    ax.tick_params(
        axis="both",
        which="both",
        length=0,
        width=0.5,
        labelsize=matplotlib.rcParams["xtick.labelsize"],
    )


def finish_axis(ax: matplotlib.axes.Axes, *, legend: bool = False) -> None:
    configure_log_axes(ax)
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


