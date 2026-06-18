from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402
import seaborn as sns  # noqa: E402

from plot_style import (
    ANNOTATION_BOLD_FONT_SIZE,
    HESSIAN_METHOD_TO_COLOUR,
    model_color,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "reactbench_relaxation"
DEFAULT_RELAXATION_CSV = (
    DEFAULT_DATA_DIR / "relaxation_results_noiserms0.035.csv"
)
DEFAULT_REACTBENCH_CSV = DEFAULT_DATA_DIR / "reactbench.csv"
DEFAULT_ZPE_CSV = DEFAULT_DATA_DIR / "zpe_classification.csv"
DEFAULT_OUTPUT = (
    ROOT_DIR
    / "plots"
    / "reactbench_relaxation"
    / "relaxation_reactbench_zpe.png"
)

ZPE_MODEL_ORDER = [
    "AlphaNet",
    "LeftNet-CF",
    "LeftNet-DF",
    "EquiformerV2",
    "HIP",
]
ZPE_BOLD_MODELS = {"HIP"}
ZPE_DISPLAY_NAMES = {
    "EquiformerV2": "EqV2",
}

AD_COLOR = "#5e859e"
HIP_COLOR = "#d96001"
AD_NO_H_COLOR = "#837d80"
PLOT_FONT_COLOR = "#2F4565"
LINE_WIDTH = 2.2
MARKER_SIZE = 5.5
PANEL_LABEL_SIZE = ANNOTATION_BOLD_FONT_SIZE
SUBPLOT_TITLE_SIZE = 15
BAR_LABEL_SIZE = 8
XTICK_LABEL_SIZE = 12
PLOT_LEGEND_SIZE = 11
PANEL_AC_LEGEND_SIZE = PLOT_LEGEND_SIZE + 1

LEGEND_CATEGORY_LABELS = {
    "AD Hessians": "AD",
    "FD Hessians": "FD",
    "Hessian Approximation": "No H",
    "HIP Hessians": "HIP",
}

METRIC_TO_LABEL = {
    "steps": "Steps to Convergence",
    "wall_time_s": "Wall Time [s]",
}

CATEGORY_TO_COLOUR = {
    "Energy-Force": HESSIAN_METHOD_TO_COLOUR["ef"],
    "Hessian Approximation": HESSIAN_METHOD_TO_COLOUR["hessian_approx"],
    "AD Hessians": HESSIAN_METHOD_TO_COLOUR["autograd"],
    "FD Hessians": HESSIAN_METHOD_TO_COLOUR["finite_difference_bz1"],
    "HIP Hessians": HESSIAN_METHOD_TO_COLOUR["prediction"],
}

METHOD_TO_CATEGORY = {
    "NaiveDescent": "Energy-Force",
    "Descent": "Energy-Force",
    "FIRE": "Hessian Approximation",
    "ConjugateGradient": "Hessian Approximation",
    "RFO-BFGS (unit init)": "Hessian Approximation",
    "RFO-BFGS (DFT init)": "Hessian Approximation",
    "RFO-BFGS (autograd init)": "AD Hessians",
    "RFO-BFGS (NumHess init)": "FD Hessians",
    "RFO-BFGS (learned init)": "HIP Hessians",
    "RFO-BFGS (learned k3)": "HIP Hessians",
    "RFO (NumHess)": "FD Hessians",
    "RFO (NumHess 4)": "FD Hessians",
    "RFO (autograd)": "AD Hessians",
    "RFO (learned)": "HIP Hessians",
}
METHOD_TO_CATEGORY["RFO-BFGS"] = "Hessian Approximation"

METHOD_TO_COLOUR = {
    method: CATEGORY_TO_COLOUR[category]
    for method, category in METHOD_TO_CATEGORY.items()
}
METHOD_TO_COLOUR["RFO-BFGS"] = CATEGORY_TO_COLOUR["Hessian Approximation"]

DO_METHOD = [
    "NaiveDescent",
    "FIRE",
    "RFO (autograd)",
    "RFO (NumHess)",
    "RFO-BFGS (unit init)",
    "RFO-BFGS (autograd init)",
    "RFO-BFGS (NumHess init)",
    "RFO (learned)",
    "RFO-BFGS (learned init)",
]

COMPETITIVE_METHODS_WALL_TIME = [
    "FIRE",
    "ConjugateGradient",
    "RFO-BFGS (unit init)",
    "RFO-BFGS",
    "RFO-BFGS (NumHess init)",
    "RFO-BFGS (learned init)",
    "RFO-BFGS (learned k3)",
    "RFO-BFGS (autograd init)",
    "RFO (learned)",
]

WALL_TIME_ANNOTATION_ONLY = [
    "Descent",
    "RFO (autograd)",
    "RFO (NumHess)",
]

PANEL_METHOD_ORDER = [
    "RFO (NumHess)",
    "RFO (autograd)",
    "FIRE",
    "RFO-BFGS",
    "RFO (learned)",
]

RENAME_METHODS_PLOT = {
    "NaiveDescent": "Descent",
    "RFO-BFGS (unit init)": "RFO-BFGS",
    "RFO-BFGS (NumHess init)": "RFO-BFGS (FD init)",
    "RFO (NumHess)": "RFO FD",
}


def _literal_dict(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, dict) else {}
    return {}


def apply_plot_style() -> None:
    sns.set_theme(
        context="talk",
        style="whitegrid",
        palette=[AD_COLOR, HIP_COLOR, AD_NO_H_COLOR],
        rc={
            "axes.edgecolor": "#E6E6E6",
            "axes.labelcolor": PLOT_FONT_COLOR,
            "axes.labelsize": 16,
            "axes.linewidth": 1.1,
            "axes.titlecolor": PLOT_FONT_COLOR,
            "axes.titlesize": SUBPLOT_TITLE_SIZE,
            "figure.facecolor": "white",
            "font.family": "sans-serif",
            "grid.color": "#E9E9E9",
            "grid.linewidth": 1.0,
            "legend.edgecolor": "none",
            "legend.fontsize": PLOT_LEGEND_SIZE,
            "legend.frameon": True,
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,
            "savefig.facecolor": "white",
            "xtick.color": PLOT_FONT_COLOR,
            "xtick.labelsize": XTICK_LABEL_SIZE,
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
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, color="#E9E9E9", linewidth=1.0)
    if ax.get_yscale() == "log":
        ax.minorticks_on()
        ax.yaxis.grid(True, which="minor", color="#F1F1F1", linewidth=0.7, alpha=0.85)
    sns.despine(ax=ax, trim=False)
    if legend:
        ax.legend(frameon=True, edgecolor="none")


def add_group_title(
    fig: plt.Figure,
    axes: list[plt.Axes],
    title: str,
    *,
    pad: float = 0.012,
) -> None:
    positions = [ax.get_position() for ax in axes]
    left = min(position.x0 for position in positions)
    right = max(position.x1 for position in positions)
    top = max(position.y1 for position in positions)
    fig.text(
        0.5 * (left + right),
        top + pad,
        title,
        ha="center",
        va="bottom",
        fontsize=SUBPLOT_TITLE_SIZE,
        color=PLOT_FONT_COLOR,
    )


def add_panel_labels(
    fig: plt.Figure,
    axes: list[plt.Axes],
    *,
    labels: str | tuple[str, ...] = "abcdef",
) -> None:
    label_list = list(labels) if not isinstance(labels, str) else list(labels)
    for panel_label, ax in zip(label_list, axes, strict=True):
        if not panel_label:
            continue
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


def _series_for_method(df: pd.DataFrame, method: str, metric: str) -> pd.Series:
    return df.loc[df["name"] == method, metric].dropna()


def _prepare_order(df: pd.DataFrame, metric: str) -> list[str]:
    if df.empty:
        return []
    methods = set(df.dropna(subset=[metric])["name"].unique())
    return [method for method in PANEL_METHOD_ORDER if method in methods]


def _display_name(method: str) -> str:
    if method == "RFO (learned)":
        return "RFO HIP"
    if method == "FIRE":
        return "FIRE"
    if method == "RFO-BFGS (learned init)":
        return "RFO-BFGS (HIP init)"
    if method == "RFO-BFGS (autograd init)":
        return "RFO-BFGS (AD init)"
    if method == "RFO (autograd)":
        return "RFO AD"
    return RENAME_METHODS_PLOT.get(method, method)


def _load_reactbench(reactbench_csv: Path) -> pd.DataFrame:
    rb_rename_metrics = {
        "gsm_success": "GSM Success",
        "converged_ts": "RFO Converged",
        "ts_success": "TS Success",
        "convert_ts": "RFO Converged and TS Success",
        "irc_success": "IRC Success",
        "intended_count": "IRC Verified",
    }
    rb_extra_metrics = {
        "predict": {
            "one negative eigenvalue and force RMS < 2.0e-03 Ha/Bohr:": {
                "correct_proposed_estimated": 642.24,
            },
            "one negative eigenvalue:": {
                "correct_proposed_estimated": 669.0,
            },
        },
        "autograd": {
            "one negative eigenvalue and force RMS < 2.0e-03 Ha/Bohr:": {
                "correct_proposed_estimated": 571.48,
            },
            "one negative eigenvalue:": {
                "correct_proposed_estimated": 621.72,
            },
        },
    }
    rb_allowed_metrics = [
        "GSM Success",
        "RFO Converged",
        "IRC Verified",
        "TS (DFT)",
        "+ Converged",
    ]

    runs_df = pd.read_csv(reactbench_csv, quotechar='"')
    records = []
    for _, row in runs_df.iterrows():
        cfg = _literal_dict(row.get("config", {}))
        summary = _literal_dict(row.get("summary", {}))
        base_method = str(cfg.get("hessian_method", "unknown"))
        calc = cfg.get("calc")
        calc_str = None if calc is None else str(calc)
        if calc_str and calc_str.lower() not in ["none", "nan", "na", ""]:
            method_label = f"{base_method}-{calc_str}"
        else:
            method_label = base_method
        for source_metric, display_metric in rb_rename_metrics.items():
            value = summary.get(source_metric)
            if value is not None:
                records.append(
                    {
                        "Metric": display_metric,
                        "Value": value,
                        "Method": method_label,
                    }
                )

    df_rb = pd.DataFrame.from_records(records)
    rb_method_key_map = {
        "predict": "predict-equiformer",
        "autograd": "autograd-equiformer",
    }
    extra_records = []
    for source_key, method_label in rb_method_key_map.items():
        metric_block = rb_extra_metrics.get(source_key, {})
        ts_block = metric_block.get("one negative eigenvalue:")
        conv_block = metric_block.get(
            "one negative eigenvalue and force RMS < 2.0e-03 Ha/Bohr:"
        )
        if ts_block is not None:
            extra_records.append(
                {
                    "Metric": "TS (DFT)",
                    "Value": ts_block["correct_proposed_estimated"],
                    "Method": method_label,
                }
            )
        if conv_block is not None:
            extra_records.append(
                {
                    "Metric": "+ Converged",
                    "Value": conv_block["correct_proposed_estimated"],
                    "Method": method_label,
                }
            )
    if extra_records:
        df_rb = pd.concat(
            [df_rb, pd.DataFrame.from_records(extra_records)],
            ignore_index=True,
        )

    df_rb = df_rb[df_rb["Metric"].isin(rb_allowed_metrics)]
    df_rb = df_rb[df_rb["Method"].isin(["predict-equiformer", "autograd-equiformer"])].copy()
    df_rb["Metric"] = pd.Categorical(
        df_rb["Metric"],
        categories=rb_allowed_metrics,
        ordered=True,
    )
    gsm_mask = df_rb["Metric"] == "GSM Success"
    gsm_values = df_rb[gsm_mask].groupby("Method", observed=False)["Value"].first()
    if len(gsm_values) >= 2:
        df_rb.loc[gsm_mask, "Value"] = gsm_values.mean()
    return df_rb


def _ordered_display_names(
    df: pd.DataFrame,
    methods: list[str],
    metric: str,
    *,
    include_annotation_methods: bool = False,
) -> list[str]:
    display_order = []
    for method in methods:
        series = _series_for_method(df, method, metric)
        has_data = not series.empty
        has_annotation = (
            include_annotation_methods
            and method in WALL_TIME_ANNOTATION_ONLY
            and has_data
        )
        if has_data or has_annotation:
            display_order.append(_display_name(method))
    return display_order


def _format_categorical_xaxis(
    ax: plt.Axes,
    *,
    bold_targets: set[str] | None = None,
) -> None:
    bold_targets = set() if bold_targets is None else bold_targets
    for label in ax.get_xticklabels():
        label.set_rotation(35)
        label.set_horizontalalignment("right")
        label.set_verticalalignment("top")
        label.set_rotation_mode("anchor")
        label.set_clip_on(False)
        label.set_color(PLOT_FONT_COLOR)
        if label.get_text() in bold_targets:
            label.set_fontweight("bold")
    ax.tick_params(axis="both", which="both", length=0)
    ax.tick_params(axis="x", pad=2)


def _distribution_plot_data(
    df: pd.DataFrame,
    metric: str,
    order: list[str],
) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    records = []
    palette = {}
    methods_plotted = []
    for method in order:
        series = _series_for_method(df, method, metric)
        if series.empty:
            continue
        display_name = _display_name(method)
        methods_plotted.append(method)
        palette[display_name] = METHOD_TO_COLOUR[method]
        for value in series.astype(float):
            records.append({"Method": display_name, "Value": value})
    return pd.DataFrame.from_records(records), palette, methods_plotted


def _plot_distribution_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    order: list[str],
    *,
    title: str | None,
    max_cycles: int,
    display_order: list[str] | None = None,
    show_stripplot: bool = True,
    show_nonconverged_markers: bool = True,
) -> list[str]:
    plot_df, palette, methods_plotted = _distribution_plot_data(df, metric, order)
    display_order = (
        _ordered_display_names(df, order, metric)
        if display_order is None
        else display_order
    )
    for method in order:
        display_name = _display_name(method)
        if display_name in display_order:
            palette.setdefault(display_name, METHOD_TO_COLOUR[method])
    if not plot_df.empty:
        sns.violinplot(
            data=plot_df,
            x="Method",
            y="Value",
            hue="Method",
            order=display_order,
            hue_order=display_order,
            palette=palette,
            inner=None,
            cut=0,
            linewidth=1.0,
            width=0.9,
            saturation=1.0,
            dodge=False,
            legend=False,
            ax=ax,
        )
        for collection in ax.collections:
            collection.set_alpha(0.14)
        if show_stripplot:
            sns.stripplot(
                data=plot_df,
                x="Method",
                y="Value",
                hue="Method",
                order=display_order,
                hue_order=display_order,
                palette=palette,
                jitter=0.28,
                size=4,
                alpha=0.3,
                linewidth=0,
                dodge=False,
                legend=False,
                ax=ax,
            )
    if show_nonconverged_markers:
        for method in methods_plotted:
            method_rows = df[df["name"] == method]
            hit_max_mask = method_rows["steps"].isin([max_cycles, max_cycles - 1])
            series_noconv = method_rows.loc[hit_max_mask, metric].dropna()
            if series_noconv.empty:
                continue
            x_pos = display_order.index(_display_name(method))
            ax.scatter(
                [x_pos] * len(series_noconv),
                series_noconv.astype(float),
                marker="x",
                s=85,
                linewidths=2.0,
                color=METHOD_TO_COLOUR[method],
                zorder=5,
            )

    bold_targets = {
        _display_name(method)
        for method in ("RFO (learned)", "RFO-BFGS (learned init)")
        if method in methods_plotted
    }
    if title:
        ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(METRIC_TO_LABEL[metric])
    _format_categorical_xaxis(ax, bold_targets=bold_targets)
    finish_axis(ax)
    return methods_plotted


def _add_wall_time_annotations(
    ax: plt.Axes,
    df_wall: pd.DataFrame,
    display_order: list[str],
) -> None:
    annotation_y = 4.38
    arrow_y = 4.0
    for method in WALL_TIME_ANNOTATION_ONLY:
        series = _series_for_method(df_wall, method, "wall_time_s")
        if series.empty:
            continue
        display_name = _display_name(method)
        if display_name not in display_order:
            continue
        color = METHOD_TO_COLOUR[method]
        x_pos = display_order.index(display_name)
        mean_value = float(series.mean())
        ax.annotate(
            f"{mean_value:.0f}s",
            xy=(x_pos, arrow_y),
            xytext=(x_pos, annotation_y),
            textcoords="data",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=color,
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "linewidth": 2.0,
                "mutation_scale": 10,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )


def _plot_reactbench_panel(ax: plt.Axes, df_rb: pd.DataFrame) -> None:
    rb_allowed_metrics = [
        "GSM Success",
        "RFO Converged",
        "IRC Verified",
        "TS (DFT)",
        "+ Converged",
    ]
    rb_display_names = {
        "predict-equiformer": "HIP",
        "autograd-equiformer": "AD",
    }
    rb_hue_order = ["HIP", "AD"]
    plot_df = df_rb.copy()
    plot_df["Hessian Method"] = plot_df["Method"].map(rb_display_names)
    palette = {
        "HIP": HESSIAN_METHOD_TO_COLOUR["predict"],
        "AD": HESSIAN_METHOD_TO_COLOUR["autograd"],
    }
    sns.barplot(
        data=plot_df,
        x="Metric",
        y="Value",
        hue="Hessian Method",
        order=rb_allowed_metrics,
        hue_order=rb_hue_order,
        palette=palette,
        saturation=1.0,
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(
            container,
            labels=[
                f"{bar.get_height():.0f}" if bar.get_height() else ""
                for bar in container
            ],
            padding=1,
            fontsize=BAR_LABEL_SIZE,
            color=PLOT_FONT_COLOR,
        )
    ax.set_title("TS Search (ReactBench)")
    ax.set_xlabel("")
    ax.set_ylabel("Success Count")
    ax.set_ylim(498.5, 920)
    _format_categorical_xaxis(ax)
    for label in ax.get_xticklabels():
        label.set_fontsize(XTICK_LABEL_SIZE - 2.5)
    finish_axis(ax)
    legend_obj = ax.get_legend()
    if legend_obj is not None:
        legend_obj.set_title(None)
        for text in legend_obj.get_texts():
            text.set_fontsize(PANEL_AC_LEGEND_SIZE)
        legend_obj.get_frame().set_alpha(0.75)
        legend_obj.get_frame().set_linewidth(0)


def _load_zpe_metrics(zpe_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(zpe_csv)
    required = {
        "model",
        "zpe_mae",
        "zpe_mae_std",
        "delta_zpe_mae",
        "delta_zpe_mae_std",
        "classification",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ZPE CSV is missing required columns: {sorted(missing)}")
    df = df.copy()
    df["model"] = df["model"].replace({"HIP-EquiformerV2": "HIP"})
    df["model"] = pd.Categorical(
        df["model"],
        categories=ZPE_MODEL_ORDER,
        ordered=True,
    )
    return df.sort_values("model").reset_index(drop=True)


def _zpe_model_palette(models: list[str]) -> dict[str, str]:
    return {
        model: (
            HESSIAN_METHOD_TO_COLOUR["alphanet"]
            if model == "AlphaNet"
            else model_color(model, HESSIAN_METHOD_TO_COLOUR["autograd"])
        )
        for model in models
    }


def _zpe_display_name(model: str) -> str:
    return ZPE_DISPLAY_NAMES.get(model, model)


def _format_zpe_bar_value(value: float) -> str:
    return f"{value:.4f}"


def _plot_zpe_metric_panel(
    ax: plt.Axes,
    df_zpe: pd.DataFrame,
    *,
    metric: str,
    title: str | None,
    ylabel: str,
    log_y: bool,
    show_bar_labels: bool = True,
) -> None:
    models = [model for model in ZPE_MODEL_ORDER if model in set(df_zpe["model"])]
    plot_df = df_zpe[df_zpe["model"].isin(models)].copy()
    palette = _zpe_model_palette(models)
    x = list(range(len(models)))
    values = plot_df.set_index("model").loc[models, metric].astype(float).to_numpy()
    colors = [palette[model] for model in models]
    bars = ax.bar(x, values, color=colors, width=0.72, edgecolor="none", zorder=2)
    if show_bar_labels:
        for bar, model, value in zip(bars, models, values, strict=True):
            if value <= 0:
                continue
            label_y = value * 1.18 if log_y else value * 1.04
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                _format_zpe_bar_value(float(value)),
                ha="center",
                va="bottom",
                fontsize=BAR_LABEL_SIZE,
                color=PLOT_FONT_COLOR,
                fontweight="bold" if model in ZPE_BOLD_MODELS else "normal",
            )
    ax.set_xticks(x)
    ax.set_xticklabels([_zpe_display_name(model) for model in models])
    if title:
        ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
        positive = values[values > 0]
        if len(positive) > 0:
            ymin = max(positive.min() * 0.35, 1e-4)
            ymax = max(values) * 3.2
            ax.set_ylim(ymin, ymax)
    _format_categorical_xaxis(ax, bold_targets=ZPE_BOLD_MODELS)
    finish_axis(ax)


def _plot_classification_panel(ax: plt.Axes, df_zpe: pd.DataFrame) -> None:
    models = [model for model in ZPE_MODEL_ORDER if model in set(df_zpe["model"])]
    plot_df = df_zpe[df_zpe["model"].isin(models)].copy()
    palette = _zpe_model_palette(models)
    x = list(range(len(models)))
    values = (
        100.0
        * plot_df.set_index("model").loc[models, "classification"].astype(float).to_numpy()
    )
    colors = [palette[model] for model in models]
    bars = ax.bar(x, values, color=colors, width=0.72, edgecolor="none", zorder=2)
    for bar, model, value in zip(bars, models, values, strict=True):
        if value <= 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.0,
            f"{value:.0f}%",
            ha="center",
            va="bottom",
            fontsize=BAR_LABEL_SIZE,
            color=PLOT_FONT_COLOR,
            fontweight="bold" if model in ZPE_BOLD_MODELS else "normal",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([_zpe_display_name(model) for model in models])
    ax.set_title("Stationary Point Class")
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy [%]")
    ax.set_ylim(50, 102)
    _format_categorical_xaxis(ax, bold_targets=ZPE_BOLD_MODELS)
    finish_axis(ax)


def plot_relaxation_reactbench_zpe(
    relaxation_csv: Path,
    reactbench_csv: Path,
    zpe_csv: Path,
    output_path: Path,
    max_cycles: int,
) -> None:
    apply_plot_style()
    df = pd.read_csv(relaxation_csv)
    df = df[df["name"].isin(DO_METHOD)].copy()
    df.loc[df["name"] == "RFO-BFGS (DFT init)", "wall_time_s"] += 5 * 60
    df.loc[df["name"] == "RFO-BFGS (unit init)", "name"] = "RFO-BFGS"
    df = df[~df["name"].str.contains("init", case=False)].copy()
    df = df.sort_values(by="steps", ascending=False)

    df_steps = df.dropna(subset=["steps"]).copy()
    df_wall = df.dropna(subset=["wall_time_s"]).copy()
    df_wall_comp = df_wall[df_wall["name"].isin(COMPETITIVE_METHODS_WALL_TIME)].copy()
    order_steps = _prepare_order(df_steps, "steps")
    wall_display_order = _ordered_display_names(
        df_wall,
        PANEL_METHOD_ORDER,
        "wall_time_s",
        include_annotation_methods=True,
    )
    df_rb = _load_reactbench(reactbench_csv)
    df_zpe = _load_zpe_metrics(zpe_csv)

    fig, axes_grid = plt.subplots(2, 3, figsize=(12, 7.2))
    axes = list(axes_grid.ravel())
    categories_all: list[str] = []
    for method in _plot_distribution_panel(
        axes[0],
        df_steps,
        "steps",
        order_steps,
        title=None,
        max_cycles=max_cycles,
    ):
        category = METHOD_TO_CATEGORY.get(method)
        if category is not None and category not in categories_all:
            categories_all.append(category)

    for method in _plot_distribution_panel(
        axes[1],
        df_wall_comp,
        "wall_time_s",
        PANEL_METHOD_ORDER,
        title=None,
        max_cycles=max_cycles,
        display_order=wall_display_order,
        show_nonconverged_markers=False,
    ):
        category = METHOD_TO_CATEGORY.get(method)
        if category is not None and category not in categories_all:
            categories_all.append(category)

    _format_categorical_xaxis(
        axes[1],
        bold_targets={_display_name("RFO (learned)")},
    )
    _add_wall_time_annotations(axes[1], df_wall, wall_display_order)

    category_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CATEGORY_TO_COLOUR[category],
            markeredgecolor=CATEGORY_TO_COLOUR[category],
            markersize=8,
            label=LEGEND_CATEGORY_LABELS.get(category, category),
        )
        for category in categories_all
    ]
    if category_handles:
        legend = axes[0].legend(
            handles=category_handles,
            frameon=True,
            edgecolor="none",
            loc="upper right",
            fontsize=PANEL_AC_LEGEND_SIZE,
        )
        legend.get_frame().set_alpha(0.75)
        legend.get_frame().set_linewidth(0)

    _plot_reactbench_panel(axes[2], df_rb)
    _plot_zpe_metric_panel(
        axes[3],
        df_zpe,
        metric="zpe_mae",
        title=None,
        ylabel=r"ZPE MAE [eV]",
        log_y=True,
    )
    _plot_zpe_metric_panel(
        axes[4],
        df_zpe,
        metric="delta_zpe_mae",
        title=None,
        ylabel=r"$\Delta$ZPE MAE [eV]",
        log_y=True,
    )
    _plot_classification_panel(axes[5], df_zpe)
    axes[0].set_ylim(0, 155)
    axes[1].set_ylim(0, 4.9)
    add_panel_labels(fig, axes, labels=("a", "", "b", "c", "", "d"))
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.16, top=0.92, wspace=0.24, hspace=0.46)
    add_group_title(fig, [axes[0], axes[1]], "Geometry Optimization")
    add_group_title(fig, [axes[3], axes[4]], "Zero-Point Energy")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(f"Saved {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--relaxation-csv",
        type=Path,
        default=DEFAULT_RELAXATION_CSV,
        help="CSV with relaxation metrics for the steps and wall-time panels.",
    )
    parser.add_argument(
        "--reactbench-csv",
        type=Path,
        default=DEFAULT_REACTBENCH_CSV,
        help="CSV with ReactBench run summaries for the TS-search panel.",
    )
    parser.add_argument(
        "--zpe-csv",
        type=Path,
        default=DEFAULT_ZPE_CSV,
        help="CSV with ZPE MAE, delta ZPE MAE, and classification metrics.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output image path.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=150,
        help="Maximum optimization cycles used to mark non-converged runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_relaxation_reactbench_zpe(
        relaxation_csv=args.relaxation_csv,
        reactbench_csv=args.reactbench_csv,
        zpe_csv=args.zpe_csv,
        output_path=args.output,
        max_cycles=args.max_cycles,
    )


if __name__ == "__main__":
    main()
