import argparse
from pathlib import Path

import pandas as pd

try:
    from hip.colours import (
        ANNOTATION_BOLD_FONT_SIZE,
        ANNOTATION_FONT_SIZE,
        AXES_FONT_SIZE,
        AXES_TITLE_FONT_SIZE,
        LEGEND_FONT_SIZE,
        TITLE_FONT_SIZE,
        HESSIAN_METHOD_TO_COLOUR,
    )
except ModuleNotFoundError:
    ANNOTATION_BOLD_FONT_SIZE = 18
    ANNOTATION_FONT_SIZE = 14
    AXES_FONT_SIZE = 12
    AXES_TITLE_FONT_SIZE = 13
    LEGEND_FONT_SIZE = 12
    TITLE_FONT_SIZE = 16
    HESSIAN_METHOD_TO_COLOUR = {
        "autograd": "#1f77b4",
        "prediction": "#d96001",
    }

PLOTLY_TEMPLATE = "plotly_white"
EIGVAL_MAE_COLUMNS = (
    "eckart_eigval_mae_ev_a2",
    "eigval_mae_eckart",
)


def _color_for_method(method):
    method_lower = str(method).lower()
    if method_lower == "prediction":
        return "#d96001"
    if method_lower == "autograd":
        return HESSIAN_METHOD_TO_COLOUR.get("autograd")
    if method_lower == "autograd_conservative":
        return "#9b59b6"
    if method_lower == "forward_pass":
        return "#68c4af"
    if "finite_difference_bz1" in method_lower:
        return "#5a5255"
    if "finite_difference_bz32" in method_lower:
        return "#ff8b94"
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


def _load_eigval_curve(csv_path):
    df = pd.read_csv(csv_path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    metric_col = next((col for col in EIGVAL_MAE_COLUMNS if col in df.columns), None)
    if "natoms" not in df.columns or metric_col is None:
        raise ValueError(
            f"Missing required columns in {csv_path}. Need 'natoms' and one of "
            f"{list(EIGVAL_MAE_COLUMNS)}."
        )
    return df.groupby("natoms")[metric_col].mean().sort_index()


def _load_curves(lambda_results):
    curves = {}
    for label, csv_path in lambda_results.items():
        if not Path(csv_path).exists():
            print(f"Skipping {label}: missing {csv_path}")
            continue
        curves[label] = _load_eigval_curve(csv_path)
    return curves


def _load_speed_tables(speed_csv):
    speed_df = pd.read_csv(speed_csv)
    avg_times = speed_df.groupby(["n_atoms", "method"])["time"].mean().unstack()
    avg_memory = speed_df.groupby(["n_atoms", "method"])["memory"].mean().unstack()

    methods = [
        "autograd",
        "autograd_conservative",
        "finite_difference_bz1",
        "forward_pass",
        "prediction",
    ]
    methods = [m for m in methods if m in avg_times.columns]
    return avg_times[methods], avg_memory[methods]


def make_plot_plotly(
    speed_csv,
    rgd1_lambda_results,
    pubchem_lambda_results,
    output_path,
    ymin_time=0.0,
    ymax_time=3700.0,
    ymin_memory=0.0,
    ymax_memory=2100.0,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    avg_times, avg_memory = _load_speed_tables(speed_csv)
    rgd1_lambda_curves = _load_curves(rgd1_lambda_results)
    pubchem_lambda_curves = _load_curves(pubchem_lambda_results)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Time per Sample (ms)",
            "Peak Memory (MB)",
            "Eigenvalues λ MAE (RGD1)",
            "Eigenvalues λ MAE (PubChem)",
        ),
        horizontal_spacing=0.05,
        vertical_spacing=0.10,
    )

    for method in avg_times.columns:
        color = _color_for_method(method)
        fig.add_trace(
            go.Scatter(
                x=avg_times.index,
                y=avg_times[method],
                mode="lines+markers",
                name=_display_name(method),
                showlegend=True,
                line=dict(color=color),
                marker=dict(color=color),
            ),
            row=1,
            col=1,
        )

    for method in avg_memory.columns:
        color = _color_for_method(method)
        dash_pattern = _dash_for_memory(method)
        if dash_pattern == "":
            mode = "markers"
            line_dict = None
        else:
            mode = "lines+markers"
            line_dict = dict(color=color, dash=dash_pattern)
        fig.add_trace(
            go.Scatter(
                x=avg_memory.index,
                y=avg_memory[method],
                mode=mode,
                name=method,
                showlegend=False,
                line=line_dict,
                marker=dict(color=color),
            ),
            row=1,
            col=2,
        )

    rgd1_label_to_method = {
        "HIP": "prediction",
        "AD": "autograd",
    }
    for label in ["HIP", "AD"]:
        if label not in rgd1_lambda_curves:
            continue
        method = rgd1_label_to_method[label]
        color = _color_for_method(method)
        fig.add_trace(
            go.Scatter(
                x=rgd1_lambda_curves[label].index,
                y=rgd1_lambda_curves[label].values,
                mode="lines+markers",
                name=label,
                showlegend=False,
                line=dict(color=color),
                marker=dict(color=color),
            ),
            row=2,
            col=1,
        )
    # Mark train-size boundary for RGD1 panel
    fig.add_vline(
        x=22,
        line_width=2.5,
        line_dash="dash",
        line_color="gray",
        row=2,
        col=1,
    )
    fig.add_annotation(
        x=16.0,
        y=0.95,
        xref="x3",
        yref="y3 domain",
        text=" Train",
        showarrow=True,
        ax=55,
        ay=0,
        arrowhead=2,
        arrowsize=1.0,
        arrowwidth=2.0,
        arrowcolor="gray",
        font=dict(size=AXES_FONT_SIZE, color="gray"),
    )

    pubchem_label_to_method = {
        "HIP": "prediction",
        "AD": "autograd",
    }
    for label in ["HIP", "AD"]:
        if label not in pubchem_lambda_curves:
            continue
        method = pubchem_label_to_method[label]
        color = _color_for_method(method)
        fig.add_trace(
            go.Scatter(
                x=pubchem_lambda_curves[label].index,
                y=pubchem_lambda_curves[label].values,
                mode="lines+markers",
                name=label,
                showlegend=False,
                line=dict(color=color),
                marker=dict(color=color),
            ),
            row=2,
            col=2,
        )

    fig.update_xaxes(title_text="Number of Atoms", title_standoff=5, row=1, col=1)
    fig.update_xaxes(title_text="Number of Atoms", title_standoff=5, row=1, col=2)
    fig.update_xaxes(title_text="Number of Atoms", title_standoff=5, row=2, col=1)
    fig.update_xaxes(title_text="Number of Atoms", title_standoff=5, row=2, col=2)
    fig.update_yaxes(title_text="", row=1, col=1)
    fig.update_yaxes(title_text="", row=1, col=2)
    fig.update_yaxes(title_text="", row=2, col=1)
    fig.update_yaxes(title_text="", row=2, col=2)

    fig.update_yaxes(range=[ymin_time, ymax_time], autorange=False, row=1, col=1)
    fig.update_yaxes(range=[ymin_memory, ymax_memory], autorange=False, row=1, col=2)
    fig.update_xaxes(range=[4.5, 21.5], autorange=False, row=1, col=1)
    fig.update_xaxes(range=[4.5, 21.5], autorange=False, row=1, col=2)

    rgd1_lambda_n = sorted(
        {
            n
            for series in rgd1_lambda_curves.values()
            for n in series.index.tolist()
        }
    )
    if rgd1_lambda_n:
        fig.update_xaxes(
            range=[9.95, max(rgd1_lambda_n) + 0.5],
            autorange=False,
            row=2,
            col=1,
        )
    pubchem_lambda_n = sorted(
        {
            n
            for series in pubchem_lambda_curves.values()
            for n in series.index.tolist()
        }
    )
    if pubchem_lambda_n:
        fig.update_xaxes(
            range=[min(pubchem_lambda_n) - 0.5, max(pubchem_lambda_n) + 0.5],
            autorange=False,
            row=2,
            col=2,
        )

    fig.update_traces(line=dict(width=3))
    fig.update_xaxes(
        tickfont=dict(size=AXES_FONT_SIZE), title_font=dict(size=AXES_TITLE_FONT_SIZE)
    )
    fig.update_yaxes(
        tickfont=dict(size=AXES_FONT_SIZE), title_font=dict(size=AXES_TITLE_FONT_SIZE)
    )
    fig.update_annotations(font=dict(size=ANNOTATION_FONT_SIZE))
    for ann in fig.layout.annotations:
        ann.update(font=dict(size=TITLE_FONT_SIZE))

    dom1 = fig.layout.xaxis.domain
    dom2 = fig.layout.xaxis2.domain
    dom3 = fig.layout.xaxis3.domain
    dom4 = fig.layout.xaxis4.domain
    y_top = 0.999
    y_bottom = fig.layout.yaxis3.domain[1]
    fig.add_annotation(
        x=dom1[0],
        y=y_top,
        xref="paper",
        yref="paper",
        text="<b>a</b>",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=ANNOTATION_BOLD_FONT_SIZE),
    )
    fig.add_annotation(
        x=dom2[0],
        y=y_top,
        xref="paper",
        yref="paper",
        text="<b>b</b>",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=ANNOTATION_BOLD_FONT_SIZE),
    )
    fig.add_annotation(
        x=dom3[0],
        y=y_bottom,
        xref="paper",
        yref="paper",
        text="<b>c</b>",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=ANNOTATION_BOLD_FONT_SIZE),
    )
    fig.add_annotation(
        x=dom4[0],
        y=y_bottom,
        xref="paper",
        yref="paper",
        text="<b>d</b>",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=ANNOTATION_BOLD_FONT_SIZE),
    )

    legend_x = dom1[0] + 0.005
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        margin=dict(l=10, r=0, b=10, t=30),
        width=800,
        height=760,
        legend=dict(
            x=legend_x,
            y=0.995,
            xanchor="left",
            yanchor="top",
            orientation="v",
            bgcolor="rgba(255,255,255,0.6)",
            font=dict(size=LEGEND_FONT_SIZE - 2),
        ),
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(output_path, width=800, height=760, scale=2)
    print(f"Plot saved to {output_path}")


def make_plot_seaborn(
    speed_csv,
    rgd1_lambda_results,
    pubchem_lambda_results,
    output_path,
    ymin_time=0.0,
    ymax_time=3700.0,
    ymin_memory=0.0,
    ymax_memory=2100.0,
):
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns

        sns.set_theme(style="whitegrid", context="paper")
    except ModuleNotFoundError:
        plt.rcParams.update(
            {
                "axes.grid": True,
                "axes.labelsize": AXES_TITLE_FONT_SIZE,
                "font.size": AXES_FONT_SIZE,
                "legend.fontsize": LEGEND_FONT_SIZE,
                "xtick.labelsize": AXES_FONT_SIZE,
                "ytick.labelsize": AXES_FONT_SIZE,
            }
        )

    avg_times, avg_memory = _load_speed_tables(speed_csv)
    rgd1_lambda_curves = _load_curves(rgd1_lambda_results)
    pubchem_lambda_curves = _load_curves(pubchem_lambda_results)

    fig, axes = plt.subplots(2, 2, figsize=(8, 7.6))
    ax_time, ax_memory, ax_rgd1, ax_pubchem = axes.ravel()

    for method in avg_times.columns:
        color = _color_for_method(method)
        ax_time.plot(
            avg_times.index,
            avg_times[method],
            marker="o",
            linewidth=2,
            markersize=3,
            label=_display_name(method),
            color=color,
        )

    for method in avg_memory.columns:
        color = _color_for_method(method)
        linestyle = ":" if _dash_for_memory(method) == "dot" else "-"
        if method == "prediction":
            ax_memory.plot(
                avg_memory.index,
                avg_memory[method],
                marker="o",
                linestyle="None",
                markersize=3,
                color=color,
            )
        else:
            ax_memory.plot(
                avg_memory.index,
                avg_memory[method],
                marker="o",
                linewidth=2,
                markersize=3,
                linestyle=linestyle,
                color=color,
            )

    label_to_method = {"HIP": "prediction", "AD": "autograd"}
    for label in ["HIP", "AD"]:
        if label in rgd1_lambda_curves:
            color = _color_for_method(label_to_method[label])
            ax_rgd1.plot(
                rgd1_lambda_curves[label].index,
                rgd1_lambda_curves[label].values,
                marker="o",
                linewidth=2,
                markersize=3,
                color=color,
            )
        if label in pubchem_lambda_curves:
            color = _color_for_method(label_to_method[label])
            ax_pubchem.plot(
                pubchem_lambda_curves[label].index,
                pubchem_lambda_curves[label].values,
                marker="o",
                linewidth=2,
                markersize=3,
                color=color,
            )

    ax_rgd1.axvline(22, linewidth=1.5, linestyle="--", color="gray")
    ax_rgd1.annotate(
        "Train",
        xy=(22, 0.95),
        xycoords=("data", "axes fraction"),
        xytext=(-55, 0),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "gray"},
        color="gray",
        va="center",
    )

    titles = (
        "Time per Sample (ms)",
        "Peak Memory (MB)",
        "Eigenvalues λ MAE (RGD1)",
        "Eigenvalues λ MAE (PubChem)",
    )
    for panel_label, ax, title in zip("abcd", axes.ravel(), titles):
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)
        ax.set_xlabel("Number of Atoms")
        ax.text(
            -0.08,
            1.04,
            panel_label,
            transform=ax.transAxes,
            fontsize=ANNOTATION_BOLD_FONT_SIZE,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    ax_time.set_ylim(ymin_time, ymax_time)
    ax_memory.set_ylim(ymin_memory, ymax_memory)
    ax_time.set_xlim(4.5, 21.5)
    ax_memory.set_xlim(4.5, 21.5)
    if rgd1_lambda_curves:
        max_n = max(n for series in rgd1_lambda_curves.values() for n in series.index)
        ax_rgd1.set_xlim(9.95, max_n + 0.5)
    if pubchem_lambda_curves:
        ns = [n for series in pubchem_lambda_curves.values() for n in series.index]
        ax_pubchem.set_xlim(min(ns) - 0.5, max(ns) + 0.5)

    ax_time.legend(loc="upper left", framealpha=0.6)
    fig.tight_layout(pad=0.4)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output_path}")


def make_plot(
    speed_csv,
    rgd1_lambda_results,
    pubchem_lambda_results,
    output_path,
    ymin_time=0.0,
    ymax_time=3700.0,
    ymin_memory=0.0,
    ymax_memory=2100.0,
    backend="auto",
):
    kwargs = dict(
        speed_csv=speed_csv,
        rgd1_lambda_results=rgd1_lambda_results,
        pubchem_lambda_results=pubchem_lambda_results,
        output_path=output_path,
        ymin_time=ymin_time,
        ymax_time=ymax_time,
        ymin_memory=ymin_memory,
        ymax_memory=ymax_memory,
    )
    if backend == "plotly":
        make_plot_plotly(**kwargs)
        return
    if backend == "seaborn":
        make_plot_seaborn(**kwargs)
        return
    if backend != "auto":
        raise ValueError(f"Unknown backend: {backend}")

    try:
        make_plot_plotly(**kwargs)
    except (ImportError, ModuleNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Plotly backend unavailable ({exc}); falling back to seaborn.")
        make_plot_seaborn(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone 2x2 speed/memory/lambda plot")
    parser.add_argument(
        "--speed_csv",
        type=str,
        default="results_speed2/ts1x-val.lmdb_speed_comparison_extended_10_r100.0_rh100.0.csv",
    )
    parser.add_argument(
        "--rgd1_hip_metrics_csv",
        type=str,
        default="results_evalhorm/hesspred_v2_RGD1_predict_metrics.csv",
    )
    parser.add_argument(
        "--rgd1_ad_metrics_csv",
        type=str,
        default="results_evalhorm/eqv2_RGD1_autograd_metrics.csv",
    )
    parser.add_argument(
        "--rgd1_ad_ef_metrics_csv",
        type=str,
        default="results_evalhorm/eqv2_orig_RGD1_autograd_metrics.csv",
    )
    parser.add_argument(
        "--pubchem_hip_metrics_csv",
        type=str,
        default="results_size_eval/hesspred_v2_dft_geometries_predict_metrics.csv",
    )
    parser.add_argument(
        "--pubchem_ad_metrics_csv",
        type=str,
        default="results_size_eval/eqv2_dft_geometries_autograd_metrics.csv",
    )
    parser.add_argument(
        "--pubchem_ad_ef_metrics_csv",
        type=str,
        default="results_size_eval/eqv2_orig_dft_geometries_autograd_metrics.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results_speed2/speed_memory_lambda_scaling.png",
    )
    parser.add_argument("--ymin_time", type=float, default=0.0)
    parser.add_argument("--ymax_time", type=float, default=3700.0)
    parser.add_argument("--ymin_memory", type=float, default=0.0)
    parser.add_argument("--ymax_memory", type=float, default=2100.0)
    parser.add_argument(
        "--backend",
        choices=("auto", "plotly", "seaborn"),
        default="auto",
        help="Plotting backend. 'auto' tries Plotly first, then seaborn/Matplotlib.",
    )
    args = parser.parse_args()

    make_plot(
        speed_csv=args.speed_csv,
        rgd1_lambda_results={
            "HIP": args.rgd1_hip_metrics_csv,
            "AD": args.rgd1_ad_metrics_csv,
            "AD (E-F)": args.rgd1_ad_ef_metrics_csv,
        },
        pubchem_lambda_results={
            "HIP": args.pubchem_hip_metrics_csv,
            "AD": args.pubchem_ad_metrics_csv,
            "AD (E-F)": args.pubchem_ad_ef_metrics_csv,
        },
        output_path=args.output,
        ymin_time=args.ymin_time,
        ymax_time=args.ymax_time,
        ymin_memory=args.ymin_memory,
        ymax_memory=args.ymax_memory,
        backend=args.backend,
    )
