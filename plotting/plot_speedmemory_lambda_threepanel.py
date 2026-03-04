import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hip.colours import (
    ANNOTATION_BOLD_FONT_SIZE,
    ANNOTATION_FONT_SIZE,
    AXES_FONT_SIZE,
    AXES_TITLE_FONT_SIZE,
    LEGEND_FONT_SIZE,
    TITLE_FONT_SIZE,
    HESSIAN_METHOD_TO_COLOUR,
)

PLOTLY_TEMPLATE = "plotly_white"


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


def _load_eigval_curve(csv_path):
    df = pd.read_csv(csv_path)
    required_cols = {"natoms", "eigval_mae_eckart"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Missing required columns in {csv_path}. Need {sorted(required_cols)}."
        )
    return df.groupby("natoms")["eigval_mae_eckart"].mean().sort_index()


def make_plot(
    speed_csv,
    pubchem_lambda_results,
    output_path,
    ymin_time=0.0,
    ymax_time=3700.0,
    ymin_memory=0.0,
    ymax_memory=2100.0,
):
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
    avg_times = avg_times[methods]
    avg_memory = avg_memory[methods]

    pubchem_lambda_curves = {}
    for label, csv_path in pubchem_lambda_results.items():
        if not Path(csv_path).exists():
            print(f"Skipping {label}: missing {csv_path}")
            continue
        pubchem_lambda_curves[label] = _load_eigval_curve(csv_path)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Time per Sample (ms)",
            "Peak Memory (MB)",
            "Eigenvalue λ MAE (PubChem)",
        ),
        horizontal_spacing=0.03,
    )

    for method in avg_times.columns:
        color = _color_for_method(method)
        if method == "prediction":
            name = "HIP Hessians (ours)"
        elif method == "autograd":
            name = "AD Hessians (direct force)"
        elif method == "autograd_conservative":
            name = "AD Hessians (conservative)"
        elif method == "forward_pass":
            name = "Forward Pass"
        elif "finite_difference_bz1" in method:
            name = "FD Hessians"
        else:
            name = method
        fig.add_trace(
            go.Scatter(
                x=avg_times.index,
                y=avg_times[method],
                mode="lines+markers",
                name=name,
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

    pubchem_label_to_method = {
        "HIP": "prediction",
        "AD": "autograd",
    }
    for label in ["HIP", "AD"]:
        if label not in pubchem_lambda_curves:
            continue
        method = pubchem_label_to_method[label]
        color = _color_for_method(method)
        display_name = "HIP Hessians (ours)" if label == "HIP" else "AD Hessians (direct force)"
        fig.add_trace(
            go.Scatter(
                x=pubchem_lambda_curves[label].index,
                y=pubchem_lambda_curves[label].values,
                mode="lines+markers",
                name=display_name,
                showlegend=False,
                line=dict(color=color),
                marker=dict(color=color),
            ),
            row=1,
            col=3,
        )

    fig.update_xaxes(title_text="Number of Atoms", title_standoff=5, row=1, col=1)
    fig.update_xaxes(title_text="Number of Atoms", title_standoff=5, row=1, col=2)
    fig.update_xaxes(title_text="Number of Atoms", title_standoff=5, row=1, col=3)
    fig.update_yaxes(title_text="", row=1, col=1)
    fig.update_yaxes(title_text="", row=1, col=2)
    fig.update_yaxes(title_text="", row=1, col=3)

    fig.update_yaxes(range=[ymin_time, ymax_time], autorange=False, row=1, col=1)
    fig.update_yaxes(range=[ymin_memory, ymax_memory], autorange=False, row=1, col=2)
    fig.update_xaxes(range=[4.5, 21.5], autorange=False, row=1, col=1)
    fig.update_xaxes(range=[4.5, 21.5], autorange=False, row=1, col=2)

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
            row=1,
            col=3,
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
    y_top = 0.999
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
        y=y_top,
        xref="paper",
        yref="paper",
        text="<b>c</b>",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=ANNOTATION_BOLD_FONT_SIZE),
    )

    legend_x = dom1[0] + 0.005
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        margin=dict(l=0, r=0, b=10, t=30),
        width=1200,
        height=380,
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
    fig.write_image(output_path, width=1200, height=380, scale=2)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Three-panel speed/memory/lambda plot")
    parser.add_argument(
        "--speed_csv",
        type=str,
        default="results_speed2/ts1x-val.lmdb_speed_comparison_extended_10_r100.0_rh100.0.csv",
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
        "--output",
        type=str,
        default="results_speed2/speed_memory_lambda_scaling_3panel.png",
    )
    parser.add_argument("--ymin_time", type=float, default=0.0)
    parser.add_argument("--ymax_time", type=float, default=3700.0)
    parser.add_argument("--ymin_memory", type=float, default=0.0)
    parser.add_argument("--ymax_memory", type=float, default=2100.0)
    args = parser.parse_args()

    make_plot(
        speed_csv=args.speed_csv,
        pubchem_lambda_results={
            "HIP": args.pubchem_hip_metrics_csv,
            "AD": args.pubchem_ad_metrics_csv,
        },
        output_path=args.output,
        ymin_time=args.ymin_time,
        ymax_time=args.ymax_time,
        ymin_memory=args.ymin_memory,
        ymax_memory=args.ymax_memory,
    )
