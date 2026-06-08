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
    HESSIAN_METHOD_TO_COLOUR,
    LEGEND_FONT_SIZE,
    TITLE_FONT_SIZE,
)


PLOTLY_TEMPLATE = "plotly_white"

DEFAULT_INPUTS = (
    "metadata/orca_hessian_eval_hip_v2_predict.parquet",
    "metadata/orca_hessian_eval_eqv2_autograd.parquet",
)

REQUIRED_COLUMNS = {
    "natoms",
    "hessian_method",
    "hessian_mae_ev_a2",
    "cart_eigval_mae_ev_a2",
    "time_ms",
    "memory_mb",
}

METRICS = (
    ("hessian_mae_ev_a2", "Hessian MAE", "MAE (eV/A^2)"),
    ("cart_eigval_mae_ev_a2", "Cartesian Eigenvalue MAE", "MAE (eV/A^2)"),
    ("time_ms", "Time", "Time (ms)"),
    ("memory_mb", "Memory", "Peak Memory (MB)"),
)

METHOD_LABELS = {
    "predict": "HIP Hessians",
    "prediction": "HIP Hessians",
    "autograd": "EquiformerV2 AD",
}


def rgba_from_hex(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def label_for_method(method):
    method = str(method)
    return METHOD_LABELS.get(method.lower(), method)


def load_results(input_paths, only_successful=True):
    frames = []
    for input_path in input_paths:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file does not exist: {path}")

        df = pd.read_parquet(path)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise ValueError(f"{path} is missing required columns: {missing_columns}")

        if only_successful and "status" in df.columns:
            df = df[df["status"] == "ok"].copy()

        df["source_file"] = path.name
        df["method_key"] = df["hessian_method"].astype(str)
        df["method_label"] = df["method_key"].map(label_for_method)
        frames.append(df)

    if not frames:
        raise ValueError("No input files provided.")

    results = pd.concat(frames, ignore_index=True)
    if results.empty:
        raise ValueError("No rows remain after filtering input results.")

    return results


def aggregate_results(results_df):
    aggregations = {
        metric: ["mean", "std", "min", "max", "count"] for metric, _, _ in METRICS
    }
    aggregated = results_df.groupby(
        ["natoms", "method_key", "method_label"], as_index=False
    ).agg(aggregations)

    aggregated.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in aggregated.columns
    ]
    return aggregated.sort_values(["natoms", "method_key"])


def spread_bounds(method_stats, metric, spread):
    mean = method_stats[f"{metric}_mean"]
    if spread == "none":
        return None, None
    if spread == "std":
        spread_values = method_stats[f"{metric}_std"].fillna(0.0)
        return mean - spread_values, mean + spread_values
    if spread == "sem":
        spread_values = method_stats[f"{metric}_std"].fillna(0.0) / (
            method_stats[f"{metric}_count"].clip(lower=1) ** 0.5
        )
        return mean - spread_values, mean + spread_values
    if spread == "minmax":
        return method_stats[f"{metric}_min"], method_stats[f"{metric}_max"]

    raise ValueError(f"Unknown spread: {spread}")


def add_metric_traces(fig, stats_df, metric, row, col, show_legend, spread):
    for method_key in sorted(stats_df["method_key"].unique()):
        method_stats = stats_df[stats_df["method_key"] == method_key]
        if method_stats.empty:
            continue

        method_label = method_stats["method_label"].iloc[0]
        color = HESSIAN_METHOD_TO_COLOUR.get(method_key)
        if color is None:
            color = HESSIAN_METHOD_TO_COLOUR.get(method_key.lower(), "#636EFA")
        fill_color = rgba_from_hex(color, 0.18)

        lower, upper = spread_bounds(method_stats, metric, spread)
        if lower is not None and upper is not None:
            fig.add_trace(
                go.Scatter(
                    x=method_stats["natoms"],
                    y=lower,
                    mode="lines",
                    line=dict(width=0, color=fill_color),
                    hoverinfo="skip",
                    legendgroup=method_label,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=method_stats["natoms"],
                    y=upper,
                    mode="lines",
                    line=dict(width=0, color=fill_color),
                    fill="tonexty",
                    fillcolor=fill_color,
                    hoverinfo="skip",
                    legendgroup=method_label,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        fig.add_trace(
            go.Scatter(
                x=method_stats["natoms"],
                y=method_stats[f"{metric}_mean"],
                mode="lines+markers",
                name=method_label,
                legendgroup=method_label,
                showlegend=show_legend,
                line=dict(color=color, width=3),
                marker=dict(color=color, size=6),
                customdata=method_stats[f"{metric}_count"],
                hovertemplate=(
                    "N=%{x}<br>"
                    "mean=%{y:.4g}<br>"
                    "samples=%{customdata}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )


def plot_orca_hessian_eval(results_df, output_path, spread="std"):
    stats_df = aggregate_results(results_df)

    height = 700
    width = 900
    subplot_titles = [title for _, title, _ in METRICS]
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    positions = ((1, 1), (1, 2), (2, 1), (2, 2))
    for index, ((metric, _, yaxis_title), (row, col)) in enumerate(
        zip(METRICS, positions)
    ):
        add_metric_traces(
            fig,
            stats_df,
            metric=metric,
            row=row,
            col=col,
            show_legend=index == 0,
            spread=spread,
        )
        fig.update_xaxes(title_text="Number of Atoms (N)", row=row, col=col)
        fig.update_yaxes(title_text=yaxis_title, row=row, col=col)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        width=width,
        height=height,
        margin=dict(l=10, r=10, b=10, t=45),
        legend=dict(
            x=0.5,
            y=1.02,
            xanchor="center",
            yanchor="bottom",
            orientation="h",
            bgcolor="rgba(255,255,255,0.6)",
            font=dict(size=LEGEND_FONT_SIZE),
        ),
    )
    fig.update_xaxes(
        tickfont=dict(size=AXES_FONT_SIZE), title_font=dict(size=AXES_TITLE_FONT_SIZE)
    )
    fig.update_yaxes(
        tickfont=dict(size=AXES_FONT_SIZE), title_font=dict(size=AXES_TITLE_FONT_SIZE)
    )
    fig.update_annotations(font=dict(size=ANNOTATION_FONT_SIZE))

    for annotation in fig.layout.annotations:
        if annotation.text in subplot_titles:
            annotation.update(font=dict(size=TITLE_FONT_SIZE))

    panel_labels = ("<b>a</b>", "<b>b</b>", "<b>c</b>", "<b>d</b>")
    domains = (
        (fig.layout.xaxis.domain[0], fig.layout.yaxis.domain[1]),
        (fig.layout.xaxis2.domain[0], fig.layout.yaxis2.domain[1]),
        (fig.layout.xaxis3.domain[0], fig.layout.yaxis3.domain[1]),
        (fig.layout.xaxis4.domain[0], fig.layout.yaxis4.domain[1]),
    )
    for label, (x_pos, y_pos) in zip(panel_labels, domains):
        fig.add_annotation(
            x=x_pos,
            y=y_pos + 0.015,
            xref="paper",
            yref="paper",
            text=label,
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=ANNOTATION_BOLD_FONT_SIZE),
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_path = output_path.with_suffix(".html")
    fig.write_html(html_path)
    print(f"Interactive plot saved to\n{html_path}")

    try:
        fig.write_image(output_path, width=width, height=height, scale=2)
        print(f"Plot saved to\n{output_path}")
    except RuntimeError as exc:
        message = str(exc).strip().splitlines()[0]
        if "Kaleido requires Google Chrome" in str(exc):
            message = "Kaleido requires Google Chrome; run `uv run plotly_get_chrome`."
        print(f"Warning: could not write PNG to {output_path}: {message}")

    return stats_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot ORCA Hessian eval MAE, timing, and memory by atom count."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=list(DEFAULT_INPUTS),
        help="Parquet eval files to load.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results_speed/orca_hessian_eval_mae_time_memory.png",
        help="Output PNG path. A matching HTML file is also written.",
    )
    parser.add_argument(
        "--spread",
        choices=("std", "sem", "minmax", "none"),
        default="std",
        help="Spread shown around each mean line.",
    )
    parser.add_argument(
        "--include_failed",
        action="store_true",
        help="Include rows whose status is not 'ok'.",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default=None,
        help="Optional path for aggregated mean/spread values.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = load_results(args.inputs, only_successful=not args.include_failed)
    summary = plot_orca_hessian_eval(results, args.output, spread=args.spread)

    if args.summary_csv is not None:
        summary_path = Path(args.summary_csv)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"Summary saved to\n{summary_path}")
