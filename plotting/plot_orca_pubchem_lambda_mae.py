import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

try:
    from hip.colours import (
        ANNOTATION_BOLD_FONT_SIZE,
        AXES_FONT_SIZE,
        AXES_TITLE_FONT_SIZE,
        HESSIAN_METHOD_TO_COLOUR,
        LEGEND_FONT_SIZE,
        TITLE_FONT_SIZE,
    )
except ModuleNotFoundError:
    ANNOTATION_BOLD_FONT_SIZE = 18
    AXES_FONT_SIZE = 12
    AXES_TITLE_FONT_SIZE = 13
    HESSIAN_METHOD_TO_COLOUR = {
        "autograd": "#1f77b4",
        "prediction": "#d96001",
    }
    LEGEND_FONT_SIZE = 12
    TITLE_FONT_SIZE = 16


PLOTLY_TEMPLATE = "plotly_white"

DEFAULT_INPUTS = (
    "metadata/orca_hessian_eval_hip_v2_predict.parquet",
    "metadata/orca_hessian_eval_eqv2_autograd.parquet",
)

DEFAULT_METRIC = "eckart_eigval_mae_hartree_bohr2"
ECKART_METRIC_EV_A2 = "eckart_eigval_mae_ev_a2"
HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_BOHR2_TO_EV_A2 = HARTREE_TO_EV / (BOHR_TO_ANGSTROM**2)

METHOD_LABELS = {
    "predict": "HIP Hessians (ours)",
    "prediction": "HIP Hessians (ours)",
    "autograd": "AD Hessians (direct force)",
}

METHOD_ORDER = ("autograd", "predict", "prediction")
OUTLIER_RATIO = 5.0
OUTLIER_MAD_FACTOR = 8.0


def label_for_method(method):
    method = str(method)
    return METHOD_LABELS.get(method.lower(), method)


def color_for_method(method):
    method = str(method).lower()
    if method in {"predict", "prediction"}:
        return "#d96001"
    return HESSIAN_METHOD_TO_COLOUR.get(method, "#636EFA")


def load_results(input_paths, metric, only_successful=True):
    required_columns = {"natoms", "hessian_method", metric}
    frames = []
    for input_path in input_paths:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file does not exist: {path}")

        df = pd.read_parquet(path)
        missing = required_columns - set(df.columns)
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise ValueError(f"{path} is missing required columns: {missing_columns}")

        if only_successful and "status" in df.columns:
            df = df[df["status"] == "ok"].copy()

        df["method_key"] = df["hessian_method"].astype(str)
        df["method_label"] = df["method_key"].map(label_for_method)
        frames.append(df)

    if not frames:
        raise ValueError("No input files provided.")

    results = pd.concat(frames, ignore_index=True)
    if results.empty:
        raise ValueError("No rows remain after filtering input results.")

    return results


def aggregate_results(results_df, metric):
    stats = (
        results_df.groupby(["natoms", "method_key", "method_label"], as_index=False)[
            metric
        ]
        .mean()
        .sort_values(["natoms", "method_key"])
    )
    return stats


def detect_outliers(results_df, metric, ratio=OUTLIER_RATIO, mad_factor=OUTLIER_MAD_FACTOR):
    grouped = results_df.groupby(["method_key", "natoms"])[metric]
    median = grouped.transform("median")
    abs_deviation = (results_df[metric] - median).abs()
    mad = abs_deviation.groupby([results_df["method_key"], results_df["natoms"]]).transform(
        "median"
    )
    threshold = pd.concat(
        [median * ratio, median + mad_factor * mad],
        axis=1,
    ).max(axis=1)

    outliers_df = results_df.copy()
    outliers_df["group_median"] = median
    outliers_df["group_mad"] = mad
    outliers_df["outlier_threshold"] = threshold
    outliers_df["ratio_to_median"] = outliers_df[metric] / median
    outliers_df["is_outlier"] = outliers_df[metric] > threshold
    return outliers_df


def add_plot_metric(results_df, metric):
    results_df = results_df.copy()
    if metric == DEFAULT_METRIC:
        results_df[ECKART_METRIC_EV_A2] = results_df[metric] * HARTREE_BOHR2_TO_EV_A2
        return results_df, ECKART_METRIC_EV_A2
    return results_df, metric


def plot_pubchem_lambda_mae(
    results_df,
    output_path,
    metric=DEFAULT_METRIC,
    title="Eigenvalues λ MAE (PubChem)",
    width=520,
    height=420,
    show_panel_label=True,
    show_legend=False,
    exclude_outliers=True,
    outliers_csv=None,
):
    outliers_df = detect_outliers(results_df, metric)
    outliers_df, plot_metric = add_plot_metric(outliers_df, metric)
    if outliers_csv is not None:
        outliers_path = Path(outliers_csv)
        outliers_path.parent.mkdir(parents=True, exist_ok=True)
        outliers_df[outliers_df["is_outlier"]].to_csv(outliers_path, index=False)
        print(f"Outliers saved to\n{outliers_path}")

    if exclude_outliers:
        excluded = outliers_df["is_outlier"].sum()
        if excluded:
            print(f"Excluding {excluded} outlier rows.")
        results_df = outliers_df[~outliers_df["is_outlier"]].copy()
    else:
        results_df = outliers_df.copy()

    stats_df = aggregate_results(results_df, plot_metric)
    fig = go.Figure()

    method_keys = sorted(
        stats_df["method_key"].unique(),
        key=lambda method: (
            METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER),
            method,
        ),
    )
    for method_key in method_keys:
        method_stats = stats_df[stats_df["method_key"] == method_key]
        method_label = method_stats["method_label"].iloc[0]
        color = color_for_method(method_key)
        fig.add_trace(
            go.Scatter(
                x=method_stats["natoms"],
                y=method_stats[plot_metric],
                mode="lines+markers",
                name=method_label,
                showlegend=show_legend,
                line=dict(color=color, width=3),
                marker=dict(color=color, size=6),
            )
        )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        width=width,
        height=height,
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=TITLE_FONT_SIZE)),
        margin=dict(l=10, r=10, b=10, t=45),
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.6)",
            font=dict(size=LEGEND_FONT_SIZE - 2),
        ),
    )
    fig.update_xaxes(
        title_text="Number of Atoms",
        tickfont=dict(size=AXES_FONT_SIZE),
        title_font=dict(size=AXES_TITLE_FONT_SIZE),
    )
    fig.update_yaxes(
        title_text="",
        tickfont=dict(size=AXES_FONT_SIZE),
        title_font=dict(size=AXES_TITLE_FONT_SIZE),
        rangemode="tozero",
    )

    if show_panel_label:
        fig.add_annotation(
            x=0.0,
            y=1.0,
            xref="paper",
            yref="paper",
            text="<b>d</b>",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=ANNOTATION_BOLD_FONT_SIZE),
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(output_path, width=width, height=height, scale=2)
    print(f"Plot saved to\n{output_path}")

    return stats_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot ORCA PubChem eigenvalue lambda MAE by atom count."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=list(DEFAULT_INPUTS),
        help="Parquet ORCA eval files to load.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results_speed/orca_pubchem_lambda_mae.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=DEFAULT_METRIC,
        help=(
            "Metric column to plot from the ORCA eval files. The default "
            "Hartree/Bohr^2 Eckart metric is converted to eV/A^2 for plotting."
        ),
    )
    parser.add_argument(
        "--include_failed",
        action="store_true",
        help="Include rows whose status is not 'ok'.",
    )
    parser.add_argument(
        "--include_outliers",
        action="store_true",
        help="Include robustly detected eigenvalue MAE outliers.",
    )
    parser.add_argument(
        "--outliers_csv",
        type=str,
        default=None,
        help="Optional path to save robustly detected outlier rows.",
    )
    parser.add_argument(
        "--no_panel_label",
        action="store_true",
        help="Do not draw the bold panel label 'd'.",
    )
    parser.add_argument(
        "--show_legend",
        action="store_true",
        help="Show the method legend.",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default=None,
        help="Optional path for aggregated values.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = load_results(
        args.inputs,
        metric=args.metric,
        only_successful=not args.include_failed,
    )
    summary = plot_pubchem_lambda_mae(
        results,
        args.output,
        metric=args.metric,
        show_panel_label=not args.no_panel_label,
        show_legend=args.show_legend,
        exclude_outliers=not args.include_outliers,
        outliers_csv=args.outliers_csv,
    )

    if args.summary_csv is not None:
        summary_path = Path(args.summary_csv)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"Summary saved to\n{summary_path}")
