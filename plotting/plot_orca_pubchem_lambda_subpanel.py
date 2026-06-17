"""
Plot the standalone PubChem lambda MAE subpanel from ORCA evaluation CSVs.

Usage:
    .venv-plotting/bin/python plotting/plot_orca_pubchem_lambda_subpanel.py
"""

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

# try:
#     from hip.colours import (
#         ANNOTATION_BOLD_FONT_SIZE,
#         AXES_FONT_SIZE,
#         AXES_TITLE_FONT_SIZE,
#         HESSIAN_METHOD_TO_COLOUR,
#         TITLE_FONT_SIZE,
#     )
# except ModuleNotFoundError:
ANNOTATION_BOLD_FONT_SIZE = 18
AXES_FONT_SIZE = 12
AXES_TITLE_FONT_SIZE = 13
HESSIAN_METHOD_TO_COLOUR = {
    "autograd": "#1f77b4",
    "prediction": "#d96001",
}
TITLE_FONT_SIZE = 16


RESULTS = {
    "HIP": "results_eval_largehessians_orca_hip_v2/metrics.csv",
    "AD": "results_eval_largehessians_orca_hf_horm_eqv2_autograd/metrics.csv",
}

DEFAULT_METRIC = "eckart_eigval_mae_ev_a2"
DEFAULT_OUTPUT = "results_speed/orca_pubchem_lambda_subpanel.png"
OUTLIER_METRIC = "hessian_mae_ev_a2"
OUTLIER_MODIFIED_Z_THRESHOLD = 10.0
PLOTLY_TEMPLATE = "plotly_white"

METHOD_TO_COLOUR = {
    "autograd": HESSIAN_METHOD_TO_COLOUR.get("autograd", "#1f77b4"),
    "prediction": "#d96001",
}


def _color_for_method(method):
    return METHOD_TO_COLOUR.get(method, "#cfcfcf")


def _load_metrics(results, metric, only_successful=True):
    frames = {}
    required_columns = {"natoms", metric, OUTLIER_METRIC}

    for label, path in results.items():
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing input for {label}: {csv_path}")

        df = pd.read_csv(csv_path)
        missing = required_columns - set(df.columns)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"{csv_path} is missing required columns: {missing_text}")

        if only_successful and "status" in df.columns:
            df = df[df["status"] == "ok"].copy()

        frames[label] = df.reset_index(drop=True)

    return frames


def _modified_z_outliers(df, metric, threshold):
    if metric not in df.columns or "natoms" not in df.columns:
        return pd.Series(False, index=df.index)

    values = pd.to_numeric(df[metric], errors="coerce")
    medians = values.groupby(df["natoms"]).transform("median")
    abs_deviation = (values - medians).abs()
    mads = abs_deviation.groupby(df["natoms"]).transform("median")
    modified_z = 0.6745 * (values - medians) / mads
    return modified_z.abs() > threshold


def _remove_outlier_samples(dfs, output_dir=None):
    outlier_rows = []
    outlier_sample_names = set()

    for label, df in dfs.items():
        mask = _modified_z_outliers(
            df,
            OUTLIER_METRIC,
            OUTLIER_MODIFIED_Z_THRESHOLD,
        )
        if not mask.any():
            continue

        label_outliers = df[mask].copy()
        label_outliers.insert(0, "label", label)
        outlier_rows.append(label_outliers)

        if "sample_name" in label_outliers.columns:
            outlier_sample_names.update(label_outliers["sample_name"].dropna())

    if not outlier_rows:
        print("No outliers removed")
        return dfs

    outliers = pd.concat(outlier_rows, ignore_index=True)
    if output_dir is not None:
        outliers_path = Path(output_dir) / "orca_pubchem_lambda_subpanel_removed_outliers.csv"
        outliers_path.parent.mkdir(parents=True, exist_ok=True)
        outliers.to_csv(outliers_path, index=False)
        print(f"Saved removed outlier rows to {outliers_path}")

    print(
        f"Removing {len(outlier_sample_names)} outlier samples identified by "
        f"per-natoms {OUTLIER_METRIC} modified z-score > "
        f"{OUTLIER_MODIFIED_Z_THRESHOLD}"
    )

    filtered = {}
    for label, df in dfs.items():
        if "sample_name" not in df.columns:
            filtered[label] = df
            continue
        filtered[label] = df[~df["sample_name"].isin(outlier_sample_names)].copy()
        print(f"Filtered {label}: {len(df)} -> {len(filtered[label])} samples")

    return filtered


def _load_eigval_curve(df, metric):
    grouped = (
        df.groupby("natoms")[metric]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("natoms")
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    return grouped["natoms"], grouped["mean"], grouped["std"]


def _hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def make_plot(
    results,
    output,
    metric=DEFAULT_METRIC,
    title="Eigenvalues λ MAE (PubChem)",
    only_successful=True,
    filter_outliers=True,
    write_outliers=True,
    width=400,
    height=380,
):
    dfs = _load_metrics(results, metric, only_successful=only_successful)
    if filter_outliers:
        outlier_dir = Path(output).parent if write_outliers else None
        dfs = _remove_outlier_samples(dfs, output_dir=outlier_dir)

    fig = go.Figure()
    label_to_method = {"HIP": "prediction", "AD": "autograd"}

    for label in ("AD", "HIP"):
        if label not in dfs:
            continue

        method = label_to_method[label]
        color = _color_for_method(method)
        x, y, std = _load_eigval_curve(dfs[label], metric)
        upper = y + std
        lower = y - std
        band_x = pd.concat([x, x.iloc[::-1]], ignore_index=True)
        band_y = pd.concat([upper, lower.iloc[::-1]], ignore_index=True)

        fig.add_trace(
            go.Scatter(
                x=band_x,
                y=band_y,
                fill="toself",
                fillcolor=_hex_to_rgba(color, 0.18),
                line=dict(color="rgba(255,255,255,0)", width=0),
                hoverinfo="skip",
                name=f"{label} std",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=label,
                showlegend=False,
                line=dict(color=color, width=3),
                marker=dict(color=color),
            )
        )

    pubchem_lambda_n = sorted(
        {n for df in dfs.values() for n in df["natoms"].dropna().tolist()}
    )
    if pubchem_lambda_n:
        fig.update_xaxes(
            range=[min(pubchem_lambda_n) - 0.5, max(pubchem_lambda_n) + 0.5],
            autorange=False,
        )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        width=width,
        height=height,
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=TITLE_FONT_SIZE)),
        margin=dict(l=10, r=0, b=10, t=30),
    )
    fig.update_xaxes(
        title_text="Number of Atoms",
        title_standoff=5,
        tickfont=dict(size=AXES_FONT_SIZE),
        title_font=dict(size=AXES_TITLE_FONT_SIZE),
    )
    fig.update_yaxes(
        title_text="",
        tickfont=dict(size=AXES_FONT_SIZE),
        title_font=dict(size=AXES_TITLE_FONT_SIZE),
    )
    fig.add_annotation(
        x=0.0,
        y=0.999,
        xref="paper",
        yref="paper",
        text="<b>d</b>",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=ANNOTATION_BOLD_FONT_SIZE),
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(output_path, width=width, height=height, scale=2)
    print(f"Plot saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the standalone ORCA PubChem lambda MAE subpanel."
    )
    parser.add_argument("--hip_csv", default=RESULTS["HIP"], help="HIP metrics CSV.")
    parser.add_argument("--ad_csv", default=RESULTS["AD"], help="AD metrics CSV.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output image path.")
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help="Metric column to plot.",
    )
    parser.add_argument(
        "--include_failed",
        action="store_true",
        help="Include rows whose status is not 'ok'.",
    )
    parser.add_argument(
        "--include_outliers",
        action="store_true",
        help="Include samples flagged by the plot_size.py modified-z outlier filter.",
    )
    parser.add_argument(
        "--no_outliers_csv",
        action="store_true",
        help="Do not write a CSV containing removed outlier rows.",
    )
    parser.add_argument("--width", type=int, default=400, help="Output figure width.")
    parser.add_argument("--height", type=int, default=380, help="Output figure height.")
    parser.add_argument(
        "--title",
        default="Eigenvalues λ MAE (PubChem)",
        help="Plot title.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_plot(
        results={"HIP": args.hip_csv, "AD": args.ad_csv},
        output=args.output,
        metric=args.metric,
        title=args.title,
        only_successful=not args.include_failed,
        filter_outliers=not args.include_outliers,
        write_outliers=not args.no_outliers_csv,
        width=args.width,
        height=args.height,
    )
