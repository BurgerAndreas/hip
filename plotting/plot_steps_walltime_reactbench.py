from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hip.colours import (
    ANNOTATION_BOLD_FONT_SIZE,
    ANNOTATION_FONT_SIZE,
    HESSIAN_METHOD_TO_COLOUR,
    LEGEND_FONT_SIZE,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "reactbench_relaxation"
DEFAULT_RELAXATION_CSV = (
    DEFAULT_DATA_DIR / "relaxation_results_noiserms0.035.csv"
)
DEFAULT_REACTBENCH_CSV = DEFAULT_DATA_DIR / "reactbench.csv"
DEFAULT_OUTPUT = (
    ROOT_DIR
    / "plots"
    / "reactbench_relaxation"
    / "steps_walltime_reactbench_plotly.png"
)

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


def _hex_to_rgba(hex_colour: str, alpha: float) -> str:
    hex_colour = hex_colour.lstrip("#")
    r, g, b = tuple(int(hex_colour[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


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
        return "Fire"
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


def plot_steps_walltime_reactbench(
    relaxation_csv: Path,
    reactbench_csv: Path,
    output_path: Path,
    max_cycles: int,
) -> None:
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
    order_wall_comp = _prepare_order(df_wall_comp, "wall_time_s")
    df_rb = _load_reactbench(reactbench_csv)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Steps to Convergence",
            "Wall Time [s] (Subset)",
            "TS Search (ReactBench)",
        ),
        horizontal_spacing=0.06,
    )

    categories_all: list[str] = []
    for col_idx, (df_i, metric_i, order_i) in enumerate(
        (
            (df_steps, "steps", order_steps),
            (df_wall_comp, "wall_time_s", order_wall_comp),
        ),
        start=1,
    ):
        display_order = []
        method_to_display_name = {}
        methods_plotted = []
        for method in order_i:
            series = _series_for_method(df_i, method, metric_i)
            if series.empty:
                continue
            display_name = _display_name(method)
            color = METHOD_TO_COLOUR[method]
            display_order.append(display_name)
            method_to_display_name[method] = display_name
            methods_plotted.append(method)
            method_rows = df_i[df_i["name"] == method]
            hit_max_mask = method_rows["steps"].isin([max_cycles, max_cycles - 1])
            series_noconv = method_rows.loc[hit_max_mask, metric_i].dropna()

            fig.add_trace(
                go.Violin(
                    y=series.astype(float),
                    name=display_name,
                    line_color=color,
                    fillcolor=_hex_to_rgba(color, 0.1),
                    opacity=1.0,
                    line_width=1,
                    width=0.9,
                    box_visible=False,
                    meanline_visible=False,
                    spanmode="hard",
                    points="all",
                    jitter=0.5,
                    pointpos=0,
                    marker=dict(color=color, opacity=0.3, size=4),
                    showlegend=False,
                ),
                row=1,
                col=col_idx,
            )
            if not series_noconv.empty:
                fig.add_trace(
                    go.Scatter(
                        x=[display_name] * len(series_noconv),
                        y=series_noconv.astype(float),
                        mode="markers",
                        marker=dict(symbol="x", color=color, opacity=1.0, size=14),
                        showlegend=False,
                        hovertemplate="not converged: %{y}<extra></extra>",
                    ),
                    row=1,
                    col=col_idx,
                )

        for method in methods_plotted:
            category = METHOD_TO_CATEGORY.get(method)
            if category is not None and category not in categories_all:
                categories_all.append(category)

        bold_targets = {
            method_to_display_name[method]
            for method in ("RFO (learned)", "RFO-BFGS (learned init)")
            if method in method_to_display_name
        }
        ticktext = [
            f"<b>{name}</b>" if name in bold_targets else name
            for name in display_order
        ]
        fig.update_xaxes(
            categoryorder="array",
            categoryarray=display_order,
            tickvals=display_order,
            ticktext=ticktext,
            tickangle=-25,
            row=1,
            col=col_idx,
        )
        fig.update_yaxes(
            title_text=METRIC_TO_LABEL[metric_i],
            row=1,
            col=col_idx,
        )

    if not df_wall_comp.empty:
        wall_y_max = float(df_wall_comp["wall_time_s"].max())
    else:
        wall_y_max = 100.0

    anno_entries = []
    for method in WALL_TIME_ANNOTATION_ONLY:
        series = _series_for_method(df_wall, method, "wall_time_s")
        if series.empty:
            continue
        anno_entries.append((_display_name(method), float(series.mean()), method))

    if anno_entries:
        existing_categories = list(fig.layout.xaxis2.categoryarray or [])
        existing_ticks = list(fig.layout.xaxis2.ticktext or [])
        tick_map = dict(zip(existing_categories, existing_ticks, strict=False))
        for display_name, _, _ in anno_entries:
            tick_map[display_name] = display_name
        desired_display_order = [_display_name(method) for method in PANEL_METHOD_ORDER]
        category_set = set(existing_categories)
        category_set.update(display_name for display_name, _, _ in anno_entries)
        merged_categories = [
            display_name
            for display_name in desired_display_order
            if display_name in category_set
        ]
        merged_categories.extend(
            category
            for category in category_set
            if category not in set(merged_categories)
        )
        merged_ticks = [tick_map.get(category, category) for category in merged_categories]

        for display_name, mean_value, method in anno_entries:
            color = METHOD_TO_COLOUR[method]
            fig.add_trace(
                go.Scatter(
                    x=[display_name],
                    y=[None],
                    mode="markers",
                    marker=dict(size=0, opacity=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=2,
            )
            fig.add_annotation(
                x=display_name,
                y=wall_y_max * 0.45,
                xref="x2",
                yref="y2",
                text=f"<b>{mean_value:.0f}s</b>",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=2.5,
                arrowcolor=color,
                ax=0,
                ay=40,
                font=dict(size=18, color=color),
                xanchor="center",
                yanchor="top",
            )
        fig.update_xaxes(
            categoryorder="array",
            categoryarray=merged_categories,
            tickvals=merged_categories,
            ticktext=merged_ticks,
            row=1,
            col=2,
        )

    for category in categories_all:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color=CATEGORY_TO_COLOUR[category], size=10),
                name=category,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    rb_display_names = {
        "predict-equiformer": "HIP Hessians",
        "autograd-equiformer": "AD Hessians",
    }
    rb_render_order = ["predict-equiformer", "autograd-equiformer"]
    default_colorway = px.colors.qualitative.Plotly
    for method_key in rb_render_order:
        sub = df_rb[df_rb["Method"] == method_key].sort_values("Metric")
        if sub.empty:
            continue
        base_key = method_key.split("-")[0]
        colour = (
            HESSIAN_METHOD_TO_COLOUR.get(base_key)
            or default_colorway[rb_render_order.index(method_key) % len(default_colorway)]
        )
        fig.add_trace(
            go.Bar(
                x=sub["Metric"],
                y=sub["Value"],
                name=rb_display_names[method_key],
                marker=dict(color=colour),
                text=[f"{value:.0f}" for value in sub["Value"]],
                textposition="outside",
                textfont=dict(size=ANNOTATION_FONT_SIZE + 1),
                cliponaxis=False,
                opacity=1.0,
                legend="legend2",
            ),
            row=1,
            col=3,
        )

    rb_allowed_metrics = [
        "GSM Success",
        "RFO Converged",
        "IRC Verified",
        "TS (DFT)",
        "+ Converged",
    ]
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=rb_allowed_metrics,
        tickangle=-25,
        showgrid=False,
        row=1,
        col=3,
    )
    fig.update_yaxes(title_text="Success Count", row=1, col=3)

    height = 400
    width = int(height * 3.0)
    fig.update_layout(
        template="plotly_white",
        showlegend=True,
        barmode="group",
        bargap=0.22,
        bargroupgap=0.06,
        height=height,
        width=width,
        margin=dict(l=0, r=20, b=60, t=20),
        font=dict(size=13),
        legend=dict(
            x=0.30,
            y=0.95,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=LEGEND_FONT_SIZE + 1),
        ),
        legend2=dict(
            x=0.99,
            y=0.95,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=LEGEND_FONT_SIZE + 1),
        ),
    )
    fig.update_yaxes(title_standoff=10, range=[0, 155], row=1, col=1)
    fig.update_yaxes(title_standoff=10, range=[0, 4.9], row=1, col=2)
    fig.update_yaxes(title_standoff=10, range=[498.5, 920], row=1, col=3)
    fig.update_xaxes(automargin=False, row=1, col=2)
    fig.update_xaxes(automargin=False, row=1, col=3)

    for panel_i, label in enumerate(["a", "b", "c"]):
        axis_name = "xaxis" if panel_i == 0 else f"xaxis{panel_i + 1}"
        domain = getattr(fig.layout, axis_name).domain
        fig.add_annotation(
            x=domain[0],
            y=0.999,
            xref="paper",
            yref="paper",
            text=f"<b>{label}</b>",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=ANNOTATION_BOLD_FONT_SIZE + 1),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(output_path, width=width, height=height, scale=3)
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
    plot_steps_walltime_reactbench(
        relaxation_csv=args.relaxation_csv,
        reactbench_csv=args.reactbench_csv,
        output_path=args.output,
        max_cycles=args.max_cycles,
    )


if __name__ == "__main__":
    main()
