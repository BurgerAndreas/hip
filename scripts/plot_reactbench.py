import pandas as pd
import os
import wandb

import seaborn as sns

from hip.colours import (
    COLOUR_LIST,
    METHOD_TO_COLOUR,
    HESSIAN_METHOD_TO_COLOUR,
    ANNOTATION_FONT_SIZE,
    AXES_FONT_SIZE,
    AXES_TITLE_FONT_SIZE,
    LEGEND_FONT_SIZE,
)
import plotly.graph_objects as go
import plotly.express as px

api = wandb.Api()

# for .csv files
OUT_DIR = "results/"
os.makedirs(OUT_DIR, exist_ok=True)
OUTFILE = os.path.join(OUT_DIR, "reactbench.csv")

# for plots
PLOTS_DIR = "results_reactbench/plots/reactbench"
os.makedirs(PLOTS_DIR, exist_ok=True)

"""
Make a grouped bar plot
where each group is a metric
and each bar within a group is a model

colour by name of the method
name is based on hessian_method and calc fields

Metrics:
gsm_success
Number of successful GSM calculations that lead to an initial guess

converged_ts
local TS search (RS-P-RFO) converged
Convergence is defined as reaching all the following criteria within 50 steps: maximum force of $4.5e^{-4}$, RMS force of $3.0e^{-4}$, maximum step of $1.8e^{-3}$, RMS step of $1.2e^{-3}$ in atomic units (Hartree, Bohr), default in Gaussian.

ts_success
if is transition state according to frequency analysis

convert_ts
converged and ts_success
not the same as ts_success

irc_success (ignore)
IRC found two different geometries, both different from the initial transition state
(transition state, reactant, product)
This counts how many structures have energies different from the minimum.
Since one structure will always be 0 (the lowest energy):
At least 3 distinct energy levels were found (minimum + 2 others)

intended_count
converged to initial reactant and product
"""

# extra metrics, verified by DFT
extra_metrics = {
    "predict": {
        "one negative eigenvalue and force RMS < 2.0e-03 Ha/Bohr:": {
            "correct_proposed_estimated": 642.24,  # "DFT-Verified Converged and TS Success"
            "false_proposed_estimated": 26.76,
            "true_positive_rate": 0.96,
            "false_positive_rate": 0.04,
        },
        "one negative eigenvalue:": {
            "correct_proposed_estimated": 669.0,  # "DFT-Verified TS Success"
            "false_proposed_estimated": 0.0,
            "true_positive_rate": 1.0,
            "false_positive_rate": 0.0,
        },
    },
    "autograd": {
        "one negative eigenvalue and force RMS < 2.0e-03 Ha/Bohr:": {
            "correct_proposed_estimated": 571.48,
            "false_proposed_estimated": 56.519999999999996,
            "true_positive_rate": 0.91,
            "false_positive_rate": 0.09,
        },
        "one negative eigenvalue:": {
            "correct_proposed_estimated": 621.72,
            "false_proposed_estimated": 6.28,
            "true_positive_rate": 0.99,
            "false_positive_rate": 0.01,
        },
    },
}

rename_metrics = {
    "gsm_success": "GSM Success",
    "converged_ts": "RFO Converged",
    "ts_success": "TS Success",
    "convert_ts": "RFO Converged and TS Success",
    "irc_success": "IRC Success",  # ignore
    "intended_count": "IRC Verified",
}

# Try to use the eval export if present; otherwise derive from runs_df
try:
    df = pd.read_csv(OUTFILE, quotechar='"')
    df["Metric"] = df["Metric"].map(rename_metrics)
    print(f"Loaded eval csv from {OUTFILE}")
except Exception:
    # Project is specified by <entity/project-name>
    runs = api.runs("andreas-burger/reactbench")

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        if "final" not in run.tags:
            continue
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    runs_df = pd.DataFrame(
        {"summary": summary_list, "config": config_list, "name": name_list}
    )

    runs_df.to_csv(OUTFILE)
    print(f"Saved csv to {OUTFILE}")

    records = []
    for _, row in runs_df.iterrows():
        cfg = row.get("config", {}) or {}
        summ = row.get("summary", {}) or {}
        base_method = str(cfg.get("hessian_method", "unknown"))
        calc = cfg.get("calc")
        calc_str = None if calc is None else str(calc)
        if calc_str and calc_str.lower() not in ["none", "nan", "na", ""]:
            method_label = f"{base_method}-{calc_str}"
        else:
            method_label = base_method
        for metric_key, metric_label in rename_metrics.items():
            value = summ.get(metric_key)
            if value is None:
                continue
            records.append(
                {
                    "Metric": metric_label,
                    "Value": value,
                    "Method": method_label,
                }
            )
    df = pd.DataFrame.from_records(records)

# Append DFT-verified metrics (correct_proposed_estimated) for predict/autograd
method_key_map = {"predict": "predict-equiformer", "autograd": "autograd-equiformer"}
extra_records = []
for src_key, method_label in method_key_map.items():
    metrics_block = extra_metrics.get(src_key, {})
    ts_block = metrics_block.get("one negative eigenvalue:")
    conv_ts_block = metrics_block.get(
        "one negative eigenvalue and force RMS < 2.0e-03 Ha/Bohr:"
    )
    if ts_block is not None:
        extra_records.append(
            {
                "Metric": "DFT-Verified TS Success",
                "Value": ts_block.get("correct_proposed_estimated"),
                "Method": method_label,
            }
        )
    if conv_ts_block is not None:
        extra_records.append(
            {
                "Metric": "DFT-Verified Converged and TS Success",
                "Value": conv_ts_block.get("correct_proposed_estimated"),
                "Method": method_label,
            }
        )
if extra_records:
    df = pd.concat([df, pd.DataFrame.from_records(extra_records)], ignore_index=True)

sns.set_theme(style="whitegrid", palette="pastel")

# data
allowed_metrics = [
    "GSM Success",
    "TS Success",
    "RFO Converged",
    # "RFO Converged and TS Success",
    "IRC Verified",
    "DFT-Verified TS Success",
    "DFT-Verified Converged and TS Success",
]
df = df[df["Metric"].isin(allowed_metrics)]

# Build palette mapping from method labels to consistent colours
methods = list(pd.unique(df["Method"]))
palette = {}
colour_iter = iter(COLOUR_LIST)
for m in methods:
    colour = METHOD_TO_COLOUR.get(m)
    if colour is None:
        try:
            colour = next(colour_iter)
        except StopIteration:
            # fallback to seaborn palette cycling
            colour = sns.color_palette("pastel", len(methods)).as_hex()[
                methods.index(m) % len(methods)
            ]
    palette[m] = colour

# order = allowed_metrics
# fig, ax = plt.subplots(figsize=(10, 6))
# sns.barplot(
#     data=df,
#     x="Metric",
#     y="Value",
#     hue="Method",
#     order=order,
#     palette=palette,
#     ax=ax,
# )
# ax.set_xlabel("")
# ax.set_ylabel("Count")
# plt.setp(ax.get_xticklabels(), rotation=-25, ha="right", fontsize=12)
# ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
# plt.tight_layout()

# outfile = os.path.join(PLOTS_DIR, "reactbench.png")
# plt.savefig(outfile, dpi=300)
# print(f"Saved plot to {outfile}")

###################################################################################
# Build a Plotly lollipop plot for two specific methods
desired_methods = ["predict-equiformer", "autograd-equiformer"]
method_display_name = {
    "predict-equiformer": "Predicted Hessians (Ours)",
    "autograd-equiformer": "Autograd Hessians",
}

df_plot = df[df["Method"].isin(desired_methods)].copy()
if df_plot.empty:
    print("No data available for Plotly lollipop plot (predict/autograd).")

else:
    # Ensure consistent metric ordering
    df_plot["Metric"] = pd.Categorical(
        df_plot["Metric"], categories=allowed_metrics, ordered=True
    )

    # Ensure GSM Success is identical across methods by averaging
    # should only not be the case due to randomness
    gsm_mask = df_plot["Metric"] == "GSM Success"
    gsm_vals = df_plot[gsm_mask].groupby("Method")["Value"].first()
    if len(gsm_vals) >= 2:
        gsm_mean = gsm_vals.mean()
        df_plot.loc[gsm_mask, "Value"] = gsm_mean

    fig = go.Figure()
    default_colorway = px.colors.qualitative.Plotly

    # Draw background first (wider), then foreground on top (narrower)
    render_order = ["predict-equiformer", "autograd-equiformer"]
    for method_key in render_order:
        sub = df_plot[df_plot["Method"] == method_key].sort_values("Metric")
        if sub.empty:
            continue

        # Prefer explicit colours: METHOD_TO_COLOUR by full key, else HESSIAN_METHOD_TO_COLOUR by prefix
        base_key = method_key.split("-")[0]
        colour = (
            METHOD_TO_COLOUR.get(method_key)
            or HESSIAN_METHOD_TO_COLOUR.get(base_key)
            or default_colorway[
                desired_methods.index(method_key) % len(default_colorway)
            ]
        )
        display = method_display_name.get(method_key, method_key)
        is_background = method_key == render_order[0]

        # Vertical stems per category (x,0)->(x,y)
        xs, ys = [], []
        for _, r in sub.iterrows():
            xs.extend([r["Metric"], r["Metric"], None])
            ys.extend([0, r["Value"], None])

        LINE_WIDTH = 16
        MARKER_SIZE = 22
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(
                    color=colour, width=(12 if is_background else LINE_WIDTH * 0.5)
                ),
                showlegend=False,
                hoverinfo="skip",
                # opacity=(0.6 if is_background else 1.0),
                opacity=1.0,
            )
        )

        # Lollipop heads; only show GSM Success label for foreground
        show_text = []
        for _, r in sub.iterrows():
            if r["Metric"] == "GSM Success" and is_background:
                show_text.append("")
            else:
                show_text.append(f"{r['Value']:.0f}")

        # Shift TS Success labels vertically to reduce overlap between methods
        text_positions = []
        for _, r in sub.iterrows():
            if r["Metric"] == "TS Success":
                text_positions.append(
                    "middle right" if is_background else "bottom right"
                )
            else:
                text_positions.append("middle right")

        fig.add_trace(
            go.Scatter(
                x=sub["Metric"],
                y=sub["Value"],
                mode="markers+text",
                name=display,
                marker=dict(
                    color=colour,
                    size=(MARKER_SIZE if is_background else MARKER_SIZE * (2 / 3)),
                ),
                text=show_text,
                texttemplate="%{text}",
                textposition=text_positions,
                textfont=dict(size=ANNOTATION_FONT_SIZE),
                cliponaxis=False,
                # opacity=(0.75 if is_background else 1.0),
                opacity=1.0,
            )
        )

    fig.update_layout(
        xaxis_title="",
        xaxis=dict(
            categoryorder="array",
            categoryarray=allowed_metrics,
            tickangle=-25,
            tickfont=dict(size=AXES_FONT_SIZE),
        ),
        yaxis=dict(
            title=dict(text="Success Count", font=dict(size=AXES_TITLE_FONT_SIZE)),
            range=[398.5, 920],
            tickfont=dict(size=AXES_FONT_SIZE),
        ),
        margin=dict(l=40, r=10, t=0, b=10),
        template="plotly_white",
        legend=dict(
            title=dict(text=""),
            font=dict(size=LEGEND_FONT_SIZE),
            x=0.45,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.5)",
        ),
    )

    # outfile_html = os.path.join(PLOTS_DIR, "reactbench_lollipop.html")
    # fig.write_html(outfile_html, include_plotlyjs="cdn")
    # print(f"Saved Plotly lollipop to {outfile_html}")
    outfile = os.path.join(PLOTS_DIR, "reactbench_lollipop_wide.png")
    fig.write_image(outfile, width=1000, height=600, scale=3)
    print(f"Saved Plotly lollipop to {outfile}")
    outfile = os.path.join(PLOTS_DIR, "reactbench_lollipop_square.png")
    fig.write_image(outfile, width=600, height=600, scale=3)
    print(f"Saved Plotly lollipop to {outfile}")
