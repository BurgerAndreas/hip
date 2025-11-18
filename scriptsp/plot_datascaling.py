import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from hip.colours import (
    ANNOTATION_FONT_SIZE,
    ANNOTATION_BOLD_FONT_SIZE,
    AXES_FONT_SIZE,
    AXES_TITLE_FONT_SIZE,
    LEGEND_FONT_SIZE,
    TITLE_FONT_SIZE,
)

PLOTLY_TEMPLATE = "plotly_white"

# Set seaborn style for poster
sns.set_style("whitegrid")
# sns.set_palette("husl")
sns.set_context("poster")

base_dir = "plots/datascaling"
os.makedirs(base_dir, exist_ok=True)

# https://wandb.ai/andreas-burger/hip/workspace?nw=fdkkquw19gl

# Store dataframes for combined plot
df_energy = None
df_force = None

for losstype in ["Loss E", "Loss F", "MAE Hessian"]:
    human_name = {"Loss E": "Energy", "Loss F": "Force", "MAE Hessian": "Hessian"}[
        losstype
    ]

    # Load the CSV file
    df = pd.read_csv(f"scaling/wandb_datascaling_loss_{human_name.lower()}2.csv")

    # Print basic info about the dataset
    print("Original dataset shape:", df.shape)
    print("Column names:")
    print(df.columns.tolist())

    # Filter columns to keep only those ending with "val-Loss E"
    val_loss_columns = [col for col in df.columns if col.endswith(f"val-{losstype}")]

    # Create a new dataframe with only the filtered columns
    df_filtered = df[val_loss_columns].copy()

    # Extract statistics for each method
    stats_data = []
    for col in val_loss_columns:
        # Remove NaN values for calculation
        clean_data = df_filtered[col].dropna()
        if len(clean_data) > 0:
            last_val = clean_data.iloc[-1]
            min_val = clean_data.min()
            max_val = clean_data.max()

            # Remove "val-Loss E" from the method name
            method_name = col.replace(f" - val-{losstype}", "")

            # Check if method ends with "EF"
            is_ef = method_name.endswith("EF")

            # Extract dataset size from the beginning of the method name
            # Find the numeric part at the beginning (including scientific notation)
            match = re.match(r"^([0-9.]+(?:e[+-]?[0-9]+)?)", method_name)
            if match:
                dataset_size = float(match.group(1))
            else:
                dataset_size = None

            stats_data.append(
                {
                    "Method": method_name,
                    "Last_Value": last_val,
                    "Min_Value": min_val,
                    "Max_Value": max_val,
                    "ef": is_ef,
                    "Dataset size": dataset_size,
                }
            )

    # Create new dataframe with methods as rows and statistics as columns
    df_stats = pd.DataFrame(stats_data)
    df_stats = df_stats.set_index("Method")

    # Store for combined plot
    if human_name == "Energy":
        df_energy = df_stats.copy()
    elif human_name == "Force":
        df_force = df_stats.copy()

    # Save the filtered data
    csvfname = os.path.join(base_dir, f"loss_{human_name.lower()}.csv")
    df_stats.to_csv(csvfname, index=False)
    print(f"Filtered data saved to '{csvfname}'")

    print("\nStatistics for each method:")
    print(df_stats)

    # Create log-log plot of min_value vs dataset size
    plt.figure(figsize=(10, 8))

    # Filter out rows where dataset size is not None
    plot_data = df_stats.dropna(subset=["Dataset size"])

    # Create scatter plot with different colors for EF vs non-EF methods
    ef_data = plot_data[plot_data["ef"]]
    efh_data = plot_data[~plot_data["ef"]]

    plt.scatter(
        ef_data["Dataset size"],
        ef_data["Min_Value"],
        label="Energy-Force",
        marker="o",
        s=100,
        alpha=0.7,
    )
    plt.scatter(
        efh_data["Dataset size"],
        efh_data["Min_Value"],
        label="Energy-Force-Hessian",
        marker="s",
        s=100,
        alpha=0.7,
    )

    # # Add horizontal line at 5% above the lowest loss
    # min_loss = plot_data["Min_Value"].min()
    # max_loss = plot_data["Max_Value"].max()
    # threshold = min_loss * 1.05
    # plt.axhline(
    #     y=threshold,
    #     color="darkgray",
    #     linestyle="--",
    #     alpha=0.7,
    #     label=r"5% MAE$_{\text{min}}$",
    # )
    # threshold = (max_loss - min_loss) * 0.01 + min_loss
    # plt.axhline(
    #     y=threshold,
    #     color="lightgray",
    #     linestyle="--",
    #     alpha=0.7,
    #     label=r"1% MAE$_{\text{min-max}}$",
    # )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of Training Samples")
    plt.ylabel(f"{human_name} MAE (validation)")
    plt.title(f"{human_name} Error vs Dataset Size")
    plt.legend(
        # title='Training Loss',
        frameon=True,
        edgecolor="none",
        fontsize=12,
    )
    plt.grid(True, alpha=0.3, which="major")
    plt.grid(True, alpha=0.1, which="minor")
    plt.tight_layout()
    fname = os.path.join(base_dir, f"log_log_{human_name.lower()}_mae.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    # plt.show()

    print(f"\nLog-log plot saved as '{fname}'")

# Create combined plotly plot with Energy on left and Force on right
if df_energy is not None and df_force is not None:
    height = 400
    width = height * 2

    # Filter out rows where dataset size is not None
    plot_data_energy = df_energy.dropna(subset=["Dataset size"])
    plot_data_force = df_force.dropna(subset=["Dataset size"])

    # Split by EF vs EFH
    ef_energy = plot_data_energy[plot_data_energy["ef"]]
    efh_energy = plot_data_energy[~plot_data_energy["ef"]]
    ef_force = plot_data_force[plot_data_force["ef"]]
    efh_force = plot_data_force[~plot_data_force["ef"]]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Energy", "Force"),
        horizontal_spacing=0.05,
        vertical_spacing=0.0,
    )

    # Energy plot (left)
    fig.add_trace(
        go.Scatter(
            x=ef_energy["Dataset size"],
            y=ef_energy["Min_Value"],
            mode="markers",
            name="Energy-Force",
            legend="legend",
            showlegend=True,
            marker=dict(color="#1b85b8", symbol="circle", size=10),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=efh_energy["Dataset size"],
            y=efh_energy["Min_Value"],
            mode="markers",
            name="Energy-Force-Hessian",
            legend="legend",
            showlegend=True,
            marker=dict(color="#d96009", symbol="square", size=10),
        ),
        row=1,
        col=1,
    )

    # Force plot (right)
    fig.add_trace(
        go.Scatter(
            x=ef_force["Dataset size"],
            y=ef_force["Min_Value"],
            mode="markers",
            name="Energy-Force",
            showlegend=False,
            marker=dict(color="#1b85b8", symbol="circle", size=10),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=efh_force["Dataset size"],
            y=efh_force["Min_Value"],
            mode="markers",
            name="Energy-Force-Hessian",
            showlegend=False,
            marker=dict(color="#d96009", symbol="square", size=10),
        ),
        row=1,
        col=2,
    )

    # Get subplot domains for legend placement
    dom1 = fig.layout.xaxis.domain if hasattr(fig.layout, "xaxis") else [0.0, 0.45]
    dom2 = fig.layout.xaxis2.domain if hasattr(fig.layout, "xaxis2") else [0.55, 1.0]
    x2 = 0.5 * (dom2[0] + dom2[1])

    # Add axis titles
    fig.update_xaxes(
        title_text="Number of Training Samples",
        title_standoff=5,
        row=1,
        col=1,
        type="log",
    )
    fig.update_yaxes(
        title_text="Energy MAE (validation)", title_standoff=10, row=1, col=1, type="log"
    )
    fig.update_xaxes(
        title_text="Number of Training Samples",
        title_standoff=5,
        row=1,
        col=2,
        type="log",
    )
    fig.update_yaxes(title_text="", title_standoff=0, row=1, col=2, type="log")

    # Configure y-axis for log scale
    yaxis_config = {
        "type": "log",
        "showgrid": True,
        "dtick": 1,
        "exponentformat": "power",
        "showexponent": "all",
        "minor": dict(showgrid=True, gridcolor="#eee"),
    }
    yaxis2_config = {
        "type": "log",
        "showgrid": True,
        "dtick": 1,
        "exponentformat": "power",
        "showexponent": "all",
        "minor": dict(showgrid=True, gridcolor="#eee"),
    }

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        margin=dict(l=10, r=0, b=0, t=20),
        width=width,
        height=height,
        yaxis=yaxis_config,
        yaxis2=yaxis2_config,
        legend=dict(
            x=x2 - 0.08,
            y=0.999,
            xanchor="center",
            yanchor="top",
            orientation="v",
            bgcolor="rgba(255,255,255,0.6)",
            font=dict(size=LEGEND_FONT_SIZE),
        ),
    )

    # Increase line width for markers
    fig.update_traces(marker=dict(line=dict(width=1, color="white")))

    # Increase global font sizes
    fig.update_xaxes(
        tickfont=dict(size=AXES_FONT_SIZE), title_font=dict(size=AXES_TITLE_FONT_SIZE)
    )
    fig.update_yaxes(
        tickfont=dict(size=AXES_FONT_SIZE), title_font=dict(size=AXES_TITLE_FONT_SIZE)
    )
    fig.update_annotations(font=dict(size=ANNOTATION_FONT_SIZE))

    # Set subplot title fonts
    for ann in fig.layout.annotations:
        if ann.text in ("Energy", "Force"):
            ann.update(font=dict(size=TITLE_FONT_SIZE))

    # Add subplot panel labels (a, b)
    fig.add_annotation(
        x=dom1[0],
        y=0.999,
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
        y=0.999,
        xref="paper",
        yref="paper",
        text="<b>b</b>",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=ANNOTATION_BOLD_FONT_SIZE),
    )

    # Add y-axis title for right subplot
    fig.add_annotation(
        x=max(dom2[0] - 0.019, 0.0),
        y=0.5,
        xref="paper",
        yref="paper",
        text="Force MAE (validation)",
        textangle=-90,
        showarrow=False,
        xanchor="center",
        yanchor="middle",
        font=dict(size=AXES_TITLE_FONT_SIZE),
    )

    output_path = Path(base_dir) / "datascaling_energy_force.png"
    fig.write_image(output_path, width=width, height=height, scale=2)
    print(f"\nCombined plot saved to \n {output_path}")
