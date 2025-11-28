import pandas as pd
import plotly.graph_objects as go

# Import data from table.py
data = {
    "Hessian": [
        "AD",
        "AD",
        "AD",
        "AD",
        "Predicted",
        "Predicted",
    ],
    "Model": [
        "AlphaNet",
        "LEFTNet-CF",
        "LEFTNet-DF",
        "EquiformerV2",
        "HIP-EquiformerV2",
        "HIP-EquiformerV2 (end-to-end)",
    ],
    "Hessian_eV_A2": [0.385, 0.150, 0.197, 0.074, 0.030, 0.020],
    "Eigenvalues_eV_A2": [2.871, 0.684, 0.669, 0.242, 0.063, 0.041],
    "CosSim_evec1": [0.900, 0.943, 0.930, 0.915, 0.982, 0.982],
    "eval1_eV_A2": [0.284, 0.112, 0.148, 0.097, 0.031, 0.031],
    "Time_ms": [767.0, 1110.7, 341.3, 633.0, 38.5, 31.4],
}

df_hessian_ef = pd.DataFrame(
    [
        {"Model": "AlphaNet (E-F)", "Hessian_eV_A2": 0.433},
        {"Model": "EquiformerV2 (E-F)", "Hessian_eV_A2": 2.252},
        {"Model": "LEFTNet-CF (E-F)", "Hessian_eV_A2": 0.366},
        {"Model": "LEFTNet-DF (E-F)", "Hessian_eV_A2": 1.648},
    ]
)

df = pd.DataFrame(data)

# Define color mapping for models (using colors from plot_mae.py)
color_map = {
    "HIP": "#d96001",  # prediction color (HIP)
    "Equiformer": "#5a5255",
    "Leftnet": "#444f97",
    "AlphaNet": "#7f7f7f",  # default gray for unmatched
}


# Assign colors based on model names (case-insensitive)
def get_color(model_name):
    model_lower = model_name.lower()
    for key, color in color_map.items():
        if key.lower() in model_lower:
            return color
    return "#7f7f7f"  # default gray for unmatched models


# Define metrics to plot with their axis titles and output filenames
metrics = [
    ("Hessian_eV_A2", "Hessian MAE [eV/Å²]", "hessian_mae_table.png"),
    ("Eigenvalues_eV_A2", "Eigenvalues MAE [eV/Å²]", "eigenvalues_mae_table.png"),
    ("CosSim_evec1", "Cosine Similarity (eigenvector 1)", "cossim_evec1_table.png"),
    ("eval1_eV_A2", "Eigenvalue 1 MAE [eV/Å²]", "eval1_mae_table.png"),
    ("Time_ms", "Time [ms]", "time_table.png"),
]

for metric_col, xaxis_title, output_filename in metrics:
    # Extract model names and values
    model_names = df["Model"].values
    values = df[metric_col].values

    bar_colors = [get_color(name) for name in model_names]

    # Sort by values (low to high, except for CosSim which should be high to low since higher is better)
    reverse = metric_col == "CosSim_evec1"
    sorted_data = sorted(
        zip(model_names, values, bar_colors), key=lambda x: x[1], reverse=reverse
    )
    model_names, values, bar_colors = [list(x) for x in zip(*sorted_data)]

    # Create horizontal bar plot
    fig = go.Figure()

    # Format text based on metric type
    if metric_col == "Time_ms":
        text_values = [f"{v:.0f}" for v in values]
    elif metric_col == "CosSim_evec1":
        text_values = [f"{v:.2f}" for v in values]
    else:
        text_values = [f"{v:.3f}" for v in values]

    fig.add_trace(
        go.Bar(
            x=values,
            y=model_names,
            orientation="h",
            text=text_values,
            textposition="outside",
            marker_color=bar_colors,
            width=0.5,
        )
    )

    # Add model name labels above bars
    annotations = []
    for i, (name, val) in enumerate(zip(model_names, values)):
        annotations.append(
            dict(
                x=0,  # Position at the start of the bar
                y=name,  # Use the model name as y position (categorical)
                text=name,
                showarrow=False,
                xref="x",
                yref="y",
                xanchor="left",
                yanchor="bottom",
                yshift=20,  # Move above the bar
                font=dict(size=12),
            )
        )

    max_value = max(values)
    fig.update_layout(
        xaxis_title=xaxis_title,
        xaxis={
            "gridcolor": "#e0e0e0",
            "showgrid": True,
            "range": [0, max_value * 1.15],  # Add 15% padding for text labels
        },
        yaxis={"showgrid": False, "showticklabels": False},
        plot_bgcolor="white",
        height=600,
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=30),
        annotations=annotations,
    )

    # write to png
    fname = f"plots/{output_filename}"
    fig.write_image(fname, width=700, height=600, scale=2)
    print(f"Plot saved to {fname}")
