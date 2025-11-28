import pandas as pd
import plotly.graph_objects as go
from hip.colours import METHOD_TO_COLOUR

# Define color mapping for models (using colors from speed_comparison_extended.py)
color_map = {
    "mace-off": "#cfcbc5",  #'#cfcbc5',
    "mace-omol": "#d6c8e8",
    "esen": "#ff8b94",  # finite_difference_bz32 color
    "uma": "#68c4af",  # forward_pass color
    "HIP": "#d96001",  # prediction color (HIP)
    # #5a5255 # #636EFA
    "Equiformer": "#5a5255",  # METHOD_TO_COLOUR.get('eqv2', '#5a5255'),
    "Leftnet": "#444f97",
}

# rename some model names
rename_map = {
    "mace_omol_extra_large": "MACE-OMol XL",
    "esen_sm-conserving": "eSEN-CF Small",
    "uma_s": "UMA Small",
    "mace_off_large": "MACE-OFF Large",
    "mace_off_small": "MACE-OFF Small",
    "mace_off_medium": "MACE-OFF Medium",
}


# Assign colors based on model names (case-insensitive)
def get_color(model_name):
    model_lower = model_name.lower()
    for key, color in color_map.items():
        if key.lower() in model_lower:
            return color
    return "#7f7f7f"  # default gray for unmatched models


# Process both CSV files
csv_files = [
    ("force_mae.csv", "forces_mae", "Force MAE [eV/Å]", "force_mae.png"),
    ("hessian_mae.csv", "hessian_mae", "Hessian MAE", "hessian_mae.png"),
]

for csv_file, suffix, xaxis_title, output_filename in csv_files:
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Filter out MIN and MAX columns, keep only the main columns
    main_columns = [
        col
        for col in df.columns
        if not col.endswith("__MIN") and not col.endswith("__MAX")
    ]
    df_filtered = df[main_columns]

    # Extract model names from column names (remove " - {suffix}" suffix)
    model_names = [
        col.replace(f" - {suffix}", "") for col in main_columns if col != "Step"
    ]
    values = df_filtered.iloc[0][main_columns[1:]].values

    # Apply rename map
    model_names = [rename_map.get(name, name) for name in model_names]

    bar_colors = [get_color(name) for name in model_names]

    # Sort by values (high to low)
    sorted_data = sorted(
        zip(model_names, values, bar_colors), key=lambda x: x[1], reverse=False
    )
    model_names, values, bar_colors = [list(x) for x in zip(*sorted_data)]

    # Create horizontal bar plot
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=values,
            y=model_names,
            orientation="h",
            text=[f"{v:.4f}" for v in values],
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
                yshift=15,  # Move above the bar
                font=dict(size=10),
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
