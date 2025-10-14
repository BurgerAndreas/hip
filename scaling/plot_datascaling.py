import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import re

# Set seaborn style for poster
sns.set_style("whitegrid")
# sns.set_palette("husl")
sns.set_context("poster")

base_dir = "plots/datascaling"
os.makedirs(base_dir, exist_ok=True)

# https://wandb.ai/andreas-burger/hip/workspace?nw=fdkkquw19gl

for losstype in ["Loss E", "Loss F", "MAE Hessian"]:
    human_name = {"Loss E": "Energy", "Loss F": "Force", "MAE Hessian": "Hessian"}[
        losstype
    ]

    # Load the CSV file
    df = pd.read_csv(f"scaling/wandb_datascaling_loss_{human_name.lower()}.csv")

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
    ef_data = plot_data[plot_data["ef"] == True]
    efh_data = plot_data[plot_data["ef"] == False]

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

    ###############################################################33
    # Marginal Improvement Threshold
    ###############################################################33

    # Fit polynomial to energy-force-hessian points and add marginal improvement line
    if len(efh_data) > 2:  # Need at least 3 points for degree 2 polynomial
        # Sort data by dataset size for polynomial fitting
        sorted_data = efh_data.sort_values("Dataset size")
        x_fit = np.log10(sorted_data["Dataset size"].values)
        y_fit = np.log10(sorted_data["Min_Value"].values)

        # Fit degree 2 polynomial
        poly_coeffs = np.polyfit(x_fit, y_fit, 2)
        poly_func = np.poly1d(poly_coeffs)

        # Calculate slope threshold: less than 1% error improvement for 10x data increase
        # In log space: slope = (log(y2) - log(y1)) / (log(x2) - log(x1))
        # For 10x increase: slope = (log(y2) - log(y1)) / log(10)
        # 1% improvement: y2 = 0.99 * y1, so log(y2) - log(y1) = log(0.99)
        # slope_threshold = log(0.99) / log(10) ≈ -0.0044
        slope_threshold = np.log(0.99) / np.log(10)

        # Plot polynomial fit
        x_plot = np.linspace(x_fit.min(), x_fit.max(), 100)
        y_plot = poly_func(x_plot)
        plt.plot(
            10**x_plot,
            10**y_plot,
            color="C1",  # Same color as second scatter plot (EFH)
            alpha=0.5,
            linewidth=2,
            label="Quadratic fit",
        )

        # Add marginal improvement threshold line
        # Find where slope becomes less negative than threshold
        x_range = np.linspace(x_fit.min(), x_fit.max(), 1000)
        y_range = poly_func(x_range)
        slopes = np.gradient(y_range, x_range)

        # Find intersection point where slope equals threshold
        slope_diff = slopes - slope_threshold
        sign_changes = np.where(np.diff(np.sign(slope_diff)))[0]

        if len(sign_changes) > 0:
            # Find the first point where slope becomes less negative than threshold
            threshold_idx = sign_changes[0]
            threshold_x = x_range[threshold_idx]
            threshold_y = y_range[threshold_idx]

            # Plot vertical line at threshold point
            plt.axvline(
                x=10**threshold_x,
                color="orange",
                linestyle=":",
                alpha=0.8,
                label=r"Slope < 10x data, 1% MAE",
            )

    # Add horizontal line at 5% above the lowest loss
    min_loss = plot_data["Min_Value"].min()
    max_loss = plot_data["Max_Value"].max()
    threshold = min_loss * 1.05
    plt.axhline(
        y=threshold,
        color="darkgray",
        linestyle="--",
        alpha=0.7,
        label=r"5% MAE$_{\text{min}}$",
    )
    threshold = (max_loss - min_loss) * 0.01 + min_loss
    plt.axhline(
        y=threshold,
        color="lightgray",
        linestyle="--",
        alpha=0.7,
        label=r"1% MAE$_{\text{min-max}}$",
    )

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
