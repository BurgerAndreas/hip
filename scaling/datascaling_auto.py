import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import re
import wandb
from tqdm import tqdm
from datetime import datetime
import yaml

def flatten_dict(d, parent_key="", sep="."):
    """Recursively flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# Set seaborn style for poster
sns.set_style("whitegrid")
# sns.set_palette("husl")
sns.set_context("poster")

base_dir = "plots/datascaling"
os.makedirs(base_dir, exist_ok=True)

# https://wandb.ai/andreas-burger/hip/workspace?nw=fdkkquw19gl

force_download = False

# Check if parquet file exists
parquet_file = "wandb_datascaling.parquet"

if os.path.exists(parquet_file) and not force_download:
    print("Loading existing parquet file...")
    runs_df = pd.read_parquet(parquet_file)
else:
    print("Downloading data from wandb...")
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs("andreas-burger/hip")

    list_of_runs = []
    for run in tqdm(runs):
        # filter by training.split=random
        if "split" not in run.config.get("training", {}): 
            continue
        if run.config["training"]["split"] != "random":
            continue
        # if pltrainer.limit_val_batches == 80, skip
        if run.config["pltrainer"]["limit_val_batches"] == 80:
            continue
        
        # # Check runtime - skip runs that have been running for less than 24 hours
        # # Calculate runtime in hours
        # if run.state == "running":
        #     # For running jobs, calculate time since start
        #     created_at = datetime.fromisoformat(run.created_at.replace('Z', '+00:00'))
        #     runtime_hours = (datetime.now(created_at.tzinfo) - created_at).total_seconds() / 3600
        #     if runtime_hours < 24:
        #         continue
        # else:
        #     # For finished jobs, check if they ran for at least 24 hours
        #     run.created_at
        #     else:
        #         # If we can't determine runtime, skip the run
        #         continue
            
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_dict = run.summary._json_dict
        # Keep only columns containing "val"
        summary_dict = {k: v for k, v in summary_dict.items() if "val" in k}
        
        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_dict = {k: v for k, v in run.config.items() if not k.startswith("_")}
        # Recursively flatten dict
        config_dict = flatten_dict(config_dict)
        # Keep relevant config columns
        config_dict = {
            k: v for k, v in config_dict.items() if any(keyword in k for keyword in ["weight", "split", "dataset", "size"])
        }

        if "val-Loss E" not in summary_dict:
            # old runs or runs without required metrics
            continue
        
        print("Found:", run.name)

        # Get history for minimum values
        keys = ['val-Loss E', 'val-Loss F']
        if config_dict.get("training.hessian_loss_weight", 0) > 0:
            keys += ['val-MAE Hessian']
            
        # Get history data
        history = run.history(samples=5_000, keys=keys, x_axis='_step', pandas=True, stream='default')
        # get the minimum value for each key using pandas
        min_history = {k: float(history[k].min()) for k in keys if k in history.columns}
        
        # Overwrite summary with min values
        summary_dict.update(min_history)

        # .name is the human-readable name of the run.
        list_of_runs.append(
            {
                "name": run.name,
                **config_dict,
                **summary_dict,
            }
        )

    runs_df = pd.DataFrame(list_of_runs)

    # Save the downloaded data
    runs_df.to_parquet(parquet_file)
    print("Data saved to parquet file.")

# List the columns
print("DataFrame columns:")
print(runs_df.columns.tolist())

for losstype in ["Loss E", "Loss F", "MAE Hessian"]:
    human_name = {"Loss E": "Energy", "Loss F": "Force", "MAE Hessian": "Hessian"}[
        losstype
    ]
    

    # Work with wandb data
    df = runs_df.copy()

    
    # Print basic info about the dataset
    print(f"\nProcessing {losstype} data...")
    print("Original dataset shape:", df.shape)
    print("Column names:")
    print(df.columns.tolist())

    # Filter for the specific loss type
    loss_col = f"val-{losstype}"
    if loss_col not in df.columns:
        print(f"Column {loss_col} not found, skipping {losstype}")
        continue
        
    # Extract statistics for each method
    stats_data = []
    for idx, row in df.iterrows():
        method_name = row["name"]
        loss_value = row[loss_col]
        
        if pd.notna(loss_value):
            # Check if method ends with "EF" (Energy-Force only)
            # is_ef = method_name.endswith("EF")
            is_ef = row["training.hessian_loss_weight"] < 0.1
            
            # Extract dataset size from the method name
            # Find the numeric part at the beginning (including scientific notation)
            match = re.match(r"^([0-9.]+(?:e[+-]?[0-9]+)?)", method_name)
            if match:
                dataset_size = float(match.group(1))
            else:
                dataset_size = None

            stats_data.append(
                {
                    "Method": method_name,
                    "Last_Value": loss_value,
                    "Min_Value": loss_value,  # Already min from history
                    "Max_Value": loss_value,  # Same as min since we took min from history
                    "ef": is_ef,
                    "Dataset size": dataset_size,
                }
            )

    # Create new dataframe with methods as rows and statistics as columns
    df_stats = pd.DataFrame(stats_data)
    df_stats = df_stats.set_index("Method")

    # Save the filtered data
    csvfname = os.path.join(base_dir, f"loss_{human_name.lower()}.csv")
    # df_stats.to_csv(csvfname, index=False)
    # print(f"Filtered data saved to '{csvfname}'")

    print("\nStatistics for each method:")
    print(df_stats)

    # Filter out rows where dataset size is not None
    plot_data = df_stats.dropna(subset=["Dataset size"])
    
    # plot_data = df_stats[df_stats["training.hessian_loss_weight"] > 0.1]
    # ef_data = df_stats[df_stats["training.hessian_loss_weight"] < 0.1]
    
    # Create log-log plot of min_value vs dataset size
    plt.figure(figsize=(10, 8))

    # Create scatter plot with different colors for EF vs non-EF methods
    ef_data = plot_data[plot_data["ef"]]
    efh_data = plot_data[~plot_data["ef"]]
    
    # if there is two runs with the x value, pick the one with the lowest y value
    ef_data = ef_data.sort_values("Dataset size").groupby("Dataset size").first().reset_index()
    efh_data = efh_data.sort_values("Dataset size").groupby("Dataset size").first().reset_index()

    # if with_ef:
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
