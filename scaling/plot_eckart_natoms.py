import pandas as pd
import wandb
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm


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

base_dir = "scaling/plots/natoms_scaling"
os.makedirs(base_dir, exist_ok=True)

force_download = True

# Check if parquet file exists
parquet_file = "wandb_natoms_scaling.parquet"

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
        if (
            "split" not in run.config["training"]
            or run.config["training"]["split"] != "size"
        ):
            continue
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_dict = run.summary._json_dict
        # Keep only columns containing "val"
        summary_dict = {k: v for k, v in summary_dict.items() if "val" in k}

        if "eval_8-Loss E" in summary_dict:
            # old runs
            continue
        if "val-MAE Val1 Eckart" not in summary_dict:
            # old runs
            continue

        # # try to find minimal value in history
        # # Returns: An iterable collection over history records (dict)
        # keys = [k for k in summary_dict.keys()]
        # print("keys", keys)
        # history = run.scan_history(keys=keys)
        # tqdm.write("Got history")
        # print("history", history)
        # # Compute minima per metric column
        # min_history = {}
        # for c in keys:
        #     c_values = [row[c] for row in history]
        #     print("c_values", c_values)
        #     min_history[c] = float(np.nanmin(c_values))
        #     if np.nanmin(c_values) is not None:
        #         print(f"-> Min value for {c}: {min_history[c]}")
        #     else:
        #         print(f"-> No non-NaN values found for {c}")
        
        # # over write last value with min value
        # summary_dict.update(min_history)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_dict = {k: v for k, v in run.config.items() if not k.startswith("_")}
        # Recursively flatten dict
        config_dict = flatten_dict(config_dict)
        # Keep only columns containing "weight" or "split"
        config_dict = {
            k: v for k, v in config_dict.items() if "weight" in k or "split" in k or "equal_samples_per_size" in k
        }

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

loss_labels_axis = {
    "Loss E": "Energy MAE [eV]", "Loss F": "Force MAE [eV/A]", "MAE Hessian": "Hessian MAE [eV/A^2]"
}
loss_labels = {
    "Loss E": "Energy MAE", "Loss F": "Force MAE", "MAE Hessian": "Hessian MAE"
}

# Filter to keep only columns containing loss types
for loss_type in ["Loss E", "Loss F", "MAE Hessian"]:
    df = runs_df.copy()
    matching_cols = [col for col in df.columns if loss_type in col]

    # Keep only loss-related columns plus name and hessian weight
    columns_to_keep = ["name", "training.splitsize", "training.hessian_loss_weight", "training.equal_samples_per_size"] + matching_cols
    df = df[columns_to_keep]
    

    print(f"\nDataFrame shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    # map training.equal_samples_per_size nan to False
    df["training.equal_samples_per_size"] = (
        df["training.equal_samples_per_size"].fillna(False).infer_objects(copy=False)
    )

    ##########################################################
    # Plot evaluation metrics
    """Plot eval-{x}-{loss_type} for x from 0 to 20, with each method as a separate line.
    Only plot lines with hessian_loss_weight > 0
    """
    # Filter to only include rows where training.hessian_loss_weight > 0
    df_ef = df[df["training.hessian_loss_weight"] > 0]

    plt.figure(figsize=(12, 8))

    # Find all eval columns for this loss type
    eval_cols = [col for col in df_ef.columns if "eval_" in col and loss_type in col]

    if not eval_cols:
        print(f"No eval columns found for {loss_type}")
        print(df_ef.columns)
        continue

    # Extract x values from column names (eval_{x}-{loss_type})
    x_values = []
    for col in eval_cols:
        try:
            # Extract number between eval_x- and -{loss_type}
            x_val = int(col.split("eval_")[1].split("-")[0])
            x_values.append(x_val)
        except (IndexError, ValueError):
            continue

    if not x_values:
        print(f"Could not extract x values for {loss_type}")
        continue

    # Sort columns by x value
    sorted_cols = sorted(zip(x_values, eval_cols))
    x_vals = [x for x, _ in sorted_cols]
    sorted_eval_cols = [col for _, col in sorted_cols]

    # Prepare data for seaborn
    plot_data = []
    for idx, row in df_ef.iterrows():
        splitsize = row.get("training.splitsize", f"Split {idx}")
        for i, col in enumerate(sorted_eval_cols):
            val = row[col]
            if pd.notna(val) and val is not None:
                plot_data.append(
                    {
                        "X Value": x_vals[i],
                        "Value": float(val),
                        "Split Size": splitsize,
                        "Method": row["name"],
                        "Equal Samples Per Size": bool(row["training.equal_samples_per_size"]),
                    }
                )

    if plot_data:
        plot_df = pd.DataFrame(plot_data)
        plot_df = plot_df[~plot_df["Equal Samples Per Size"]]
        # Sort by split size for consistent ordering
        plot_df = plot_df.sort_values("Split Size", ascending=False)
        sns.lineplot(
            data=plot_df,
            x="X Value",
            y="Value",
            # hue='Split Size',
            hue="Method",
            marker="o",
            linewidth=2,
            markersize=4,
            palette="viridis",
        )

    plt.xlabel("Number of atoms during validation")
    plt.ylabel(f"{loss_labels_axis[loss_type]}")
    plt.title(f"{loss_labels[loss_type]} per atom size")
    plt.legend(title="Training Atoms", loc="upper left", frameon=True, edgecolor="none")
    plt.grid(True, alpha=0.3)
    plt.xlim(8, 20)
    plt.tight_layout(pad=0.0)

    # Save the plot
    filename = os.path.join(
        base_dir, f"natoms_eval_{loss_type.replace(' ', '_').lower()}_plot.png"
    )
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {filename}")
    plt.cla()
    plt.clf()
    plt.close()
    
    
    ##########################################################
    # Plot only specific split sizes
    # training.hessian_loss_weight > 0 as solid lines
    # training.hessian_loss_weight < 0.1 as dashed lines
    # color by "Split Size", label by "Method"
    
    plt.figure(figsize=(12, 8))
    
    # Filter for specific split sizes
    target_split_sizes = [10, 14, 18, 20]  # Adjust these values as needed
    df_filtered = df[df["training.splitsize"].isin(target_split_sizes)]
    
    if df_filtered.empty:
        print(f"No data found for target split sizes {target_split_sizes}")
        continue
    
    # Use all available x values instead of filtering
    eval_cols = [col for col in df_filtered.columns if "eval_" in col and loss_type in col]
    if not eval_cols:
        print(f"No eval columns found for {loss_type}")
        continue
    
    # Extract x values from column names
    x_values = []
    for col in eval_cols:
        try:
            x_val = int(col.split("eval_")[1].split("-")[0])
            x_values.append(x_val)
        except (IndexError, ValueError):
            continue
    
    if not x_values:
        print(f"Could not extract x values for {loss_type}")
        continue
    
    # Sort columns by x value
    sorted_cols = sorted(zip(x_values, eval_cols))
    x_vals = [x for x, _ in sorted_cols]
    sorted_eval_cols = [col for _, col in sorted_cols]
    
    # Prepare data for both hessian > 0 and hessian == 0
    plot_data = []
    
    # Data for hessian > 0 (solid lines)
    df_hessian_positive = df_filtered[df_filtered["training.hessian_loss_weight"] > 0]
    for idx, row in df_hessian_positive.iterrows():
        splitsize = row.get("training.splitsize", f"Split {idx}")
        for i, col in enumerate(sorted_eval_cols):
            val = row[col]
            if pd.notna(val) and val is not None:
                plot_data.append(
                    {
                        "X Value": x_vals[i],
                        "Value": float(val),
                        "Split Size": splitsize,
                        "Method": row["name"],
                        "Line Style": "Solid",
                        "Equal Samples Per Size": bool(row["training.equal_samples_per_size"]),
                    }
                )
    
    # Data for hessian < 0.1 (dashed lines)
    df_hessian_zero = df_filtered[df_filtered["training.hessian_loss_weight"] < 0.1]
    for idx, row in df_hessian_zero.iterrows():
        splitsize = row.get("training.splitsize", f"Split {idx}")
        for i, col in enumerate(sorted_eval_cols):
            val = row[col]
            if pd.notna(val) and val is not None:
                plot_data.append(
                    {
                        "X Value": x_vals[i],
                        "Value": float(val),
                        "Split Size": splitsize,
                        "Method": row["name"],
                        "Line Style": "Dashed",
                        "Equal Samples Per Size": bool(row["training.equal_samples_per_size"]),
                    }
                )
    
    if plot_data:
        plot_df = pd.DataFrame(plot_data)
        
        # Plot solid lines first (hessian > 0) - color by Split Size only
        solid_df = plot_df[plot_df["Line Style"] == "Solid"]
        solid_df = solid_df[~solid_df["Equal Samples Per Size"]]
        if not solid_df.empty:
            sns.lineplot(
                data=solid_df,
                x="X Value",
                y="Value",
                hue="Split Size",
                marker="o",
                linewidth=2,
                markersize=4,
                palette="viridis",
                linestyle="-"
            )
        
        # Plot dashed lines (hessian == 0) with correct colors but single legend entry
        dashed_df = plot_df[plot_df["Line Style"] == "Dashed"]
        dashed_df = dashed_df[~dashed_df["Equal Samples Per Size"]]
        print("Number of dashed lines:", len(dashed_df))
        if not dashed_df.empty:
            # Get unique split sizes for dashed lines to assign colors
            unique_split_sizes = dashed_df["Split Size"].unique()
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_split_sizes)))
            split_size_to_color = dict(zip(unique_split_sizes, colors))
            
            # Plot each dashed line with its correct color
            for split_size in unique_split_sizes:
                split_data = dashed_df[dashed_df["Split Size"] == split_size]
                plt.plot(
                    split_data["X Value"],
                    split_data["Value"],
                    linestyle="--",
                    color=split_size_to_color[split_size],
                    linewidth=2,
                    marker="o",
                    markersize=4,
                    label=None  # No label for individual lines
                )
            
            # Add a single dummy "E+F" entry to the legend
            plt.plot([], [], linestyle="--", color="gray", linewidth=2, label="E+F")
    
    plt.xlabel("Number of atoms during validation")
    plt.ylabel(f"{loss_labels_axis[loss_type]}")
    plt.xlim(x_vals[0], x_vals[-1])
    plt.title(f"{loss_labels[loss_type]} per atom size (trained w/wo Hessian)")
    plt.legend(title="Training Atoms", frameon=True, edgecolor="none")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(pad=0.0)
    
    # Save the plot
    filename = os.path.join(
        base_dir, f"natoms_eval_{loss_type.replace(' ', '_').lower()}_ef_plot.png"
    )
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {filename}")
    plt.cla()
    plt.clf()
    plt.close()

    ##########################################################
    # Plot only specific split sizes
    # training.equal_samples_per_size == True as solid lines
    # training.equal_samples_per_size == False as dashed lines
    # color by "Split Size", label by "Method"
    # only plot the sizes that exist for equal_samples_per_size == True

    plt.figure(figsize=(12, 8))

    # Determine which split sizes exist for equal_samples_per_size == True
    df_eq_true = df[df["training.equal_samples_per_size"]]
    # Filter to only include rows where training.hessian_loss_weight > 0
    df_eq_true = df_eq_true[df_eq_true["training.hessian_loss_weight"] > 0]
    target_split_sizes = sorted(df_eq_true["training.splitsize"].dropna().unique().tolist())

    if len(target_split_sizes) == 0:
        print("No data found where training.equal_samples_per_size == True")
        continue

    # Filter for only those split sizes
    df_filtered = df[df["training.splitsize"].isin(target_split_sizes)]
    # Filter to only include rows where training.hessian_loss_weight > 0
    df_filtered = df_filtered[df_filtered["training.hessian_loss_weight"] > 0]

    # Use all available x values instead of filtering
    eval_cols = [col for col in df_filtered.columns if "eval_" in col and loss_type in col]
    if not eval_cols:
        print(f"No eval columns found for {loss_type}")
        continue

    # Extract x values from column names
    x_values = []
    for col in eval_cols:
        try:
            x_val = int(col.split("eval_")[1].split("-")[0])
            x_values.append(x_val)
        except (IndexError, ValueError):
            continue

    if not x_values:
        print(f"Could not extract x values for {loss_type}")
        continue

    # Sort columns by x value
    sorted_cols = sorted(zip(x_values, eval_cols))
    x_vals = [x for x, _ in sorted_cols]
    sorted_eval_cols = [col for _, col in sorted_cols]

    # Prepare data for both equal_samples_per_size True (dotted) and False (solid)
    plot_data = []

    # Dotted: equal_samples_per_size == True
    for idx, row in df_filtered[df_filtered["training.equal_samples_per_size"]].iterrows():
        splitsize = row.get("training.splitsize", f"Split {idx}")
        for i, col in enumerate(sorted_eval_cols):
            val = row[col]
            if pd.notna(val) and val is not None:
                plot_data.append(
                    {
                        "X Value": x_vals[i],
                        "Value": float(val),
                        "Split Size": splitsize,
                        "Method": row["name"],
                        "Line Style": "Dotted",
                    }
                )

    # Solid: equal_samples_per_size == False (but only those split sizes that exist for True)
    for idx, row in df_filtered[~df_filtered["training.equal_samples_per_size"]].iterrows():
        splitsize = row.get("training.splitsize", f"Split {idx}")
        for i, col in enumerate(sorted_eval_cols):
            val = row[col]
            if pd.notna(val) and val is not None:
                plot_data.append(
                    {
                        "X Value": x_vals[i],
                        "Value": float(val),
                        "Split Size": splitsize,
                        "Method": row["name"],
                        "Line Style": "Solid",
                    }
                )

    if plot_data:
        plot_df = pd.DataFrame(plot_data)

        # Plot solid lines first (Unequal samples) - color by Split Size only
        solid_df = plot_df[plot_df["Line Style"] == "Solid"]
        if not solid_df.empty:
            sns.lineplot(
                data=solid_df,
                x="X Value",
                y="Value",
                hue="Split Size",
                marker="o",
                linewidth=2,
                markersize=4,
                palette="viridis",
                linestyle="-",
            )

        # Plot dotted lines (Equal samples) with matching colors but single legend entry for linestyle
        dotted_df = plot_df[plot_df["Line Style"] == "Dotted"]
        if not dotted_df.empty:
            unique_split_sizes = dotted_df["Split Size"].unique()
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_split_sizes)))
            split_size_to_color = dict(zip(unique_split_sizes, colors))

            for split_size in unique_split_sizes:
                split_data = dotted_df[dotted_df["Split Size"] == split_size]
                plt.plot(
                    split_data["X Value"],
                    split_data["Value"],
                    linestyle=":",
                    color=split_size_to_color[split_size],
                    linewidth=2,
                    marker="o",
                    markersize=4,
                    label=None,
                )

            # Add a single dummy legend entry to indicate dotted = equal samples
            plt.plot([], [], linestyle=":", color="gray", linewidth=2, label="Stratified")

    plt.xlabel("Number of atoms during validation")
    plt.ylabel(f"{loss_labels_axis[loss_type]}")
    plt.xlim(x_vals[0], x_vals[-1])
    plt.title(f"{loss_labels[loss_type]} per size (random vs equal-samples-per-size splits)")
    plt.legend(title="Training Atoms", frameon=True, edgecolor="none")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(pad=0.0)

    # Save the plot
    filename = os.path.join(
        base_dir, f"natoms_eval_{loss_type.replace(' ', '_').lower()}_equalsamplesizes_plot.png"
    )
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {filename}")
    plt.cla()
    plt.clf()
    plt.close()

    ##########################################################
    # Plot "val-{loss_type}" vs "training.splitsize"
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=df_ef, x="training.splitsize", y=f"val-{loss_type}", marker="o", linewidth=0
    )
    plt.xlabel("Max number of atoms during training")
    plt.ylabel(f"{loss_labels_axis[loss_type]}")
    plt.title(f"{loss_labels[loss_type]} per atom size during training")
    # plt.legend(title='Training Atoms', loc='upper left', frameon=True, edgecolor="none")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(pad=0.0)

    # Save the plot
    filename = os.path.join(
        base_dir, f"natoms_val_{loss_type.replace(' ', '_').lower()}_plot.png"
    )
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {filename}")
    plt.cla()
    plt.clf()
    plt.close()
    
