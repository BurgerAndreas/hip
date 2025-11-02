#!/usr/bin/env python3
"""
Plot histogram of number of atoms for t1x train and validation datasets.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Set seaborn style for poster
sns.set_style("whitegrid")
# sns.set_palette("husl")
sns.set_context("poster")

base_dir = "scaling/plots/natoms_histogram"
os.makedirs(base_dir, exist_ok=True)


def plot_natoms_histogram(which_val="t1x"):
    """Plot histogram comparing natoms distribution between train and val datasets."""

    # Read the CSV files
    print("Loading datasets...")
    train_df = pd.read_csv(
        "/ssd/Code/hip/metadata/dataset_metadata_ts1x_hess_train_big.csv"
    )
    if which_val == "t1x":
        val_df = pd.read_csv("/ssd/Code/hip/metadata/dataset_metadata_ts1x-val.csv")
    elif which_val == "rgd1":
        val_df = pd.read_csv("/ssd/Code/hip/metadata/dataset_metadata_RGD1.csv")
    else:
        raise ValueError(f"Invalid validation dataset: {which_val}")

    print(f"Train dataset: {len(train_df)} samples")
    print(f"Validation dataset: {len(val_df)} samples")

    # Get natoms data
    train_natoms = train_df["natoms"].values
    val_natoms = val_df["natoms"].values

    # Print basic statistics
    print(
        f"\nTrain natoms - Min: {train_natoms.min()}, Max: {train_natoms.max()}, Mean: {train_natoms.mean():.2f}"
    )
    print(
        f"Val {which_val.upper()} natoms - Min: {val_natoms.min()}, Max: {val_natoms.max()}, Mean: {val_natoms.mean():.2f}"
    )

    # Create bin edges that align with integers
    min_natoms = int(min(train_natoms.min(), val_natoms.min()))
    max_natoms = int(max(train_natoms.max(), val_natoms.max()))
    bin_edges = np.arange(min_natoms - 0.5, max_natoms + 1.5, 1)

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot histograms with integer-aligned bins
    plt.hist(
        train_natoms,
        bins=bin_edges,
        alpha=0.5,
        label=f"Train (n={len(train_natoms)})",
        density=True,
        histtype="bar",
        edgecolor="black",
        linewidth=0.5,
    )
    plt.hist(
        val_natoms,
        bins=bin_edges,
        alpha=0.5,
        label=f"Validation {which_val.upper()} (n={len(val_natoms)})",
        density=True,
        histtype="bar",
        edgecolor="black",
        linewidth=0.5,
    )

    # Customize the plot
    plt.xlabel("Number of Atoms")
    plt.ylabel("Density")
    plt.title("Distribution of Number of Atoms")
    plt.legend(frameon=True, edgecolor="none")
    plt.grid(True, alpha=0.3)

    # Set x-axis ticks to integers that align with histogram bars
    tick_positions = np.arange(min_natoms, max_natoms + 1, 2)
    plt.xticks(ticks=tick_positions, labels=tick_positions)

    plt.tight_layout(pad=0.0)

    # Save the plot
    output_path = os.path.join(base_dir, f"natoms_histogram_{which_val}_train_val.png")
    plt.savefig(output_path, dpi=500, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")


if __name__ == "__main__":
    plot_natoms_histogram(which_val="t1x")
    plot_natoms_histogram(which_val="rgd1")
