import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

from hip.ff_lmdb import LmdbDataset
from hip.path_config import fix_dataset_path
from hip.t1x_dft_dataloader import get_molecular_reference_energy


def load_atomization_energies(dataset_path, dataset_name, max_samples=10000):
    """Load atomization energies from a dataset.

    Args:
        dataset_path: Path to the LMDB dataset
        dataset_name: Name of the dataset (for labeling)
        max_samples: Maximum number of samples to load

    Returns:
        DataFrame with columns: natoms, atomization_energy, dataset
    """
    print(f"\nLoading {dataset_name} from {dataset_path}...")

    dataset = LmdbDataset(fix_dataset_path(dataset_path))
    n_samples = min(max_samples, len(dataset))

    print(f"Loading {n_samples} samples from {dataset_name} (total: {len(dataset)})...")

    data_list = []

    # Sample random indices
    indices = np.random.choice(len(dataset), size=n_samples, replace=False)

    for idx in tqdm(indices, desc=f"Loading {dataset_name}"):
        batch = dataset[idx]

        n_atoms = batch.pos.shape[0]

        # Get atomization energy
        if hasattr(batch, "ae") and batch.ae is not None:
            # T1x dataset has atomization energy directly
            atomization_energy = batch.ae.item()
        elif hasattr(batch, "energy") and batch.energy is not None:
            # RGD1 dataset: compute from energy
            atomic_numbers = batch.z.cpu().numpy().tolist()
            molecular_reference_energy = get_molecular_reference_energy(
                atomic_numbers,
                # fix
                # constant_per_atom=-3.911, #4.6,
                # constant=-0.686,
            )
            energy = batch.energy.item()
            atomization_energy = energy - molecular_reference_energy
        else:
            print(f"Warning: Sample {idx} has neither .ae nor .energy, skipping")
            continue

        data_list.append(
            {
                "natoms": n_atoms,
                "atomization_energy": atomization_energy,
                "dataset": dataset_name,
            }
        )

    df = pd.DataFrame(data_list)
    print(f"Loaded {len(df)} samples from {dataset_name}")
    print(
        f"  Atomization energy range: [{df['atomization_energy'].min():.4f}, {df['atomization_energy'].max():.4f}] eV"
    )
    print(f"  Number of atoms range: [{df['natoms'].min()}, {df['natoms'].max()}]")

    return df


def plot_atomization_energies(df_t1x, df_rgd1, output_dir="plots/eval_horm"):
    """Plot atomization energies vs number of atoms for both datasets."""
    os.makedirs(output_dir, exist_ok=True)

    # Combine dataframes
    df_combined = pd.concat([df_t1x, df_rgd1], ignore_index=True)

    sns.set_theme(style="whitegrid")

    # Create figure with subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

    # Left plot: Scatter with regression lines
    ax1 = axes[0]

    # Plot T1x
    sns.scatterplot(
        data=df_t1x,
        x="natoms",
        y="atomization_energy",
        alpha=0.3,
        s=10,
        label="T1x Train",
        ax=ax1,
    )

    # Plot RGD1
    sns.scatterplot(
        data=df_rgd1,
        x="natoms",
        y="atomization_energy",
        alpha=0.3,
        s=10,
        label="RGD1",
        ax=ax1,
    )

    # Add regression lines
    z_t1x = np.polyfit(df_t1x["natoms"], df_t1x["atomization_energy"], 1)
    p_t1x = np.poly1d(z_t1x)
    x_line_t1x = np.linspace(df_t1x["natoms"].min(), df_t1x["natoms"].max(), 100)
    ax1.plot(
        x_line_t1x,
        p_t1x(x_line_t1x),
        "b--",
        alpha=0.7,
        linewidth=2,
        label=f"T1x fit: y={z_t1x[0]:.3f}x+{z_t1x[1]:.3f}",
    )

    z_rgd1 = np.polyfit(df_rgd1["natoms"], df_rgd1["atomization_energy"], 1)
    p_rgd1 = np.poly1d(z_rgd1)
    x_line_rgd1 = np.linspace(df_rgd1["natoms"].min(), df_rgd1["natoms"].max(), 100)
    ax1.plot(
        x_line_rgd1,
        p_rgd1(x_line_rgd1),
        "r--",
        alpha=0.7,
        linewidth=2,
        label=f"RGD1 fit: y={z_rgd1[0]:.3f}x+{z_rgd1[1]:.3f}",
    )

    ax1.set_xlabel("Number of Atoms")
    ax1.set_ylabel("Atomization Energy (eV)")
    ax1.set_title("Atomization Energy vs Number of Atoms")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right plot: Distribution comparison
    ax2 = axes[1]

    sns.histplot(
        data=df_combined,
        x="atomization_energy",
        hue="dataset",
        bins=50,
        alpha=0.6,
        kde=True,
        ax=ax2,
    )

    ax2.set_xlabel("Atomization Energy (eV)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of Atomization Energies")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = os.path.join(output_dir, "atomization_energy_comparison_t1x_rgd1.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved plot to {output_path}")

    plt.close()

    # Print statistics
    print("\n" + "=" * 60)
    print("ATOMIZATION ENERGY STATISTICS")
    print("=" * 60)

    for name, df in [("T1x Train", df_t1x), ("RGD1", df_rgd1)]:
        print(f"\n{name}:")
        print(f"  Mean: {df['atomization_energy'].mean():.6f} eV")
        print(f"  Std:  {df['atomization_energy'].std():.6f} eV")
        print(f"  Min:  {df['atomization_energy'].min():.6f} eV")
        print(f"  Max:  {df['atomization_energy'].max():.6f} eV")
        print(
            f"  Linear fit: y = {z_t1x[0] if name == 'T1x Train' else z_rgd1[0]:.6f} * natoms + {z_t1x[1] if name == 'T1x Train' else z_rgd1[1]:.6f}"
        )

    # Per-atom statistics
    df_t1x_per_atom = df_t1x.copy()
    df_t1x_per_atom["atomization_energy_per_atom"] = (
        df_t1x_per_atom["atomization_energy"] / df_t1x_per_atom["natoms"]
    )

    df_rgd1_per_atom = df_rgd1.copy()
    df_rgd1_per_atom["atomization_energy_per_atom"] = (
        df_rgd1["atomization_energy"] / df_rgd1["natoms"]
    )

    print("\n" + "=" * 60)
    print("ATOMIZATION ENERGY PER ATOM STATISTICS")
    print("=" * 60)

    for name, df in [("T1x Train", df_t1x_per_atom), ("RGD1", df_rgd1_per_atom)]:
        print(f"\n{name}:")
        print(f"  Mean: {df['atomization_energy_per_atom'].mean():.6f} eV/atom")
        print(f"  Std:  {df['atomization_energy_per_atom'].std():.6f} eV/atom")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare atomization energies between T1x and RGD1"
    )
    parser.add_argument(
        "--t1x_path",
        type=str,
        default="ts1x_hess_train_big.lmdb",
        help="Path to T1x train dataset (default: ts1x_hess_train_big.lmdb)",
    )
    parser.add_argument(
        "--rgd1_path",
        type=str,
        default="RGD1.lmdb",
        help="Path to RGD1 dataset (default: RGD1.lmdb)",
    )
    parser.add_argument(
        "--max_samples",
        "-m",
        type=int,
        default=1000,
        help="Maximum number of samples to load from each dataset (default: 10000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load datasets
    df_t1x = load_atomization_energies(
        args.t1x_path, "T1x Train", max_samples=args.max_samples
    )
    df_rgd1 = load_atomization_energies(
        args.rgd1_path, "RGD1", max_samples=args.max_samples
    )

    # Plot comparison
    plot_atomization_energies(df_t1x, df_rgd1)

    print("\nDone!")
