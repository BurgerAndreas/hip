import argparse
import pandas as pd
import os
import numpy as np
from torch_geometric.loader import DataLoader as TGDataLoader

from hip.ff_lmdb import LmdbDataset
from hip.path_config import fix_dataset_path
from nets.prediction_utils import Z_TO_ATOM_SYMBOL

try:
    from rdkit import Chem

    RDKIT_AVAILABLE = True
except ImportError:
    print("Warning: RDKit not available. SMILES extraction will be skipped.")
    RDKIT_AVAILABLE = False


def generate_smiles_from_coords(atomic_numbers, coordinates, atomic_symbols):
    """
    Generate SMILES string from atomic coordinates using RDKit.

    Args:
        atomic_numbers: Array of atomic numbers
        coordinates: Array of atomic coordinates (N, 3)
        atomic_symbols: List of atomic symbols

    Returns:
        str: SMILES string or None if generation fails
    """
    if not RDKIT_AVAILABLE:
        return None

    try:
        # Create RDKit molecule from coordinates
        mol = Chem.RWMol()

        # Add atoms
        for atomic_num in atomic_numbers:
            mol.AddAtom(Chem.Atom(int(atomic_num)))

        # Add conformer with coordinates
        conf = Chem.Conformer(len(atomic_numbers))
        for i, coord in enumerate(coordinates):
            conf.SetAtomPosition(i, (float(coord[0]), float(coord[1]), float(coord[2])))
        mol.AddConformer(conf)

        # Convert to regular molecule first
        mol = Chem.Mol(mol)

        # Try to perceive bonds using distance-based approach first
        try:
            # Use RDKit's built-in bond perception
            mol = Chem.SanitizeMol(mol)
        except Exception:
            # If sanitization fails, try manual bond perception
            mol = Chem.RWMol(mol)
            # Add bonds based on distance
            for i in range(len(atomic_numbers)):
                for j in range(i + 1, len(atomic_numbers)):
                    dist = np.linalg.norm(coordinates[i] - coordinates[j])
                    # Use more conservative distance thresholds
                    if dist < 1.8:  # Angstroms - typical single bond length
                        mol.AddBond(i, j, Chem.BondType.SINGLE)
            mol = Chem.Mol(mol)

        # Try to add hydrogens and sanitize
        try:
            mol = Chem.AddHs(mol)
            mol = Chem.SanitizeMol(mol)
        except Exception:
            # If adding hydrogens fails, try without
            try:
                mol = Chem.SanitizeMol(mol)
            except Exception:
                # If sanitization still fails, return None
                return None

        # Convert to SMILES
        smiles = Chem.MolToSmiles(mol)

        # Validate the SMILES
        if smiles and len(smiles) > 0:
            # Try to parse it back to make sure it's valid
            test_mol = Chem.MolFromSmiles(smiles)
            if test_mol is not None:
                return smiles

        return None

    except Exception:
        # # If RDKit method fails, try a simpler approach
        # try:
        #     # Create molecule with just atoms and basic connectivity
        #     mol = Chem.RWMol()
        #     for atomic_num in atomic_numbers:
        #         mol.AddAtom(Chem.Atom(int(atomic_num)))

        #     # Add basic bonds based on distance (very crude)
        #     for i in range(len(atomic_numbers)):
        #         for j in range(i + 1, len(atomic_numbers)):
        #             dist = np.linalg.norm(coordinates[i] - coordinates[j])
        #             # Simple distance-based bonding (this is very approximate)
        #             if dist < 2.0:  # Angstroms
        #                 mol.AddBond(i, j, Chem.BondType.SINGLE)

        #     mol = Chem.Mol(mol)
        #     smiles = Chem.MolToSmiles(mol)

        #     if smiles and len(smiles) > 0:
        #         return smiles

        # except Exception:
        #     pass

        return None


def compare_formulas_between_datasets(train_csv_path, val_csv_path, output_file=None):
    """
    Compare formulas between training and validation datasets.
    Find training samples with formulas that don't appear in validation set.

    Args:
        train_csv_path: Path to training dataset metadata parquet file
        val_csv_path: Path to validation dataset metadata parquet file
        output_file: Output file for unique training indices (auto-generated if None)

    Returns:
        tuple: (unique_indices, common_formulas, unique_formulas)
    """
    print(f"Loading training data from {train_csv_path}")
    df_train = pd.read_parquet(train_csv_path)

    print(f"Loading validation data from {val_csv_path}")
    df_val = pd.read_parquet(val_csv_path)

    print(f"Training samples: {len(df_train)}")
    print(f"Validation samples: {len(df_val)}")

    # Get unique formulas from each dataset
    train_formulas = set(df_train["formula"].unique())
    val_formulas = set(df_val["formula"].unique())

    print(f"Unique formulas in training: {len(train_formulas)}")
    print(f"Unique formulas in validation: {len(val_formulas)}")

    # Find common and unique formulas
    common_formulas = train_formulas.intersection(val_formulas)
    unique_to_train = train_formulas - val_formulas
    unique_to_val = val_formulas - train_formulas

    print(f"Common formulas: {len(common_formulas)}")
    print(f"Formulas unique to training: {len(unique_to_train)}")
    print(f"Formulas unique to validation: {len(unique_to_val)}")

    # Find training indices with formulas not in validation set
    unique_train_indices = df_train[df_train["formula"].isin(unique_to_train)][
        "index"
    ].tolist()

    print(f"Training samples with unique formulas: {len(unique_train_indices)}")

    # Save results
    if output_file is None:
        output_file = "unique_training_indices.parquet"

    # Create DataFrame with unique training samples
    df_unique_train = df_train[df_train["index"].isin(unique_train_indices)].copy()
    df_unique_train.to_parquet(output_file, index=False)
    print(f"Saved unique training samples to {output_file}")

    # Save summary statistics
    summary_file = output_file.replace(".parquet", "_summary.txt")
    with open(summary_file, "w") as f:
        f.write("Dataset Formula Comparison Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Training samples: {len(df_train)}\n")
        f.write(f"Validation samples: {len(df_val)}\n")
        f.write(f"Unique formulas in training: {len(train_formulas)}\n")
        f.write(f"Unique formulas in validation: {len(val_formulas)}\n")
        f.write(f"Common formulas: {len(common_formulas)}\n")
        f.write(f"Formulas unique to training: {len(unique_to_train)}\n")
        f.write(f"Formulas unique to validation: {len(unique_to_val)}\n")
        f.write(f"Training samples with unique formulas: {len(unique_train_indices)}\n")
        f.write(
            f"Percentage of training samples with unique formulas: {len(unique_train_indices) / len(df_train) * 100:.2f}%\n\n"
        )

        f.write("Top 10 most common formulas in training:\n")
        train_formula_counts = df_train["formula"].value_counts().head(10)
        for formula, count in train_formula_counts.items():
            f.write(f"  {formula}: {count}\n")

        f.write("\nTop 10 most common formulas in validation:\n")
        val_formula_counts = df_val["formula"].value_counts().head(10)
        for formula, count in val_formula_counts.items():
            f.write(f"  {formula}: {count}\n")

        f.write("\nTop 10 most common unique training formulas:\n")
        unique_train_formula_counts = df_unique_train["formula"].value_counts().head(10)
        for formula, count in unique_train_formula_counts.items():
            f.write(f"  {formula}: {count}\n")

    print(f"Saved summary to {summary_file}")

    return unique_train_indices, common_formulas, unique_to_train


def extract_dataset_metadata(lmdb_path, max_samples=None, output_file=None):
    """
    Extract dataset metadata including number of atoms, formula, and SMILES.

    Args:
        lmdb_path: Path to the LMDB dataset
        max_samples: Maximum number of samples to process (None for all)
        output_file: Output parquet file path (auto-generated if None)

    Returns:
        pandas.DataFrame: DataFrame with metadata
    """
    print(f"Loading dataset from {lmdb_path}")
    dataset = LmdbDataset(fix_dataset_path(lmdb_path))
    dataloader = TGDataLoader(dataset, batch_size=1, shuffle=False)

    if max_samples is not None:
        n_total_samples = min(max_samples, len(dataloader))
    else:
        n_total_samples = len(dataloader)

    print(f"Processing {n_total_samples} samples...")

    metadata_list = []

    for idx, batch in enumerate(dataloader):
        if idx >= n_total_samples:
            break

        # Extract basic information
        n_atoms = batch.pos.shape[0]

        # Get atomic numbers and convert to symbols
        atomic_numbers = batch.z.cpu().numpy()
        atomic_symbols = [Z_TO_ATOM_SYMBOL[z] for z in atomic_numbers]

        # Create molecular formula
        from collections import Counter

        symbol_counts = Counter(atomic_symbols)
        formula_parts = []
        for symbol in sorted(symbol_counts.keys()):
            count = symbol_counts[symbol]
            if count == 1:
                formula_parts.append(symbol)
            else:
                formula_parts.append(f"{symbol}{count}")
        formula = "".join(formula_parts)

        # Extract SMILES - first try from dataset, then generate with RDKit
        smiles = None

        # Try to get SMILES from dataset first
        if hasattr(batch, "smiles") and batch.smiles is not None:
            smiles = batch.smiles
        elif hasattr(batch, "y") and hasattr(batch.y, "smiles"):
            smiles = batch.y.smiles
        elif hasattr(batch, "data") and hasattr(batch.data, "smiles"):
            smiles = batch.data.smiles

        # If no SMILES found and RDKit is available, generate from coordinates
        if smiles is None and RDKIT_AVAILABLE:
            smiles = generate_smiles_from_coords(
                atomic_numbers=atomic_numbers,
                coordinates=batch.pos.cpu().numpy(),
                atomic_symbols=atomic_symbols,
            )

        # If still no SMILES, try a simple molecular formula-based approach
        if smiles is None:
            smiles = ""

        # Extract other potentially useful metadata
        energy = None
        if hasattr(batch, "ae") and batch.ae is not None:
            energy = batch.ae.item()
        elif hasattr(batch, "y") and hasattr(batch.y, "energy"):
            energy = batch.y.energy.item()

        # Store metadata
        metadata = {
            "index": idx,
            "natoms": n_atoms,
            "formula": formula,
            "smiles": smiles,
            "energy": energy,
            "atomic_symbols": " ".join(atomic_symbols),
            "atomic_numbers": " ".join(map(str, atomic_numbers)),
        }

        metadata_list.append(metadata)

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{n_total_samples} samples")

    # Create DataFrame
    df_metadata = pd.DataFrame(metadata_list)

    # Generate output filename if not provided
    if output_file is None:
        dataset_name = lmdb_path.split("/")[-1].split(".")[0]
        output_file = f"dataset_metadata_{dataset_name}.parquet"

    # Create output directory if it doesn't exist
    os.makedirs(
        os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
        exist_ok=True,
    )

    # Save to parquet
    df_metadata.to_parquet(output_file, index=False)
    print(f"Saved metadata to {output_file}")

    # Print summary statistics
    print(f"\nDataset Summary:")
    print(f"Total samples: {len(df_metadata)}")
    print(
        f"Number of atoms range: {df_metadata['natoms'].min()} - {df_metadata['natoms'].max()}"
    )
    print(f"Unique formulas: {df_metadata['formula'].nunique()}")
    print(f"Samples with SMILES: {df_metadata['smiles'].notna().sum()}")

    # Show most common formulas
    print(f"\nTop 10 most common formulas:")
    formula_counts = df_metadata["formula"].value_counts().head(10)
    for formula, count in formula_counts.items():
        print(f"  {formula}: {count}")

    return df_metadata


def main():
    """
    uv run scripts/extract_dataset_metadata.py --dataset ts1x-val.lmdb
    uv run scripts/extract_dataset_metadata.py --dataset ts1x_hess_train_big.lmdb
    uv run scripts/extract_dataset_metadata.py --dataset RGD1.lmdb

    # Compare formulas between datasets
    uv run scripts/extract_dataset_metadata.py --compare --train_csv dataset_metadata_ts1x_hess_train_big.parquet --val_csv dataset_metadata_ts1x-val.parquet
    """
    parser = argparse.ArgumentParser(
        description="Extract dataset metadata for split design"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        help="Dataset file name (e.g., ts1x-val.lmdb, ts1x_hess_train_big.lmdb, RGD1.lmdb)",
    )
    parser.add_argument(
        "--max_samples",
        "-m",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all samples)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output parquet file path (auto-generated if not provided)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare formulas between training and validation datasets",
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        help="Path to training dataset parquet file (for comparison)",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        help="Path to validation dataset parquet file (for comparison)",
    )

    args = parser.parse_args()

    if args.compare:
        if not args.train_csv or not args.val_csv:
            print("Error: --compare requires both --train_csv and --val_csv")
            return

        # Compare formulas between datasets
        unique_indices, common_formulas, unique_formulas = (
            compare_formulas_between_datasets(
                train_csv_path=args.train_csv,
                val_csv_path=args.val_csv,
                output_file=args.output,
            )
        )

        print(f"\nFormula comparison completed!")
        print(f"Unique training indices: {len(unique_indices)}")
        print(f"Common formulas: {len(common_formulas)}")
        print(f"Unique training formulas: {len(unique_formulas)}")

    else:
        if not args.dataset:
            print("Error: --dataset is required when not using --compare")
            return

        # Extract metadata
        df_metadata = extract_dataset_metadata(
            lmdb_path=args.dataset,
            max_samples=args.max_samples,
            output_file=args.output,
        )

        print(f"\nMetadata extraction completed!")
        print(f"DataFrame shape: {df_metadata.shape}")
        print(f"Columns: {list(df_metadata.columns)}")


if __name__ == "__main__":
    main()
