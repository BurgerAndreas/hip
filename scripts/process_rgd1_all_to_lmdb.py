#!/usr/bin/env python3
"""
Script to convert RGD1 dataset from HDF5 format to LMDB format for faster training.

Converts the RGD1 dataset (RGD1_CHNO.h5, RGD1_RPs.h5, RGD1CHNO_AMsmiles.csv, RandP_smiles.txt)
into a single LMDB file that can be used with the existing LmdbDataset classes.

The script processes all reactions and creates PyTorch Geometric data objects with:
- Transition state geometries and energies
- Reactant and product geometries and energies
- Atom-mapped SMILES
- Activation energies and reaction enthalpies
- Individual molecule data (DFT and xTB)

Usage:
    python scripts/create_rgd1_lmdb.py --output-path data/rgd1.lmdb --method all_ts
"""

import argparse
import os
import pickle
import lmdb
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import Data
from ocpmodels.units import hartree_to_ev

from hip.rgd1_load_raw import collect_raw_reaction_data


def convert_reaction_to_pyg_data(reaction_data):
    """
    Convert a single reaction dictionary to PyTorch Geometric Data object.

    Args:
        reaction_data (dict): Reaction data from collect_raw_reaction_data()

    Returns:
        torch_geometric.data.Data: PyTorch Geometric data object
    """
    # Convert elements to atomic numbers
    element_to_z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
    atomic_numbers = torch.tensor(
        [element_to_z[el] for el in reaction_data["elements"]], dtype=torch.long
    )

    # Create PyTorch Geometric data object
    data = Data()

    # Basic structure data
    data.atomic_numbers = atomic_numbers
    data.pos = torch.tensor(reaction_data["TS_geometry"], dtype=torch.float32)
    data.natoms = len(atomic_numbers)

    # Energies (convert from Hartree to eV: 1 Hartree = 27.2114 eV)
    data.energy = torch.tensor(
        reaction_data["TS_E"] * hartree_to_ev, dtype=torch.float32
    )
    data.energy_reactant = torch.tensor(
        reaction_data["R_E"] * hartree_to_ev, dtype=torch.float32
    )
    data.energy_product = torch.tensor(
        reaction_data["P_E"] * hartree_to_ev, dtype=torch.float32
    )

    # Enthalpies
    data.enthalpy_ts = torch.tensor(
        reaction_data["TS_H"] * hartree_to_ev, dtype=torch.float32
    )
    data.enthalpy_reactant = torch.tensor(
        reaction_data["R_H"] * hartree_to_ev, dtype=torch.float32
    )
    data.enthalpy_product = torch.tensor(
        reaction_data["P_H"] * hartree_to_ev, dtype=torch.float32
    )

    # Gibbs free energies
    data.gibbs_ts = torch.tensor(
        reaction_data["TS_F"] * hartree_to_ev, dtype=torch.float32
    )
    data.gibbs_reactant = torch.tensor(
        reaction_data["R_F"] * hartree_to_ev, dtype=torch.float32
    )
    data.gibbs_product = torch.tensor(
        reaction_data["P_F"] * hartree_to_ev, dtype=torch.float32
    )

    # Geometries
    data.pos_reactant = torch.tensor(reaction_data["R_geometry"], dtype=torch.float32)
    data.pos_product = torch.tensor(reaction_data["P_geometry"], dtype=torch.float32)

    # SMILES strings
    data.smiles_reactant = reaction_data["Rsmiles"]
    data.smiles_product = reaction_data["Psmiles"]

    # Reaction ID
    data.reaction_id = reaction_data["reaction_id"]

    # CSV data if available
    if "atom_mapped_reactant_smiles" in reaction_data:
        data.atom_mapped_reactant_smiles = reaction_data["atom_mapped_reactant_smiles"]
        data.atom_mapped_product_smiles = reaction_data["atom_mapped_product_smiles"]
        data.activation_energy_forward = torch.tensor(
            reaction_data["activation_energy_forward"], dtype=torch.float32
        )
        data.activation_energy_backward = torch.tensor(
            reaction_data["activation_energy_backward"], dtype=torch.float32
        )
        data.gibbs_energy_forward = torch.tensor(
            reaction_data["gibbs_energy_forward"], dtype=torch.float32
        )
        data.gibbs_energy_backward = torch.tensor(
            reaction_data["gibbs_energy_backward"], dtype=torch.float32
        )
        data.enthalpy_change = torch.tensor(
            reaction_data["enthalpy_change"], dtype=torch.float32
        )

    # Individual molecule data
    data.reactant_molecules = reaction_data["reactant_molecules"]
    data.product_molecules = reaction_data["product_molecules"]

    return data


def create_rgd1_lmdb(
    output_path, raw_data_dir="rgd1_raw", method="all_ts", map_size_gb=10
):
    """
    Create LMDB dataset from RGD1 raw data.

    Args:
        output_path (str): Path to output LMDB file
        raw_data_dir (str): Directory containing RGD1 raw data files
        method (str): Method for handling multiple TS per reaction ('unique_ts', 'first_ts', 'all_ts')
        map_size_gb (int): LMDB map size in GB
    """
    print(f"Processing RGD1 dataset with method: {method}")
    print(f"Raw data directory: {raw_data_dir}")
    print(f"Output LMDB path: {output_path}")

    # Process RGD1 reactions
    print("\nProcessing RGD1 reactions")
    processed_reactions = collect_raw_reaction_data(
        raw_data_dir=raw_data_dir, method=method
    )

    if not processed_reactions:
        raise RuntimeError("No reactions were successfully processed!")

    print(f"\nSuccessfully processed {len(processed_reactions)} reactions")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Clean up old database files if they exist
    if os.path.exists(output_path):
        os.remove(output_path)
    if os.path.exists(f"{output_path}-lock"):
        os.remove(f"{output_path}-lock")

    # Open LMDB environment
    map_size = map_size_gb * 1024 * 1024 * 1024  # Convert GB to bytes
    env = lmdb.open(output_path, map_size=map_size, subdir=False)

    print("\nConverting to PyTorch Geometric format and writing to LMDB")
    num_samples_written = 0

    with env.begin(write=True) as txn:
        for idx, reaction_data in tqdm(
            enumerate(processed_reactions), total=len(processed_reactions)
        ):
            try:
                # Convert to PyTorch Geometric format
                pyg_data = convert_reaction_to_pyg_data(reaction_data)

                # Add sample ID
                pyg_data.id = f"rgd1_{idx}"

                # Serialize and write to LMDB
                txn.put(
                    f"{idx}".encode("ascii"),
                    pickle.dumps(pyg_data, protocol=pickle.HIGHEST_PROTOCOL),
                )
                num_samples_written += 1

            except Exception as e:
                print(
                    f"Error processing reaction {idx} ({reaction_data.get('reaction_id', 'unknown')}): {e}"
                )
                continue

        # Write metadata
        txn.put(
            "length".encode("ascii"),
            pickle.dumps(num_samples_written, protocol=pickle.HIGHEST_PROTOCOL),
        )

    env.close()

    print("\nSuccessfully created LMDB dataset!")
    print(f"Output path: {output_path}")
    print(f"Total samples written: {num_samples_written}")
    print(f"File size: {os.path.getsize(output_path) / (1024**3):.2f} GB")

    return output_path


def verify_lmdb_dataset(lmdb_path):
    """
    Verify the created LMDB dataset by loading a few samples.

    Args:
        lmdb_path (str): Path to LMDB file
    """
    print(f"\nVerifying LMDB dataset: {lmdb_path}")

    # Import here to avoid circular imports
    from hip.ff_lmdb import LmdbDataset

    try:
        dataset = LmdbDataset(lmdb_path)
        print(f"Dataset loaded successfully with {len(dataset)} samples")

        # Check first sample
        first_sample = dataset[0]
        print(f"First sample keys: {list(first_sample.keys())}")
        print(f"First sample reaction ID: {first_sample.reaction_id}")
        print(f"First sample atomic numbers shape: {first_sample.atomic_numbers.shape}")
        print(f"First sample positions shape: {first_sample.pos.shape}")
        print(f"First sample energy: {first_sample.energy.item():.4f} eV")
        print(
            f"First sample activation energy: {first_sample.activation_energy.item():.4f} eV"
        )

        # Check a few more samples
        for i in range(1, min(5, len(dataset))):
            sample = dataset[i]
            print(
                f"Sample {i} - Reaction ID: {sample.reaction_id}, Energy: {sample.energy.item():.4f} eV"
            )

        print("Dataset verification completed successfully!")

    except Exception as e:
        print(f"Error verifying dataset: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RGD1 dataset to LMDB format")
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/rgd1.lmdb",
        help="Path to output LMDB file (default: data/rgd1.lmdb)",
    )
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default="rgd1_raw",
        help="Directory containing RGD1 raw data files (default: rgd1_raw)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["unique_ts", "first_ts", "all_ts"],
        default="all_ts",
        help="Method for handling multiple TS per reaction (default: all_ts)",
    )
    parser.add_argument(
        "--map-size-gb", type=int, default=10, help="LMDB map size in GB (default: 10)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the created dataset after creation",
    )

    args = parser.parse_args()

    # Create LMDB dataset
    output_path = create_rgd1_lmdb(
        output_path=args.output_path,
        raw_data_dir=args.raw_data_dir,
        method=args.method,
        map_size_gb=args.map_size_gb,
    )

    # Verify if requested
    if args.verify:
        verify_lmdb_dataset(output_path)
