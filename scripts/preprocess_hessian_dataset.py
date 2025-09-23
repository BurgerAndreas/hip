"""
Load LMDB dataset, do HessianGraphTransform, save to new LMDB dataset.
"""

import argparse
import os
import pickle
import lmdb
import torch
import copy
from tqdm import tqdm
from hip.ff_lmdb import LmdbDataset
from hip.path_config import (
    DATASET_DIR_HORM_EIGEN,
    fix_dataset_path,
    remove_dir_recursively,
)
from ocpmodels.hessian_graph_transform import HessianGraphTransform
import numpy as np


def create_preprocessed_dataset(
    dataset_file,
    output_lmdb_path,
    cutoff=12.0,
    cutoff_hessian=100.0,
    max_neighbors=20,
    use_pbc=False,
):
    """
    Creates a new dataset with precomputed graph and indices for Hessian prediction.
    Saves the new dataset to a new file.
    Adds all keys of the old dataset to the new dataset.
    Ensures that the ordering (indexing) between the old and the new dataset is also the same.

    Args:
        save_hessian (bool): If True, saves the Hessian of the original dataset again.
            The Hessian is by far the largest part of the dataset.
            If False, removes the Hessian from the original dataset.
    """
    # ---- Config ----
    summary = []

    input_lmdb_path = fix_dataset_path(dataset_file)

    # Clean up old database files if they exist
    successfully_removed = remove_dir_recursively(output_lmdb_path)
    remove_dir_recursively(f"{output_lmdb_path}-lock")
    if not successfully_removed:
        raise RuntimeError(f"Output database file {output_lmdb_path} already exists!")

    print(f"\nProcessing {input_lmdb_path} -> {output_lmdb_path}")

    # ---- Load dataset ----
    # Settings taken from EquiformerV2.yaml in HORM
    transform = HessianGraphTransform(
        cutoff=cutoff,
        cutoff_hessian=cutoff_hessian,
        max_neighbors=max_neighbors,
        use_pbc=use_pbc,
    )
    dataset = LmdbDataset(input_lmdb_path, transform=transform)
    print(f"Loaded dataset with {len(dataset)} samples from {input_lmdb_path}")

    # ---- Print keys of first sample ----
    # first_sample = dataset[0]
    # print("Keys in first sample (dataset):", list(first_sample.keys()))
    # print("Shapes per key:")
    # for key in first_sample.keys():
    #     print(key)
    #     print(f"{key}: {first_sample[key].shape}")

    # ---- Prepare output LMDB ----
    map_size = 10 * os.path.getsize(input_lmdb_path)  # generous size
    out_env = lmdb.open(output_lmdb_path, map_size=map_size, subdir=False)

    # ---- Collect dataset statistics ----
    nedges = []
    nedges_hessian = []
    natoms = []

    # ---- Main loop ----
    print("")
    num_samples_written = 0
    with out_env.begin(write=True) as txn:
        for sample_idx in tqdm(range(len(dataset)), total=len(dataset)):
            try:
                # Get the original sample
                original_sample = dataset[sample_idx]
                data_copy = copy.deepcopy(original_sample)

                # save statistics
                nedges.append(data_copy.nedges)
                nedges_hessian.append(data_copy.nedges_hessian)
                natoms.append(data_copy.natoms)

                txn.put(
                    f"{sample_idx}".encode("ascii"),
                    pickle.dumps(data_copy, protocol=pickle.HIGHEST_PROTOCOL),
                )
                num_samples_written += 1

            except Exception as e:
                print(f"Error processing sample {sample_idx}: {e}", flush=True)
                exit()

        # end of loop
        txn.put(
            "length".encode("ascii"),
            pickle.dumps(num_samples_written, protocol=pickle.HIGHEST_PROTOCOL),
        )
    out_env.close()
    print(f"Done. New dataset written to {output_lmdb_path}")
    summary.append((dataset_file, len(dataset), output_lmdb_path))

    # ---- Print dataset statistics ----
    try:
        avg_nedges = np.mean(nedges)
        avg_nedges_hessian = np.mean(nedges_hessian)
        avg_natoms = np.mean(natoms)
    except:
        avg_nedges = torch.mean(torch.stack(nedges))
        avg_nedges_hessian = torch.mean(torch.stack(nedges_hessian))
        avg_natoms = torch.mean(torch.stack(natoms))
    print(f"Number of edges: {avg_nedges}")
    print(f"Number of edges (hessian): {avg_nedges_hessian}")
    print(f"Number of atoms: {avg_natoms}")
    # save to file
    with open(output_lmdb_path.replace(".lmdb", "_stats.txt"), "w") as f:
        f.write(f"Number of edges: {avg_nedges}\n")
        f.write(f"Number of edges (hessian): {avg_nedges_hessian}\n")
        f.write(f"Number of atoms: {avg_natoms}\n")

    print("\nDataset processed.")
    for fname, n, outpath in summary:
        print(f"{fname}: {n} samples -> {outpath}")
    return summary


def test_dataset(dataset_file):
    input_lmdb_path = fix_dataset_path(dataset_file)

    # ---- Load dataset ----
    dataset = LmdbDataset(input_lmdb_path)
    print(f"\nLoaded dataset with {len(dataset)} samples from {input_lmdb_path}")

    # ---- Print keys of first sample ----
    first_sample = dataset[0]
    print("Shapes per key:")
    for key in first_sample.keys():
        print(key)
        print(f"{key}: {first_sample[key].shape}")

    return


if __name__ == "__main__":
    """Try:
    uv run scripts/preprocess_hessian_dataset.py --dataset-file data/sample_100.lmdb

    uv run scripts/preprocess_hessian_dataset.py --dataset-file ts1x-val.lmdb
    uv run scripts/preprocess_hessian_dataset.py --dataset-file RGD1.lmdb
    uv run scripts/preprocess_hessian_dataset.py --dataset-file ts1x_hess_train_big.lmdb
    """
    parser = argparse.ArgumentParser(
        description="Create hesspred dataset with precomputed graph and indices for Hessian prediction"
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        default="ts1x-val.lmdb",
        help="Name of the dataset file to process (default: ts1x-val.lmdb)",
    )
    # Settings taken from EquiformerV2.yaml in HORM
    # cutoff=12.0, cutoff_hessian=100.0, max_neighbors=20, use_pbc=False
    parser.add_argument(
        "--r",
        type=float,
        default=5.0,
        help="Cutoff radius for the graph (default: 12.0)",
    )
    parser.add_argument(
        "--rh",
        type=float,
        default=100.0,
        help="Cutoff radius for the hessian graph (default: 100.0)",
    )
    parser.add_argument(
        "--maxn",
        type=int,
        default=32,
        help="Maximum number of neighbors for the graph (default: 20)",
    )
    parser.add_argument(
        "--pbc",
        action="store_true",
        help="Use periodic boundary conditions (default: False)",
    )
    args = parser.parse_args()

    suffix = f"r{int(args.r)}_rh{int(args.rh)}_maxn{int(args.maxn)}"
    if args.pbc:
        suffix += "_pbc"
    output_lmdb_path = args.dataset_file.replace(".lmdb", f"-{suffix}.lmdb")

    create_preprocessed_dataset(
        dataset_file=args.dataset_file,
        output_lmdb_path=output_lmdb_path,
        cutoff=args.r,
        cutoff_hessian=args.rh,
        max_neighbors=args.maxn,
        use_pbc=args.pbc,
    )

    # test_dataset(dataset_file=output_lmdb_path)
