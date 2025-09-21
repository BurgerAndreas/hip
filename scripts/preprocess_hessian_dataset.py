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


def create_preprocessed_dataset(dataset_file="ts1x-val.lmdb", debug=False):
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    summary = []

    input_lmdb_path = fix_dataset_path(dataset_file)
    if debug:
        output_lmdb_path = input_lmdb_path.replace(".lmdb", "-hesspred-DEBUG.lmdb")
    else:
        output_lmdb_path = input_lmdb_path.replace(".lmdb", "-hesspred100.lmdb")

    # Clean up old database files if they exist
    successfully_removed = remove_dir_recursively(output_lmdb_path)
    remove_dir_recursively(f"{output_lmdb_path}-lock")
    if not successfully_removed:
        raise RuntimeError(f"Output database file {output_lmdb_path} already exists!")

    print(f"\nProcessing {input_lmdb_path} -> {output_lmdb_path}")

    # ---- Load dataset ----
    # Settings taken from EquiformerV2.yaml
    transform = HessianGraphTransform(
        cutoff=12.0, cutoff_hessian=100.0, max_neighbors=20, use_pbc=False
    )
    dataset = LmdbDataset(input_lmdb_path, transform=transform)
    print(f"Loaded dataset with {len(dataset)} samples from {input_lmdb_path}")

    # ---- Print keys of first sample ----
    first_sample = dataset[0]
    print("Keys in first sample (dataset):", list(first_sample.keys()))
    print("Shapes per key:")
    for key in first_sample.keys():
        print(key)
        print(f"{key}: {first_sample[key].shape}")

    # ---- Prepare output LMDB ----
    map_size = 10 * os.path.getsize(input_lmdb_path)  # generous size
    out_env = lmdb.open(output_lmdb_path, map_size=map_size, subdir=False)

    # ---- Main loop ----
    print("")
    num_samples_written = 0
    with out_env.begin(write=True) as txn:
        for sample_idx in tqdm(range(len(dataset)), total=len(dataset)):
            try:
                # Get the original sample
                original_sample = dataset[sample_idx]
                data_copy = copy.deepcopy(original_sample)

                # # Compute smallest eigenvalues and eigenvectors from DFT Hessian
                # dft_hessian = original_sample.hessian  # Shape should be [3*N * 3*N]

                # # Memory movement overhead is not worth it
                # # dft_hessian = dft_hessian.to(device)

                # n_atoms = original_sample.pos.shape[0]  # [N]
                # dft_hessian = dft_hessian.reshape(
                #     n_atoms * 3, n_atoms * 3
                # )  # [N*3, N*3]

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

    print("\nDataset processed.")
    for fname, n, outpath in summary:
        print(f"{fname}: {n} samples -> {outpath}")
    return summary


def test_dataset(dataset_file="ts1x-val-hesspred.lmdb"):
    input_lmdb_path = fix_dataset_path(dataset_file)
    input_lmdb_path = input_lmdb_path.replace(".lmdb", "-hesspred.lmdb")

    print(f"Testing {input_lmdb_path}")

    # ---- Load dataset ----
    dataset = LmdbDataset(input_lmdb_path)
    print(f"Loaded dataset with {len(dataset)} samples from {input_lmdb_path}")

    # ---- Print keys of first sample ----
    first_sample = dataset[0]
    print("Keys in first sample (dataset):", list(first_sample.keys()))
    print("Shapes per key:")
    for key in first_sample.keys():
        print(key)
        print(f"{key}: {first_sample[key].shape}")

    return


if __name__ == "__main__":
    """Try:
    python scripts/preprocess_hessian_dataset.py --dataset-file data/sample_100.lmdb

    python scripts/preprocess_hessian_dataset.py --dataset-file ts1x-val.lmdb
    python scripts/preprocess_hessian_dataset.py --dataset-file RGD1.lmdb
    python scripts/preprocess_hessian_dataset.py --dataset-file ts1x_hess_train_big.lmdb
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
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode (default: False)"
    )
    args = parser.parse_args()

    create_preprocessed_dataset(dataset_file=args.dataset_file, debug=args.debug)

    test_dataset(dataset_file=args.dataset_file)
